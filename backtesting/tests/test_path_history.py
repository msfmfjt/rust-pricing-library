"""Hand-computed path settlement and immutable fixing replay."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
import unittest
from rust_pricing_backtest import (PathContract, PathObservation, Contract, Position, SettlementFixing,
    BacktestConfig, BacktestError, run_backtest)
from rust_pricing_backtest.domain import data
from rust_pricing_backtest.path_history import native_product
from rust_pricing_backtest.serialization import input_json, load_input
from rust_pricing_backtest.validation import validate_document
from test_backtest import snapshot, START, FixtureProvider


def fixing(day,value):
    at=datetime.combine(START+timedelta(days=day),datetime.min.time(),timezone.utc)+timedelta(hours=12)
    return SettlementFixing(at.date(),value,at,at,'hand_checked','r1')


def fixture(kind='arithmetic_asian',side='call'):
    days=[-1,0,1,3];values=[98.,110.,94.,106.]
    weights=[.1,.2,.3,.4] if kind=='arithmetic_asian' else [1.]*4
    c=PathContract('target',START+timedelta(days=3),100.,side=side,multiplier=2.,
        product_type=kind,observations=tuple(PathObservation(START+timedelta(days=d),f'f{i}',w) for i,(d,w) in enumerate(zip(days,weights))))
    h=Contract('hedge',c.expiry,100.,lot=.1)
    history=[snapshot(i,prices={'hedge':5.},settlement_fixings={f'f{j}':fixing(d,v)
        for j,(d,v) in enumerate(zip(days,values)) if (i==0 and d<=0) or d==i}) for i in range(4)]
    provider=FixtureProvider({'target':(5.,.5,(10.,20.,0.,0.)),'hedge':(5.,.5,(10.,20.,0.,0.))})
    return c,h,history,provider


class PathHistoryTests(unittest.TestCase):
    def test_unused_path_candidates_do_not_require_missing_fixings(self):
        c,_,_,_=fixture()
        unused=replace(c,contract_id='unused')
        target=Contract('target',c.expiry,100.)
        provider=FixtureProvider({'target':(5.,.5,(10.,20.,0.,0.))})
        for strategy,prices in [('unhedged',{'unused':5.}),('delta',{'unused':5.}),
                                ('vega_kt',{}),('scalar_vega',{}),
                                ('vega_kt',{'unused':(1.,9.)})]:
            with self.subTest(strategy=strategy,prices=prices):
                cfg=BacktestConfig(strategy=strategy)
                history=[snapshot(i,prices=prices) for i in range(4)]
                args=dict(config=cfg,history=history,portfolio=[Position('target',-1.,5.)],provider=provider)
                expected=run_backtest(contracts=[target],**args)
                actual=run_backtest(contracts=[target,unused],**args)
                self.assertEqual(actual.ledger,expected.ledger)
                self.assertEqual(actual.decisions,expected.decisions)
                self.assertEqual(actual.checkpoint['state']['path_fixings'],{})

    def test_path_candidate_requires_history_when_first_evaluated(self):
        c,_,_,_=fixture()
        candidate=replace(c,contract_id='candidate')
        target=Contract('target',c.expiry,100.)
        # The provider does not inspect path history; the engine must enforce it.
        provider=FixtureProvider({'target':(5.,.5,(10.,20.,0.,0.)),
                                  'candidate':(5.,.5,(10.,20.,0.,0.))})
        history=[snapshot(0),snapshot(1,prices={'candidate':5.}),snapshot(2,prices={'candidate':5.})]
        args=dict(config=BacktestConfig(strategy='vega_kt'),contracts=[target,candidate],
                  portfolio=[Position('target',-1.,5.)],provider=provider)
        partial=run_backtest(history=history,stop_after=1,**args)
        failed=run_backtest(history=history,on_error='return',**args)
        self.assertEqual(failed.failure['code'],'missing_path_fixing')
        self.assertEqual(failed.checkpoint,partial.checkpoint)
        repaired=[history[0],replace(history[1],settlement_fixings={
            'f0':fixing(-1,98.),'f1':fixing(0,110.),'f2':fixing(1,94.)}),history[2]]
        resumed=run_backtest(history=repaired,checkpoint=failed.checkpoint,**args)
        self.assertEqual(resumed.to_json(),run_backtest(history=repaired,**args).to_json())
        self.assertIn('candidate',resumed.decisions[1]['candidates'])

    def test_available_candidate_history_is_retained_before_first_quote(self):
        c,_,history,_=fixture()
        candidate=replace(c,contract_id='candidate')
        target=Contract('target',c.expiry,100.)
        history=[replace(s,option_quotes=snapshot(i,prices={'candidate':5.} if i else {}).option_quotes)
                 for i,s in enumerate(history)]
        provider=FixtureProvider({'target':(5.,.5,(10.,20.,0.,0.)),
                                  'candidate':(5.,.5,(10.,20.,0.,0.))})
        args=dict(config=BacktestConfig(strategy='vega_kt',ridge=0),history=history,
                  contracts=[target,candidate],portfolio=[Position('target',-1.,5.)],provider=provider)
        partial=run_backtest(stop_after=1,**args)
        self.assertEqual(set(partial.checkpoint['state']['path_fixings']),{'f0','f1'})
        whole=run_backtest(**args)
        self.assertEqual(run_backtest(checkpoint=partial.checkpoint,**args).to_json(),whole.to_json())
        self.assertIn('candidate',whole.decisions[1]['candidates'])
        self.assertTrue(whole.summary['fully_settled'])

    def test_weighted_asian_and_call_put_extrema_settle_from_fixings_not_spot(self):
        for kind,side,payoff in [('arithmetic_asian','call',2.4),('arithmetic_asian','put',0.),
                                 ('fixed_lookback','call',10.),('fixed_lookback','put',6.)]:
            c,h,history,provider=fixture(kind,side)
            cfg=BacktestConfig(strategy='unhedged')
            whole=run_backtest(cfg,history,[c,h],[Position('target',-3.,5.)],provider)
            fixed=next(x for x in whole.settlements if x['event'] in ['fixed','fixed_zero'])
            self.assertAlmostEqual(fixed['settlement']['payoff_per_unit'],payoff)
            self.assertAlmostEqual(fixed['amount'],-6*payoff)
            self.assertEqual(len(fixed['settlement']['observations']),4)
            self.assertEqual(len(whole.checkpoint['state']['path_fixings']),4)
            self.assertEqual(whole.manifest['schema_version'],12)
            self.assertTrue(whole.summary['fully_settled'])
            for d in [json.loads(whole.to_json()),json.loads(input_json(whole)),whole.checkpoint]:
                self.assertEqual(validate_document(d)['status'],'valid')
            partial=run_backtest(cfg,history,[c,h],[Position('target',-3.,5.)],provider,stop_after=2)
            resumed=run_backtest(**load_input(input_json(whole)),provider=provider,checkpoint=partial.checkpoint)
            self.assertEqual(resumed.to_json(),whole.to_json())

    def test_missing_future_revised_or_wrong_date_fixing_rolls_back_and_repairs(self):
        c,h,history,provider=fixture();cfg=BacktestConfig(strategy='vega_kt',ridge=0)
        args=dict(config=cfg,contracts=[c,h],portfolio=[Position('target',-1.,5.)],provider=provider)
        partial=run_backtest(history=history,stop_after=1,**args)
        bads=[('missing_path_fixing',replace(history[1],settlement_fixings={})),
              ('path_fixing_revision',replace(history[1],settlement_fixings={**history[1].settlement_fixings,'f0':fixing(-1,99.)})),
              ('settlement_date_mismatch',replace(history[1],settlement_fixings={'f2':fixing(0,94.)})),
              ('future_settlement_fixing',replace(history[1],settlement_fixings={'f2':fixing(2,94.)}))]
        for code,bad in bads:
            failed=run_backtest(history=[history[0],bad,*history[2:]],on_error='return',**args)
            self.assertEqual(failed.failure['code'],code)
            self.assertEqual(failed.checkpoint,partial.checkpoint)
            repaired=run_backtest(history=history,checkpoint=failed.checkpoint,**args)
            self.assertEqual(repaired.to_json(),run_backtest(history=history,**args).to_json())
        # A vanilla with the same strike/expiry is a different payoff and remains eligible.
        self.assertIn('hedge',partial.decisions[0]['candidates'])

    def test_native_history_is_bound_to_contract_weights_dates_and_extrema(self):
        c,_,history,_=fixture()
        p=native_product(c,history[0])
        self.assertEqual([o['value']['type'] for o in p['observations']],['known','known','unknown','unknown'])
        self.assertEqual([o['weight'] for o in p['observations']],[.1,.2,.3,.4])
        c,_,history,_=fixture('fixed_lookback','put')
        today=native_product(c,history[0])
        self.assertEqual(today['historical_extremum'],98.)
        self.assertEqual(today['valuation_date_fixing'],110.)
        first_only=replace(c,observations=c.observations[1:])
        self.assertNotIn('historical_extremum',native_product(first_only,history[0]))
        self.assertEqual(native_product(first_only,history[0])['valuation_date_fixing'],110.)
        later=replace(history[2],settlement_fixings={'f0':fixing(-1,98.),'f1':fixing(0,110.),'f2':fixing(1,94.)})
        self.assertEqual(native_product(c,later)['historical_extremum'],94.)
        with self.assertRaisesRegex(BacktestError,'path_history_required'):c.payoff(100.)

    def test_path_contract_and_wire_reject_ambiguous_schedules(self):
        c,h,history,provider=fixture()
        for obs in [(),c.observations[::-1],c.observations[:-1],
                    (replace(c.observations[0],weight=.2),*c.observations[1:])]:
            with self.assertRaises(BacktestError):replace(c,observations=obs)
        with self.assertRaises(BacktestError):Contract('bad',c.expiry,100.,settlement_rule='path_observations')
        whole=run_backtest(BacktestConfig(strategy='unhedged'),history,[c,h],[Position('target',-1.,5.)],provider)
        old=json.loads(input_json(whole));old['schema_version']=7;old['config']['market_shock_log_moneyness']=[-.2,.2]
        with self.assertRaisesRegex(BacktestError,'schema_fields'):load_input(json.dumps(old))
        with self.assertRaises(BacktestError):validate_document(old)


if __name__=='__main__':unittest.main()
