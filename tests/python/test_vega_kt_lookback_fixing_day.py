"""Independent fixed-observation oracles and fixing-day native Gamma replays."""
from copy import deepcopy
from dataclasses import replace
from datetime import date,datetime,timedelta,timezone
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backtesting'));sys.path.insert(0,str(ROOT/'scripts'))
from multi_step_cases import model
from test_vega_kt_path_ledger import fixture,provider
from rust_pricing_backtest import (PathContract,PathObservation,SettlementFixing,Snapshot,Quote,RiskBasis,
    BacktestConfig,Position,BacktestError,run_backtest,compare_strategies)
from rust_pricing_backtest.domain import data
from rust_pricing_backtest.path_history import native_product,resolve
from rust_pricing_backtest.serialization import input_json,load_input
from rust_pricing_backtest.validation import validate_document


def ordinary(p):
    p=deepcopy(p);p['risk'].pop('vega_kt',None);p['risk']['vega']=False
    return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate()


def bs_request(side,today,past=None,spot=100.,fully_fixed=False):
    start=date(2026,1,5);end=start+timedelta(days=365)
    dates=([str(start-timedelta(days=1))] if past is not None else [])+[str(start)]
    if not fully_fixed:dates.append(str(end))
    product=rp.Product.fixed_lookback(1,2,100.,1.,side,dates,str(end),
        historical_extremum=past,valuation_date_fixing=today)
    curve=rp.DiscountCurve(10,[0.,1.],[1.,1.]);dividend=rp.DiscountCurve(11,[0.,1.],[1.,1.])
    return rp.PricingRequest(str(start),product,rp.Market.equity(2,1,spot,curve,dividend),
        rp.Model.black_scholes(.2),rp.Engine.pseudo_monte_carlo(7,32768,antithetic=True),
        rp.RiskRequest(delta=True,gamma_relative_bump=.01,vega=True))


class LookbackFixingDayTests(unittest.TestCase):
    evidence={}

    @classmethod
    def setUpClass(cls):
        cls.provider=provider();cls.evidence=dict(oracles=[],panels=[],replays=[])

    def test_fixed_observation_matches_cash_plus_effective_strike_vanilla(self):
        # For a fixed maximum H: max(max(H,S)-K,0)=(H-K)+ +(S-max(H,K))+.
        # For a fixed minimum L: max(K-min(L,S),0)=(K-L)+ +(min(L,K)-S)+.
        for side,today,past in [('call',110.,None),('put',90.,None),
                                ('call',110.,120.),('put',90.,80.),
                                ('call',90.,None),('put',110.,None)]:
            request=bs_request(side,today,past);p=json.loads(request.to_json())
            self.assertEqual(p['product']['valuation_date_fixing'],today)
            self.assertEqual(rp.PricingRequest.from_json(request.to_json()).to_json(),request.to_json())
            actual=rp.PricingPlan.compile(request,worker_threads=4,reduction_block_size=256).evaluate()
            extremum=(max if side=='call' else min)([today]+([] if past is None else [past]))
            strike=(max if side=='call' else min)(100.,extremum)
            cash=max((extremum-100.)*(1 if side=='call' else -1),0.)
            q=deepcopy(p);q['product']=dict(type='european_vanilla',underlying_id=1,currency_id=2,
                expiry='2027-01-05',strike=strike,notional=1.,side=dict(type=side))
            expected=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(q)),worker_threads=4,reduction_block_size=256).evaluate()
            self.assertAlmostEqual(actual.value,cash+expected.value,places=11)
            for attr in ['delta_raw','gamma_raw','vega_raw']:
                self.assertAlmostEqual(getattr(actual,attr),getattr(expected,attr),places=11)
            cdf=lambda x:.5*math.erfc(-x/math.sqrt(2.))
            d1=(math.log(100./strike)+.02)/.2;d2=d1-.2
            call=100*cdf(d1)-strike*cdf(d2)
            analytic=cash+call+(strike-100 if side=='put' else 0.)
            self.assertLessEqual(abs(actual.value-analytic),6*actual.standard_error+1e-10)
            self.assertEqual(actual.diagnostics.lookback_valuation_date_fixing,today)
            self.assertEqual(actual.diagnostics.lookback_historical_extremum,past)
            self.assertEqual(rp.PricingResult.from_json(actual.to_json()).to_json(),actual.to_json())
            import jsonschema
            jsonschema.validate(p,json.loads(rp.request_json_schema()))
            jsonschema.validate(json.loads(actual.to_json()),json.loads(rp.result_json_schema()))
            self.evidence['oracles'].append(dict(side=side,today=today,past=past,effective_strike=strike,
                cash=cash,value=actual.value,analytic_value=analytic,standard_error=actual.standard_error,
                delta=actual.delta_raw,gamma=actual.gamma_raw,vega=actual.vega_raw,vanilla_identity=True))

    def test_fully_fixed_today_has_zero_risks_while_legacy_today_tracks_spot(self):
        for side,today,past in [('call',110.,None),('put',90.,None),('call',110.,120.),('put',90.,80.)]:
            for spot in [95.,105.]:
                request=bs_request(side,today,past,spot,fully_fixed=True)
                result=rp.PricingPlan.compile(request,worker_threads=4,reduction_block_size=256).evaluate()
                extrema=(max if side=='call' else min)([today]+([] if past is None else [past]))
                self.assertEqual(result.value,max((extrema-100)*(1 if side=='call' else -1),0.))
                self.assertEqual((result.standard_error,result.delta_raw,result.gamma_raw,result.vega_raw),(0.,0.,0.,0.))
        for side,spot,delta in [('call',110.,1.),('put',90.,-1.)]:
            p=json.loads(bs_request(side,None,None,spot,fully_fixed=True).to_json())
            self.assertNotIn('valuation_date_fixing',p['product'])
            result=ordinary(p)
            self.assertEqual((result.value,result.delta_raw,result.gamma_raw),(10.,delta,0.))

    def test_fixing_date_value_and_schema_validation(self):
        p=json.loads(bs_request('call',110.).to_json())
        for invalid in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(rp.ValidationError):bs_request('call',invalid)
        q=deepcopy(p);q['product']['monitoring_dates']=q['product']['monitoring_dates'][1:]
        with self.assertRaisesRegex(rp.ValidationError,'valuation-date fixing requires'):
            rp.PricingRequest.from_json(json.dumps(q))
        for version in [1,2]:
            q=deepcopy(p);q['schema_version']=version
            with self.assertRaisesRegex(rp.ValidationError,'not available in wire schema version'):
                rp.PricingRequest.from_json(json.dumps(q))

    def test_four_six_interval_today_only_and_past_histories_pass_gamma(self):
        for intervals in [4,6]:
            for side,today,past in [('call',110.,98.),('put',90.,102.)]:
                for has_past in [False,True]:
                    p=model(intervals,samples=4096);day=date.fromisoformat(p['valuation_date'])
                    dates=[day+timedelta(days=round(t*365)) for t in p['model']['local_variance_grid']['time_nodes'][1:]]
                    dates=([day-timedelta(days=1)] if has_past else [])+[day]+dates
                    c=PathContract('target',dates[-1],100.,side=side,product_type='fixed_lookback',
                        observations=tuple(PathObservation(d,f'f{i}') for i,d in enumerate(dates)))
                    at=datetime.combine(day,datetime.min.time(),timezone.utc)+timedelta(hours=16)
                    fixes={}
                    for i,d in enumerate(dates):
                        if d>day:continue
                        fixed_at=datetime.combine(d,datetime.min.time(),timezone.utc)+timedelta(hours=12)
                        fixes[f'f{i}']=SettlementFixing(d,today if d==day else past,fixed_at,fixed_at,'synthetic_path','r1')
                    b=p['risk']['vega_kt'];s=Snapshot(f'{intervals}-{side}-{has_past}',at,Quote(100.,100.,at,at),{},
                        RiskBasis(day,tuple(date.fromisoformat(d) for d in b['maturity_nodes']),tuple(b['log_forward_moneyness_nodes'])),
                        at,'synthetic_path',settlement_fixings=fixes)
                    p['product']=native_product(c,s);s=replace(s,pricing_requests={'target':json.dumps(p)})
                    risk=self.provider.evaluate(s,c);audit=self.provider.gamma_audit(s,c);raw=ordinary(p)
                    self.assertEqual(audit['status'],'passed')
                    self.assertEqual((risk.price,risk.delta),(raw.value,raw.delta_raw))
                    self.assertEqual(p['product']['valuation_date_fixing'],today)
                    self.assertEqual('historical_extremum' in p['product'],has_past)
                    self.evidence['panels'].append(dict(intervals=intervals,side=side,has_past=has_past,today=today,
                        risk=data(risk),audit=audit))
                    for change in ['wrong','missing']:
                        q=deepcopy(p)
                        if change=='wrong':q['product']['valuation_date_fixing']+=1.
                        else:q['product'].pop('valuation_date_fixing')
                        with self.assertRaisesRegex(BacktestError,'path_request_mismatch'):
                            self.provider.evaluate(replace(s,pricing_requests={'target':json.dumps(q)}),c)
                    print(f'Accepted fixing-day Gamma: {intervals} intervals, {side}, past={has_past}',flush=True)

    def test_fixing_day_ledgers_settle_and_resume_with_atomic_failure(self):
        for side in ['call','put']:
            c,h,history=fixture('fixed_lookback',side,on_fixing_days=True)
            cfg=BacktestConfig(ridge=1e-4,market_shock_log_moneyness=(-.2,0.,.2));positions=[Position('target',-1.,5.)]
            results=compare_strategies(cfg,history,[c,h],positions,self.provider);book={}
            for s in history[:-1]:
                s=resolve(s,[c,h],book);risk=self.provider.evaluate(s,c);audit=self.provider.gamma_audit(s,c)
                self.assertEqual(audit['status'],'passed')
                raw=ordinary(json.loads(s.pricing_requests['target']))
                self.assertEqual((risk.price,risk.delta),(raw.value,raw.delta_raw))
                self.evidence['panels'].append(dict(side=side,at=s.at.isoformat(),intervals=4,risk=data(risk),audit=audit))
            for strategy,whole in results.items():
                self.assertTrue(whole.summary['fully_settled'])
                target=next(x for x in whole.settlements if x['contract_id']=='target' and x['event'] in ['fixed','fixed_zero'])
                self.assertEqual(target['settlement']['payoff_per_unit'],10. if side=='call' else 6.)
                partial=run_backtest(replace(cfg,strategy=strategy),history,[c,h],positions,self.provider,stop_after=2)
                resumed=run_backtest(**load_input(input_json(whole)),provider=self.provider,checkpoint=partial.checkpoint)
                self.assertEqual(resumed.to_json(),whole.to_json())
                for doc in [json.loads(whole.to_json()),json.loads(input_json(whole)),whole.checkpoint]:
                    self.assertEqual(validate_document(doc)['status'],'valid')
                self.evidence['replays'].append(dict(side=side,strategy=strategy,summary=whole.summary,resume_exact=True))
            first=run_backtest(cfg,history,[c,h],positions,self.provider,stop_after=1)
            fixing=history[1].settlement_fixings['f1']
            for code,fixes in [('missing_path_fixing',{}),('future_settlement_fixing',{'f1':replace(fixing,available_at=history[1].at+timedelta(hours=1))}),
                    ('path_fixing_revision',{**history[1].settlement_fixings,'f0':replace(history[0].settlement_fixings['f0'],value=99.)})]:
                bad=replace(history[1],settlement_fixings=fixes)
                failed=run_backtest(cfg,[history[0],bad,*history[2:]],[c,h],positions,self.provider,on_error='return')
                self.assertEqual(failed.failure['code'],code);self.assertEqual(failed.checkpoint,first.checkpoint)
                repaired=run_backtest(cfg,history,[c,h],positions,self.provider,checkpoint=failed.checkpoint)
                self.assertEqual(repaired.to_json(),results['vega_kt'].to_json())
            if side=='call':
                self.evidence['sample_result']=json.loads(results['vega_kt'].to_json())
                self.evidence['sample_request']=json.loads(input_json(results['vega_kt']))
            print(f'Accepted {side} fixing-day ledger: four strategies settled and resumed',flush=True)


if __name__=='__main__':unittest.main()
