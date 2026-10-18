"""Independent shared-error fixtures for integrated decisions and replay."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import unittest
from unittest.mock import patch

import numpy as np

from rust_pricing_backtest import BacktestConfig, BacktestError, BacktestResult, Contract, Position, run_backtest, compare_strategies
from rust_pricing_backtest.domain import fingerprint, require
from rust_pricing_backtest.serialization import input_json, load_input
from rust_pricing_backtest.validation import validate_document
from rust_pricing_backtest.ensemble import ScenarioResults, aggregate_scenarios
from test_backtest import START, EXPIRY, FixtureProvider, snapshot

IDENTITY = 'a'*64


class JointFixture(FixtureProvider):
    identity = 'shared_error_fixture_v1'
    validation = dict(status='synthetic_fixture')

    def __init__(self, hedge_loading=.5):
        super().__init__({'target': (5., .5, (10.,20.,0.,0.)), 'hedge': (3., .2, (5.,10.,0.,0.))})
        self.hedge_loading = hedge_loading
        self.fail_at = None
        self.calls = []

    def evaluate(self, s, c):
        return replace(super().evaluate(s,c), reporting_iv=(.2,)*4, reporting_forwards=(s.spot.mid,)*2)

    def joint_market_risk(self, s, contracts, risks):
        self.calls.append(s.snapshot_id)
        require(s.snapshot_id != self.fail_at, 'injected_joint_failure', 'injected after the preceding order fill')
        ids = sorted(risks);q=s.basis.size
        # One shared unit-variance error: target loading 1, hedge loading 1/2.
        # Mean covariance = outer(loadings); 4 independent observations.
        loadings = np.concatenate([[0.]+[self.hedge_loading if cid=='hedge' else 1.]*q for cid in ids])
        covariance = np.outer(loadings,loadings)*4
        requests = {cid:json.loads(s.pricing_requests[cid]) for cid in ids}
        r=dict(document_kind='portfolio_market_risk',schema_version=1,status='passed',
            implementation_identity=IDENTITY,instrument_order=ids,independent_sampling_units=4,
            request_hashes={cid:fingerprint(p) for cid,p in requests.items()},quantities={cid:0. for cid in ids},
            coordinate_order=[dict(maturity=(d-s.at.date()).days/365.,log_moneyness=k) for d in s.basis.maturity_dates for k in s.basis.log_moneyness],
            instruments={cid:dict(price=risk.price,frozen_grid_delta=risk.delta,buckets=list(risk.buckets)) for cid,risk in risks.items()},
            price_and_market_sample_covariance=covariance.tolist(),market_mean_covariance=covariance.tolist(),
            net_buckets=[0.]*q,net_price=0.,net_price_and_market_mean_covariance=np.zeros((q+1,q+1)).tolist(),
            net_bucket_standard_errors=[0.]*q)
        r['report_hash']=fingerprint(r)
        return r


def fixture(**changes):
    history=[]
    for i in range(5):
        s=snapshot(i,prices={'hedge':3.})
        basis=replace(s.basis,maturity_dates=(START+timedelta(days=7),START+timedelta(days=21)))
        p=dict(valuation_date=s.at.date().isoformat(),market=dict(spot=s.spot.mid),
            model=dict(type='local_volatility',reporting_iv_basis=dict(maturity_nodes=[(d-s.at.date()).days/365. for d in basis.maturity_dates],
                log_forward_moneyness_nodes=list(basis.log_moneyness),implied_volatilities=[.2]*4)),engine={},risk={})
        history.append(replace(s,basis=basis,pricing_requests={cid:json.dumps(p) for cid in ('target','hedge')}))
    return dict(config=BacktestConfig(ridge=0.,max_joint_bucket_standard_error=1.1,**changes),history=history,
        contracts=[Contract('target',EXPIRY,100.),Contract('hedge',EXPIRY,105.)],portfolio=[Position('target',-1.,5.)],provider=JointFixture())


class JointReplayTests(unittest.TestCase):
    def setUp(self):
        for target,value in [('rust_pricing_backtest.portfolio_market_risk.implementation_identity',IDENTITY),
                             ('rust_pricing_backtest.portfolio_market_risk.spot_iv_response',(np.full(4,.002),{}))]:
            p=patch(target,return_value=value);p.start();self.addCleanup(p.stop)

    def test_integrated_gate_preserves_orders_ledger_and_includes_covariance(self):
        args=fixture();r=run_backtest(**args)
        old=run_backtest(**(args|{'config':replace(args['config'],max_joint_bucket_standard_error=None)}))
        self.assertEqual(r.ledger,old.ledger);self.assertEqual(r.orders,old.orders);self.assertEqual(r.fills,old.fills)
        held=r.risks[0]['joint_market']['held'];proposed=r.decisions[0]['joint_market']
        self.assertEqual(held['bucket_standard_errors'],[1.]*4)
        np.testing.assert_array_equal(proposed['bucket_standard_errors'],[0.]*4)
        self.assertEqual(proposed['quantities'],{'hedge':2.,'target':-1.})
        self.assertEqual(args['provider'].calls,['s0','s1','s2','s3'])
        self.assertEqual(r.greek_pnl[1]['market_iv']['status'],'available')
        self.assertEqual(r.greek_pnl[-1]['market_iv']['status'],'unavailable')
        self.assertEqual(r.summary['joint_market']['pnl_status'],'partial')
        for row in r.greek_pnl:
            self.assertAlmostEqual(row['market_iv']['reconciliation_error'],0.)
            self.assertAlmostEqual(row['market_iv']['cumulative_reconciliation_error'],0.)
        self.assertNotIn('max_joint_bucket_standard_error',old.manifest['config'])
        self.assertEqual(old.manifest['schema_version'],10)

    def test_proposed_error_can_exceed_limit_even_when_held_error_passes(self):
        args=fixture();args['provider']=JointFixture(hedge_loading=2.)
        r=run_backtest(**args,on_error='return')
        self.assertEqual(r.failure['code'],'joint_risk_quality')
        self.assertIn('3.0 exceeds 1.1',r.failure['message'])
        self.assertEqual((r.ledger,r.orders,r.fills,r.decisions),((),(),(),()))
        self.assertEqual(r.checkpoint['state']['next_index'],0)
        self.assertEqual(r.checkpoint['state']['cash'],5.)
        self.assertEqual(validate_document(json.loads(r.to_json()))['status'],'valid')

    def test_signed_multipliers_are_applied_once_to_joint_errors(self):
        args=fixture();args['config']=replace(args['config'],max_joint_bucket_standard_error=10.1)
        args['contracts']=[replace(c,multiplier=10. if c.contract_id=='target' else 2.) for c in args['contracts']]
        r=run_backtest(**args)
        self.assertEqual(r.risks[0]['joint_market']['held']['bucket_standard_errors'],[10.]*4)
        proposed=r.decisions[0]['joint_market']
        self.assertEqual(proposed['quantities'],{'hedge':20.,'target':-10.})
        np.testing.assert_array_equal(proposed['bucket_standard_errors'],[0.]*4)

    def test_resealed_report_cannot_substitute_different_decision_greeks(self):
        args=fixture();method=args['provider'].joint_market_risk
        def wrong(*values):
            r=method(*values);r.pop('report_hash');r['instruments']['target']['buckets'][0]+=1.
            r['report_hash']=fingerprint(r);return r
        args['provider'].joint_market_risk=wrong
        r=run_backtest(**args,on_error='return')
        self.assertEqual(r.failure['code'],'joint_risk_mismatch')
        self.assertEqual(r.orders,());self.assertEqual(r.ledger,())

    def test_failure_after_fills_rolls_back_and_repair_resumes_exactly(self):
        args=fixture();whole=run_backtest(**args);part=run_backtest(**args,stop_after=1)
        args['provider'].fail_at='s1'
        failed=run_backtest(**args,on_error='return')
        self.assertEqual(failed.failure['code'],'injected_joint_failure')
        self.assertEqual(failed.checkpoint,part.checkpoint)
        self.assertEqual(failed.fills,())
        args['provider'].fail_at=None
        resumed=run_backtest(**args,checkpoint=failed.checkpoint)
        self.assertEqual(resumed.to_json(),whole.to_json())
        changed=replace(args['config'],max_joint_bucket_standard_error=2.)
        with self.assertRaisesRegex(BacktestError,'checkpoint_mismatch'):
            run_backtest(**(args|{'config':changed}),checkpoint=part.checkpoint)
        corrupt=deepcopy(part.checkpoint)
        corrupt['state']['risks'][0]['joint_market']['report']['instruments']['target']['price']+=1
        corrupt['state_hash']=fingerprint(corrupt['state'])
        with self.assertRaisesRegex(BacktestError,'portfolio_report'):
            run_backtest(**args,checkpoint=corrupt)

    def test_versions_roundtrip_and_required_evidence(self):
        args=fixture();r=run_backtest(**args,stop_after=2)
        self.assertEqual(r.manifest['schema_version'],13);self.assertEqual(r.checkpoint['schema_version'],11)
        document=json.loads(input_json(r));self.assertEqual(document['schema_version'],9)
        for obj in [document,json.loads(r.to_json()),r.checkpoint]:
            self.assertEqual(validate_document(obj)['status'],'valid')
        self.assertEqual(BacktestResult.from_json(r.to_json()).to_json(),r.to_json())
        replay=run_backtest(**load_input(input_json(r)),provider=JointFixture(),stop_after=2)
        self.assertEqual(replay.to_json(),r.to_json())
        for key in ('joint_market',):
            bad=json.loads(r.to_json());del bad['risks'][0][key]
            with self.assertRaisesRegex(BacktestError,'schema_fields'):validate_document(bad)
        bad=json.loads(r.to_json());del bad['greek_pnl'][0]['market_iv']
        with self.assertRaisesRegex(BacktestError,'schema_fields'):validate_document(bad)
        document['schema_version']=8
        with self.assertRaisesRegex(BacktestError,'schema_fields'):load_input(json.dumps(document))

    def test_four_strategy_ensemble_and_fixed_shock_gate(self):
        args=fixture(market_shock_log_moneyness=(-.2,.2))
        results=compare_strategies(**args)
        for r in results.values():
            self.assertEqual(validate_document(json.loads(r.to_json()))['status'],'valid')
            self.assertEqual(r.summary['joint_market']['proposed_checks'],4)
        report=aggregate_scenarios([ScenarioResults('one','one','fixture',results)])
        self.assertEqual(report.manifest['schema_version'],5)
        self.assertEqual(validate_document(json.loads(report.to_json()))['status'],'valid')

    def test_limit_types_and_absent_provider_fail_before_entry(self):
        for value in (-1.,float('nan'),float('inf'),True,'1'):
            with self.assertRaises(BacktestError):BacktestConfig(max_joint_bucket_standard_error=value)
        args=fixture();args['provider']=FixtureProvider(args['provider'].values)
        with self.assertRaisesRegex(BacktestError,'joint_provider_scope'):run_backtest(**args)

    def test_missing_surface_is_explicit_and_does_not_change_prior_decisions(self):
        args=fixture();whole=run_backtest(**args)
        args['history'][2]=replace(args['history'][2],basis=replace(args['history'][2].basis,
            maturity_dates=(START+timedelta(days=8),START+timedelta(days=22))))
        p=args['provider'];evaluate=p.evaluate
        p.evaluate=lambda s,c: replace(evaluate(s,c),reporting_iv=None,reporting_forwards=None) if s.snapshot_id=='s2' else evaluate(s,c)
        r=run_backtest(**args)
        self.assertEqual(r.orders[:2],whole.orders[:2])
        row=r.greek_pnl[2]['market_iv'];self.assertEqual(row['status'],'unavailable')
        self.assertEqual(row['reason']['code'],'pnl_surface')
        self.assertEqual(row['available_greek_pnl'],0.)
        self.assertEqual(row['unexplained_pnl'],row['actual_asset_pnl'])


if __name__=='__main__':unittest.main()
