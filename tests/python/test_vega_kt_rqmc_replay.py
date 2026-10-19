"""Actual RQMC path replay, sample covariance, rollback and exact resume."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from test_vega_kt_joint_replay import fixture as six_fixture
from rust_pricing_backtest import RustPricingProvider,run_backtest,BacktestError
from rust_pricing_backtest.domain import fingerprint
from rust_pricing_backtest.serialization import input_json,load_input
from rust_pricing_backtest.validation import validate_document
from rust_pricing_backtest.portfolio_market_risk import implementation_identity

OUT=ROOT/'benchmark-results/vega-kt-rqmc-provider/native-replay'


def fixture():
    args=six_fixture();history=[]
    for snapshot in args['history']:
        requests={}
        for cid,raw in snapshot.pricing_requests.items():
            p=json.loads(raw);points=128 if p['product']['type']=='arithmetic_asian' else 256
            p['engine']=dict(type='randomized_quasi_monte_carlo',points_per_scramble=points,scramble_count=16,
                master_scramble_seed=100043,variance_reduction=dict(antithetic=True,brownian_bridge=False))
            requests[cid]=json.dumps(p)
        history.append(replace(snapshot,pricing_requests=requests))
    args['history']=history
    return args


def provider(workers=4):
    return RustPricingProvider(numerical_scope='multi_step_custom_basis_gamma',variance_basis_policy='coupled_row_hats',
                               sampling_policy='rqmc_path_16_scrambles',worker_threads=workers)


class RqmcReplayTests(unittest.TestCase):
    def test_actual_path_ledger_covariance_resume_and_failure(self):
        self.assertFalse(OUT.exists(),'fresh evidence directory required')
        args=fixture();p=provider();result=run_backtest(**args,provider=p,on_error='return');result.write(OUT/'api')
        self.assertEqual(result.status,'completed',result.failure)
        self.assertTrue(result.summary['fully_settled'])
        self.assertEqual(result.summary['risk_validation_status'],'runtime_checked_multi_step_rqmc')
        self.assertEqual(result.summary['joint_market']['covered_intervals'],1)
        self.assertEqual(result.greek_pnl[1]['market_iv']['status'],'available')
        for row in result.risks[:-1]:
            joint=row['joint_market']['report'];self.assertEqual(joint['independent_sampling_units'],16)
            self.assertEqual(joint['instrument_engines']['target']['points_per_scramble'],128)
            self.assertEqual(joint['instrument_engines']['hedge']['points_per_scramble'],256)
            np.testing.assert_allclose(np.array(joint['market_sample_covariance'])/16,joint['market_mean_covariance'],rtol=1e-12,atol=1e-12)
            self.assertLessEqual(row['joint_market']['held']['max_bucket_standard_error'],10.)
        partial=run_backtest(**args,provider=p,stop_after=1);partial.write(OUT/'partial')
        resumed=run_backtest(**load_input(input_json(result)),provider=provider(),checkpoint=partial.checkpoint)
        self.assertEqual(resumed.to_json(),result.to_json())
        for doc in (json.loads(result.to_json()),json.loads(input_json(result)),result.checkpoint):
            self.assertEqual(validate_document(doc)['status'],'valid')
        # A failed native audit may not release a risk or place a new order.
        rejected=provider()
        from rust_pricing_backtest.multi_step_extension import evaluate_extended_market_gamma
        def fail(*a,**kw):
            audit=evaluate_extended_market_gamma(*a,**kw)
            audit['status']='path_gamma_chain_failed';audit['estimate']=None
            return audit
        with patch('rust_pricing_backtest.multi_step_extension.evaluate_extended_market_gamma',side_effect=fail):
            failed=run_backtest(**args,provider=rejected,on_error='return')
        self.assertEqual(failed.failure['code'],'multi_step_gamma_acceptance_failure')
        self.assertFalse(failed.orders);self.assertFalse(failed.ledger);failed.write(OUT/'rejected')
        # Provider sampling policy must be bound into checkpoint identity.
        changed=RustPricingProvider(numerical_scope='multi_step_custom_basis_gamma',variance_basis_policy='coupled_row_hats',
                                    sampling_policy='rqmc_256x16',worker_threads=4)
        self.assertNotEqual(changed.identity,p.identity)
        with self.assertRaises(BacktestError):run_backtest(**args,provider=changed,checkpoint=partial.checkpoint)
        d=dict(status='passed',implementation_identity=implementation_identity(),
               native_sample_units=16,mixed_points_paired=True,resume_byte_identical=True,
               failed_native_audit_atomic=True,checkpoint_sampling_policy_bound=True,
               result_hash=fingerprint(json.loads(result.to_json())))
        d['report_hash']=fingerprint(d);(OUT/'report.json').write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':unittest.main()
