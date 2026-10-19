"""Public all-row Gamma facade, immutable inputs and explicit failed estimates."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backtesting'));sys.path.insert(0,str(ROOT/'scripts'))
import rust_pricing as rp
from multi_step_cases import model
from rust_pricing_backtest.multi_step_extension import evaluate_extended_market_gamma,MultiStepExtensionCalibrator
from rust_pricing_backtest.domain import BacktestError
from test_vega_kt_selection import remove_plan_fingerprints


class ExtensionGammaTests(unittest.TestCase):
    def test_public_facade_covariance_workers_and_price_are_consistent(self):
        p=model(4,'asian',samples=4096);original=deepcopy(p)
        a=evaluate_extended_market_gamma(p,[80.,90.,100.,110.,120.],worker_threads=1)
        b=evaluate_extended_market_gamma(p,[80.,90.,100.,110.,120.],worker_threads=4)
        self.assertEqual(a['status'],'path_gamma_chain_passed');self.assertEqual(p,original)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b))
        self.assertEqual([r['variance_row'] for r in a['gamma_audit']['rows']],[0,1,2,3])
        self.assertEqual(a['extension_audit']['extension'],'all_rows')
        covariance=np.array(a['market_sample_covariance']);mean=np.array(a['market_mean_covariance'])
        np.testing.assert_array_equal(mean,covariance/4096)
        np.testing.assert_allclose(np.diag(mean),[v['standard_error']**2 for v in a['estimate']],atol=1e-12,rtol=1e-12)
        p['risk'].pop('vega_kt');p['risk']['vega']=False
        plain=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate()
        self.assertEqual(plain.value,a['gamma_audit']['native_price']['mean'])
        self.assertTrue(a['diagnostic_only']);self.assertEqual(a['full_market_iv_bucket_acceptance'],'not_established')
        json.dumps(a,allow_nan=False)

    def test_unaccepted_precision_and_rank_do_not_release_estimates(self):
        a=evaluate_extended_market_gamma(model(4,'lookback',samples=32),[80.,100.,120.],subdivisions=16,relative_tolerance=1e-10)
        self.assertNotEqual(a['status'],'path_gamma_chain_passed')
        for name in ['estimate','market_sample_covariance','market_mean_covariance']:self.assertIsNone(a[name])
        sparse=evaluate_extended_market_gamma(model(4,'asian',samples=2048),[80.,90.,100.,110.,120.],worker_threads=4)
        self.assertEqual(sparse['status'],'validation_failed')
        self.assertIsNone(sparse['estimate'])
        self.assertFalse(sparse['gamma_audit']['rows'][2]['conditional_gamma_validation_passed'])
        with self.assertRaisesRegex(BacktestError,'extension_rank'):
            evaluate_extended_market_gamma(model(4),[100.],extension='custom',variance_basis=np.zeros((132,132)))
        with self.assertRaisesRegex(rp.PricingError,'budget exceeded'):
            evaluate_extended_market_gamma(model(6,'asian',samples=4096),[80.,90.,100.,110.,120.])

    def test_custom_time_basis_preserves_the_preset_and_same_path_covariance(self):
        p=model(4,'lookback',samples=2048);c=MultiStepExtensionCalibrator(p,extension='linear_time')
        a=evaluate_extended_market_gamma(p,[90.,100.,110.],extension='linear_time',worker_threads=4)
        b=evaluate_extended_market_gamma(p,[90.,100.,110.],extension='custom',variance_basis=c.basis,worker_threads=4)
        self.assertEqual(a['status'],'path_gamma_chain_passed');self.assertEqual(b['status'],a['status'])
        for name in ['estimate','calibration_tangent','gamma_audit','market_sample_covariance','market_mean_covariance']:
            self.assertEqual(a[name],b[name])
        self.assertEqual(a['extension_audit']['basis_hash'],b['extension_audit']['basis_hash'])


if __name__=='__main__':unittest.main()
