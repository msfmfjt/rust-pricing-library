import json
import sys
import unittest
from pathlib import Path

import numpy as np
import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'backtesting'))
from multi_step_cases import model
from rust_pricing_backtest.multi_step_full_smile import (
    MultiStepSmileCalibrator, evaluate_multi_step_market_gamma,
)
from test_vega_kt_selection import remove_plan_fingerprints


class PathCovarianceTests(unittest.TestCase):
    def native(self, p, directions, workers=1, block=256):
        return rp.PricingPlan.compile(
            rp.PricingRequest.from_json(json.dumps(p)),
            worker_threads=workers, reduction_block_size=block,
        ).evaluate_vega_kt_multi_step_gamma_chain_audit(
            [1, 3], [80., 100., 120.], variance_risk_directions=directions,
            subdivisions=512, gamma_relative_standard_error_limit=.5,
        )

    def test_signed_dependent_zero_readouts_and_antithetic_units(self):
        for antithetic in [True, False]:
            p = model(4, 'asian', samples=1024)
            p['engine']['variance_reduction'].update(antithetic=antithetic, brownian_bridge=True)
            t, _ = MultiStepSmileCalibrator(p).all_maturity_iv_tangent()
            a, b = t.T.reshape(10, 2, 33)[[2, 7]]
            directions = np.array([a, b, a-2*b, -a, np.zeros_like(a)]).tolist()
            r = self.native(p, directions)
            self.assertEqual(r['sampling_unit'], 'antithetic_pair' if antithetic else 'path')
            cov = np.array(r['joint_sample_covariance']).reshape(15, 15)
            moments = r['native_directions'] + r['recovered_directions'] + r['direction_differences']
            np.testing.assert_allclose(cov.diagonal(), [x['sample_variance'] for x in moments], rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(cov.diagonal()/1024, [x['standard_error']**2 for x in moments], rtol=1e-12, atol=1e-12)
            for start in [0, 5, 10]:
                np.testing.assert_allclose(cov[start+2], cov[start]-2*cov[start+1], rtol=1e-10, atol=1e-8)
                np.testing.assert_allclose(cov[start+3], -cov[start], rtol=1e-10, atol=1e-8)
                np.testing.assert_array_equal(cov[start+4], 0.)
            np.testing.assert_allclose(cov[10:], cov[5:10]-cov[:5], rtol=1e-10, atol=1e-8)
            self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-9*np.linalg.norm(cov, 2))
            replay = self.native(p, directions, workers=4)
            self.assertEqual(remove_plan_fingerprints(r), remove_plan_fingerprints(replay))
            p['product']['notional'] = 2.5
            scaled = self.native(p, directions)
            np.testing.assert_allclose(scaled['joint_sample_covariance'], 6.25*cov.ravel(), rtol=1e-9, atol=1e-8)

    def test_market_covariance_layout_and_failed_estimate_suppression(self):
        for samples in [1024, 2]:
            p = model(4, 'asian', samples=samples, seed=1)
            r = evaluate_multi_step_market_gamma(p, [80., 100., 120.],
                    subdivisions=512, gamma_relative_standard_error_limit=.5)
            cov = np.array(r['gamma_audit']['joint_sample_covariance']).reshape(30, 30)[10:20, 10:20]
            if samples == 2:
                self.assertIsNone(r['estimate'])
                self.assertIsNone(r['market_sample_covariance'])
                self.assertIsNone(r['market_mean_covariance'])
                self.assertTrue(np.isfinite(cov).all())
            else:
                self.assertIsNotNone(r['estimate'])
                np.testing.assert_array_equal(r['market_sample_covariance'], cov)
                np.testing.assert_array_equal(r['market_mean_covariance'], cov/samples)
            json.dumps(r, allow_nan=False)

    def test_covariance_storage_budget_is_checked_before_sampling(self):
        p = model(4, samples=1024)
        # Existing moment components fit the cap; the 96x96 covariance exceeds it.
        with self.assertRaisesRegex(rp.PricingError, 'budget'):
            self.native(p, np.zeros((32, 2, 33)).tolist(), block=1)


if __name__ == '__main__':
    unittest.main()
