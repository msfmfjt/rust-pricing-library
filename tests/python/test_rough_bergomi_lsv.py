"""Deterministic-rate rough Bergomi LSV through the public Python plan."""
import unittest

import rust_pricing as rp
import test_lsv


def compile_rough(request, *, hurst=0.2, vol_of_vol=1.4, trace=True, workers=2):
    return rp.RoughBergomiLsvPlan.compile(
        request, hurst=hurst, vol_of_vol=vol_of_vol, correlation=-0.7,
        particle_count=256, calibration_seed=42, log_bandwidth=0.35,
        minimum_effective_samples=5.0, retain_reverse_trace=trace,
        worker_threads=workers, reduction_block_size=32)


class RoughBergomiLsvTest(unittest.TestCase):
    def setUp(self):
        self.request = test_lsv.BergomiLsvSmokeTest().request()

    def test_price_calibration_risk_and_replay(self):
        plan = compile_rough(self.request)
        result = plan.evaluate_local_variance_risk()
        self.assertEqual(result.price.value, plan.evaluate().value)
        self.assertEqual(result.price.scheme, "rough-bergomi-lsv-hybrid-kappa1-log-euler-v1")
        self.assertEqual(result.price.independent_sampling_units, 4)
        self.assertEqual(len(result.node_adjoints), 9)
        self.assertEqual(len(result.standard_errors), 9)
        self.assertEqual(len(plan.squared_leverage), 9)
        other = compile_rough(self.request, workers=3)
        self.assertEqual(other.evaluate_local_variance_risk().node_adjoints, result.node_adjoints)
        self.assertNotEqual(
            compile_rough(self.request, hurst=0.3).plan_fingerprint,
            compile_rough(self.request, hurst=0.2, workers=2).plan_fingerprint)

    def test_flat_target_is_repriced_near_black(self):
        # The LV target is flat 20%: a calibrated rough-LSV must reprice it
        # within Monte Carlo and particle error.
        lv = test_lsv.BergomiLsvSmokeTest().compile(self.request).evaluate()
        rough = compile_rough(self.request).evaluate()
        self.assertLess(abs(rough.value - lv.value),
                        5.0 * (rough.standard_error + lv.standard_error) + 0.02 * lv.value)

    def test_invalid_inputs_and_missing_trace_are_explicit(self):
        with self.assertRaises(rp.ValidationError):
            compile_rough(self.request, hurst=0.7)
        with self.assertRaises(rp.ValidationError):
            compile_rough(self.request, vol_of_vol=-1.0)
        plan = compile_rough(self.request, trace=False)
        self.assertGreater(plan.evaluate().value, 0.0)
        with self.assertRaises(rp.PricingError):
            plan.evaluate_local_variance_risk()


if __name__ == "__main__":
    unittest.main()
