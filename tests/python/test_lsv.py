import json
import unittest
from pathlib import Path
import rust_pricing as rp


class BergomiLsvSmokeTest(unittest.TestCase):
    def request(self, vega=False):
        data = json.loads(Path("fixtures/v1/pricing_request.golden.json").read_text())
        data["model"] = {"type": "local_volatility", "local_variance_grid": {
            "time_nodes": [0.0, 0.5, 1.0],
            "log_forward_moneyness_nodes": [-0.5, 0.0, 0.5],
            "shape": [3, 3], "values": [0.04] * 9, "floor": 1e-8, "cap": 4.0}}
        data["engine"] = {"type": "randomized_quasi_monte_carlo", "points_per_scramble": 64,
                          "scramble_count": 4, "master_scramble_seed": 91,
                          "variance_reduction": {"antithetic": True, "brownian_bridge": True}}
        data["risk"]["vega"] = vega
        return rp.PricingRequest.from_json(json.dumps(data))

    def compile(self, request, trace=True):
        return rp.BergomiLsvPlan.compile(
            request, mean_reversion=2.0, vol_of_vol=0.6, correlation=-0.5,
            particle_count=256, calibration_seed=42, log_bandwidth=0.35,
            minimum_effective_samples=5.0, retain_reverse_trace=trace,
            worker_threads=2, reduction_block_size=32)

    def test_wheel_price_calibration_and_risk(self):
        plan = self.compile(self.request())
        result = plan.evaluate_local_variance_risk()
        self.assertEqual(result.price.value, plan.evaluate().value)
        self.assertGreater(result.price.standard_error, 0.0)
        self.assertEqual(result.price.independent_sampling_units, 4)
        self.assertEqual(result.price.evaluated_paths, 512)
        self.assertEqual(len(result.node_adjoints), 9)
        self.assertEqual(len(result.standard_errors), 9)
        self.assertEqual(result.coordinate, "relative_dupire_variance_nodes_in_f")
        self.assertEqual(result.price.uncertainty_scope, "pricing_conditional_on_calibration")
        self.assertEqual(plan.plan_fingerprint, self.compile(self.request()).plan_fingerprint)
        self.assertEqual(len(plan.squared_leverage), 9)
        self.assertEqual(len(plan.extrapolated_moment_nodes), 3)
        with self.assertRaises(AttributeError):
            result.price.value = 1.0

    def test_unsupported_risk_and_missing_trace_are_explicit(self):
        with self.assertRaises(rp.PricingError):
            self.compile(self.request(vega=True))
        plan = self.compile(self.request(), trace=False)
        self.assertGreater(plan.evaluate().value, 0.0)
        with self.assertRaises(rp.PricingError):
            plan.evaluate_local_variance_risk()
