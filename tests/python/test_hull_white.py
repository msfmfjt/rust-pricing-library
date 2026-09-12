import json
import math
from pathlib import Path
import unittest

import rust_pricing as rp


class HullWhiteTest(unittest.TestCase):
    def payload(self):
        data = json.loads(Path("fixtures/v1/pricing_request.golden.json").read_text())
        data["engine"] = {
            "type": "randomized_quasi_monte_carlo", "points_per_scramble": 64,
            "scramble_count": 4, "master_scramble_seed": 612,
            "variance_reduction": {"antithetic": True, "brownian_bridge": True},
        }
        return data

    def request(self, target=None):
        data = self.payload()
        if target is not None:
            data["model"] = {
                "type": "local_volatility", "local_variance_grid": {
                    "time_nodes": target.time_nodes,
                    "log_forward_moneyness_nodes": target.log_moneyness_nodes,
                    "shape": [3, 3], "values": [0.04] * 9, "floor": 1e-8, "cap": 4.0,
                },
            }
        return rp.PricingRequest.from_json(json.dumps(data))

    def compile_lsv(self, target, workers=1):
        return rp.HullWhiteEquityPlan.compile_lsv(
            self.request(target), target, rp.HullWhiteModel(0.2, [0.0], [0.005]),
            vol_mean_reversion=2.0, vol_of_vol=0.25,
            equity_vol_correlation=-0.5, equity_rate_correlation=0.25,
            vol_rate_correlation=-0.1, particle_count=256, calibration_seed=712,
            log_bandwidth=0.35, minimum_effective_samples=5.0,
            worker_threads=workers, reduction_block_size=32,
        )

    def test_rate_model_curve_fit_bond_parity_and_validation(self):
        curve = rp.DiscountCurve(10, [0.0, 1.0, 5.0], [1.0, 1.01, 0.9])
        rates = rp.HullWhiteModel(0.0, [0.0, 0.5], [0.01, 0.02])
        self.assertAlmostEqual(rates.bond_price(curve, 0.0, 1.0, 0.0), 1.01, places=14)
        call = rates.bond_option(curve, 1.0, 5.0, 0.9)
        put = rates.bond_option(curve, 1.0, 5.0, 0.9, is_call=False)
        self.assertAlmostEqual(call - put, 0.9 - 0.9 * 1.01, places=14)
        self.assertEqual(rates.mean_reversion, 0.0)
        copied = rates.volatilities
        copied[0] = 999.0
        self.assertEqual(rates.volatilities, [0.01, 0.02])
        with self.assertRaises(AttributeError):
            rates.mean_reversion = 1.0
        for a, t, v in [(-0.1, [0.0], [0.01]), (0.1, [0.1], [0.01]),
                        (0.1, [0.0], [-0.01]), (math.nan, [0.0], [0.01])]:
            with self.assertRaises(rp.ValidationError):
                rp.HullWhiteModel(a, t, v)
        with self.assertRaises(rp.PricingError):
            rates.bond_price(curve, 2.0, 1.0, 0.0)

    def test_bs_price_replay_and_unsupported_risk(self):
        rates = rp.HullWhiteModel(0.2, [0.0], [0.01])
        def compile_request(request):
            return rp.HullWhiteEquityPlan.compile_bs(
                request, rates, equity_rate_correlation=-0.3,
                maximum_step=0.2, worker_threads=2,
            )
        plan = compile_request(self.request())
        result = plan.evaluate()
        self.assertGreater(result.value, 0.0)
        self.assertGreater(result.standard_error, 0.0)
        self.assertEqual(result.independent_sampling_units, 4)
        self.assertEqual(result.evaluated_paths, 512)
        self.assertEqual(result.uncertainty_scope, "pricing_only")
        self.assertIsNone(result.calibration_method)
        self.assertIsNone(result.calibration_seed)
        self.assertIsNone(plan.squared_leverage)
        self.assertEqual(plan.plan_fingerprint, compile_request(self.request()).plan_fingerprint)
        with self.assertRaises(AttributeError):
            result.value = 0.0
        data = self.payload()
        data["risk"]["vega"] = True
        with self.assertRaises(rp.PricingError):
            compile_request(rp.PricingRequest.from_json(json.dumps(data)))

    def test_paired_targets_and_discounted_calibration(self):
        times, nodes = [0.0, 0.5, 1.0], [-0.5, 0.0, 0.5]
        target = rp.HullWhiteLsvTarget.flat(0.2, times, nodes)
        explicit = rp.HullWhiteLsvTarget.from_grid(target.model, target.forward_log_densities)
        self.assertEqual(explicit.forward_log_densities, target.forward_log_densities)
        essvi = rp.HullWhiteLsvTarget.from_essvi(
            [rp.EssviSlice(0.5, 0.02, 0.1, -0.03), rp.EssviSlice(1.0, 0.04, 0.2, -0.06)],
            0.04, times, nodes,
        )
        # Independent strike finite difference of the eSSVI Black call at T=1.
        def call(strike):
            k = math.log(strike)
            w = 0.5 * (0.04 - 0.06*k + math.sqrt(0.04**2 - 2*0.04*0.06*k + 0.2**2*k*k))
            root = math.sqrt(w)
            d1 = -k/root + 0.5*root
            cdf = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            return cdf(d1) - strike*cdf(d1-root)
        h = 1e-4
        density = (call(1.0+h) - 2*call(1.0) + call(1.0-h)) / h**2
        self.assertAlmostEqual(essvi.forward_log_densities[7], density, delta=2e-6)
        plan, replay = self.compile_lsv(target), self.compile_lsv(target, workers=3)
        result = plan.evaluate()
        self.assertEqual(result.value, replay.evaluate().value)
        self.assertEqual(result.standard_error, replay.evaluate().standard_error)
        self.assertEqual(result.uncertainty_scope, "pricing_conditional_on_calibration")
        self.assertEqual(result.calibration_seed, 712)
        self.assertEqual(len(plan.squared_leverage), 9)
        self.assertEqual(len(plan.minimum_effective_samples), 3)
        self.assertEqual(len(plan.fallback_nodes), 3)
        self.assertAlmostEqual(plan.calibration_discount_means[0], 1.0)
        self.assertAlmostEqual(plan.calibration_discounted_equity_means[0], 100.0)
        with self.assertRaises(rp.ValidationError):
            rp.HullWhiteLsvTarget.flat(0.2, times, nodes, cap=0.03)
        with self.assertRaises(rp.ValidationError):
            rp.HullWhiteLsvTarget.from_grid(target.model, [-1.0] * 9)
        with self.assertRaises(rp.ValidationError):
            rp.HullWhiteLsvTarget.from_grid(rp.Model.black_scholes(0.2), [1.0])
        # Model and density inputs must describe the same market target.
        different = rp.HullWhiteLsvTarget.flat(0.3, times, nodes)
        with self.assertRaises(rp.PricingError):
            self.compile_lsv(different)
