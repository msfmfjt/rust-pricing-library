"""Fixed-model Spot Delta and recalibrated local-variance-node AAD contracts."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import unittest

import rust_pricing as rp
from test_rough_volatility_families import models

ROOT = Path(__file__).resolve().parents[2]


def request(lsv=False, shift=0.0, node_shift=0.0, digital=False):
    data = json.loads((ROOT / "fixtures/v1/pricing_request.golden.json").read_text())
    data["market"]["spot"] += shift
    data["market"]["discrete_dividends"] = [
        {"event_id": 1, "ex_time": 0.25, "quote": {"type": "fixed_cash", "amount": 3.0}},
        {"event_id": 2, "ex_time": 1.5, "quote": {"type": "fixed_cash", "amount": 4.0}},
    ]
    data["engine"] = {"type": "randomized_quasi_monte_carlo", "points_per_scramble": 64,
                      "scramble_count": 4, "master_scramble_seed": 819,
                      "variance_reduction": {"antithetic": True, "brownian_bridge": True}}
    if lsv:
        data["model"] = {"type": "local_volatility", "local_variance_grid": {
            "time_nodes": [0.0, 0.25, 0.5, 0.75, 1.0],
            "log_forward_moneyness_nodes": [-0.6, -0.2, 0.13, 0.45, 0.8],
            "shape": [5, 5], "values": [0.04 + 0.002*(i % 5) + node_shift*math.cos(0.7*i)
                                       for i in range(25)], "floor": 1e-8, "cap": 4.0}}
    if digital:
        data["product"] = {"type": "digital", "underlying_id": 1, "currency_id": 2,
                           "expiry": "2027-09-04", "strike": 100.0, "payout": 1.0,
                           "side": {"type": "call"}, "payoff": {"type": "cash"}}
    return rp.PricingRequest.from_json(json.dumps(data))


def lsv_plan(model, shift=0.0, trace=True, workers=1):
    return rp.RoughFamilyLsvPlan.compile(
        request(True, node_shift=shift), model, particle_count=128,
        calibration_seed=429, log_bandwidth=0.5, minimum_effective_samples=3.0,
        retain_reverse_trace=trace, worker_threads=workers, reduction_block_size=64)


class RoughFamilyAadLsvTest(unittest.TestCase):
    def test_initial_state_vjp_all_models_and_seed_validation(self):
        for m in models():
            p = rp.RoughVolatilityPathPlan.compile(m, [0.0, 0.1, 0.4, 1.0])
            z = p.pseudo_shocks(17, 2)
            seeds = [0.2, -0.1, 0.5, 1.0]
            got = p.reverse_initial_forward(100.0, z, seeds)
            eps = 1e-3
            up = p.evolve_path(100.0+eps, z)
            dn = p.evolve_path(100.0-eps, z)
            expected = sum(s*(u-d)/(2*eps) for s, u, d in zip(seeds, up.forwards, dn.forwards))
            self.assertAlmostEqual(got, expected, delta=1e-8)
            with self.assertRaises(rp.PricingError):
                p.reverse_initial_forward(100.0, z, [1.0])
            with self.assertRaises(rp.PricingError):
                p.reverse_initial_forward(100.0, z, [math.nan]*4)

    def test_spot_delta_prices_cash_mapping_workers_and_frozen_properties(self):
        for m in models():
            def run(shift=0.0, workers=1):
                return rp.RoughVolatilityPlan.compile(request(shift=shift), m,
                    maximum_step=0.25, worker_threads=workers, reduction_block_size=64)
            p = run()
            got = p.evaluate_delta()
            self.assertEqual(got.price.value, p.evaluate().value)
            self.assertEqual(got.price.standard_error, p.evaluate().standard_error)
            self.assertEqual(got.coordinate, "physical_spot_fixed_model_fixed_cash_dividends")
            self.assertEqual(got.method, "rough-fixed-model-spot-delta-discrete-vjp-v1")
            self.assertGreaterEqual(got.delta_standard_error, 0.0)
            self.assertEqual(got.price.independent_sampling_units, 4)
            other = run(workers=3).evaluate_delta()
            self.assertEqual((got.delta, got.delta_standard_error),
                             (other.delta, other.delta_standard_error))
            for eps in [1e-4, 5e-5]:
                expected = (run(eps).evaluate().value-run(-eps).evaluate().value)/(2*eps)
                self.assertAlmostEqual(got.delta, expected, delta=2e-8)
            with self.assertRaises(AttributeError):
                got.delta = 0.0

    def test_lsv_all_families_full_recalibration_vjp_and_owned_arrays(self):
        for m in models():
            p = lsv_plan(m)
            r = p.evaluate_local_variance_risk()
            self.assertEqual(r.price.value, p.evaluate().value)
            self.assertEqual(r.coordinate, "relative_dupire_variance_nodes_in_f")
            self.assertEqual(r.price.uncertainty_scope, "pricing_conditional_on_calibration")
            self.assertEqual(len(r.node_adjoints), 25)
            self.assertEqual(len(r.standard_errors), 25)
            self.assertEqual(p.time_nodes[-1], 1.0)  # future cash stays in reserve, not time grid
            self.assertEqual(r.node_adjoints, lsv_plan(m, workers=3).evaluate_local_variance_risk().node_adjoints)
            direction = [math.cos(0.7*i) for i in range(25)]
            got = sum(a*d for a, d in zip(r.node_adjoints, direction))
            for eps in [1e-7, 5e-8]:
                expected = (lsv_plan(m, eps, False).evaluate().value -
                            lsv_plan(m, -eps, False).evaluate().value)/(2*eps)
                self.assertAlmostEqual(got, expected, delta=4e-6*(1+abs(expected)))
            copy = p.squared_leverage
            copy[0] = -1.0
            self.assertGreater(p.squared_leverage[0], 0.0)
            with self.assertRaises(AttributeError):
                p.squared_leverage = []
            with self.assertRaises(rp.PricingError):
                lsv_plan(m, trace=False).evaluate_local_variance_risk()

    def test_unsupported_nonlog_lsv_and_discontinuous_pure_risk(self):
        curve = rp.ForwardVarianceCurve.constant(0.04)
        for beta in [0.0, 0.5]:
            m = rp.RoughVolatilityModel.rough_sabr(hurst=0.2, vol_of_vol=0.3,
                correlation=-0.6, beta=beta, forward_variance=curve)
            with self.assertRaises(rp.PricingError):
                lsv_plan(m)
        # Price is allowed, unsmoothed pathwise derivative of a digital is not.
        product = rp.Product.digital(1, 2, "2027-09-04", 100.0, 1.0, "call", "cash")
        from test_rough_volatility_families import request as family_request
        p = rp.RoughVolatilityPlan.compile(family_request(product=product), models()[0],
            maximum_step=0.25, worker_threads=1)
        self.assertTrue(math.isfinite(p.evaluate().value))
        with self.assertRaises(rp.PricingError):
            p.evaluate_delta()

    def test_asian_delta_and_ssvi_derived_lsv_target(self):
        curve = rp.DiscountCurve(10, [0.0, 1.0, 2.0], [1.0, 0.97, 0.97**2])
        repo = rp.DiscountCurve(11, [0.0, 1.0, 2.0], [1.0, 0.99, 0.99**2])
        obs = [rp.AsianObservation.unknown(d, 1.0/3.0) for d in
               ["2026-12-04", "2027-03-04", "2027-09-04"]]
        product = rp.Product.arithmetic_asian(1, 2, 100.0, 1.0, "call", obs, "2027-12-04")
        engine = rp.Engine.randomized_quasi_monte_carlo(64, 12, scramble_count=4,
                                                        antithetic=True, brownian_bridge=True)
        def req(spot, carrier):
            market = rp.Market.equity(2, 1, spot, curve, repo,
                discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.5, 3.0),
                                   rp.DividendEvent.fixed_cash(2, 1.5, 4.0)])
            return rp.PricingRequest("2026-09-04", product, market, carrier, engine, rp.RiskRequest())
        for model in models():
            def pure(shift):
                return rp.RoughVolatilityPlan.compile(req(100.0+shift, rp.Model.black_scholes(0.2)),
                    model, maximum_step=0.25, worker_threads=1)
            delta = pure(0.0).evaluate_delta()
            self.assertEqual(delta.price.value, pure(0.0).evaluate().value)
            for e in [1e-4, 5e-5]:
                self.assertAlmostEqual(delta.delta,
                    (pure(e).evaluate().value-pure(-e).evaluate().value)/(2*e), delta=2e-8)
            times = [i/8 for i in range(9)]
            xs = [-0.6, -0.3, 0.0, 0.3, 0.6]
            target = rp.Model.local_volatility_from_standard_ssvi_power_law(
                [0.25, 0.5, 1.0], [0.01, 0.02, 0.04], 0.04,
                -0.4, 0.3, 0.5, times, xs, 1e-8, 4.0)
            p = rp.RoughFamilyLsvPlan.compile(req(100.0, target), model, particle_count=128,
                calibration_seed=429, log_bandwidth=0.5, minimum_effective_samples=3.0,
                retain_reverse_trace=True, worker_threads=1)
            risk = p.evaluate_local_variance_risk()
            self.assertEqual(risk.price.value, p.evaluate().value)
            self.assertEqual(len(risk.node_adjoints), len(times)*len(xs))
            self.assertTrue(all(math.isfinite(a) for a in risk.node_adjoints))
            self.assertEqual(risk.coordinate, "relative_dupire_variance_nodes_in_f")

    def test_added_stub_contract_members_cannot_be_removed(self):
        spec = importlib.util.spec_from_file_location("aad_wheel_contract", ROOT / "scripts/smoke_test_wheel.py")
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        text = (ROOT / "rust_pricing.pyi").read_text()
        check.exported_stub_api(text.encode())
        for cls_name, member in [("RoughVolatilityPlan", "evaluate_delta"),
                ("RoughVolatilityPathPlan", "reverse_initial_forward"),
                ("RoughFamilyLsvPlan", "evaluate_local_variance_risk"),
                ("RoughVolatilityDelta", "delta_standard_error")]:
            tree = ast.parse(text)
            cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls_name)
            cls.body = [c for c in cls.body if not isinstance(c, ast.FunctionDef) or c.name != member]
            with self.assertRaises(RuntimeError):
                check.verify_stub_static_shape(tree)


if __name__ == "__main__":
    unittest.main()
