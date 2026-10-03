"""Public extension API, independent path boundaries and payoff integration."""
import ast
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import unittest

import rust_pricing as rp


def models(stochastic=True, variance=0.04):
    eta = 0.3 if stochastic else 0.0
    curve = rp.ForwardVarianceCurve.constant(variance)
    heston = rp.RoughVolatilityModel.rough_heston(
        hurst=0.1, initial_variance=variance, mean_reversion=1.2,
        long_run_variance=variance, vol_of_vol=eta, correlation=-0.6)
    log_vol = math.log(math.sqrt(variance)) if variance else -350.0
    return [
        heston,
        rp.RoughVolatilityModel.lifted_heston_from_rough(heston, factors=12),
        rp.RoughVolatilityModel.quadratic_rough_heston(
            hurst=0.1, initial_state=0.1, mean_reversion=1.2, vol_of_vol=eta,
            quadratic=0.3 if stochastic else 0.0, shift=0.1, variance_floor=variance),
        rp.RoughVolatilityModel.mixed_rough_bergomi(
            hurst=0.1, correlation=-0.6, weights=[0.3, 0.7], vol_of_vols=[eta, 2*eta],
            forward_variance=curve),
        rp.RoughVolatilityModel.rough_sabr(
            hurst=0.1, vol_of_vol=eta, correlation=-0.6, beta=1.0, forward_variance=curve),
        rp.RoughVolatilityModel.rfsv(
            hurst=0.1, mean_reversion=1.2, vol_of_log_vol=eta/5,
            mean_log_vol=log_vol, initial_log_vol=log_vol),
    ]


def request(product=None, risk=None, qmc=True, cash=False, sigma=0.2, flat=False):
    product = product or rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call")
    engine = (rp.Engine.randomized_quasi_monte_carlo(
        64, 612, scramble_count=4, antithetic=True, brownian_bridge=True) if qmc
        else rp.Engine.pseudo_monte_carlo(612, 64, antithetic=True, brownian_bridge=True))
    ex_time = (dt.date(2027, 3, 5)-dt.date(2026, 9, 4)).days/365
    return rp.PricingRequest(
        "2026-09-04", product,
        rp.Market.equity(2, 1, 100.0,
            rp.DiscountCurve(10, [0.0, 1.0, 2.0], [1.0, 1.0 if flat else 0.95, 1.0 if flat else 0.95**2]),
            rp.DiscountCurve(11, [0.0, 1.0, 2.0], [1.0, 1.0 if flat else 0.98, 1.0 if flat else 0.98**2]),
            discrete_dividends=[rp.DividendEvent.fixed_cash(1, ex_time, 6.0)] if cash else []),
        rp.Model.black_scholes(sigma), engine, risk or rp.RiskRequest())


def plan(model, req=None, workers=1):
    return rp.RoughVolatilityPlan.compile(
        req or request(), model, maximum_step=0.25,
        worker_threads=workers, reduction_block_size=64)


class RoughVolatilityFamiliesTest(unittest.TestCase):
    def test_factories_paths_and_frozen_ownership(self):
        expected = ["rough_heston", "lifted_heston", "quadratic_rough_heston",
                    "mixed_rough_bergomi", "rough_sabr", "rfsv"]
        self.assertEqual([m.name for m in models()], expected)
        for model in models():
            with self.subTest(model=model.name):
                with self.assertRaises(AttributeError):
                    model.name = "other"
                p = rp.RoughVolatilityPathPlan.compile(model, [0.0, 0.13, 0.5, 1.0])
                z = p.pseudo_shocks(17, 2)
                self.assertEqual(len(z), p.random_dimension)
                self.assertEqual(z, p.pseudo_shocks(17, 2))
                path = p.evolve_path(100.0, z)
                self.assertEqual(len(path.forwards), 4)
                self.assertTrue(all(math.isfinite(v) and v >= 0 for v in path.variances))
                copy = path.variances
                copy[0] = -10.0
                self.assertNotEqual(path.variances[0], -10.0)
                with self.assertRaises(AttributeError):
                    path.variances = []
                altered = list(z)
                altered[-1] += 0.7
                self.assertEqual(path.forwards, p.evolve_path(100.0, altered).forwards)

    def test_covariance_and_one_cell_reference_fixture(self):
        refs = json.loads((Path(__file__).resolve().parents[2] /
                           "fixtures/rough-volatility/reference.json").read_text())
        for row in refs["heston_one_step"]:
            args = {key: row[key] for key in ["hurst", "initial_variance", "mean_reversion",
                                            "long_run_variance", "vol_of_vol", "correlation"]}
            model = rp.RoughVolatilityModel.rough_heston(**args)
            p = rp.RoughVolatilityPathPlan.compile(model, [0.0, row["dt"]])
            result = p.evolve_path(100.0, row["normals"])
            self.assertAlmostEqual(result.latent_states[1], row["raw_variance"], delta=3e-13)
            self.assertAlmostEqual(result.forwards[1], row["forward"], delta=2e-12)
        for row in refs["rfsv_correlations"]:
            model = rp.RoughVolatilityModel.rfsv(hurst=row["hurst"], mean_reversion=row["lag"]/0.1,
                                                vol_of_log_vol=0.1, mean_log_vol=-5.0)
            p = rp.RoughVolatilityPathPlan.compile(model, [0.0, 0.1])
            result = p.evolve_path(100.0, [0.0, 1.0, 0.0])
            correlation = (result.latent_states[1]+5)/(result.latent_states[0]+5)
            self.assertAlmostEqual(correlation, row["correlation"], delta=2e-10)

    def test_price_counts_replay_and_authoritative_model_level(self):
        for model in models():
            for qmc in [False, True]:
                with self.subTest(model=model.name, qmc=qmc):
                    r = request(qmc=qmc, cash=True)
                    p = plan(model, r)
                    a = p.evaluate()
                    b = plan(model, r, workers=3).evaluate()
                    self.assertEqual((a.value, a.standard_error), (b.value, b.standard_error))
                    self.assertEqual(a.independent_sampling_units, 4 if qmc else 64)
                    self.assertEqual(a.evaluated_paths, 512 if qmc else 128)
                    self.assertEqual(a.uncertainty_scope, "pricing_only")
                    self.assertEqual(a.cash_dividend_model, "escrowed-hw-bonds-v1")
                    self.assertLess(p.risky_spot, 100.0)
                    # sigma in the old JSON carrier is not a second model parameter.
                    other = plan(model, request(qmc=qmc, cash=True, sigma=0.8)).evaluate()
                    self.assertEqual((a.value, a.standard_error), (other.value, other.standard_error))

    def test_pre_dividend_monitoring_and_post_dividend_expiry(self):
        product = rp.Product.barrier(1, 2, "2027-09-04", 90.0, 97.0, 1.0, "call",
                                    "up", "knock_in", "discrete", ["2027-03-05"], "2027-09-04")
        r = request(product=product, cash=True, flat=True)
        # With no diffusion, pre-dividend spot is 100 and post-dividend spot is
        # 94. The pre-dividend observation hits 97; terminal intrinsic is 4.
        for model in models(False, variance=0.0):
            price = plan(model, r).evaluate()
            self.assertAlmostEqual(price.value, 4.0, delta=2e-12)
            self.assertLess(price.standard_error, 1e-12)

    def test_digital_payment_discounting_is_applied_once(self):
        products = [rp.Product.digital(1, 2, "2027-09-04", 100.0, 10.0, "call", "cash",
                                      payment_date=date) for date in ["2027-09-04", "2028-03-04"]]
        additional_time = (dt.date(2028, 3, 4)-dt.date(2027, 9, 4)).days/365
        for model in models():
            a = plan(model, request(product=products[0])).evaluate().value
            b = plan(model, request(product=products[1])).evaluate().value
            self.assertAlmostEqual(b, a * 0.95**additional_time, delta=2e-12)

    def test_explicit_unsupported_and_invalid_inputs(self):
        with self.assertRaises(rp.ValidationError):
            rp.ForwardVarianceCurve.constant(math.nan)
        with self.assertRaises(rp.ValidationError):
            rp.RoughVolatilityModel.rfsv(hurst=0.1, mean_reversion=0.0,
                                         vol_of_log_vol=0.1, mean_log_vol=-1.0)
        for model in models():
            with self.assertRaises(rp.PricingError):
                plan(model, request(risk=rp.RiskRequest(delta=True)))
            with self.assertRaises(rp.ValidationError):
                rp.RoughVolatilityPathPlan.compile(model, [0.0, 1.0, 0.5])
            p = rp.RoughVolatilityPathPlan.compile(model, [0.0, 1.0])
            with self.assertRaises(rp.PricingError):
                p.evolve_path(100.0, [math.nan] * p.random_dimension)
        heston = models()[0]
        with self.assertRaises(rp.PricingError):
            rp.RoughVolatilityPlan.compile(request(), heston,
                                            maximum_step=1e-12, worker_threads=1)


    def test_wheel_stub_contract_covers_and_rejects_new_api_drift(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "rough_wheel_contract", root / "scripts/smoke_test_wheel.py")
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        stub = (root / "rust_pricing.pyi").read_text()
        api = check.exported_stub_api(stub.encode())
        names = ("ForwardVarianceCurve", "RoughVolatilityModel",
                 "RoughVolatilityPath", "RoughVolatilityPathPlan", "RoughVolatilityPlan")
        for name in names:
            self.assertIn(name, api["symbols"])
            tree = ast.parse(stub)
            tree.body = [node for node in tree.body
                         if not isinstance(node, ast.ClassDef) or node.name != name]
            with self.subTest(missing_class=name), self.assertRaises(RuntimeError):
                check.verify_stub_static_shape(tree)
        tree = ast.parse(stub)
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "RoughVolatilityModel")
        factory = next(n for n in model.body if isinstance(n, ast.FunctionDef)
                       and n.name == "rfsv")
        factory.args.kw_defaults[-1] = ast.Constant(value=0.0)
        with self.assertRaisesRegex(RuntimeError, "signature changed"):
            check.verify_stub_static_shape(tree)
        tree = ast.parse(stub)
        curve = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "ForwardVarianceCurve")
        factory = next(n for n in curve.body if isinstance(n, ast.FunctionDef)
                       and n.name == "constant")
        factory.decorator_list = []
        with self.assertRaises(RuntimeError):
            check.verify_stub_static_shape(tree)


if __name__ == "__main__":
    unittest.main()
