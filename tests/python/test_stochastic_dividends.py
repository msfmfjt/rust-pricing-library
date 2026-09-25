"""Public Buehler pricing, independent conditional Black oracle and API limits."""
import json
import math
import unittest
from pathlib import Path

import numpy as np
import rust_pricing as rp


def make_request(sigma=0.2, dividends=((1.4, 25.0),), strike=100.0, points=2048, risk=None):
    return rp.PricingRequest(
        "2026-09-04",
        rp.Product.european_vanilla(1, 2, "2027-09-04", strike, 1.0, "call"),
        rp.Market.equity(
            2, 1, 100.0,
            rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
            rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
            discrete_dividends=[rp.DividendEvent.fixed_cash(i+1, t, d)
                                for i, (t, d) in enumerate(dividends)],
        ), rp.Model.black_scholes(sigma),
        rp.Engine.randomized_quasi_monte_carlo(points, 612, scramble_count=8,
            antithetic=True, brownian_bridge=True), risk or rp.RiskRequest(),
    )


def make_lsv_request(dividends=((0.5, 6.0), (1.4, 3.0)), points=64):
    data = json.loads(Path("fixtures/v1/pricing_request.golden.json").read_text())
    data["model"] = {"type": "local_volatility", "local_variance_grid": {
        "time_nodes": [0.0, 0.5, 1.0],
        "log_forward_moneyness_nodes": [-0.5, 0.0, 0.5],
        "shape": [3, 3], "values": [0.04] * 9, "floor": 1e-8, "cap": 4.0}}
    data["market"]["discrete_dividends"] = [
        {"event_id": i + 1, "ex_time": t, "quote": {"type": "fixed_cash", "amount": d}}
        for i, (t, d) in enumerate(dividends)
    ]
    data["engine"] = {
        "type": "randomized_quasi_monte_carlo",
        "points_per_scramble": points,
        "scramble_count": 4,
        "master_scramble_seed": 91,
        "variance_reduction": {"antithetic": True, "brownian_bridge": True},
    }
    return rp.PricingRequest.from_json(json.dumps(data))


def compile_lsv(request=None, two_factor=False, worker_threads=2, **kwargs):
    common = dict(
        dividend_mean_reversion=0.7,
        equity_linkage=0.6,
        dividend_volatility=0.35,
        equity_dividend_correlation=-0.25,
        particle_count=256,
        calibration_seed=42,
        log_bandwidth=0.35,
        minimum_effective_samples=5.0,
        maximum_step=0.25,
        worker_threads=worker_threads,
        reduction_block_size=32,
    )
    if two_factor:
        return rp.StochasticDividendPlan.compile_bergomi_two_factor_lsv(
            request or make_lsv_request(),
            mean_reversions=[0.8, 2.1],
            vol_of_vol=0.3,
            mixing_weight=0.35,
            spot_correlations=[-0.4, -0.2],
            factor_correlation=0.3,
            dividend_volatility_correlations=[0.15, -0.1],
            **(common | kwargs),
        )
    return rp.StochasticDividendPlan.compile_bergomi_lsv(
        request or make_lsv_request(),
        mean_reversion=0.8,
        vol_of_vol=0.3,
        correlation=-0.4,
        dividend_volatility_correlation=0.15,
        **(common | kwargs),
    )


def compile_plan(request=None, **kwargs):
    return rp.StochasticDividendPlan.compile_bs(
        request or make_request(),
        **(dict(mean_reversion=0.0, equity_linkage=0.6, dividend_volatility=0.45,
                equity_dividend_correlation=-0.35, maximum_step=1.0,
                worker_threads=1, reduction_block_size=64) | kwargs),
    )


def conditional_black(order):
    # kappa=0: S_T=A*f_T+B*Y_T. Condition on dividend normal, then use
    # a univariate lognormal formula for residual equity. No library pricing,
    # forward, state transition or random number helpers are used here.
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    growth = 0.98/0.95
    a, b = (100-25/growth**1.4)*growth, 25*growth/growth**1.4
    sigma, nu, rho = 0.2, 0.45, -0.35
    root = sigma*math.sqrt(1-rho*rho)
    cdf = lambda x: 0.5*math.erfc(-x/math.sqrt(2))
    total = 0.0
    for z, weight in zip(nodes*math.sqrt(2), weights/math.sqrt(math.pi)):
        y = math.exp(-0.5*nu*nu+nu*z)
        forward = a*math.exp(-0.5*(sigma*rho)**2+sigma*rho*z)
        strike = 100-b*y
        if strike <= 0:
            call = forward-strike
        else:
            d1 = math.log(forward/strike)/root+0.5*root
            call = forward*cdf(d1)-strike*cdf(d1-root)
        total += weight*call
    return 0.95*total


class StochasticDividendTest(unittest.TestCase):
    def test_independent_price_oracle_and_uncertainty_metadata(self):
        expected = conditional_black(96)
        self.assertAlmostEqual(expected, conditional_black(128), delta=2e-7)
        p = compile_plan()
        result = p.evaluate()
        self.assertLess(abs(result.value-expected), 6*result.standard_error+0.002)
        self.assertEqual(result.independent_sampling_units, 8)
        self.assertEqual(result.evaluated_paths, 32768)
        self.assertEqual(result.uncertainty_scope, "pricing_only")
        self.assertEqual(result.scheme, "buehler-cash-positive-split-v1")
        self.assertEqual(result.plan_fingerprint, p.plan_fingerprint)
        self.assertEqual(p.random_factor_count, 2)
        self.assertEqual(p.time_nodes, [0.0, 1.0])
        self.assertAlmostEqual(p.risky_spot, 100-25/(0.98/0.95)**1.4, delta=3e-14)
        with self.assertRaises(AttributeError):
            result.value = 1.0
        with self.assertRaises(AttributeError):
            p.risky_spot = 1.0

    def test_fixed_cash_limit_and_expiry_dividend_are_discounted_once(self):
        request = make_request(sigma=0.0, dividends=((1.0, 10.0),), strike=80.0, points=32)
        p = compile_plan(request, mean_reversion=2.0, equity_linkage=0.0,
                         dividend_volatility=0.0, maximum_step=0.125)
        result = p.evaluate()
        self.assertAlmostEqual(result.value, 0.95*(100*0.98/0.95-10-80), delta=1e-13)
        self.assertAlmostEqual(result.standard_error, 0.0, delta=1e-13)

    def test_worker_replay_and_configuration_identity(self):
        request = make_request(dividends=((0.5, 6.0), (1.4, 3.0)), points=128)
        p = compile_plan(request, mean_reversion=0.7, maximum_step=0.125)
        a = p.evaluate()
        b = compile_plan(request, mean_reversion=0.7, maximum_step=0.125,
                         worker_threads=3).evaluate()
        self.assertEqual(a.value, b.value)
        self.assertEqual(a.standard_error, b.standard_error)
        self.assertEqual(a.value, p.evaluate().value)
        for name, value in [("mean_reversion", 0.8), ("equity_linkage", 0.7),
                            ("dividend_volatility", 0.5), ("equity_dividend_correlation", 0.1),
                            ("maximum_step", 0.0625)]:
            changed = compile_plan(request, **(dict(mean_reversion=0.7, maximum_step=0.125)
                                               | {name: value}))
            self.assertNotEqual(p.plan_fingerprint, changed.plan_fingerprint)

    def test_residual_lsv_price_surface_and_risk_boundary(self):
        p = compile_lsv()
        result = p.evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertGreater(result.standard_error, 0.0)
        self.assertEqual(result.uncertainty_scope, "pricing_conditional_on_calibration")
        self.assertEqual(result.scheme,
                         "buehler-bergomi-1f-residual-lsv-joint-ou-positive-split-v1")
        self.assertEqual(p.random_factor_count, 3)
        self.assertEqual(p.lsv_time_nodes, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(p.lsv_log_moneyness_nodes, [-0.5, 0.0, 0.5])
        self.assertEqual(len(p.lsv_squared_leverage), 15)
        self.assertTrue(all(x > 0.0 for x in p.lsv_squared_leverage))
        self.assertAlmostEqual(p.lsv_initial_residual_equity, p.risky_spot, delta=1e-13)

        # Calibration and pricing reductions are deterministic across worker counts.
        q = compile_lsv(worker_threads=3)
        self.assertEqual(q.evaluate().value, result.value)
        self.assertEqual(q.evaluate().standard_error, result.standard_error)

        # The existing fixed-parameter reverse/Gamma are not valid once leverage
        # is calibrated and therefore must never be silently reused.
        with self.assertRaises(rp.PricingError):
            p.evaluate_aad()
        with self.assertRaises(rp.PricingError):
            p.evaluate_gamma(gamma_relative_bump=0.01)

    def test_two_factor_residual_lsv_is_explicit(self):
        p = compile_lsv(two_factor=True)
        result = p.evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertEqual(result.scheme,
                         "buehler-bergomi-2f-residual-lsv-joint-ou-positive-split-v1")
        self.assertEqual(p.random_factor_count, 4)
        self.assertEqual(len(p.lsv_squared_leverage), 15)

    def test_validation_and_unsupported_risk_are_not_silently_ignored(self):
        for name, value in [("mean_reversion", -1.0), ("equity_linkage", 1.1),
                            ("dividend_volatility", math.nan),
                            ("equity_dividend_correlation", -1.1), ("worker_threads", 0)]:
            with self.subTest(name=name), self.assertRaises(rp.ValidationError):
                compile_plan(**{name: value})
        for step in [0.0, -1.0, math.nan, math.inf]:
            with self.assertRaises(rp.PricingError):
                compile_plan(maximum_step=step)
        with self.assertRaises(rp.PricingError):
            compile_plan(make_request(risk=rp.RiskRequest(delta=True)))
        self.assertTrue(hasattr(compile_plan(), "evaluate_aad"))


if __name__ == "__main__":
    unittest.main()
