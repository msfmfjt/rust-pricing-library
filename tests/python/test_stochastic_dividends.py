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


def make_lsv_request(
    dividends=((0.5, 6.0), (1.4, 3.0)),
    points=64,
    local_variances=None,
    spot=100.0,
    discount_factors=(1.0, 0.95),
    repo_spread_factors=(1.0, 0.98),
):
    data = json.loads(Path("fixtures/v1/pricing_request.golden.json").read_text())
    data["market"]["spot"] = spot
    data["market"]["discount_curve"]["discount_factors"] = list(discount_factors)
    data["market"]["dividend_curve"]["discount_factors"] = list(repo_spread_factors)
    data["model"] = {"type": "local_volatility", "local_variance_grid": {
        "time_nodes": [0.0, 0.5, 1.0],
        "log_forward_moneyness_nodes": [-0.5, 0.0, 0.5],
        "shape": [3, 3], "values": list(local_variances or [0.04] * 9),
        "floor": 1e-8, "cap": 4.0}}
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


def make_lsv_vegakt_request(points=32, full_bucket_covariance=True, pseudo=False):
    half = 182.0 / 365.0
    engine = (
        rp.Engine.pseudo_monte_carlo(
            91,
            points,
            antithetic=True,
            brownian_bridge=True,
        )
        if pseudo
        else rp.Engine.randomized_quasi_monte_carlo(
            points,
            91,
            scramble_count=4,
            antithetic=True,
            brownian_bridge=True,
        )
    )
    return rp.PricingRequest(
        "2026-09-04",
        rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
        rp.Market.equity(
            2,
            1,
            100.0,
            rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
            rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
            discrete_dividends=[
                rp.DividendEvent.fixed_cash(1, half, 6.0),
                rp.DividendEvent.fixed_cash(2, 0.8, 3.0),
            ],
        ),
        rp.Model.local_volatility_from_grid_with_reporting_basis(
            [0.0, half, 1.0],
            [-0.5, 0.0, 0.5],
            [0.04] * 9,
            1e-8,
            4.0,
            [half, 1.0],
            [-0.5, 0.0, 0.5],
            [0.2] * 6,
        ),
        engine,
        rp.RiskRequest(
            vega_kt_maturity_nodes=["2027-03-05", "2027-09-04"],
            vega_kt_log_forward_moneyness_nodes=[-0.5, 0.0, 0.5],
            vega_kt_relative_density_threshold=1e-8,
            vega_kt_full_bucket_covariance=full_bucket_covariance,
        ),
    )


def compile_lsv(request=None, two_factor=False, rough=False, worker_threads=2, **kwargs):
    common = dict(
        dividend_mean_reversion=0.7,
        equity_linkage=0.6,
        dividend_volatility=0.35,
        equity_dividend_correlation=-0.25,
        particle_count=256,
        calibration_seed=42,
        log_bandwidth=0.35,
        minimum_effective_samples=5.0,
        retain_reverse_trace=True,
        maximum_step=0.25,
        worker_threads=worker_threads,
        reduction_block_size=32,
    )
    if two_factor and rough:
        raise ValueError("choose at most one of two_factor or rough")
    if rough:
        params = (
            dict(
                hurst=0.1,
                vol_of_vol=0.6,
                correlation=-0.4,
                dividend_volatility_correlation=0.15,
            )
            | common
            | kwargs
        )
        return rp.StochasticDividendPlan.compile_rough_bergomi_lsv(
            request or make_lsv_request(),
            **params,
        )
    if two_factor:
        params = (
            dict(
                mean_reversions=[0.8, 2.1],
                vol_of_vol=0.3,
                mixing_weight=0.35,
                spot_correlations=[-0.4, -0.2],
                factor_correlation=0.3,
                dividend_volatility_correlations=[0.15, -0.1],
            )
            | common
            | kwargs
        )
        return rp.StochasticDividendPlan.compile_bergomi_two_factor_lsv(
            request or make_lsv_request(),
            **params,
        )
    params = (
        dict(
            mean_reversion=0.8,
            vol_of_vol=0.3,
            correlation=-0.4,
            dividend_volatility_correlation=0.15,
        )
        | common
        | kwargs
    )
    return rp.StochasticDividendPlan.compile_bergomi_lsv(
        request or make_lsv_request(),
        **params,
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
        q_risk = q.evaluate_local_variance_risk()
        p_risk = p.evaluate_local_variance_risk()
        self.assertEqual(q_risk.node_adjoints, p_risk.node_adjoints)
        self.assertEqual(q_risk.standard_errors, p_risk.standard_errors)

        q_spot = q.evaluate_lsv_spot_risk()
        p_spot = p.evaluate_lsv_spot_risk()
        self.assertEqual(q_spot.delta, p_spot.delta)
        self.assertEqual(q_spot.delta_standard_error, p_spot.delta_standard_error)

        # The existing fixed-parameter reverse/Gamma are not valid once leverage
        # is calibrated and therefore must never be silently reused.
        with self.assertRaises(rp.PricingError):
            p.evaluate_aad()
        with self.assertRaises(rp.PricingError):
            p.evaluate_gamma(gamma_relative_bump=0.01)
        with self.assertRaises(rp.PricingError):
            p.evaluate_vega_kt()

        local_risk = p.evaluate_local_variance_risk()
        self.assertAlmostEqual(local_risk.price.value, result.value, delta=2e-13)
        self.assertEqual(local_risk.time_nodes, [0.0, 0.5, 1.0])
        self.assertEqual(local_risk.log_moneyness_nodes, [-0.5, 0.0, 0.5])
        self.assertEqual(len(local_risk.node_adjoints), 9)
        self.assertTrue(all(math.isfinite(x) for x in local_risk.node_adjoints))
        self.assertIsNotNone(local_risk.standard_errors)
        self.assertEqual(len(local_risk.standard_errors), 9)
        self.assertEqual(
            local_risk.method,
            "buehler-residual-lsv-path-and-discrete-particle-vjp-v1",
        )
        self.assertEqual(
            local_risk.coordinate,
            "relative_dupire_variance_nodes_in_residual_equity",
        )

        spot_risk = p.evaluate_lsv_spot_risk()
        self.assertAlmostEqual(spot_risk.price.value, result.value, delta=2e-13)
        self.assertTrue(math.isfinite(spot_risk.delta))
        self.assertGreaterEqual(spot_risk.delta_standard_error, 0.0)
        self.assertEqual(
            spot_risk.method,
            "buehler-residual-lsv-scale-invariant-spot-reverse-v1",
        )
        self.assertEqual(
            spot_risk.coordinate,
            "physical_spot_with_residual_lsv_reanchoring",
        )

        # The reverse trace is an explicit memory/capability choice. Missing it
        # must fail before valuation rather than silently freeze leverage.
        with self.assertRaises(rp.PricingError):
            compile_lsv(retain_reverse_trace=False).evaluate_local_variance_risk()

        # Spot Delta uses scale invariance of the relative LSV calibration and
        # therefore does not require the particle calibration reverse trace.
        no_trace_spot = compile_lsv(retain_reverse_trace=False).evaluate_lsv_spot_risk()
        self.assertTrue(math.isfinite(no_trace_spot.delta))

    def test_residual_lsv_dividend_model_risk_keeps_calibration_fixed(self):
        request = make_lsv_request(points=16)
        common = dict(
            particle_count=64,
            reduction_block_size=16,
        )
        bumps = dict(
            dividend_mean_reversion_bump=0.02,
            equity_linkage_bump=0.02,
            dividend_volatility_bump=0.01,
            equity_dividend_correlation_bump=0.02,
        )
        plan = compile_lsv(request, worker_threads=1, **common)
        risk = plan.evaluate_lsv_dividend_model_risk(**bumps)

        self.assertEqual(
            risk.parameter_labels,
            [
                "dividend_mean_reversion",
                "equity_linkage",
                "dividend_volatility",
                "equity_dividend_correlation",
            ],
        )
        self.assertEqual(
            risk.parameter_bumps,
            [
                bumps["dividend_mean_reversion_bump"],
                bumps["equity_linkage_bump"],
                bumps["dividend_volatility_bump"],
                bumps["equity_dividend_correlation_bump"],
            ],
        )
        self.assertEqual(
            risk.method,
            "buehler-residual-lsv-common-noise-fixed-calibration-dividend-model-v1",
        )
        self.assertEqual(
            risk.coordinate,
            "buehler_dividend_model_parameters_with_fixed_residual_lsv_calibration",
        )
        self.assertTrue(all(math.isfinite(x) for x in risk.derivatives))
        self.assertTrue(all(x >= 0.0 for x in risk.standard_errors))
        self.assertAlmostEqual(risk.price.value, plan.evaluate().value, delta=2e-13)

        scenarios = [
            (
                dict(dividend_mean_reversion=0.7-bumps["dividend_mean_reversion_bump"]),
                dict(dividend_mean_reversion=0.7+bumps["dividend_mean_reversion_bump"]),
                bumps["dividend_mean_reversion_bump"],
            ),
            (
                dict(equity_linkage=0.6-bumps["equity_linkage_bump"]),
                dict(equity_linkage=0.6+bumps["equity_linkage_bump"]),
                bumps["equity_linkage_bump"],
            ),
            (
                dict(dividend_volatility=0.35-bumps["dividend_volatility_bump"]),
                dict(dividend_volatility=0.35+bumps["dividend_volatility_bump"]),
                bumps["dividend_volatility_bump"],
            ),
            (
                dict(
                    equity_dividend_correlation=-0.25
                    - bumps["equity_dividend_correlation_bump"]
                ),
                dict(
                    equity_dividend_correlation=-0.25
                    + bumps["equity_dividend_correlation_bump"]
                ),
                bumps["equity_dividend_correlation_bump"],
            ),
        ]
        finite_differences = []
        for down_kwargs, up_kwargs, bump in scenarios:
            down = compile_lsv(
                request,
                worker_threads=1,
                **(common | down_kwargs),
            )
            up = compile_lsv(
                request,
                worker_threads=1,
                **(common | up_kwargs),
            )
            self.assertEqual(down.lsv_squared_leverage, plan.lsv_squared_leverage)
            self.assertEqual(up.lsv_squared_leverage, plan.lsv_squared_leverage)
            finite_differences.append(
                (up.evaluate().value - down.evaluate().value) / (2.0 * bump)
            )

        for actual, expected in zip(risk.derivatives, finite_differences):
            self.assertAlmostEqual(actual, expected, delta=3e-10)

        parallel = compile_lsv(request, worker_threads=3, **common)
        parallel_risk = parallel.evaluate_lsv_dividend_model_risk(**bumps)
        self.assertEqual(parallel_risk.derivatives, risk.derivatives)
        self.assertEqual(parallel_risk.standard_errors, risk.standard_errors)

        no_trace = compile_lsv(
            request,
            worker_threads=2,
            retain_reverse_trace=False,
            **common,
        ).evaluate_lsv_dividend_model_risk(**bumps)
        self.assertTrue(all(math.isfinite(x) for x in no_trace.derivatives))

        two_factor = compile_lsv(
            request,
            two_factor=True,
            worker_threads=2,
            **common,
        ).evaluate_lsv_dividend_model_risk(**bumps)
        self.assertEqual(two_factor.parameter_labels, risk.parameter_labels)
        self.assertTrue(all(math.isfinite(x) for x in two_factor.derivatives))

        with self.assertRaises(rp.PricingError):
            compile_plan().evaluate_lsv_dividend_model_risk(**bumps)
        with self.assertRaises(rp.PricingError):
            plan.evaluate_lsv_dividend_model_risk(
                **(bumps | {"equity_linkage_bump": 0.7})
            )

    def test_residual_lsv_market_risk_matches_full_recompile(self):
        request = make_lsv_request(points=16)
        common = dict(
            particle_count=64,
            reduction_block_size=16,
        )
        plan = compile_lsv(request, worker_threads=1, **common)
        risk = plan.evaluate_lsv_market_risk()
        spot_risk = plan.evaluate_lsv_spot_risk()

        self.assertEqual(risk.delta, spot_risk.delta)
        self.assertEqual(risk.delta_standard_error, spot_risk.delta_standard_error)
        self.assertEqual(risk.cash_times, [0.5, 1.4])
        self.assertEqual(risk.discount_times, [0.0, 1.0])
        self.assertEqual(risk.repo_spread_times, [0.0, 1.0])
        self.assertEqual(
            risk.method,
            "buehler-residual-lsv-scale-invariant-market-reverse-v1",
        )
        self.assertEqual(
            risk.coordinate,
            "spot_cash_and_log_df_curves_with_residual_lsv_reanchoring",
        )
        self.assertAlmostEqual(risk.discount_log_df_adjoints[0], 0.0, delta=1e-14)
        self.assertAlmostEqual(risk.repo_spread_log_df_adjoints[0], 0.0, delta=1e-14)
        self.assertAlmostEqual(
            risk.discount_node_dv01[1],
            -1e-4 * risk.discount_log_df_adjoints[1],
            delta=1e-14,
        )
        self.assertAlmostEqual(
            risk.repo_spread_node_dv01[1],
            -1e-4 * risk.repo_spread_log_df_adjoints[1],
            delta=1e-14,
        )

        # Cash amounts only change funded residual equity and the affine physical
        # reconstruction. The relative LSV calibration is re-anchored, not refit
        # to a different shape.
        cash_bump = 1e-4
        for index in range(2):
            down_dividends = [(0.5, 6.0), (1.4, 3.0)]
            up_dividends = list(down_dividends)
            down_dividends[index] = (
                down_dividends[index][0],
                down_dividends[index][1] - cash_bump,
            )
            up_dividends[index] = (
                up_dividends[index][0],
                up_dividends[index][1] + cash_bump,
            )
            down = compile_lsv(
                make_lsv_request(points=16, dividends=tuple(down_dividends)),
                worker_threads=1,
                **common,
            )
            up = compile_lsv(
                make_lsv_request(points=16, dividends=tuple(up_dividends)),
                worker_threads=1,
                **common,
            )
            self.assertTrue(
                all(
                    abs(a - b) <= 3e-13
                    for a, b in zip(plan.lsv_squared_leverage, down.lsv_squared_leverage)
                )
            )
            self.assertTrue(
                all(
                    abs(a - b) <= 3e-13
                    for a, b in zip(plan.lsv_squared_leverage, up.lsv_squared_leverage)
                )
            )
            fd = (up.evaluate().value - down.evaluate().value) / (2.0 * cash_bump)
            self.assertAlmostEqual(risk.cash_mean_adjoints[index], fd, delta=3e-8)

        # Curve risk is raw dPrice/dlogDF. Full recompiles change carry, reserve
        # funding and payment discount while preserving relative leverage values.
        log_df_bump = 1e-5
        for field, base_df, actual in [
            ("discount", 0.95, risk.discount_log_df_adjoints[1]),
            ("repo", 0.98, risk.repo_spread_log_df_adjoints[1]),
        ]:
            down_df = base_df * math.exp(-log_df_bump)
            up_df = base_df * math.exp(log_df_bump)
            kwargs_down = {}
            kwargs_up = {}
            if field == "discount":
                kwargs_down["discount_factors"] = (1.0, down_df)
                kwargs_up["discount_factors"] = (1.0, up_df)
            else:
                kwargs_down["repo_spread_factors"] = (1.0, down_df)
                kwargs_up["repo_spread_factors"] = (1.0, up_df)
            down = compile_lsv(
                make_lsv_request(points=16, **kwargs_down),
                worker_threads=1,
                **common,
            )
            up = compile_lsv(
                make_lsv_request(points=16, **kwargs_up),
                worker_threads=1,
                **common,
            )
            self.assertTrue(
                all(
                    abs(a - b) <= 4e-13
                    for a, b in zip(plan.lsv_squared_leverage, down.lsv_squared_leverage)
                )
            )
            self.assertTrue(
                all(
                    abs(a - b) <= 4e-13
                    for a, b in zip(plan.lsv_squared_leverage, up.lsv_squared_leverage)
                )
            )
            fd = (up.evaluate().value - down.evaluate().value) / (2.0 * log_df_bump)
            self.assertAlmostEqual(actual, fd, delta=5e-8)

        parallel = compile_lsv(request, worker_threads=3, **common).evaluate_lsv_market_risk()
        self.assertEqual(parallel.delta, risk.delta)
        self.assertEqual(parallel.cash_mean_adjoints, risk.cash_mean_adjoints)
        self.assertEqual(parallel.discount_log_df_adjoints, risk.discount_log_df_adjoints)
        self.assertEqual(parallel.repo_spread_log_df_adjoints, risk.repo_spread_log_df_adjoints)
        self.assertEqual(parallel.cash_mean_standard_errors, risk.cash_mean_standard_errors)
        self.assertEqual(
            parallel.discount_log_df_standard_errors,
            risk.discount_log_df_standard_errors,
        )
        self.assertEqual(
            parallel.repo_spread_log_df_standard_errors,
            risk.repo_spread_log_df_standard_errors,
        )

        no_trace = compile_lsv(
            request,
            worker_threads=2,
            retain_reverse_trace=False,
            **common,
        ).evaluate_lsv_market_risk()
        self.assertTrue(math.isfinite(no_trace.delta))

        two_factor = compile_lsv(
            request,
            two_factor=True,
            worker_threads=2,
            **common,
        ).evaluate_lsv_market_risk()
        self.assertTrue(math.isfinite(two_factor.delta))
        self.assertEqual(len(two_factor.cash_mean_adjoints), 2)

        with self.assertRaises(rp.PricingError):
            compile_plan().evaluate_lsv_market_risk()

    def test_residual_lsv_gamma_reanchors_surface_and_matches_recompiled_deltas(self):
        request = make_lsv_request(points=16)
        common = dict(
            particle_count=64,
            reduction_block_size=16,
        )
        bump = 0.1
        plan = compile_lsv(request, worker_threads=1, **common)
        risk = plan.evaluate_lsv_gamma(gamma_absolute_bump=bump)
        delta = plan.evaluate_lsv_spot_risk()

        self.assertEqual(risk.delta, delta.delta)
        self.assertEqual(risk.delta_standard_error, delta.delta_standard_error)
        self.assertEqual(risk.spot_bumps, [0.5 * bump, bump, 2.0 * bump])
        self.assertEqual(
            risk.method,
            "buehler-residual-lsv-scale-invariant-aad-delta-gamma-v1",
        )
        self.assertTrue(all(math.isfinite(x) for x in risk.gamma_estimates))
        self.assertTrue(all(x >= 0.0 for x in risk.gamma_standard_errors))

        for actual, h in zip(risk.gamma_estimates, risk.spot_bumps):
            down = compile_lsv(
                make_lsv_request(points=16, spot=100.0 - h),
                worker_threads=1,
                **common,
            )
            up = compile_lsv(
                make_lsv_request(points=16, spot=100.0 + h),
                worker_threads=1,
                **common,
            )
            self.assertTrue(
                all(
                    abs(a - b) <= 3e-13
                    for a, b in zip(plan.lsv_squared_leverage, down.lsv_squared_leverage)
                )
            )
            self.assertTrue(
                all(
                    abs(a - b) <= 3e-13
                    for a, b in zip(plan.lsv_squared_leverage, up.lsv_squared_leverage)
                )
            )
            expected = (
                up.evaluate_lsv_spot_risk().delta
                - down.evaluate_lsv_spot_risk().delta
            ) / (2.0 * h)
            self.assertAlmostEqual(actual, expected, delta=3e-8)

        parallel = compile_lsv(request, worker_threads=3, **common)
        parallel_risk = parallel.evaluate_lsv_gamma(gamma_absolute_bump=bump)
        self.assertEqual(parallel_risk.delta, risk.delta)
        self.assertEqual(parallel_risk.gamma_estimates, risk.gamma_estimates)
        self.assertEqual(
            parallel_risk.gamma_standard_errors,
            risk.gamma_standard_errors,
        )
        self.assertEqual(
            parallel_risk.bump_difference_standard_errors,
            risk.bump_difference_standard_errors,
        )

        no_trace = compile_lsv(
            request,
            worker_threads=2,
            retain_reverse_trace=False,
            **common,
        ).evaluate_lsv_gamma(gamma_absolute_bump=bump)
        self.assertTrue(math.isfinite(no_trace.gamma))

        two_factor = compile_lsv(
            request,
            two_factor=True,
            worker_threads=2,
            **common,
        ).evaluate_lsv_gamma(gamma_relative_bump=0.001)
        self.assertTrue(math.isfinite(two_factor.gamma))

        with self.assertRaises(rp.PricingError):
            compile_plan().evaluate_lsv_gamma(gamma_absolute_bump=bump)
        with self.assertRaises(rp.ValidationError):
            plan.evaluate_lsv_gamma()
        with self.assertRaises(rp.ValidationError):
            plan.evaluate_lsv_gamma(
                gamma_absolute_bump=bump,
                gamma_relative_bump=0.001,
            )

    def test_residual_lsv_spot_reverse_matches_full_recompile_fd(self):
        h = 1.0e-3
        common = dict(
            particle_count=128,
            reduction_block_size=16,
            worker_threads=2,
        )
        base = compile_lsv(make_lsv_request(points=128, spot=100.0), **common)
        risk = base.evaluate_lsv_spot_risk()
        up = compile_lsv(make_lsv_request(points=128, spot=100.0 + h), **common)
        down = compile_lsv(make_lsv_request(points=128, spot=100.0 - h), **common)

        # Relative-coordinate particle calibration is homogeneous in F0. The
        # leverage values stay fixed while the surface anchor follows residual
        # equity, which is exactly the convention used by the analytic Delta.
        self.assertTrue(
            all(
                abs(a - b) <= 2e-13
                for a, b in zip(base.lsv_squared_leverage, up.lsv_squared_leverage)
            )
        )
        self.assertTrue(
            all(
                abs(a - b) <= 2e-13
                for a, b in zip(base.lsv_squared_leverage, down.lsv_squared_leverage)
            )
        )
        self.assertAlmostEqual(
            up.lsv_initial_residual_equity - down.lsv_initial_residual_equity,
            2.0 * h,
            delta=5e-13,
        )

        fd = (up.evaluate().value - down.evaluate().value) / (2.0 * h)
        tolerance = max(2.0e-4, 2.0e-3 * abs(fd))
        self.assertAlmostEqual(risk.delta, fd, delta=tolerance)

    def test_rough_residual_lsv_price_and_recalibrated_local_variance_risk(self):
        base_values = [0.04] * 9
        request = make_lsv_request(points=16, local_variances=base_values)
        common = dict(
            rough=True,
            particle_count=64,
            reduction_block_size=16,
        )
        plan = compile_lsv(request, worker_threads=1, **common)
        price = plan.evaluate()
        risk = plan.evaluate_local_variance_risk()

        self.assertTrue(math.isfinite(price.value))
        self.assertEqual(
            plan.scheme,
            "buehler-rough-bergomi-residual-lsv-joint-hybrid-positive-split-v1",
        )
        self.assertEqual(plan.random_factor_count, 4)
        self.assertEqual(len(plan.lsv_squared_leverage), 15)
        self.assertEqual(len(risk.node_adjoints), 9)
        self.assertTrue(all(math.isfinite(x) for x in risk.node_adjoints))
        self.assertEqual(
            risk.coordinate,
            "relative_dupire_variance_nodes_in_residual_equity",
        )

        index = 4
        bump = 1.0e-5
        plus = list(base_values)
        minus = list(base_values)
        plus[index] += bump
        minus[index] -= bump
        up = compile_lsv(
            make_lsv_request(points=16, local_variances=plus),
            worker_threads=1,
            **common,
        ).evaluate().value
        down = compile_lsv(
            make_lsv_request(points=16, local_variances=minus),
            worker_threads=1,
            **common,
        ).evaluate().value
        fd = (up - down) / (2.0 * bump)
        tolerance = max(8.0e-4, 8.0e-3 * abs(fd))
        self.assertAlmostEqual(risk.node_adjoints[index], fd, delta=tolerance)

        parallel = compile_lsv(request, worker_threads=3, **common)
        parallel_risk = parallel.evaluate_local_variance_risk()
        self.assertEqual(parallel.lsv_squared_leverage, plan.lsv_squared_leverage)
        self.assertEqual(parallel_risk.node_adjoints, risk.node_adjoints)
        self.assertEqual(parallel_risk.standard_errors, risk.standard_errors)

        spot = plan.evaluate_lsv_spot_risk()
        gamma = plan.evaluate_lsv_gamma(gamma_absolute_bump=0.1)
        market = plan.evaluate_lsv_market_risk()
        self.assertTrue(math.isfinite(spot.delta))
        self.assertTrue(math.isfinite(gamma.gamma))
        self.assertEqual(market.delta, spot.delta)

        with self.assertRaises(rp.PricingError):
            compile_lsv(
                request,
                retain_reverse_trace=False,
                worker_threads=1,
                **{k: v for k, v in common.items() if k != "rough"},
                rough=True,
            ).evaluate_local_variance_risk()

    def test_residual_lsv_local_variance_reverse_matches_recalibrated_fd(self):
        base_values = [0.04] * 9
        plan = compile_lsv(make_lsv_request(points=32, local_variances=base_values),
                           particle_count=128, reduction_block_size=16)
        risk = plan.evaluate_local_variance_risk()
        index = 4
        bump = 1.0e-5
        plus = list(base_values)
        minus = list(base_values)
        plus[index] += bump
        minus[index] -= bump
        up = compile_lsv(make_lsv_request(points=32, local_variances=plus),
                         particle_count=128, reduction_block_size=16).evaluate().value
        down = compile_lsv(make_lsv_request(points=32, local_variances=minus),
                           particle_count=128, reduction_block_size=16).evaluate().value
        fd = (up - down) / (2.0 * bump)
        tolerance = max(5.0e-4, 5.0e-3 * abs(fd))
        self.assertAlmostEqual(risk.node_adjoints[index], fd, delta=tolerance)

    def test_residual_lsv_vegakt_uses_recalibrated_target_risk(self):
        request = make_lsv_vegakt_request()
        p = compile_lsv(
            request,
            particle_count=128,
            reduction_block_size=16,
            worker_threads=1,
        )
        local_risk = p.evaluate_local_variance_risk()
        vega_kt = p.evaluate_vega_kt()

        self.assertEqual(len(vega_kt.coordinates), 6)
        self.assertEqual(len(vega_kt.raw_buckets), 6)
        self.assertTrue(all(math.isfinite(x) for x in vega_kt.raw_buckets))
        self.assertEqual(vega_kt.covariance_layout, "full_bucket_matrix_row_major")
        self.assertIsNotNone(vega_kt.full_bucket_covariance)
        self.assertEqual(len(vega_kt.full_bucket_covariance), 36)
        self.assertEqual(vega_kt.policy_label, "equation_11_first_order_v1")
        self.assertTrue(
            all(
                estimate.sample_variance is not None
                for estimate in vega_kt.estimates
            )
        )

        # The VegaKT pre-projection is the parallel Local-volatility derivative
        # of the original residual-equity target after the particle-calibration VJP.
        expected_parallel_vega = sum(
            2.0 * math.sqrt(0.04) * adjoint
            for adjoint in local_risk.node_adjoints
        )
        self.assertAlmostEqual(
            vega_kt.projection.pre_projection,
            expected_parallel_vega,
            delta=2e-11,
        )
        self.assertAlmostEqual(
            sum(vega_kt.raw_buckets) + vega_kt.projection.signed_residual,
            vega_kt.projection.pre_projection,
            delta=2e-11,
        )

        # Calibration, VJP, reporting projection and scramble covariance are
        # deterministic across worker counts.
        q = compile_lsv(
            request,
            particle_count=128,
            reduction_block_size=16,
            worker_threads=3,
        )
        q_vega_kt = q.evaluate_vega_kt()
        self.assertEqual(q_vega_kt.raw_buckets, vega_kt.raw_buckets)
        self.assertEqual(
            [estimate.sample_variance for estimate in q_vega_kt.estimates],
            [estimate.sample_variance for estimate in vega_kt.estimates],
        )
        self.assertEqual(
            q_vega_kt.full_bucket_covariance,
            vega_kt.full_bucket_covariance,
        )

        with self.assertRaises(rp.PricingError):
            compile_lsv(
                request,
                particle_count=128,
                reduction_block_size=16,
                retain_reverse_trace=False,
            ).evaluate_vega_kt()

    def test_residual_lsv_pseudo_vegakt_is_point_estimate(self):
        request = make_lsv_vegakt_request(
            points=32,
            full_bucket_covariance=False,
            pseudo=True,
        )
        vega_kt = compile_lsv(
            request,
            particle_count=64,
            reduction_block_size=16,
            worker_threads=2,
        ).evaluate_vega_kt()

        self.assertEqual(vega_kt.covariance_layout, "price_and_bucket_variance_only")
        self.assertIsNone(vega_kt.full_bucket_covariance)
        self.assertTrue(
            all(
                estimate.sample_variance is None
                and estimate.price_covariance is None
                for estimate in vega_kt.estimates
            )
        )
        self.assertAlmostEqual(
            sum(vega_kt.raw_buckets) + vega_kt.projection.signed_residual,
            vega_kt.projection.pre_projection,
            delta=2e-11,
        )

    def test_residual_lsv_bergomi_parameter_risk_recalibrates_full_model(self):
        request = make_lsv_request(points=32)
        common = dict(
            particle_count=64,
            reduction_block_size=16,
        )
        k_bump = 0.02
        nu_bump = 0.01
        plan = compile_lsv(request, worker_threads=1, **common)
        risk = plan.evaluate_lsv_bergomi_parameter_risk(
            mean_reversion_bump=k_bump,
            vol_of_vol_bump=nu_bump,
        )

        self.assertEqual(
            risk.parameter_labels,
            ["bergomi_mean_reversion", "bergomi_vol_of_vol"],
        )
        self.assertEqual(risk.parameter_bumps, [k_bump, nu_bump])
        self.assertEqual(
            risk.method,
            "buehler-residual-lsv-common-noise-full-recalibration-bergomi-1f-v1",
        )
        self.assertEqual(
            risk.coordinate,
            "one_factor_bergomi_parameters_with_full_residual_lsv_recalibration",
        )
        self.assertTrue(all(math.isfinite(x) for x in risk.derivatives))
        self.assertTrue(all(x >= 0.0 for x in risk.standard_errors))
        self.assertAlmostEqual(risk.price.value, plan.evaluate().value, delta=2e-13)

        # Independent full recompiles use the same calibration seed and valuation
        # random numbers. They must reproduce the dedicated common-noise risk.
        k_down = compile_lsv(
            request,
            worker_threads=1,
            mean_reversion=0.8 - k_bump,
            **common,
        ).evaluate()
        k_up = compile_lsv(
            request,
            worker_threads=1,
            mean_reversion=0.8 + k_bump,
            **common,
        ).evaluate()
        nu_down = compile_lsv(
            request,
            worker_threads=1,
            vol_of_vol=0.3 - nu_bump,
            **common,
        ).evaluate()
        nu_up = compile_lsv(
            request,
            worker_threads=1,
            vol_of_vol=0.3 + nu_bump,
            **common,
        ).evaluate()
        k_fd = (k_up.value - k_down.value) / (2.0 * k_bump)
        nu_fd = (nu_up.value - nu_down.value) / (2.0 * nu_bump)
        self.assertAlmostEqual(risk.mean_reversion_derivative, k_fd, delta=2e-10)
        self.assertAlmostEqual(risk.vol_of_vol_derivative, nu_fd, delta=2e-10)

        # Calibration and valuation reductions remain deterministic across worker
        # counts even though all four bumped particle calibrations are rerun.
        parallel = compile_lsv(request, worker_threads=3, **common)
        parallel_risk = parallel.evaluate_lsv_bergomi_parameter_risk(
            mean_reversion_bump=k_bump,
            vol_of_vol_bump=nu_bump,
        )
        self.assertEqual(parallel_risk.derivatives, risk.derivatives)
        self.assertEqual(parallel_risk.standard_errors, risk.standard_errors)

        # This is a forward full-recalibration risk, not a calibration VJP.
        no_trace = compile_lsv(
            request,
            worker_threads=2,
            retain_reverse_trace=False,
            **common,
        ).evaluate_lsv_bergomi_parameter_risk(
            mean_reversion_bump=k_bump,
            vol_of_vol_bump=nu_bump,
        )
        self.assertTrue(all(math.isfinite(x) for x in no_trace.derivatives))

        with self.assertRaises(rp.PricingError):
            compile_lsv(request, two_factor=True, **common).evaluate_lsv_bergomi_parameter_risk(
                mean_reversion_bump=k_bump,
                vol_of_vol_bump=nu_bump,
            )
        with self.assertRaises(rp.PricingError):
            plan.evaluate_lsv_bergomi_parameter_risk(
                mean_reversion_bump=1.0,
                vol_of_vol_bump=nu_bump,
            )

    def test_residual_lsv_correlation_risk_uses_selective_recalibration(self):
        request = make_lsv_request(points=32)
        common = dict(
            particle_count=64,
            reduction_block_size=16,
        )
        equity_vol_bump = 0.02
        dividend_vol_bump = 0.02
        plan = compile_lsv(request, worker_threads=1, **common)
        risk = plan.evaluate_lsv_correlation_risk(
            equity_volatility_correlation_bump=equity_vol_bump,
            dividend_volatility_correlation_bump=dividend_vol_bump,
        )

        self.assertEqual(
            risk.parameter_labels,
            ["equity_volatility_correlation", "dividend_volatility_correlation"],
        )
        self.assertEqual(risk.correlation_bumps, [equity_vol_bump, dividend_vol_bump])
        self.assertEqual(
            risk.method,
            "buehler-residual-lsv-common-noise-bergomi-1f-correlation-v1",
        )
        self.assertEqual(
            risk.coordinate,
            "one_factor_bergomi_correlations_with_selective_residual_lsv_recalibration",
        )
        self.assertTrue(all(math.isfinite(x) for x in risk.derivatives))
        self.assertTrue(all(x >= 0.0 for x in risk.standard_errors))
        self.assertAlmostEqual(risk.price.value, plan.evaluate().value, delta=2e-13)

        equity_down_plan = compile_lsv(
            request,
            worker_threads=1,
            correlation=-0.4 - equity_vol_bump,
            **common,
        )
        equity_up_plan = compile_lsv(
            request,
            worker_threads=1,
            correlation=-0.4 + equity_vol_bump,
            **common,
        )
        equity_fd = (
            equity_up_plan.evaluate().value - equity_down_plan.evaluate().value
        ) / (2.0 * equity_vol_bump)
        self.assertAlmostEqual(
            risk.equity_volatility_correlation_derivative,
            equity_fd,
            delta=2e-10,
        )

        dividend_down_plan = compile_lsv(
            request,
            worker_threads=1,
            dividend_volatility_correlation=0.15 - dividend_vol_bump,
            **common,
        )
        dividend_up_plan = compile_lsv(
            request,
            worker_threads=1,
            dividend_volatility_correlation=0.15 + dividend_vol_bump,
            **common,
        )
        self.assertEqual(dividend_down_plan.lsv_squared_leverage, plan.lsv_squared_leverage)
        self.assertEqual(dividend_up_plan.lsv_squared_leverage, plan.lsv_squared_leverage)
        dividend_fd = (
            dividend_up_plan.evaluate().value - dividend_down_plan.evaluate().value
        ) / (2.0 * dividend_vol_bump)
        self.assertAlmostEqual(
            risk.dividend_volatility_correlation_derivative,
            dividend_fd,
            delta=2e-10,
        )

        parallel = compile_lsv(request, worker_threads=3, **common)
        parallel_risk = parallel.evaluate_lsv_correlation_risk(
            equity_volatility_correlation_bump=equity_vol_bump,
            dividend_volatility_correlation_bump=dividend_vol_bump,
        )
        self.assertEqual(parallel_risk.derivatives, risk.derivatives)
        self.assertEqual(parallel_risk.standard_errors, risk.standard_errors)

        no_trace = compile_lsv(
            request,
            worker_threads=2,
            retain_reverse_trace=False,
            **common,
        ).evaluate_lsv_correlation_risk(
            equity_volatility_correlation_bump=equity_vol_bump,
            dividend_volatility_correlation_bump=dividend_vol_bump,
        )
        self.assertTrue(all(math.isfinite(x) for x in no_trace.derivatives))

        with self.assertRaises(rp.PricingError):
            compile_lsv(request, two_factor=True, **common).evaluate_lsv_correlation_risk(
                equity_volatility_correlation_bump=equity_vol_bump,
                dividend_volatility_correlation_bump=dividend_vol_bump,
            )
        with self.assertRaises(rp.PricingError):
            plan.evaluate_lsv_correlation_risk(
                equity_volatility_correlation_bump=0.7,
                dividend_volatility_correlation_bump=dividend_vol_bump,
            )

        # Individual bumped entries can remain inside [-1, 1] while the
        # full instantaneous 3x3 Brownian matrix leaves the PSD domain.
        with self.assertRaises(rp.PricingError):
            plan.evaluate_lsv_correlation_risk(
                equity_volatility_correlation_bump=equity_vol_bump,
                dividend_volatility_correlation_bump=0.84,
            )

    def test_residual_lsv_two_factor_parameter_risk_recalibrates_full_model(self):
        request = make_lsv_request(points=16)
        common = dict(
            particle_count=64,
            reduction_block_size=16,
        )
        k_bumps = [0.02, 0.03]
        nu_bump = 0.01
        theta_bump = 0.02
        plan = compile_lsv(request, two_factor=True, worker_threads=1, **common)
        risk = plan.evaluate_lsv_bergomi_two_factor_parameter_risk(
            mean_reversion_bumps=k_bumps,
            vol_of_vol_bump=nu_bump,
            mixing_weight_bump=theta_bump,
        )

        self.assertEqual(
            risk.parameter_labels,
            [
                "bergomi_mean_reversion[0]",
                "bergomi_mean_reversion[1]",
                "bergomi_vol_of_vol",
                "bergomi_mixing_weight",
            ],
        )
        self.assertEqual(risk.parameter_bumps, [k_bumps[0], k_bumps[1], nu_bump, theta_bump])
        self.assertEqual(
            risk.method,
            "buehler-residual-lsv-common-noise-full-recalibration-bergomi-2f-v1",
        )
        self.assertEqual(
            risk.coordinate,
            "two_factor_bergomi_parameters_with_full_residual_lsv_recalibration",
        )
        self.assertTrue(all(math.isfinite(x) for x in risk.derivatives))
        self.assertTrue(all(x >= 0.0 for x in risk.standard_errors))
        self.assertAlmostEqual(risk.price.value, plan.evaluate().value, delta=2e-13)

        scenarios = [
            (dict(mean_reversions=[0.8-k_bumps[0], 2.1]), dict(mean_reversions=[0.8+k_bumps[0], 2.1]), k_bumps[0]),
            (dict(mean_reversions=[0.8, 2.1-k_bumps[1]]), dict(mean_reversions=[0.8, 2.1+k_bumps[1]]), k_bumps[1]),
            (dict(vol_of_vol=0.3-nu_bump), dict(vol_of_vol=0.3+nu_bump), nu_bump),
            (dict(mixing_weight=0.35-theta_bump), dict(mixing_weight=0.35+theta_bump), theta_bump),
        ]
        finite_differences = []
        for down_kwargs, up_kwargs, bump in scenarios:
            down = compile_lsv(
                request,
                two_factor=True,
                worker_threads=1,
                **(common | down_kwargs),
            ).evaluate()
            up = compile_lsv(
                request,
                two_factor=True,
                worker_threads=1,
                **(common | up_kwargs),
            ).evaluate()
            finite_differences.append((up.value-down.value)/(2.0*bump))
        for actual, expected in zip(risk.derivatives, finite_differences):
            self.assertAlmostEqual(actual, expected, delta=3e-10)

        parallel = compile_lsv(request, two_factor=True, worker_threads=3, **common)
        parallel_risk = parallel.evaluate_lsv_bergomi_two_factor_parameter_risk(
            mean_reversion_bumps=k_bumps,
            vol_of_vol_bump=nu_bump,
            mixing_weight_bump=theta_bump,
        )
        self.assertEqual(parallel_risk.derivatives, risk.derivatives)
        self.assertEqual(parallel_risk.standard_errors, risk.standard_errors)

        no_trace = compile_lsv(
            request,
            two_factor=True,
            worker_threads=2,
            retain_reverse_trace=False,
            **common,
        ).evaluate_lsv_bergomi_two_factor_parameter_risk(
            mean_reversion_bumps=k_bumps,
            vol_of_vol_bump=nu_bump,
            mixing_weight_bump=theta_bump,
        )
        self.assertTrue(all(math.isfinite(x) for x in no_trace.derivatives))

        with self.assertRaises(rp.PricingError):
            compile_lsv(request, **common).evaluate_lsv_bergomi_two_factor_parameter_risk(
                mean_reversion_bumps=k_bumps,
                vol_of_vol_bump=nu_bump,
                mixing_weight_bump=theta_bump,
            )
        with self.assertRaises(rp.PricingError):
            plan.evaluate_lsv_bergomi_two_factor_parameter_risk(
                mean_reversion_bumps=k_bumps,
                vol_of_vol_bump=nu_bump,
                mixing_weight_bump=0.4,
            )
        with self.assertRaises((rp.ValidationError, ValueError)):
            plan.evaluate_lsv_bergomi_two_factor_parameter_risk(
                mean_reversion_bumps=[k_bumps[0]],
                vol_of_vol_bump=nu_bump,
                mixing_weight_bump=theta_bump,
            )

    def test_residual_lsv_two_factor_correlation_risk_uses_selective_recalibration(self):
        request = make_lsv_request(points=16)
        common = dict(
            particle_count=64,
            reduction_block_size=16,
        )
        spot_bumps = [0.02, 0.02]
        factor_bump = 0.02
        dividend_bumps = [0.02, 0.02]
        plan = compile_lsv(request, two_factor=True, worker_threads=1, **common)
        risk = plan.evaluate_lsv_bergomi_two_factor_correlation_risk(
            spot_volatility_correlation_bumps=spot_bumps,
            factor_correlation_bump=factor_bump,
            dividend_volatility_correlation_bumps=dividend_bumps,
        )

        self.assertEqual(
            risk.parameter_labels,
            [
                "spot_volatility_correlation[0]",
                "spot_volatility_correlation[1]",
                "volatility_factor_correlation",
                "dividend_volatility_correlation[0]",
                "dividend_volatility_correlation[1]",
            ],
        )
        self.assertEqual(
            risk.correlation_bumps,
            [spot_bumps[0], spot_bumps[1], factor_bump, dividend_bumps[0], dividend_bumps[1]],
        )
        self.assertEqual(
            risk.method,
            "buehler-residual-lsv-common-noise-bergomi-2f-correlation-v1",
        )
        self.assertEqual(
            risk.coordinate,
            "two_factor_bergomi_correlations_with_selective_residual_lsv_recalibration",
        )
        self.assertTrue(all(math.isfinite(x) for x in risk.derivatives))
        self.assertTrue(all(x >= 0.0 for x in risk.standard_errors))
        self.assertAlmostEqual(risk.price.value, plan.evaluate().value, delta=2e-13)

        scenarios = [
            (
                dict(spot_correlations=[-0.4-spot_bumps[0], -0.2]),
                dict(spot_correlations=[-0.4+spot_bumps[0], -0.2]),
                spot_bumps[0],
                True,
            ),
            (
                dict(spot_correlations=[-0.4, -0.2-spot_bumps[1]]),
                dict(spot_correlations=[-0.4, -0.2+spot_bumps[1]]),
                spot_bumps[1],
                True,
            ),
            (
                dict(factor_correlation=0.3-factor_bump),
                dict(factor_correlation=0.3+factor_bump),
                factor_bump,
                True,
            ),
            (
                dict(dividend_volatility_correlations=[0.15-dividend_bumps[0], -0.1]),
                dict(dividend_volatility_correlations=[0.15+dividend_bumps[0], -0.1]),
                dividend_bumps[0],
                False,
            ),
            (
                dict(dividend_volatility_correlations=[0.15, -0.1-dividend_bumps[1]]),
                dict(dividend_volatility_correlations=[0.15, -0.1+dividend_bumps[1]]),
                dividend_bumps[1],
                False,
            ),
        ]
        finite_differences = []
        for down_kwargs, up_kwargs, bump, recalibrates in scenarios:
            down = compile_lsv(
                request,
                two_factor=True,
                worker_threads=1,
                **(common | down_kwargs),
            )
            up = compile_lsv(
                request,
                two_factor=True,
                worker_threads=1,
                **(common | up_kwargs),
            )
            if not recalibrates:
                self.assertEqual(down.lsv_squared_leverage, plan.lsv_squared_leverage)
                self.assertEqual(up.lsv_squared_leverage, plan.lsv_squared_leverage)
            finite_differences.append(
                (up.evaluate().value-down.evaluate().value)/(2.0*bump)
            )
        for actual, expected in zip(risk.derivatives, finite_differences):
            self.assertAlmostEqual(actual, expected, delta=3e-10)

        parallel = compile_lsv(request, two_factor=True, worker_threads=3, **common)
        parallel_risk = parallel.evaluate_lsv_bergomi_two_factor_correlation_risk(
            spot_volatility_correlation_bumps=spot_bumps,
            factor_correlation_bump=factor_bump,
            dividend_volatility_correlation_bumps=dividend_bumps,
        )
        self.assertEqual(parallel_risk.derivatives, risk.derivatives)
        self.assertEqual(parallel_risk.standard_errors, risk.standard_errors)

        no_trace = compile_lsv(
            request,
            two_factor=True,
            worker_threads=2,
            retain_reverse_trace=False,
            **common,
        ).evaluate_lsv_bergomi_two_factor_correlation_risk(
            spot_volatility_correlation_bumps=spot_bumps,
            factor_correlation_bump=factor_bump,
            dividend_volatility_correlation_bumps=dividend_bumps,
        )
        self.assertTrue(all(math.isfinite(x) for x in no_trace.derivatives))

        with self.assertRaises(rp.PricingError):
            compile_lsv(request, **common).evaluate_lsv_bergomi_two_factor_correlation_risk(
                spot_volatility_correlation_bumps=spot_bumps,
                factor_correlation_bump=factor_bump,
                dividend_volatility_correlation_bumps=dividend_bumps,
            )
        with self.assertRaises((rp.ValidationError, ValueError)):
            plan.evaluate_lsv_bergomi_two_factor_correlation_risk(
                spot_volatility_correlation_bumps=[spot_bumps[0]],
                factor_correlation_bump=factor_bump,
                dividend_volatility_correlation_bumps=dividend_bumps,
            )

    def test_two_factor_residual_lsv_is_explicit(self):
        p = compile_lsv(two_factor=True)
        result = p.evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertEqual(result.scheme,
                         "buehler-bergomi-2f-residual-lsv-joint-ou-positive-split-v1")
        self.assertEqual(p.random_factor_count, 4)
        self.assertEqual(len(p.lsv_squared_leverage), 15)
        risk = p.evaluate_local_variance_risk()
        self.assertEqual(len(risk.node_adjoints), 9)
        self.assertTrue(all(math.isfinite(x) for x in risk.node_adjoints))

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
