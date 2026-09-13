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

    def compile_lsv(self, target, workers=1, retain_reverse_trace=False):
        return rp.HullWhiteEquityPlan.compile_lsv(
            self.request(target), target, rp.HullWhiteModel(0.2, [0.0], [0.005]),
            vol_mean_reversion=2.0, vol_of_vol=0.25,
            equity_vol_correlation=-0.5, equity_rate_correlation=0.25,
            vol_rate_correlation=-0.1, particle_count=256, calibration_seed=712,
            log_bandwidth=0.35, minimum_effective_samples=5.0,
            worker_threads=workers, reduction_block_size=32,
            retain_reverse_trace=retain_reverse_trace,
        )

    def test_market_iv_vegakt_api_recalibration_and_ownership(self):
        maturities = [0.4, 1.0]
        xs = [-0.8, -0.3, 0.0, 0.4, 0.8]
        vols = [0.23, 0.224, 0.22, 0.216, 0.22, 0.236, 0.23, 0.226, 0.222, 0.226]
        def target(quotes):
            return rp.HullWhiteLsvTarget.from_market_iv(maturities, xs, quotes,
                                                       [0.0, 0.25, 0.5, 0.75, 1.0], xs)
        def compile_quotes(quotes, workers=1):
            t = target(quotes)
            request = rp.PricingRequest(
                "2026-09-04",
                rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
                rp.Market.equity(2, 1, 100.0,
                    rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
                    rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
                    discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.5, 6.0)]),
                t.model, rp.Engine.randomized_quasi_monte_carlo(
                    64, 612, scramble_count=4, antithetic=True, brownian_bridge=True),
                rp.RiskRequest(),
            )
            return rp.HullWhiteEquityPlan.compile_lsv(
                request, t, rp.HullWhiteModel(0.2, [0.0], [0.005]),
                vol_mean_reversion=2.0, vol_of_vol=0.25,
                equity_vol_correlation=-0.5, equity_rate_correlation=0.25,
                vol_rate_correlation=-0.1, particle_count=256, calibration_seed=712,
                log_bandwidth=0.35, minimum_effective_samples=5.0,
                worker_threads=workers, reduction_block_size=32,
                retain_reverse_trace=True, cash_dividend_model="escrowed",
            )
        self.assertTrue(target(vols).supports_vega_kt)
        plan = compile_quotes(vols)
        risk = plan.evaluate_aad()
        self.assertEqual(risk.price.value, plan.evaluate().value)
        self.assertEqual(len(risk.vega_kt_raw), 10)
        self.assertEqual(len(risk.vega_kt_standard_errors), 10)
        self.assertEqual(risk.vega_kt_maturity_nodes, maturities)
        self.assertEqual(risk.vega_kt_log_moneyness_nodes, xs)
        self.assertEqual(risk.vega_kt_implied_volatilities, vols)
        self.assertEqual(risk.vega_kt_method, "natural-cubic-w-linear-time-v1")
        self.assertEqual(risk.vega_kt_market_scaled, [v*0.01 for v in risk.vega_kt_raw])
        self.assertAlmostEqual(risk.vega, sum(risk.vega_kt_raw), places=12)
        self.assertTrue(math.isfinite(risk.parallel_vega_standard_error))
        self.assertEqual(risk.parameter_labels[-1], "parallel_market_iv")
        self.assertEqual(len(risk.parameter_labels), len(risk.derivatives))
        self.assertEqual(len(risk.derivatives), len(risk.standard_errors))
        for i in [2, 7]:
            up, down = vols.copy(), vols.copy()
            up[i] += 1e-7
            down[i] -= 1e-7
            fd = (compile_quotes(up).evaluate().value-compile_quotes(down).evaluate().value)/2e-7
            self.assertAlmostEqual(risk.vega_kt_raw[i], fd, delta=2e-5)
        replay = compile_quotes(vols, workers=3).evaluate_aad()
        self.assertEqual(risk.derivatives, replay.derivatives)
        self.assertEqual(risk.standard_errors, replay.standard_errors)
        copied = risk.vega_kt_raw
        copied[0] = 999.0
        self.assertNotEqual(risk.vega_kt_raw[0], 999.0)
        with self.assertRaises(AttributeError):
            risk.vega_kt_raw = []
        flat = rp.HullWhiteLsvTarget.flat(0.2, [0.0, 0.5, 1.0], [-0.5, 0.0, 0.5])
        self.assertFalse(flat.supports_vega_kt)
        old_risk = self.compile_lsv(flat, retain_reverse_trace=True).evaluate_aad()
        self.assertIsNone(old_risk.vega_kt_raw)
        self.assertIsNone(old_risk.vega_kt_standard_errors)
        self.assertIsNone(old_risk.vega_kt_method)
        self.assertEqual(old_risk.vega_kt_maturity_nodes, [])

    def test_market_iv_rejects_malformed_or_inconsistent_targets(self):
        def build(times, xs, values, grid_xs=None, floor=1e-8, cap=4.0):
            return rp.HullWhiteLsvTarget.from_market_iv(times, xs, values,
                [0.0, 0.5, 1.0], grid_xs or xs, floor=floor, cap=cap)
        for t, x, v in [([0.0, 1.0], [-0.5, 0.5], [0.2]*4),
                         ([0.5, 1.0], [-0.5, 0.5], [0.2]*3),
                         ([0.5, 1.0], [-0.5, 0.5], [math.nan]*4),
                         ([0.5, 1.0], [-0.5, 0.5], [0.5, 0.5, 0.1, 0.1])]:
            with self.assertRaises(rp.ValidationError):
                build(t, x, v)
        with self.assertRaises(rp.ValidationError):
            build([0.5, 1.0], [-0.5, 0.5], [0.2]*4, [-0.6, 0.0, 0.6])
        with self.assertRaises(rp.ValidationError):
            build([0.5, 1.0], [-0.5, 0.5], [0.2]*4, cap=0.03)

    def test_bs_aad_fields_and_finite_difference(self):
        rates = rp.HullWhiteModel(0.2, [0.0], [0.01])
        def compile_request(data):
            return rp.HullWhiteEquityPlan.compile_bs(
                rp.PricingRequest.from_json(json.dumps(data)), rates,
                equity_rate_correlation=0.25, maximum_step=0.25, worker_threads=2,
            )
        data = self.payload()
        plan = compile_request(data)
        risk = plan.evaluate_aad()
        self.assertIsInstance(risk, rp.HullWhiteAadRisk)
        self.assertEqual(risk.price.value, plan.evaluate().value)
        self.assertEqual(risk.parameter_labels[:2], ["spot", "bs_volatility"])
        self.assertEqual(len(risk.derivatives), len(risk.parameter_labels))
        self.assertEqual(len(risk.standard_errors), len(risk.derivatives))
        self.assertTrue(all(math.isfinite(x) and x >= 0.0 for x in risk.standard_errors))
        self.assertFalse(plan.retains_reverse_trace)
        self.assertEqual(risk.local_variance_adjoints, [])
        self.assertEqual(risk.forward_log_density_adjoints, [])
        self.assertAlmostEqual(risk.parallel_discount_dv01, sum(risk.discount_node_dv01))
        h = 1e-5
        data["market"]["spot"] = 100.0 + h
        up = compile_request(data).evaluate().value
        data["market"]["spot"] = 100.0 - h
        down = compile_request(data).evaluate().value
        self.assertAlmostEqual(risk.delta, (up-down)/(2*h), delta=1e-6)
        copied = risk.derivatives
        copied[0] = 999.0
        self.assertEqual(risk.derivatives[0], risk.delta)
        with self.assertRaises(AttributeError):
            risk.delta = 1.0

    def test_lsv_aad_trace_and_paired_target_contract(self):
        target = rp.HullWhiteLsvTarget.flat(0.2, [0.0, 0.5, 1.0], [-0.5, 0.0, 0.5])
        no_trace = self.compile_lsv(target)
        with self.assertRaises(rp.PricingError):
            no_trace.evaluate_aad()
        plan = self.compile_lsv(target, retain_reverse_trace=True)
        self.assertTrue(plan.retains_reverse_trace)
        self.assertNotEqual(plan.plan_fingerprint, no_trace.plan_fingerprint)
        self.assertEqual(plan.evaluate().value, no_trace.evaluate().value)
        risk = plan.evaluate_aad()
        self.assertIsNone(risk.vega)
        self.assertEqual(risk.price.value, no_trace.evaluate().value)
        self.assertEqual(risk.time_nodes, target.time_nodes)
        self.assertEqual(risk.log_moneyness_nodes, target.log_moneyness_nodes)
        self.assertEqual(len(risk.local_variance_adjoints), 9)
        self.assertEqual(len(risk.forward_log_density_adjoints), 9)
        self.assertEqual(risk.method, "equity-hw-discrete-particle-vjp-v1")
        self.assertEqual(risk.uncertainty_scope, "pricing_conditional_on_calibration")
        replay = self.compile_lsv(target, workers=3, retain_reverse_trace=True).evaluate_aad()
        self.assertEqual(risk.derivatives, replay.derivatives)
        self.assertEqual(risk.standard_errors, replay.standard_errors)

    def test_cash_aad_zero_volatility_delta_and_signed_dv01(self):
        data = self.payload()
        data["model"]["volatility"] = 0.0
        data["product"]["strike"] = 80.0
        data["market"]["discrete_dividends"] = [
            {"event_id": 1, "ex_time": 1.0, "quote": {"type": "fixed_cash", "amount": 10.0}}
        ]
        plan = rp.HullWhiteEquityPlan.compile_bs(
            rp.PricingRequest.from_json(json.dumps(data)), rp.HullWhiteModel(0.2, [0.0], [0.0]),
            equity_rate_correlation=0.0, maximum_step=1.0, worker_threads=1,
            cash_dividend_model="escrowed",
        )
        risk = plan.evaluate_aad()
        self.assertAlmostEqual(risk.price.value, 12.5, places=12)
        self.assertAlmostEqual(risk.delta, 0.98, places=12)
        self.assertAlmostEqual(risk.vega, 0.0, places=12)
        self.assertAlmostEqual(risk.parallel_discount_dv01, 0.00855, places=12)
        self.assertEqual(risk.discount_times, [0.0, 1.0])
        self.assertEqual(risk.dividend_times, [0.0, 1.0])
        self.assertEqual(risk.discount_log_df_adjoints[0], 0.0)
        self.assertEqual(risk.dividend_log_df_adjoints[0], 0.0)

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

    def test_explicit_cash_dividend_model_and_metadata(self):
        data = self.payload()
        data["model"]["volatility"] = 0.0
        data["product"]["strike"] = 80.0
        data["market"]["discrete_dividends"] = [
            {"event_id": 1, "ex_time": 1.0, "quote": {"type": "fixed_cash", "amount": 10.0}}
        ]
        request = rp.PricingRequest.from_json(json.dumps(data))
        rates = rp.HullWhiteModel(0.2, [0.0], [0.0])
        kwargs = dict(equity_rate_correlation=0.0, maximum_step=1.0, worker_threads=1)
        with self.assertRaises(rp.PricingError):
            rp.HullWhiteEquityPlan.compile_bs(request, rates, **kwargs)
        plan = rp.HullWhiteEquityPlan.compile_bs(
            request, rates, cash_dividend_model="escrowed", **kwargs
        )
        self.assertAlmostEqual(plan.risky_spot, 100.0 - 10.0*0.95/0.98, places=12)
        result = plan.evaluate()
        self.assertAlmostEqual(result.value, 12.5, places=12)
        self.assertEqual(plan.cash_dividend_model, "escrowed-hw-bonds-v1")
        self.assertEqual(result.cash_dividend_model, plan.cash_dividend_model)
        with self.assertRaises(AttributeError):
            plan.risky_spot = 0.0
        with self.assertRaises(rp.ValidationError):
            rp.HullWhiteEquityPlan.compile_bs(request, rates, cash_dividend_model="unknown", **kwargs)

    def test_cash_lsv_deterministic_limit_and_missing_event_time(self):
        target = rp.HullWhiteLsvTarget.flat(0.2, [0.0, 0.5, 1.0], [-0.5, 0.0, 0.5])
        data = json.loads(self.request(target).to_json())
        data["market"]["discrete_dividends"] = [
            {"event_id": 1, "ex_time": 0.5, "quote": {"type": "fixed_cash", "amount": 4.0}}
        ]
        rates = rp.HullWhiteModel(0.2, [0.0], [0.0])
        kwargs = dict(vol_mean_reversion=2.0, vol_of_vol=0.0,
                      equity_vol_correlation=-0.5, equity_rate_correlation=0.25,
                      vol_rate_correlation=-0.1, particle_count=128, calibration_seed=712,
                      log_bandwidth=0.35, minimum_effective_samples=5.0,
                      worker_threads=1, cash_dividend_model="escrowed")
        plan = rp.HullWhiteEquityPlan.compile_lsv(
            rp.PricingRequest.from_json(json.dumps(data)), target, rates, **kwargs
        )
        self.assertTrue(all(abs(v - 0.04) < 1e-15 for v in plan.squared_leverage))
        self.assertEqual(plan.evaluate().calibration_method, "lsv-hw-escrowed-quadratic-quartic-v1")
        data["market"]["discrete_dividends"][0]["ex_time"] = 0.3
        with self.assertRaises(rp.PricingError):
            rp.HullWhiteEquityPlan.compile_lsv(
                rp.PricingRequest.from_json(json.dumps(data)), target, rates, **kwargs
            )
