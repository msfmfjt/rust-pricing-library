import json
import math
import unittest
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rust_pricing


class PricingFacadeSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request_json = Path(
            "fixtures/v1/pricing_request.golden.json"
        ).read_text(encoding="utf-8")

    def test_compile_evaluate_and_serialize(self):
        request = rust_pricing.PricingRequest.from_json(self.request_json)
        self.assertTrue(request.fingerprint.startswith("blake3-256:"))

        plan = rust_pricing.PricingPlan.compile(
            request, worker_threads=2, reduction_block_size=256
        )
        self.assertTrue(plan.plan_fingerprint.startswith("blake3-256:"))
        self.assertEqual(plan.request_fingerprint, request.fingerprint)

        result = plan.evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertGreaterEqual(result.standard_error, 0.0)
        self.assertEqual(result.estimate.value, result.value)
        self.assertEqual(result.estimate.standard_error, result.standard_error)
        self.assertEqual(result.estimate.estimator, result.diagnostics.estimator)
        self.assertEqual(result.estimate.effective_sampling_units, 1024)
        self.assertEqual(result.independent_sampling_units, 1024)
        self.assertEqual(result.evaluated_paths, 2048)
        self.assertIsNone(result.delta)
        self.assertIsNone(result.delta_raw)
        self.assertIsNone(result.gamma)
        self.assertIsNone(result.vega)
        self.assertIsNone(result.vega_kt)

        payload = json.loads(result.to_json())
        self.assertEqual(payload["document_kind"], "pricing_result")
        self.assertEqual(result.diagnostics.estimator, "pseudo_monte_carlo")
        self.assertIsNone(result.diagnostics.direction_checksum)
        self.assertIsNone(result.diagnostics.scramble_checksum)
        self.assertEqual(result.diagnostics.worker_threads, 2)
        self.assertIsNone(result.diagnostics.gamma_spot_bump)
        self.assertIsNone(result.diagnostics.validation_spot_bump)
        self.assertIsNone(result.diagnostics.validation_volatility_bump)
        self.assertEqual(result.diagnostics.bump_policy_version, 1)
        self.assertIsNone(result.diagnostics.delta_validation)
        self.assertIsNone(result.diagnostics.gamma_validation)
        self.assertIsNone(result.diagnostics.vega_validation)
        self.assertEqual(
            [warning.code for warning in result.diagnostics.warnings],
            [warning.code for warning in result.warnings],
        )
        with self.assertRaises(AttributeError):
            result.diagnostics.master_seed = 99

    def test_pricing_warnings_are_immutable_and_freshly_owned(self):
        request = rust_pricing.PricingRequest.from_json(
            self.request_json.replace(
                '"times":[0.0,1.0]', '"times":[0.0,0.5]', 2
            ).replace(
                '"discount_factors":[1.0,0.95]',
                '"discount_factors":[1.0,0.975]',
                1,
            ).replace(
                '"discount_factors":[1.0,0.98]',
                '"discount_factors":[1.0,0.99]',
                1,
            )
        )
        result = rust_pricing.PricingPlan.compile(
            request, worker_threads=1, reduction_block_size=256
        ).evaluate()

        warnings = result.warnings
        self.assertEqual(
            [warning.code for warning in warnings],
            ["discount_curve_extrapolation", "dividend_curve_extrapolation"],
        )
        with self.assertRaises(AttributeError):
            warnings[0].code = "changed"
        warnings.clear()
        self.assertEqual(len(result.warnings), 2)
        self.assertEqual(len(result.diagnostics.warnings), 2)

    def test_native_builders_match_json_request_and_result(self):
        discount = rust_pricing.DiscountCurve(
            10,
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.95], dtype=np.float64),
        )
        dividend = rust_pricing.DiscountCurve(
            11,
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.98], dtype=np.float64),
        )
        product = rust_pricing.Product.european_vanilla(
            1, 2, date(2027, 9, 4), 100.0, 1.0, "call"
        )
        market = rust_pricing.Market.equity(
            2, 1, 100.0, discount, dividend
        )
        model = rust_pricing.Model.black_scholes(0.2)
        engine = rust_pricing.Engine.pseudo_monte_carlo(
            7, 1024, antithetic=True
        )
        risk = rust_pricing.RiskRequest()
        native = rust_pricing.PricingRequest(
            date(2026, 9, 4), product, market, model, engine, risk
        )
        from_json = rust_pricing.PricingRequest.from_json(self.request_json)
        self.assertEqual(native.fingerprint, from_json.fingerprint)
        self.assertEqual(native.to_json(), from_json.to_json())

        native_plan = rust_pricing.PricingPlan.compile(
            native, worker_threads=2, reduction_block_size=256
        )
        json_plan = rust_pricing.PricingPlan.compile(
            from_json, worker_threads=2, reduction_block_size=256
        )
        self.assertEqual(native_plan.plan_fingerprint, json_plan.plan_fingerprint)
        native_result = native_plan.evaluate()
        json_result = json_plan.evaluate()
        self.assertEqual(native_result.to_json(), json_result.to_json())

    def test_datetime_values_are_rejected_as_dates(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        with self.assertRaises(rust_pricing.ValidationError):
            rust_pricing.Product.european_vanilla(
                1, 2, datetime(2027, 9, 4, 12, 30), 100.0, 1.0, "call"
            )
        product = rust_pricing.Product.european_vanilla(
            1, 2, "2027-09-04", 100.0, 1.0, "call"
        )
        with self.assertRaises(rust_pricing.ValidationError):
            rust_pricing.PricingRequest(
                datetime(2026, 9, 4, 9, 0),
                product,
                rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
                rust_pricing.Model.black_scholes(0.2),
                rust_pricing.Engine.pseudo_monte_carlo(7, 1024),
                rust_pricing.RiskRequest(),
            )
        with self.assertRaises(rust_pricing.ValidationError) as captured:
            rust_pricing.Product.european_vanilla(
                1, 2, np.datetime64("2027-09-04"), 100.0, 1.0, "call"
            )
        self.assertEqual(captured.exception.issues[0].code, "invalid_date_type")

    def test_native_discrete_dividends_match_json_request(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        product = rust_pricing.Product.european_vanilla(
            1, 2, "2027-09-04", 95.0, 1.0, "call"
        )
        dividend_event = rust_pricing.DividendEvent.fixed_cash_and_proportional(
            77, 0.25, 1.5, 0.02
        )
        market = rust_pricing.Market.equity(
            2,
            1,
            100.0,
            discount,
            dividend,
            discrete_dividends=[dividend_event],
        )
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            product,
            market,
            rust_pricing.Model.black_scholes(0.2),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(),
        )
        payload = json.loads(request.to_json())
        self.assertEqual(
            payload["market"]["discrete_dividends"][0]["quote"]["type"],
            "fixed_cash_and_proportional",
        )
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        self.assertEqual(parsed.fingerprint, request.fingerprint)
        self.assertEqual(parsed.to_json(), request.to_json())

    def test_native_black_76_model_evaluates_and_round_trips(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.95])
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            rust_pricing.Product.european_vanilla(
                1, 2, "2027-09-04", 100.0, 1.0, "call"
            ),
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.black_76(0.2),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(),
        )
        payload = json.loads(request.to_json())
        self.assertEqual(payload["model"]["type"], "black_76")
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        self.assertEqual(parsed.to_json(), request.to_json())
        result = rust_pricing.PricingPlan.compile(
            parsed, worker_threads=2, reduction_block_size=256
        ).evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertGreaterEqual(result.standard_error, 0.0)

    def test_native_digital_product_evaluates_and_round_trips(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        product = rust_pricing.Product.digital(
            1,
            2,
            "2027-09-04",
            100.0,
            10.0,
            "call",
            "cash",
            payment_date="2027-09-05",
        )
        market = rust_pricing.Market.equity(2, 1, 100.0, discount, dividend)
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            product,
            market,
            rust_pricing.Model.black_scholes(0.2),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(),
        )
        payload = json.loads(request.to_json())
        self.assertEqual(payload["product"]["type"], "digital")
        self.assertEqual(payload["product"]["payout_kind"]["type"], "cash")
        self.assertEqual(payload["product"]["payment_date"], "2027-09-05")
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        self.assertEqual(parsed.fingerprint, request.fingerprint)
        result = rust_pricing.PricingPlan.compile(
            parsed, worker_threads=2, reduction_block_size=256
        ).evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertGreaterEqual(result.standard_error, 0.0)

        with self.assertRaises(rust_pricing.ValidationError):
            rust_pricing.PricingRequest(
                "2026-09-04",
                product,
                market,
                rust_pricing.Model.black_scholes(0.2),
                rust_pricing.Engine.pseudo_monte_carlo(7, 1024),
                rust_pricing.RiskRequest(delta=True),
            )

    def test_native_barrier_product_evaluates_and_round_trips(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        product = rust_pricing.Product.barrier(
            1,
            2,
            "2027-09-04",
            100.0,
            120.0,
            1.0,
            "call",
            "up",
            "knock_out",
            ["2027-03-04", "2027-09-04"],
            "2027-09-04",
            rebate=3.0,
        )
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            product,
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.black_scholes(0.2),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(),
        )
        payload = json.loads(request.to_json())
        self.assertEqual(payload["product"]["type"], "barrier")
        self.assertEqual(payload["product"]["direction"]["type"], "up")
        self.assertEqual(payload["product"]["style"]["type"], "knock_out")
        self.assertEqual(payload["product"]["rebate"], 3.0)
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        self.assertEqual(parsed.fingerprint, request.fingerprint)
        result = rust_pricing.PricingPlan.compile(
            parsed, worker_threads=2, reduction_block_size=256
        ).evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertGreaterEqual(result.standard_error, 0.0)

        with self.assertRaises(rust_pricing.ValidationError):
            rust_pricing.PricingRequest(
                "2026-09-04",
                rust_pricing.Product.barrier(
                    1,
                    2,
                    "2027-09-04",
                    100.0,
                    120.0,
                    1.0,
                    "call",
                    "up",
                    "knock_out",
                    ["2026-03-04", "2027-09-04"],
                    "2027-09-04",
                ),
                rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
                rust_pricing.Model.black_scholes(0.2),
                rust_pricing.Engine.pseudo_monte_carlo(7, 1024),
                rust_pricing.RiskRequest(),
            )

    def test_native_arithmetic_asian_product_evaluates_and_round_trips(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        observations = [
            rust_pricing.AsianObservation.unknown("2026-09-04", 0.25),
            rust_pricing.AsianObservation.unknown("2027-09-04", 0.75),
        ]
        product = rust_pricing.Product.arithmetic_asian(
            1, 2, 100.0, 1.0, "call", observations, "2027-09-04"
        )
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            product,
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.black_scholes(0.2),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(
                delta=True, gamma_relative_bump=0.01, vega=True
            ),
        )
        payload = json.loads(request.to_json())
        self.assertEqual(payload["product"]["type"], "arithmetic_asian")
        self.assertEqual(payload["product"]["observations"][0]["value"]["type"], "unknown")
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        self.assertEqual(parsed.fingerprint, request.fingerprint)
        result = rust_pricing.PricingPlan.compile(
            parsed, worker_threads=2, reduction_block_size=256
        ).evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertGreaterEqual(result.standard_error, 0.0)
        self.assertTrue(math.isfinite(result.delta_raw))
        self.assertTrue(math.isfinite(result.gamma_raw))
        self.assertTrue(math.isfinite(result.vega_raw))
        self.assertEqual(result.delta.raw_unit, "delta_raw")
        self.assertEqual(result.delta.market_scaled_unit, "delta_one_percent_spot")
        self.assertEqual(result.delta.raw.value, result.delta_raw)
        self.assertEqual(result.delta.raw.estimator, result.diagnostics.estimator)
        self.assertEqual(result.delta.market_scaled.value, result.delta_market_scaled)
        self.assertGreaterEqual(result.delta.raw.standard_error, 0.0)
        self.assertEqual(
            result.delta.raw.effective_sampling_units,
            result.independent_sampling_units,
        )
        self.assertEqual(result.gamma.raw_unit, "gamma_raw")
        self.assertEqual(
            result.gamma.market_scaled_unit, "gamma_one_percent_spot_squared"
        )
        self.assertEqual(result.gamma.raw.value, result.gamma_raw)
        self.assertEqual(result.vega.raw_unit, "vega_raw")
        self.assertEqual(result.vega.market_scaled_unit, "vega_one_vol_point")
        self.assertEqual(result.vega.raw.value, result.vega_raw)
        with self.assertRaises(AttributeError):
            result.delta.raw = result.gamma.raw
        for validation in (
            result.diagnostics.delta_validation,
            result.diagnostics.gamma_validation,
            result.diagnostics.vega_validation,
        ):
            self.assertIsNotNone(validation)
            self.assertTrue(math.isfinite(validation.bump_and_revalue.value))
            self.assertGreaterEqual(
                validation.bump_and_revalue.standard_error, 0.0
            )
            lower, upper = validation.bump_and_revalue.confidence_interval
            self.assertLessEqual(lower, validation.bump_and_revalue.value)
            self.assertLessEqual(validation.bump_and_revalue.value, upper)
            self.assertEqual(
                validation.bump_and_revalue.effective_sampling_units,
                result.independent_sampling_units,
            )
            self.assertTrue(math.isfinite(validation.bump_minus_primary.value))
            with self.assertRaises(AttributeError):
                validation.bump_and_revalue.value = 0.0

    def test_native_fully_fixed_arithmetic_asian_discounts_known_payoff(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        observations = [
            rust_pricing.AsianObservation.known("2026-03-04", 0.25, 95.0),
            rust_pricing.AsianObservation.known("2026-06-04", 0.75, 115.0),
        ]
        product = rust_pricing.Product.arithmetic_asian(
            1, 2, 100.0, 2.0, "call", observations, "2027-09-04"
        )
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            product,
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.black_scholes(0.2),
            rust_pricing.Engine.randomized_quasi_monte_carlo(
                16, 7, scramble_count=4, antithetic=True
            ),
            rust_pricing.RiskRequest(),
        )
        result = rust_pricing.PricingPlan.compile(
            request, worker_threads=2, reduction_block_size=256
        ).evaluate()
        self.assertAlmostEqual(result.value, 19.0, places=12)
        self.assertEqual(result.standard_error, 0.0)

    def test_native_fixed_lookback_product_evaluates_and_round_trips(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        product = rust_pricing.Product.fixed_lookback(
            1,
            2,
            100.0,
            1.0,
            "put",
            ["2026-03-04", "2027-09-04"],
            "2027-09-04",
            historical_extremum=92.0,
        )
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            product,
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.black_scholes(0.2),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(),
        )
        payload = json.loads(request.to_json())
        self.assertEqual(payload["product"]["type"], "fixed_lookback")
        self.assertEqual(payload["product"]["historical_extremum"], 92.0)
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        self.assertEqual(parsed.fingerprint, request.fingerprint)
        result = rust_pricing.PricingPlan.compile(
            parsed, worker_threads=2, reduction_block_size=256
        ).evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertGreaterEqual(result.standard_error, 0.0)

        with self.assertRaises(rust_pricing.ValidationError):
            rust_pricing.PricingRequest(
                "2026-09-04",
                rust_pricing.Product.fixed_lookback(
                    1,
                    2,
                    100.0,
                    1.0,
                    "call",
                    ["2026-03-04", "2027-09-04"],
                    "2027-09-04",
                ),
                rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
                rust_pricing.Model.black_scholes(0.2),
                rust_pricing.Engine.pseudo_monte_carlo(7, 1024),
                rust_pricing.RiskRequest(),
            )

    def test_native_local_volatility_grid_matches_json_request(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            rust_pricing.Product.european_vanilla(
                1, 2, "2027-09-04", 100.0, 1.0, "call"
            ),
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.local_volatility_from_grid(
                [0.0, 1.0],
                [-0.1, 0.0, 0.2],
                [0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
                1.0e-8,
                4.0,
            ),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(delta=True, gamma_relative_bump=0.01),
        )
        payload = json.loads(request.to_json())
        self.assertEqual(payload["model"]["type"], "local_volatility")
        self.assertEqual(payload["model"]["local_variance_grid"]["shape"], [2, 3])
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        self.assertEqual(parsed.fingerprint, request.fingerprint)
        self.assertEqual(parsed.to_json(), request.to_json())
        plan = rust_pricing.PricingPlan.compile(
            parsed, worker_threads=2, reduction_block_size=256
        )
        result = plan.evaluate()
        self.assertTrue(math.isfinite(result.value))
        self.assertGreater(result.standard_error, 0.0)
        self.assertTrue(math.isfinite(result.delta_raw))
        self.assertTrue(math.isfinite(result.gamma_raw))
        self.assertIsNone(result.vega_raw)
        self.assertEqual(result.diagnostics.delta_method, "central_bump")
        self.assertEqual(result.diagnostics.gamma_method, "central_bump")
        self.assertEqual(result.diagnostics.gamma_spot_bump, 1.0)
        self.assertEqual(result.diagnostics.validation_spot_bump, 1.0)
        self.assertIsNone(result.diagnostics.validation_volatility_bump)
        self.assertEqual(result.diagnostics.bump_policy_version, 1)

        rqmc_request = rust_pricing.PricingRequest(
            "2026-09-04",
            rust_pricing.Product.european_vanilla(
                1, 2, "2027-09-04", 100.0, 1.0, "call"
            ),
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.local_volatility_from_grid(
                [0.0, 1.0],
                [-0.1, 0.0, 0.2],
                [0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
                1.0e-8,
                4.0,
            ),
            rust_pricing.Engine.randomized_quasi_monte_carlo(
                256, 11, scramble_count=4, antithetic=True
            ),
            rust_pricing.RiskRequest(delta=True, gamma_relative_bump=0.01),
        )
        rqmc_plan = rust_pricing.PricingPlan.compile(
            rqmc_request, worker_threads=2, reduction_block_size=256
        )
        rqmc_result = rqmc_plan.evaluate()
        self.assertTrue(math.isfinite(rqmc_result.value))
        self.assertEqual(
            rqmc_result.diagnostics.estimator, "randomized_quasi_monte_carlo"
        )
        self.assertRegex(
            rqmc_result.diagnostics.direction_checksum, r"^[0-9a-f]{64}$"
        )
        self.assertRegex(
            rqmc_result.diagnostics.scramble_checksum, r"^[0-9a-f]{64}$"
        )
        self.assertTrue(math.isfinite(rqmc_result.delta_raw))
        self.assertTrue(math.isfinite(rqmc_result.gamma_raw))
        self.assertEqual(rqmc_result.diagnostics.delta_method, "central_bump")

    def test_native_local_volatility_can_materialize_from_essvi(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            rust_pricing.Product.european_vanilla(
                1, 2, "2027-09-04", 100.0, 1.0, "call"
            ),
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.local_volatility_from_essvi(
                [
                    rust_pricing.EssviSlice(0.25, 0.02, 0.1, -0.03),
                    rust_pricing.EssviSlice(1.0, 0.04, 0.2, -0.06),
                ],
                0.02,
                [0.25, 1.0],
                [-0.1, 0.0, 0.2],
                1.0e-8,
                4.0,
            ),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(),
        )
        payload = json.loads(request.to_json())
        grid = payload["model"]["local_variance_grid"]
        self.assertEqual(payload["model"]["type"], "local_volatility")
        self.assertEqual(grid["shape"], [2, 3])
        self.assertEqual(len(grid["values"]), 6)
        basis = payload["model"]["reporting_iv_basis"]
        self.assertEqual(basis["shape"], [2, 3])
        self.assertEqual(len(basis["implied_volatilities"]), 6)
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        parsed_model = json.loads(parsed.to_json())["model"]
        parsed_grid = parsed_model["local_variance_grid"]
        self.assertEqual(parsed_grid["shape"], [2, 3])
        self.assertEqual(len(parsed_grid["values"]), 6)
        self.assertEqual(parsed_model["reporting_iv_basis"]["shape"], [2, 3])

    def test_native_local_volatility_can_materialize_from_standard_ssvi(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        request = rust_pricing.PricingRequest(
            "2026-09-04",
            rust_pricing.Product.european_vanilla(
                1, 2, "2027-09-04", 100.0, 1.0, "call"
            ),
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.local_volatility_from_standard_ssvi_power_law(
                [0.25, 1.0],
                [0.02, 0.04],
                0.02,
                -0.3,
                0.5,
                0.4,
                [0.25, 1.0],
                [-0.1, 0.0, 0.2],
                1.0e-8,
                4.0,
            ),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(),
        )
        payload = json.loads(request.to_json())
        grid = payload["model"]["local_variance_grid"]
        self.assertEqual(payload["model"]["type"], "local_volatility")
        self.assertEqual(grid["shape"], [2, 3])
        self.assertEqual(len(grid["values"]), 6)
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        parsed_model = json.loads(parsed.to_json())["model"]
        self.assertEqual(parsed_model["local_variance_grid"]["shape"], [2, 3])
        self.assertEqual(parsed_model["reporting_iv_basis"]["shape"], [2, 3])

    def test_native_vega_kt_request_matches_json_request(self):
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        request = rust_pricing.PricingRequest(
            date(2026, 9, 4),
            rust_pricing.Product.european_vanilla(
                1, 2, date(2027, 9, 4), 100.0, 1.0, "call"
            ),
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.local_volatility_from_grid(
                [0.25, 1.0],
                [-0.1, 0.0, 0.2],
                [0.03, 0.04, 0.05, 0.035, 0.045, 0.055],
                1.0e-8,
                4.0,
            ),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(
                delta=True,
                vega=True,
                vega_kt_maturity_nodes=[date(2027, 3, 4), "2027-09-04"],
                vega_kt_log_forward_moneyness_nodes=[-0.2, 0.0, 0.2],
                vega_kt_relative_density_threshold=1.0e-8,
                vega_kt_full_bucket_covariance=True,
                checkpoint_interval=16,
                aad_tile_capacity=256,
            ),
        )
        payload = json.loads(request.to_json())
        self.assertEqual(payload["risk"]["vega_kt"]["maturity_nodes"][0], "2027-03-04")
        self.assertTrue(payload["risk"]["vega_kt"]["full_bucket_covariance"])
        parsed = rust_pricing.PricingRequest.from_json(request.to_json())
        self.assertEqual(parsed.fingerprint, request.fingerprint)
        self.assertEqual(parsed.to_json(), request.to_json())

    def test_native_local_volatility_evaluates_vega_kt_report(self):
        valuation = date(2026, 9, 4)
        first_maturity = date(2027, 3, 5)
        expiry = date(2027, 9, 4)
        maturity_nodes = [
            (first_maturity - valuation).days / 365.0,
            (expiry - valuation).days / 365.0,
        ]
        discount = rust_pricing.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rust_pricing.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        request = rust_pricing.PricingRequest(
            valuation,
            rust_pricing.Product.european_vanilla(
                1, 2, expiry, 100.0, 1.0, "call"
            ),
            rust_pricing.Market.equity(2, 1, 100.0, discount, dividend),
            rust_pricing.Model.local_volatility_from_grid_with_reporting_basis(
                [0.0, maturity_nodes[0], maturity_nodes[1]],
                [-0.2, 0.0, 0.2],
                [0.038, 0.04, 0.042, 0.037, 0.04, 0.044, 0.036, 0.041, 0.047],
                1.0e-8,
                4.0,
                maturity_nodes,
                [-0.2, 0.0, 0.2],
                [0.195, 0.2, 0.207, 0.19, 0.202, 0.215],
            ),
            rust_pricing.Engine.pseudo_monte_carlo(7, 1024, antithetic=True),
            rust_pricing.RiskRequest(
                delta=True,
                gamma_relative_bump=0.01,
                vega=True,
                vega_kt_maturity_nodes=[first_maturity, expiry],
                vega_kt_log_forward_moneyness_nodes=[-0.2, 0.0, 0.2],
                vega_kt_relative_density_threshold=1.0e-8,
                vega_kt_full_bucket_covariance=True,
                checkpoint_interval=16,
                aad_tile_capacity=128,
            ),
        )
        result = rust_pricing.PricingPlan.compile(
            request, worker_threads=2, reduction_block_size=256
        ).evaluate()
        self.assertTrue(math.isfinite(result.vega_raw))
        vega_kt = result.vega_kt
        self.assertIsNotNone(vega_kt)
        self.assertEqual(len(vega_kt.coordinates), 6)
        self.assertEqual(len(vega_kt.estimates), 6)
        self.assertEqual(len(vega_kt.raw_buckets), 6)
        self.assertEqual(len(vega_kt.full_bucket_covariance), 36)
        self.assertTrue(math.isfinite(vega_kt.projection.scalar_vega))
        payload = json.loads(result.to_json())
        self.assertIn("vega_kt", payload["risks"])

    def test_pricing_result_round_trips_from_json(self):
        request = rust_pricing.PricingRequest.from_json(self.request_json)
        result = rust_pricing.PricingPlan.compile(
            request, worker_threads=2, reduction_block_size=256
        ).evaluate()
        parsed = rust_pricing.PricingResult.from_json(result.to_json())

        self.assertEqual(parsed.to_json(), result.to_json())
        self.assertEqual(parsed.value, result.value)
        self.assertEqual(parsed.independent_sampling_units, result.independent_sampling_units)
        self.assertAlmostEqual(parsed.estimator_variance, result.standard_error ** 2)

    def test_pretty_json_helpers_round_trip(self):
        request = rust_pricing.PricingRequest.from_json(self.request_json)
        pretty_request = request.to_pretty_json()
        self.assertIn("\n  ", pretty_request)
        self.assertEqual(
            rust_pricing.PricingRequest.from_json(pretty_request).to_json(),
            request.to_json(),
        )

        result = rust_pricing.PricingPlan.compile(
            request, worker_threads=2, reduction_block_size=256
        ).evaluate()
        pretty_result = result.to_pretty_json()
        self.assertIn("\n  ", pretty_result)
        self.assertEqual(
            rust_pricing.PricingResult.from_json(pretty_result).to_json(),
            result.to_json(),
        )

    def test_pricing_result_from_json_error_is_structured(self):
        request = rust_pricing.PricingRequest.from_json(self.request_json)
        result = rust_pricing.PricingPlan.compile(
            request, worker_threads=2, reduction_block_size=256
        ).evaluate()
        payload = json.loads(result.to_json())
        payload["value"]["standard_error"] = -0.5
        invalid = json.dumps(payload, separators=(",", ":"))
        with self.assertRaises(rust_pricing.ValidationError) as captured:
            rust_pricing.PricingResult.from_json(invalid)

        issue = captured.exception.issues[0]
        self.assertEqual(issue.document_kind, "pricing_result")
        self.assertEqual(issue.instance_path, "/value")

    def test_runtime_docstrings_are_available(self):
        self.assertIn("discount-factor curve", rust_pricing.DiscountCurve.__doc__)
        self.assertIn("Python GIL", rust_pricing.PricingPlan.compile.__doc__)
        self.assertIn("pricing-request JSON Schema", rust_pricing.request_json_schema.__doc__)
        self.assertIn("pricing-result JSON Schema", rust_pricing.result_json_schema.__doc__)
        self.assertIn("statistical estimate", rust_pricing.DiagnosticEstimate.__doc__)
        self.assertIn("Standard error", rust_pricing.DiagnosticEstimate.standard_error.__doc__)
        self.assertIn("raw and market-scaled", rust_pricing.RiskEstimate.__doc__)
        self.assertIn("CRN bump validation", rust_pricing.RiskValidation.__doc__)
        self.assertIn(
            "bump-and-revalue", rust_pricing.RiskValidation.bump_and_revalue.__doc__
        )

    def test_bundled_json_schemas_are_exported(self):
        request_schema = json.loads(rust_pricing.request_json_schema())
        result_schema = json.loads(rust_pricing.result_json_schema())

        self.assertEqual(
            request_schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertEqual(request_schema["properties"]["document_kind"]["const"], "pricing_request")
        self.assertEqual(
            result_schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertEqual(result_schema["properties"]["document_kind"]["const"], "pricing_result")

    def test_vega_kt_result_api_is_exported(self):
        exported = [
            "VegaKtResult",
            "VegaKtCoordinate",
            "VegaKtBucketEstimate",
            "VegaKtProjection",
            "VegaKtResidualDiagnostics",
            "VegaKtReportingStats",
        ]
        for name in exported:
            self.assertTrue(hasattr(rust_pricing, name), name)
        self.assertTrue(hasattr(rust_pricing.PricingResult, "vega_kt"))

    def test_native_builder_error_is_structured(self):
        curve = rust_pricing.DiscountCurve(1, [0.0, 1.0], [1.0, 0.95])
        with self.assertRaises(rust_pricing.ValidationError) as captured:
            rust_pricing.Market.equity(2, 1, -100.0, curve, curve)
        issue = captured.exception.issues[0]
        self.assertEqual(issue.pointer, "/market/spot")
        self.assertEqual(issue.instance_path, "/market/spot")
        self.assertEqual(issue.schema_version, 1)
        self.assertEqual(issue.document_kind, "pricing_request")
        self.assertEqual(issue.code, "invalid_spot")

    def test_validation_error_has_immutable_structured_issues(self):
        invalid = self.request_json.replace('"schema_version":1', '"schema_version":99')
        with self.assertRaises(rust_pricing.ValidationError) as captured:
            rust_pricing.PricingRequest.from_json(invalid)

        issues = captured.exception.issues
        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.phase, "declared_schema")
        self.assertEqual(issue.code, "unsupported_schema_version")
        self.assertEqual(issue.to_dict()["pointer"], "")
        self.assertEqual(issue.to_dict()["instance_path"], "")
        self.assertEqual(issue.to_dict()["schema_version"], 99)
        self.assertEqual(issue.to_dict()["document_kind"], "pricing_request")
        with self.assertRaises(AttributeError):
            issue.code = "changed"
        with self.assertRaises(AttributeError):
            issue.__dict__["code"] = "changed"
        payload = issue.to_dict()
        payload["code"] = "changed"
        self.assertEqual(issue.to_dict()["code"], "unsupported_schema_version")

    def test_validation_issue_equality_compares_payload(self):
        invalid_schema = self.request_json.replace('"schema_version":1', '"schema_version":99')
        with self.assertRaises(rust_pricing.ValidationError) as first:
            rust_pricing.PricingRequest.from_json(invalid_schema)
        with self.assertRaises(rust_pricing.ValidationError) as second:
            rust_pricing.PricingRequest.from_json(invalid_schema)
        invalid_spot = self.request_json.replace('"spot":100.0', '"spot":-100.0')
        with self.assertRaises(rust_pricing.ValidationError) as third:
            rust_pricing.PricingRequest.from_json(invalid_spot)

        self.assertEqual(first.exception.issues[0], second.exception.issues[0])
        self.assertNotEqual(first.exception.issues[0], third.exception.issues[0])

    def test_json_domain_error_reports_instance_path(self):
        invalid = self.request_json.replace(
            '"valuation_date":"2026-09-04"', '"valuation_date":"2026-02-31"'
        )
        with self.assertRaises(rust_pricing.ValidationError) as captured:
            rust_pricing.PricingRequest.from_json(invalid)

        issue = captured.exception.issues[0]
        self.assertEqual(issue.phase, "domain")
        self.assertEqual(issue.instance_path, "/valuation_date")
        self.assertEqual(issue.to_dict()["instance_path"], "/valuation_date")

    def test_json_market_error_reports_instance_path(self):
        invalid = self.request_json.replace('"spot":100.0', '"spot":-100.0')
        with self.assertRaises(rust_pricing.ValidationError) as captured:
            rust_pricing.PricingRequest.from_json(invalid)

        issue = captured.exception.issues[0]
        self.assertEqual(issue.phase, "domain")
        self.assertEqual(issue.instance_path, "/market/spot")
        self.assertEqual(issue.code, "invalid_domain_value")


if __name__ == "__main__":
    unittest.main()
