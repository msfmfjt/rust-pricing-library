import json
import math
import unittest
from datetime import date
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
        self.assertEqual(result.independent_sampling_units, 1024)
        self.assertEqual(result.evaluated_paths, 2048)
        self.assertIsNone(result.delta_raw)

        payload = json.loads(result.to_json())
        self.assertEqual(payload["document_kind"], "pricing_result")
        self.assertEqual(result.diagnostics.estimator, "pseudo_monte_carlo")
        self.assertEqual(result.diagnostics.worker_threads, 2)
        self.assertEqual(
            [warning.code for warning in result.diagnostics.warnings],
            [warning.code for warning in result.warnings],
        )
        with self.assertRaises(AttributeError):
            result.diagnostics.master_seed = 99

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

    def test_runtime_docstrings_are_available(self):
        self.assertIn("discount-factor curve", rust_pricing.DiscountCurve.__doc__)
        self.assertIn("Python GIL", rust_pricing.PricingPlan.compile.__doc__)

    def test_native_builder_error_is_structured(self):
        curve = rust_pricing.DiscountCurve(1, [0.0, 1.0], [1.0, 0.95])
        with self.assertRaises(rust_pricing.ValidationError) as captured:
            rust_pricing.Market.equity(2, 1, -100.0, curve, curve)
        issue = captured.exception.issues[0]
        self.assertEqual(issue.pointer, "/market/spot")
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
        with self.assertRaises(AttributeError):
            issue.code = "changed"


if __name__ == "__main__":
    unittest.main()
