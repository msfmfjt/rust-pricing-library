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

        native_result = rust_pricing.PricingPlan.compile(
            native, worker_threads=2, reduction_block_size=256
        ).evaluate()
        json_result = rust_pricing.PricingPlan.compile(
            from_json, worker_threads=2, reduction_block_size=256
        ).evaluate()
        self.assertEqual(native_result.to_json(), json_result.to_json())

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
