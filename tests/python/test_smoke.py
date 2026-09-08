import json
import math
import unittest
from pathlib import Path

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
