"""Input-envelope versions survive request/result/specification validation."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def documents():
    for version in (1, 2, 3):
        for name, cls, pointer in (("request", rp.PricingRequest, ("market", "spot")),
                                   ("result", rp.PricingResult, ("value", "standard_error"))):
            yield cls, json.loads((ROOT / f"fixtures/v{version}/pricing_{name}.golden.json").read_text()), pointer, 3
    for version in (1, 2):
        value = json.loads((ROOT / "fixtures/specification-v1/black_scholes.golden.json").read_text())
        value["schema_version"] = version
        yield rp.PricingSpecification, value, ("market", "spot"), 2


class LegacyWireContextTests(unittest.TestCase):
    def issue(self, cls, value, version, code):
        with self.assertRaises(rp.ValidationError) as caught:
            cls.from_json(value if isinstance(value, str) else json.dumps(value))
        issue = caught.exception.issues[0]
        self.assertEqual(issue.schema_version, version)
        self.assertEqual(issue.code, code)
        self.assertEqual(caught.exception.code, code)
        return issue

    def test_domain_errors_keep_input_versions_and_pointers(self):
        for cls, value, pointer, _ in documents():
            with self.subTest(cls=cls.__name__, version=value["schema_version"]):
                value[pointer[0]][pointer[1]] = -1
                issue = self.issue(cls, value, value["schema_version"], "invalid_domain_value")
                self.assertEqual(issue.pointer, "/value" if cls is rp.PricingResult else "/" + "/".join(pointer))
                self.assertEqual(issue.document_kind, value["document_kind"])

    def test_header_and_body_schema_errors_keep_declared_versions(self):
        for cls, value, pointer, _ in documents():
            version = value["schema_version"]
            for field, replacement, expected, code in (
                ("document_kind", "wrong", version, "wrong_document_kind"),
                ("schema_version", 99, 99, "unsupported_schema_version"),
                (pointer[0], "wrong type", version, "invalid_json"),
            ):
                with self.subTest(cls=cls.__name__, version=version, field=field):
                    invalid = deepcopy(value)
                    invalid[field] = replacement
                    self.issue(cls, invalid, expected, code)

    def test_rejected_envelopes_keep_historical_fallback(self):
        for cls, value, _, fallback in documents():
            header = json.dumps({k: value[k] for k in ("document_kind", "schema_version")})
            for source in (header[:-1], header[:-1] + ',"schema_version":2}',
                           header[:-1] + ',"nested":{"x":1,"x":2}}',
                           '{"schema_version":1}', '{"document_kind":7,"schema_version":1}',
                           '{"document_kind":"pricing_request","schema_version":-1}'):
                with self.subTest(cls=cls.__name__, source=source):
                    self.issue(cls, source, fallback, "invalid_json")
            self.issue(cls, header[:-1] + ',"nested":' + '[' * 65 + '0' + ']' * 65 + '}',
                       fallback, "resource_limit")

    def test_unsupported_request_features_keep_input_version(self):
        for version in (1, 2, 3):
            value = json.loads((ROOT / f"fixtures/v{version}/pricing_request.golden.json").read_text())
            value["product"] = {"type": "graph"}
            with self.subTest(version=version):
                self.issue(rp.PricingRequest, value, version, "unsupported_schema_feature")

    def test_successful_legacy_and_rich_result_round_trips(self):
        for cls, value, _, _ in documents():
            with self.subTest(cls=cls.__name__, version=value["schema_version"]):
                parsed = cls.from_json(json.dumps(value))
                canonical = parsed.to_json()
                self.assertEqual(cls.from_json(canonical).to_json(), canonical)
                if cls is rp.PricingResult:
                    rich = json.loads(canonical)
                    self.assertIn("monte_carlo", rich)
                    rich["monte_carlo"] = "wrong type"
                    self.issue(cls, rich, rich["schema_version"], "invalid_json")


if __name__ == "__main__":
    unittest.main()
