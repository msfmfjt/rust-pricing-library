"""Report domain errors point into the actual saved report, including v1 inputs."""
import copy
import json
from pathlib import Path
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def document(version):
    path = ROOT / f"fixtures/report-v{version}/multi_asset_joint.golden.json"
    return json.loads(path.read_text())


def locate(value, pointer):
    for key in pointer.split("/")[1:]:
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


class ValuationReportValidationTests(unittest.TestCase):
    def assert_pointer(self, value, expected):
        # A diagnostic location must resolve to a real field or containing block.
        locate(value, expected)
        with self.assertRaises(rp.ValidationError) as caught:
            rp.ValuationReport.from_json(json.dumps(value))
        error = caught.exception
        self.assertIsInstance(error, ValueError)
        self.assertEqual(error.code, "invalid_domain_value")
        self.assertEqual(error.stage, "parse")
        self.assertEqual(error.field, expected)
        self.assertEqual(error.issues[0].pointer, expected)
        self.assertEqual(error.issues[0].document_kind, "valuation_report")
        self.assertEqual(error.issues[0].schema_version, value["schema_version"])

    def test_wire_errors_keep_the_validated_envelope_version(self):
        for version in (1, 2):
            for field, value, code in (
                ("standard_error", "not a number", "invalid_json"),
                ("document_kind", "pricing_task", "wrong_document_kind"),
                ("schema_version", 99, "unsupported_schema_version"),
            ):
                with self.subTest(version=version, field=field):
                    report = document(version)
                    report[field] = value
                    with self.assertRaises(rp.ValidationError) as caught:
                        rp.ValuationReport.from_json(json.dumps(report))
                    issue = caught.exception.issues[0]
                    self.assertEqual(issue.code, code)
                    self.assertEqual(issue.schema_version, report["schema_version"])
                    self.assertEqual(issue.document_kind, "valuation_report")
        report = document(1)
        report["calibration_diagnostics"] = []
        self.assert_pointer(report, "/calibration_diagnostics")

    def test_untrusted_envelopes_keep_the_current_version_fallback(self):
        # No version is trusted before syntax, duplicate-member and limit checks.
        deep = ('{"document_kind":"valuation_report","schema_version":1,"x":'
                + '[' * 65 + '0' + ']' * 65 + '}')
        for text, code in (
            ('{"document_kind":"valuation_report","schema_version":1', "invalid_json"),
            ('{"document_kind":"valuation_report","schema_version":1,"schema_version":2}',
             "invalid_json"),
            ('{"schema_version":1}', "invalid_json"),
            ('{"document_kind":"valuation_report","schema_version":-1}', "invalid_json"),
            (deep, "resource_limit"),
        ):
            with self.subTest(code=code, text=text[:80]):
                with self.assertRaises(rp.ValidationError) as caught:
                    rp.ValuationReport.from_json(text)
                self.assertEqual(caught.exception.issues[0].schema_version, 2)
                self.assertEqual(caught.exception.issues[0].code, code)

    def test_scalar_and_replay_failures_identify_their_actual_fields(self):
        for version in (1, 2):
            original = document(version)
            for field, value in (
                ("standard_error", -1.0),
                ("independent_sampling_units", 0),
                ("evaluated_paths", "0"),
                ("library_version", ""),
                ("platform", ""),
                ("plan_fingerprint", "blake3-256:invalid"),
            ):
                with self.subTest(version=version, field=field):
                    report = copy.deepcopy(original)
                    report[field] = value
                    self.assert_pointer(report, "/" + field)

    def test_risk_failures_identify_the_affected_block(self):
        for version in (1, 2):
            original = document(version)
            for index, field, value in (
                (0, "values", []),
                (1, "standard_errors", [-1.0]),
                (1, "convention", ""),
            ):
                with self.subTest(version=version, index=index, field=field):
                    report = copy.deepcopy(original)
                    report["risks"][index][field] = value
                    self.assert_pointer(report, f"/risks/{index}")
            with self.subTest(version=version, duplicate=True):
                report = copy.deepcopy(original)
                index = len(report["risks"])
                report["risks"].append(copy.deepcopy(report["risks"][0]))
                self.assert_pointer(report, f"/risks/{index}")


if __name__ == "__main__":
    unittest.main()
