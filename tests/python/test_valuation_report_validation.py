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
