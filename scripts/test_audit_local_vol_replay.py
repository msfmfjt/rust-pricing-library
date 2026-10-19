"""Mutations of retained native captures must not be silently approved."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from audit_local_vol_replay import ROOT, audit_candidate, differences


class ReplayAuditTests(unittest.TestCase):
    def setUp(self):
        self.old = ROOT / "fixtures/replay/historical/local_volatility-macos-aarch64-before-node-allocation-audit.json"
        self.current = ROOT / "fixtures/replay/historical/local_volatility-macos-aarch64-before-terminal-replication.json"
        self.document = json.loads(self.current.read_text())

    def test_reviewed_correction_is_classified_without_accepting_it(self):
        report = audit_candidate(self.old, self.current)
        self.assertEqual(report["difference_count"], 46)
        self.assertEqual(report["status"], "requires_native_numerical_review")
        self.assertFalse(report["paper_vegakt_accepted"])
        self.assertFalse(report["exact_fixture_match"])

    def test_unchanged_fixture_is_exact(self):
        report = audit_candidate(self.current, self.current)
        self.assertEqual(report["status"], "unchanged")
        self.assertEqual(report["differences"], [])

    def test_price_delta_input_and_invalid_diagnostic_changes_fail(self):
        changes = [
            ("price", lambda d: d["cases"][2]["result"]["value"].update(value=1.0)),
            ("delta", lambda d: d["cases"][2]["result"]["risks"]["delta"]["raw"].update(value=1.0)),
            ("input", lambda d: d["cases"][2]["request"].update(valuation_date="2026-01-02")),
            ("invalid_label", lambda d: d["cases"][2]["result"]["risks"]["vega_kt"].update(truncation_order=99)),
            ("unreviewed_field", lambda d: d["cases"][2]["result"]["risks"]["vega_kt"].update(unexpected=True)),
        ]
        for name, mutation in changes:
            document = copy.deepcopy(self.document)
            mutation(document)
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "candidate.json"
                path.write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))
                with self.assertRaises((ValueError, SystemExit)):
                    audit_candidate(self.old, path)

    def test_wrong_label_and_corrupted_bucket_are_rejected(self):
        for key, value in [("policy_label", "equation_11_first_order_v1"), ("raw_buckets", [0.0] * 6)]:
            document = copy.deepcopy(self.document)
            document["cases"][2]["result"]["risks"]["vega_kt"][key] = value
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "candidate.json"
                path.write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))
                with self.assertRaises(ValueError):
                    audit_candidate(self.old, path)

    def test_cross_platform_comparison_is_rejected(self):
        windows = ROOT / "fixtures/replay/local_volatility-windows-x86_64.json"
        with self.assertRaises(ValueError):
            audit_candidate(windows, self.current)

    def test_projection_accounting_cannot_diverge_from_residual_diagnostics(self):
        for field in ["scalar_vega", "signed_residual"]:
            document = copy.deepcopy(self.document)
            document["cases"][2]["result"]["risks"]["vega_kt"]["projection"][field] += 1.0
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / "candidate.json"
                candidate.write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))
                with self.assertRaises(ValueError):
                    audit_candidate(self.old, candidate)

    def test_structural_changes_and_number_types_are_visible(self):
        self.assertTrue(differences({"x": 1}, {"x": 1.0}))
        self.assertTrue(differences({"x": []}, {"x": [0]}))
        self.assertTrue(differences({"x": {}}, {"x": {"new": None}}))

    def test_cli_cannot_overwrite_the_capture_or_a_fixture(self):
        original = self.current.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "candidate.json"
            capture.write_bytes(original)
            for destination in [capture, self.current]:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/audit_local_vol_replay.py"),
                     str(capture), "--output", str(destination)],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must not overwrite", result.stderr)
            self.assertEqual(capture.read_bytes(), original)
        self.assertEqual(self.current.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
