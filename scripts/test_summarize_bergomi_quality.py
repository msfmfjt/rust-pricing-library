"""Reject incomplete or misattributed Bergomi evidence without running Monte Carlo."""

from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import summarize_bergomi_quality as summary

SOURCE = b"synthetic parser-test source, not numerical evidence\n"
COMMIT = "1" * 40
COMPLETION = "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out; finished in 1.00s"


def synthetic_panel() -> tuple[list[dict], list[dict]]:
    """Structurally complete dummy data; never used as model-quality evidence."""
    runs, reports = [], []
    for smile in summary.SMILES:
        common = {"smile": smile, "model": dict(summary.MODEL), "settings": dict(summary.SETTINGS)}
        nodes = []
        for maturity, strike in summary.COORDINATES:
            node = {key: 0.0 for key in summary.NODE_FIELDS}
            node.update(maturity=maturity, log_strike=strike, target_iv=0.2,
                        model_iv=0.2, lv_iv=0.2, minimum_ess=1000.0, quote_fallbacks=0)
            nodes.append(node)
        reports.append({
            **common, "nodes": nodes, "quote_fallbacks": 0,
            "martingales": [{"value": 100.0, "se": 0.01} for _ in range(3)],
        })
        for seed in summary.SEEDS:
            runs.append({**copy.deepcopy(common), "run": {
                "calibration_seed": seed, "pricing_seed": seed ^ summary.PRICING_SEED_MASK,
                "estimates": [{"value": 1.0, "se": 0.01} for _ in range(48)],
                "support": [{
                    "time": t, "minimum_quote_ess": 1000.0, "quote_fallbacks": 0,
                    "all_row_fallbacks": 7, "particle_mean_f": 100.0,
                } for t in summary.MATURITIES],
                "quote_support": [{
                    "maturity": t, "log_strike": x, "minimum_ess": 1000.0, "fallbacks": 0,
                } for t, x in summary.COORDINATES],
            }})
    return runs, reports


def log_of(runs: list[dict], reports: list[dict]) -> str:
    lines = ["    Finished `release` profile [optimized] target(s) in 0.01s", "running 2 tests"]
    for smile, name in zip(summary.SMILES, ("flat", "skew")):
        lines.append(f"test iv_round_trip_{name}_one_factor_bergomi ... ")
        lines.extend(summary.RUN_PREFIX + json.dumps(run) for run in runs if run["smile"] == smile)
        lines.extend(summary.QUALITY_PREFIX + json.dumps(report) for report in reports if report["smile"] == smile)
        lines.append("ok")
    lines.append(COMPLETION)
    return "\n".join(lines) + "\n"


def build(log: str, **overrides) -> dict:
    metadata = {"source_commit": COMMIT, "platform": "synthetic-test-target",
                "compiler": "synthetic-test-compiler", "test_source": SOURCE}
    metadata.update(overrides)
    return summary.build(log, **metadata)


class BergomiSummaryTests(unittest.TestCase):
    def test_complete_log_preserves_all_runs_and_source_identity(self):
        runs, reports = synthetic_panel()
        log = log_of(runs, reports)
        document = build(log)
        self.assertEqual(document["complete_runs"], runs)
        self.assertEqual(document["acceptance_reports"], reports)
        self.assertEqual(document["source_commit"], COMMIT)
        self.assertEqual(len(document["test_source_sha256"]), 64)
        self.assertNotEqual(document["test_source_sha256"], build(log, test_source=b"changed")["test_source_sha256"])
        self.assertNotEqual(document["log_sha256"], build("\n" + log)["log_sha256"])
        self.assertEqual(document, build(log))
        self.assertEqual(build(log.replace("\n", "\r\n"))["complete_runs"], runs)

    def test_missing_duplicate_or_reordered_panel_entries_are_rejected(self):
        def remove_report(runs, reports):
            reports.pop()
        def duplicate_report(runs, reports):
            reports[1] = copy.deepcopy(reports[0])
        def remove_run(runs, reports):
            runs.pop()
        def duplicate_run(runs, reports):
            runs[-1] = copy.deepcopy(runs[-2])
        def missing_quote(runs, reports):
            reports[0]["nodes"].pop()
        def duplicate_quote(runs, reports):
            reports[0]["nodes"][1] = copy.deepcopy(reports[0]["nodes"][0])
        def reordered_quote(runs, reports):
            reports[0]["nodes"].reverse()
        def missing_raw_quote(runs, reports):
            runs[0]["run"]["quote_support"].pop()
        def missing_estimate(runs, reports):
            runs[0]["run"]["estimates"].pop()
        def missing_martingale(runs, reports):
            reports[0]["martingales"].pop()
        def missing_support(runs, reports):
            runs[0]["run"]["support"].pop()
        def missing_metric(runs, reports):
            reports[0]["nodes"][0].pop("total_se_bp")
        for mutate in (remove_report, duplicate_report, remove_run, duplicate_run,
                       missing_quote, duplicate_quote, reordered_quote, missing_raw_quote,
                       missing_estimate, missing_martingale, missing_support, missing_metric):
            runs, reports = synthetic_panel()
            mutate(runs, reports)
            with self.subTest(mutation=mutate.__name__), self.assertRaises(summary.SummaryError):
                build(log_of(runs, reports))

    def test_configuration_and_seed_mismatches_are_rejected(self):
        for location, key, value in (
            ("run", "calibration_seed", 43), ("run", "pricing_seed", 42),
            ("settings", "particles", 32768), ("settings", "bandwidth", 0.02),
            ("model", "nu", 1.5),
        ):
            runs, reports = synthetic_panel()
            runs[0][location][key] = value
            with self.subTest(location=location, key=key), self.assertRaises(summary.SummaryError):
                build(log_of(runs, reports))
        runs, reports = synthetic_panel()
        reports[1]["settings"]["steps"] = 512
        with self.assertRaises(summary.SummaryError):
            build(log_of(runs, reports))

    def test_nonfinite_and_wrong_type_metrics_are_rejected_everywhere(self):
        for value in (None, float("nan"), float("inf"), -float("inf"), "0", False):
            for section in ("node", "estimate", "support"):
                runs, reports = synthetic_panel()
                if section == "node":
                    reports[0]["nodes"][0]["total_se_bp"] = value
                elif section == "estimate":
                    runs[0]["run"]["estimates"][0]["value"] = value
                else:
                    runs[0]["run"]["support"][0]["minimum_quote_ess"] = value
                with self.subTest(value=value, section=section), self.assertRaises(summary.SummaryError):
                    build(log_of(runs, reports))
        runs, reports = synthetic_panel()
        log = log_of(runs, reports).replace('"total_se_bp": 0.0', '"total_se_bp": 1e999', 1)
        with self.assertRaises(summary.SummaryError):
            build(log)

    def test_negative_uncertainty_and_quote_fallbacks_are_rejected(self):
        for location in ("estimate", "node", "quote", "support", "report"):
            runs, reports = synthetic_panel()
            if location == "estimate":
                runs[0]["run"]["estimates"][0]["se"] = -0.01
            elif location == "node":
                reports[0]["nodes"][0]["total_se_bp"] = -0.01
            elif location == "quote":
                runs[0]["run"]["quote_support"][0]["fallbacks"] = 1
            elif location == "support":
                runs[0]["run"]["support"][0]["quote_fallbacks"] = 1
            else:
                reports[0]["quote_fallbacks"] = 1
            with self.subTest(location=location), self.assertRaises(summary.SummaryError):
                build(log_of(runs, reports))

    def test_failed_truncated_or_wrong_test_execution_is_rejected(self):
        log = log_of(*synthetic_panel())
        bad_logs = (
            log.replace(COMPLETION, ""),
            log.replace(COMPLETION, "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 9 filtered out; finished in 1.00s"),
            log + COMPLETION + "\n",
            COMPLETION + "\n" + log.replace(COMPLETION, ""),
            log.replace("Finished `release` profile", "Finished `test` profile"),
            log.replace("test iv_round_trip_skew_one_factor_bergomi", "test unrelated_test"),
            log.replace("2 passed; 0 failed", "1 passed; 0 failed"),
        )
        for number, candidate in enumerate(bad_logs):
            with self.subTest(number=number), self.assertRaises(summary.SummaryError):
                build(candidate)

    def test_duplicate_json_keys_and_missing_provenance_are_rejected(self):
        log = log_of(*synthetic_panel())
        with self.assertRaises(summary.SummaryError):
            build(log.replace('"calibration_seed": 42', '"calibration_seed": 42, "calibration_seed": 42', 1))
        for override in ({"source_commit": "origin/main"}, {"source_commit": "1" * 39},
                         {"platform": " "}, {"compiler": ""}, {"test_source": b""}):
            with self.subTest(override=override), self.assertRaises(summary.SummaryError):
                build(log, **override)

    def test_cli_check_and_failed_log_never_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory)
            log, source, output = root / "run.log", root / "test.rs", root / "evidence.json"
            log.write_text(log_of(*synthetic_panel()))
            source.write_bytes(SOURCE)
            args = ["--log", str(log), "--out", str(output), "--test-file", str(source),
                    "--source-commit", COMMIT, "--platform", "synthetic-test-target",
                    "--compiler", "synthetic-test-compiler"]
            self.assertEqual(summary.main(args), 0)
            before = output.read_bytes()
            self.assertEqual(summary.main(args + ["--check"]), 0)
            source.write_bytes(b"changed source")
            self.assertEqual(summary.main(args + ["--check"]), 1)
            self.assertEqual(output.read_bytes(), before)
            log.write_text(log.read_text().replace(COMPLETION, ""))
            self.assertEqual(summary.main(args), 1)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
