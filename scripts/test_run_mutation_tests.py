"""Regression tests for mutation selection and failure/evidence handling."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_mutation_tests as runner


def completed_report():
    return {
        "cargo_mutants_version": runner.CARGO_MUTANTS_VERSION,
        "end_time": "2026-10-22T12:00:00Z",
        "total_mutants": 2, "caught": 2, "missed": 0, "timeout": 0,
        "unviable": 0, "success": 0,
        "outcomes": [{"scenario": "Baseline", "summary": "Success",
                      "phase_results": [{"phase": "Build"}, {"phase": "Test", "argv": [
                          "cargo", "test", "--package=pricing@0.1.0", "--package=pricing-numerics",
                      ]}]},
                     {"scenario": {"Mutant": {"name": "fixture.rs:1: replace first"}}, "summary": "CaughtMutant"},
                     {"scenario": {"Mutant": {"name": "fixture.rs:2: replace second"}}, "summary": "CaughtMutant"}],
    }


class OutcomeTests(unittest.TestCase):
    def test_completed_test_run(self):
        self.assertEqual(runner.check_outcomes(completed_report(), 2)["caught"], 2)

    def test_incomplete_or_inconsistent_evidence_is_rejected(self):
        cases = [
            {"end_time": None}, {"total_mutants": 1}, {"caught": 1},
            {"caught": True}, {"missed": -1}, {"success": 2, "caught": 0},
            {"unviable": 2, "caught": 0}, {"cargo_mutants_version": "unreviewed"},
            {"outcomes": []},
            {"outcomes": [{"scenario": "Baseline", "summary": "Failure"}]},
            {"outcomes": [{"scenario": "Baseline", "summary": "Success",
                           "phase_results": [{"phase": "Build"}]}]},
            {"outcomes": [{"scenario": "Baseline", "summary": "Success",
                           "phase_results": [{"phase": "Test", "argv": [
                               "cargo", "test", "--package=pricing-numerics",
                           ]}]}]},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.check_outcomes(completed_report() | changes, 2)

    def test_missed_and_timeout_are_separate_from_caught(self):
        report = completed_report() | {"caught": 0, "missed": 1, "timeout": 1}
        report["outcomes"][1]["summary"] = "MissedMutant"
        report["outcomes"][2]["summary"] = "Timeout"
        self.assertEqual(runner.check_outcomes(report, 2)["caught"], 0)

    def test_aggregate_success_cannot_hide_missing_duplicate_or_wrong_records(self):
        report = completed_report()
        baseline, first, second = report["outcomes"]
        for outcomes in ([baseline], [baseline, first, first],
                         [baseline, first, second | {"summary": "MissedMutant"}],
                         [baseline, first, second | {"summary": "Unknown"}],
                         [baseline, first, {"scenario": "Unexpected", "summary": "CaughtMutant"}],
                         [baseline, first, None], [baseline, baseline, first, second], None):
            with self.subTest(outcomes=outcomes), self.assertRaises(ValueError):
                runner.check_outcomes(report | {"outcomes": outcomes}, 2)


class SelectionTests(unittest.TestCase):
    def test_full_numerics_also_tests_pricing_consumers(self):
        args = runner.parse_args(["full", "--package", "pricing-numerics", "--shard", "0/4"])
        command = runner.selection_args(args, Path("."), Path("."))
        self.assertIn("--package=pricing-numerics", command)
        self.assertIn("--test-package=pricing", command)
        self.assertIn("--test-package=pricing-numerics", command)
        self.assertIn("--cargo-arg=--package=pricing", command)
        self.assertIn("--cargo-arg=--package=pricing-numerics", command)
        self.assertIn("--baseline=run", command)
        self.assertIn("--sharding=round-robin", command)
        self.assertFalse(any("--lib" in item for item in command))

    def test_invalid_or_ambiguous_scope_arguments_fail(self):
        for argv in (["changed"], ["smoke", "--file", "*.rs"], ["full", "--base", "main"],
                     ["--jobs", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(argv)

    def test_changed_uses_stack_merge_base_and_actual_worktree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*argv):
                return subprocess.run(["git", *argv], cwd=root, check=True, capture_output=True,
                                      encoding="utf-8").stdout.strip()

            git("init", "-b", "main")
            git("config", "user.name", "Mutation fixture")
            git("config", "user.email", "mutation@example.invalid")
            source = root / "crates/pricing/src/value.rs"
            source.parent.mkdir(parents=True)
            source.write_text("fn parent() -> u32 { 1 }\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-m", "root")
            git("switch", "-c", "stack-base")
            source.write_text("fn parent() -> u32 { 2 }\n", encoding="utf-8")
            git("commit", "-am", "parent change")
            stack_base = git("rev-parse", "HEAD")
            git("switch", "-c", "child")
            source.write_text(source.read_text() + "fn child() -> u32 { 3 }\n", encoding="utf-8")
            git("commit", "-am", "child change")
            source.write_text(source.read_text() + "fn local() -> u32 { 4 }\n", encoding="utf-8")
            # Advancing the base independently must not enter this child's diff.
            git("branch", "saved-child")
            git("stash", "push")
            git("switch", "stack-base")
            (root / "unrelated.txt").write_text("base advanced\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-m", "advance base")
            git("switch", "child")
            git("stash", "pop")
            git("config", "diff.noprefix", "true")
            git("config", "color.ui", "always")
            args = runner.parse_args(["changed", "--base", "stack-base"])
            command = runner.selection_args(args, root, root)
            diff = (root / "changes.diff").read_text()
            self.assertIn("+++ b/crates/pricing/src/value.rs", diff)
            self.assertIn("+fn child()", diff)
            self.assertIn("+fn local()", diff)
            self.assertNotIn("+fn parent()", diff)
            self.assertNotIn("unrelated.txt", diff)
            self.assertEqual((root / "merge-base.txt").read_text().strip(), stack_base)
            self.assertIn(f"--in-diff={root / 'changes.diff'}", command)

            new_file = source.with_name("new.rs")
            new_file.write_text("fn new() -> u32 { 7 }\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Stage new Rust"):
                runner.selection_args(args, root, root)
            git("add", str(new_file))
            runner.selection_args(args, root, root)
            self.assertIn("+fn new()", (root / "changes.diff").read_text())


class ExecutionTests(unittest.TestCase):
    def execute(self, argv, *, selected=2, exit_code=0, report=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = runner.parse_args(argv)

            def capture(command, cwd):
                if command[:3] == ["cargo", "mutants", "--version"]:
                    return f"cargo-mutants {runner.CARGO_MUTANTS_VERSION}\n"
                if "--list" in command:
                    return json.dumps([{}] * selected) if selected else ""
                return "fixture-head\n" if "rev-parse" in command else ""

            def run(command, **kwargs):
                output = Path(next(item.split("=", 1)[1] for item in command if item.startswith("--output=")))
                destination = output / "mutants.out"
                destination.mkdir()
                document = completed_report() if report is None else report
                (destination / "outcomes.json").write_text(json.dumps(document), encoding="utf-8")
                return subprocess.CompletedProcess(command, exit_code)

            with patch.object(runner, "capture", side_effect=capture), \
                    patch.object(runner.subprocess, "run", side_effect=run) as execute, \
                    patch.dict(runner.os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = runner.run(args, root)
            record = json.loads(next((root / "target/mutation").glob("*/run.json")).read_text())
            return code, record, execute.call_count

    def test_success_and_read_only_inventory(self):
        code, record, calls = self.execute(["smoke"])
        self.assertEqual((code, record["status"], calls), (0, "passed", 1))
        code, record, calls = self.execute(["smoke", "--list"])
        self.assertEqual((code, record["status"], calls), (0, "listed_only", 0))

    def test_empty_changed_diff_is_explicit_and_empty_smoke_is_error(self):
        code, record, calls = self.execute(["changed", "--base", "fixture"], selected=0)
        self.assertEqual((code, record["status"], calls), (0, "no_mutants", 0))
        code, record, calls = self.execute(["smoke"], selected=0)
        self.assertEqual((code, record["status"], calls), (1, "error", 0))

    def test_tool_failures_and_false_success_cannot_pass(self):
        for status in (1, 2, 3, 4, 5, 6, 70, -15):
            with self.subTest(status=status):
                self.assertNotEqual(self.execute(["smoke"], exit_code=status)[0], 0)
        for changes in ({"end_time": None}, {"missed": 1, "caught": 1},
                        {"timeout": 1, "caught": 1}):
            with self.subTest(changes=changes):
                code, record, _ = self.execute(["smoke"], report=completed_report() | changes)
                self.assertNotEqual(code, 0)
                self.assertNotEqual(record["status"], "passed")


if __name__ == "__main__":
    unittest.main()
