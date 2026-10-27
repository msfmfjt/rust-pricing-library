"""Regression tests for mutation selection and failure/evidence handling."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_mutation_tests as runner


def selected_mutants():
    return [{"name": "fixture.rs:1: replace first"}, {"name": "fixture.rs:2: replace second"}]


def phase_results(test_status="Success", build_status="Success"):
    command = ["cargo", "test", "--package=pricing@0.1.0", "--package=pricing-numerics"]
    result = [{"phase": "Build", "argv": [*command, "--no-run"], "process_status": build_status}]
    if build_status == "Success":
        result.append({"phase": "Test", "argv": command, "process_status": test_status})
    return result


def completed_report():
    return {
        "cargo_mutants_version": runner.CARGO_MUTANTS_VERSION,
        "end_time": "2026-10-22T12:00:00Z",
        "total_mutants": 2, "caught": 2, "missed": 0, "timeout": 0,
        "unviable": 0, "success": 0,
        "outcomes": [{"scenario": "Baseline", "summary": "Success",
                      "phase_results": phase_results()},
                     *[{"scenario": {"Mutant": mutant}, "summary": "CaughtMutant",
                        "phase_results": phase_results({"Failure": 101})} for mutant in selected_mutants()]],
    }


class OutcomeTests(unittest.TestCase):
    def test_completed_test_run(self):
        self.assertEqual(runner.check_outcomes(completed_report(), selected_mutants())["caught"], 2)
        self.assertEqual(runner.check_outcomes(completed_report(), selected_mutants()[::-1])["caught"], 2)

    def test_same_count_cannot_substitute_an_unselected_mutant(self):
        report = completed_report()
        report["outcomes"][1]["scenario"]["Mutant"]["name"] = "unselected.rs:1: replace unrelated"
        with self.assertRaisesRegex(ValueError, "selected inventory"):
            runner.check_outcomes(report, selected_mutants())

    def test_malformed_or_duplicate_inventory_is_rejected(self):
        for inventory in (None, {}, [{}, {}], [{"name": ""}], [{"name": 1}], [None],
                          [selected_mutants()[0]] * 2):
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                runner.check_outcomes(completed_report(), inventory)

    def test_summary_cannot_hide_missing_or_contradictory_execution(self):
        original = completed_report()
        bad_phases = [
            None, [], [None], [{"phase": "Check", "process_status": "Success"}],
            phase_results()[0:1], phase_results()[1:], phase_results()[::-1],
            phase_results({"Failure": 0}), phase_results({"Failure": True}),
            phase_results("Other"), phase_results({"Signalled": 9}),
        ]
        for phases in bad_phases:
            for index in (0, 1):
                with self.subTest(phases=phases, index=index), self.assertRaises(ValueError):
                    report = copy.deepcopy(original)
                    report["outcomes"][index]["phase_results"] = phases
                    runner.check_outcomes(report, selected_mutants())
        for index, phases in ((0, phase_results({"Failure": 101})),
                              (0, phase_results("Timeout")),
                              (1, phase_results()),
                              (1, phase_results(build_status={"Failure": 101}))):
            with self.subTest(index=index, phases=phases), self.assertRaises(ValueError):
                report = copy.deepcopy(original)
                report["outcomes"][index]["phase_results"] = phases
                runner.check_outcomes(report, selected_mutants())

    def test_each_build_and_test_must_include_the_required_packages(self):
        for index in (0, 1):
            for phase in (0, 1):
                with self.subTest(index=index, phase=phase), self.assertRaisesRegex(ValueError, "packages"):
                    report = completed_report()
                    argv = report["outcomes"][index]["phase_results"][phase]["argv"]
                    argv.remove("--package=pricing-numerics")
                    # A libtest argument cannot masquerade as Cargo selection.
                    argv.extend(["--", "--package=pricing-numerics"])
                    runner.check_outcomes(report, selected_mutants())

    def test_phase_labels_do_not_substitute_for_a_test_command(self):
        for command in (None, ["cargo", "check"], ["cargo", 7],
                        ["cargo", "test", "--package=pricing", "--package=pricing-numerics", "--no-run"]):
            with self.subTest(command=command), self.assertRaises(ValueError):
                report = completed_report()
                report["outcomes"][1]["phase_results"][1]["argv"] = command
                runner.check_outcomes(report, selected_mutants())

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
                runner.check_outcomes(completed_report() | changes, selected_mutants())

    def test_missed_and_timeout_are_separate_from_caught(self):
        report = completed_report() | {"caught": 0, "missed": 1, "timeout": 1}
        report["outcomes"][1]["summary"] = "MissedMutant"
        report["outcomes"][1]["phase_results"] = phase_results()
        report["outcomes"][2]["summary"] = "Timeout"
        report["outcomes"][2]["phase_results"] = phase_results("Timeout")
        self.assertEqual(runner.check_outcomes(report, selected_mutants())["caught"], 0)

    def test_build_timeouts_and_unviable_builds_are_not_caught(self):
        report = completed_report() | {"caught": 0, "timeout": 1, "unviable": 1}
        report["outcomes"][1]["summary"] = "Timeout"
        report["outcomes"][1]["phase_results"] = phase_results(build_status="Timeout")
        report["outcomes"][2]["summary"] = "Unviable"
        report["outcomes"][2]["phase_results"] = phase_results(build_status={"Failure": 101})
        self.assertEqual(runner.check_outcomes(report, selected_mutants())["caught"], 0)
        report["outcomes"][2]["phase_results"].append(phase_results()[1])
        with self.assertRaisesRegex(ValueError, "unsuccessful build"):
            runner.check_outcomes(report, selected_mutants())

    def test_aggregate_success_cannot_hide_missing_duplicate_or_wrong_records(self):
        report = completed_report()
        baseline, first, second = report["outcomes"]
        for outcomes in ([baseline], [baseline, first, first],
                         [baseline, first, second | {"summary": "MissedMutant"}],
                         [baseline, first, second | {"summary": "Unknown"}],
                         [baseline, first, {"scenario": "Unexpected", "summary": "CaughtMutant"}],
                         [baseline, first, None], [baseline, baseline, first, second], None):
            with self.subTest(outcomes=outcomes), self.assertRaises(ValueError):
                runner.check_outcomes(report | {"outcomes": outcomes}, selected_mutants())


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
    def execute(self, argv, *, selected=2, exit_code=0, report=None, inventory=None, unsharded=None,
                execution_error=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = runner.parse_args(argv)

            def capture(command, cwd):
                if command[:3] == ["cargo", "mutants", "--version"]:
                    return f"cargo-mutants {runner.CARGO_MUTANTS_VERSION}\n"
                if "--list" in command:
                    if unsharded is not None and not any(item.startswith("--shard=") for item in command):
                        return json.dumps(unsharded)
                    if inventory is not None:
                        return json.dumps(inventory)
                    return json.dumps(selected_mutants()[:selected]) if selected else ""
                return "fixture-head\n" if "rev-parse" in command else ""

            def run(command, **kwargs):
                if execution_error is not None:
                    raise execution_error
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

    def test_hard_termination_keeps_a_nonpassing_checkpoint(self):
        # Exit from inside a subprocess without finally/exception handlers.
        # This models a CI deadline/host kill, not a handled Ctrl-C.
        child = '''
import json, os, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import run_mutation_tests as runner
root, stage = Path(sys.argv[2]), sys.argv[3]
def capture(command, cwd):
    if command[:3] == ["cargo", "mutants", "--version"]:
        return f"cargo-mutants {runner.CARGO_MUTANTS_VERSION}"
    if "--list" in command:
        if stage == "selecting":
            os._exit(143)
        return json.dumps([{"name": "fixture.rs:1: replace first"}])
    return "fixture-head" if "rev-parse" in command else ""
def execute(*args, **kwargs):
    os._exit(143)
with patch.object(runner, "capture", side_effect=capture), \\
     patch.object(runner.subprocess, "run", side_effect=execute):
    runner.run(runner.parse_args(["smoke"]), root)
'''
        for stage in ("selecting", "running"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run(
                    [sys.executable, "-c", child, str(Path(runner.__file__).parent), temporary, stage],
                    capture_output=True, encoding="utf-8", timeout=20,
                )
                self.assertEqual(result.returncode, 143, result.stderr)
                output = next((Path(temporary) / "target/mutation").iterdir())
                record = json.loads((output / "run.json").read_text())
                self.assertEqual(record["status"], stage)
                self.assertEqual(record["head"], "fixture-head")
                self.assertTrue(record["started_at"])
                self.assertIn("--baseline=run", record["command"])
                self.assertNotIn("finished_at", record)
                self.assertNotIn("runner_exit_code", record)
                self.assertNotIn("counts", record)
                if stage == "running":
                    self.assertEqual(record["selected"], 1)
                    self.assertEqual(len(json.loads((output / "selected.json").read_text())), 1)

    def test_handled_interruptions_and_errors_finish_with_a_failure_status(self):
        for error, expected_code, status in ((KeyboardInterrupt(), 130, "interrupted"),
                                              (OSError("tool unavailable"), 1, "error")):
            with self.subTest(status=status):
                code, record, _ = self.execute(["smoke"], execution_error=error)
                self.assertEqual((code, record["status"], record["runner_exit_code"]),
                                 (expected_code, status, expected_code))
                self.assertLessEqual(record["started_at"], record["finished_at"])
                self.assertNotIn("counts", record)

    def test_interrupted_checkpoint_write_preserves_the_previous_valid_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            before, after = {"status": "selecting"}, {"status": "running", "selected": 2}
            runner.save_record(output, before)
            write_text = Path.write_text

            def partial_write(path, text, **kwargs):
                write_text(path, text[:5], **kwargs)
                raise KeyboardInterrupt

            with patch.object(Path, "write_text", partial_write), self.assertRaises(KeyboardInterrupt):
                runner.save_record(output, after)
            self.assertEqual(json.loads((output / "run.json").read_text()), before)
            runner.save_record(output, after)
            self.assertEqual(json.loads((output / "run.json").read_text()), after)
            self.assertFalse((output / "run.json.tmp").exists())

    def test_outstanding_results_preserve_timeout_and_build_failure(self):
        report = completed_report() | {"caught": 0, "timeout": 1, "unviable": 1}
        report["outcomes"][1].update(summary="Timeout", phase_results=phase_results("Timeout"),
                                     log_path="log/first.log", diff_path="diff/first.diff")
        report["outcomes"][2].update(summary="Unviable",
                                     phase_results=phase_results(build_status={"Failure": 101}),
                                     log_path="log/second.log", diff_path="diff/second.diff")
        code, record, _ = self.execute(["smoke"], report=report, exit_code=3)
        self.assertEqual((code, record["status"]), (3, "failed"))
        self.assertEqual(record["runner_exit_code"], 3)
        self.assertEqual(record["counts"]["caught"], 0)
        self.assertEqual([(entry["outcome"], entry["phase"]) for entry in record["outstanding"]],
                         [("Timeout", "Test"), ("Unviable", "Build")])
        self.assertEqual(record["outstanding"][0]["log_path"], "log/first.log")
        self.assertEqual(record["outstanding"][1]["diff_path"], "diff/second.diff")

    def test_summary_limits_visible_rows_without_dropping_retained_failures(self):
        entries = [{"name": f"fixture.rs:{i}: replace `a | b`\nwith false", "outcome": "MissedMutant",
                    "phase": "Test", "log_path": f"log/{i}.log", "diff_path": f"diff/{i}.diff"}
                   for i in range(25)]
        record = {"scope": "full", "status": "failed", "selected": 25, "outstanding": entries}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(runner.os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary)
            runner.write_summary(output, record)
            self.assertEqual(json.loads((output / "run.json").read_text())["outstanding"], entries)
            summary = (output / "summary.md").read_text()
            self.assertEqual(summary.count("| MissedMutant |"), 20)
            self.assertIn("Showing 20 of 25", summary)
            self.assertIn("`` fixture.rs:0: replace `a \\| b` with false ``", summary)
            self.assertNotIn("fixture.rs:24:", summary)

    def test_empty_changed_diff_is_explicit_and_empty_smoke_is_error(self):
        code, record, calls = self.execute(["changed", "--base", "fixture"], selected=0)
        self.assertEqual((code, record["status"], calls), (0, "no_mutants", 0))
        code, record, calls = self.execute(["smoke"], selected=0)
        self.assertEqual((code, record["status"], calls), (1, "error", 0))

    def test_empty_shard_requires_a_valid_nonempty_unsharded_inventory(self):
        args = ["full", "--shard", "9/10"]
        code, record, calls = self.execute(args, selected=0, unsharded=selected_mutants())
        self.assertEqual((code, record["status"], calls), (0, "empty_shard", 0))
        for unsharded in ([], {}, [{}, {}], [selected_mutants()[0]] * 2):
            with self.subTest(unsharded=unsharded):
                code, record, calls = self.execute(args, selected=0, unsharded=unsharded)
                self.assertEqual((code, record["status"], calls), (1, "error", 0))

    def test_inventory_is_validated_even_in_list_mode(self):
        for inventory in ([{}, {}], [selected_mutants()[0]] * 2, {"not": "a list"}):
            with self.subTest(inventory=inventory):
                code, record, calls = self.execute(["full", "--list"], inventory=inventory)
                self.assertEqual((code, record["status"], calls), (1, "error", 0))

    def test_substituted_results_fail_even_with_tool_exit_zero(self):
        report = completed_report()
        report["outcomes"][1]["scenario"]["Mutant"]["name"] = "unselected.rs:1: replace unrelated"
        code, record, calls = self.execute(["smoke"], report=report)
        self.assertEqual((code, record["status"], calls), (1, "error", 1))
        self.assertIn("selected inventory", record["error"])

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
        for summary, counter, status in (("MissedMutant", "missed", "Success"),
                                          ("Timeout", "timeout", "Timeout")):
            with self.subTest(summary=summary):
                report = completed_report() | {"caught": 1, counter: 1}
                report["outcomes"][1]["summary"] = summary
                report["outcomes"][1]["phase_results"] = phase_results(status)
                code, record, _ = self.execute(["smoke"], report=report)
                self.assertEqual((code, record["status"]), (1, "failed"))


if __name__ == "__main__":
    unittest.main()
