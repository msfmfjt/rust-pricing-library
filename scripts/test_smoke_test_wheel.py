"""Guard wheel validation coverage, process isolation and failure propagation."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import ast
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import smoke_test_wheel as wheel


ROOT = Path(__file__).resolve().parents[1]


class WheelRunnerTests(unittest.TestCase):
    def test_typed_source_contract_rejects_annotation_defaults_and_shape_drift(self):
        source = (ROOT / "rust_pricing.pyi").read_text()
        for before, after in [
            ("models: Sequence[Model], correlation: CorrelationSchedule", "models: Sequence[float], correlation: CorrelationSchedule"),
            ("equity: EquityModelSpec, *, rates: RateModelSpec | None = None", "equity: EquityModelSpec, rates: RateModelSpec | None = None"),
            ("retain_reverse_trace: bool = False", "retain_reverse_trace: bool = True"),
            ("source: PricingSpecification | MultiAssetSpecification | StochasticDividendSpecification", "source: PricingSpecification"),
            ("def rough_bergomi(model: RoughBergomiModel) -> VolatilityFactorSpec", "def rough_bergomi(model: RoughBergomiModel) -> Model"),
        ]:
            with self.subTest(before=before), self.assertRaises(RuntimeError):
                wheel.verify_stub_static_shape(ast.parse(source.replace(before, after)))

    def test_additive_risk_and_error_contracts_reject_missing_or_changed_types(self):
        source = (ROOT / "rust_pricing.pyi").read_text()
        wheel.verify_stub_static_shape(ast.parse(source))
        for before, after in [
            ("    causes: list[str]", "    causes: str"),
            ("    reason: str | None\n", ""),
            ("def parameter_kinds(self) -> list[str]", "def parameter_kinds(self) -> str"),
            ("def parameter_units(self) -> list[str]", "def parameter_units(self, value: str) -> list[str]"),
        ]:
            with self.subTest(before=before), self.assertRaises(RuntimeError):
                wheel.verify_stub_static_shape(ast.parse(source.replace(before, after)))

    def test_cli_keeps_full_default_and_rejects_invalid_group_use(self):
        args = wheel.parse_arguments([])
        self.assertEqual((args.wheel, args.phase, args.group), (None, "full", "all"))
        args = wheel.parse_arguments(["built.whl", "--phase", "validation", "--group", "providers"])
        self.assertEqual((args.wheel, args.phase, args.group), ("built.whl", "validation", "providers"))
        for arguments in (["--group", "unknown"], ["--group", "providers"],
                          ["--phase", "full", "--group", "providers"],
                          ["--phase", "package", "--group", "remaining"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    wheel.parse_arguments(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_all_phases_preserve_common_package_verification(self):
        for phase, group in [("full", "all"), ("package", "all"),
                             ("validation", "custom-basis"), ("validation", "remaining")]:
            with self.subTest(phase=phase, group=group):
                with patch.object(wheel, "selected_wheel", return_value=Path("built.whl")) as select, \
                        patch.object(wheel, "prepare_wheel", return_value=Path("venv/python")) as prepare, \
                        patch.object(wheel, "run_validation") as validate:
                    wheel.main(["built.whl", "--phase", phase, "--group", group])
                    select.assert_called_once_with("built.whl")
                    prepare.assert_called_once_with(Path("built.whl"))
                    if phase == "package":
                        validate.assert_not_called()
                    else:
                        validate.assert_called_once_with(Path("venv/python"), group, None, None)

    def test_partition_is_disjoint_complete_and_includes_new_modules(self):
        named = sorted(set().union(*wheel.NAMED_TEST_GROUPS.values()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in [*named, "test_runner_first.py", "helper.py"]:
                (root / name).touch()
            modules = wheel.discover_test_modules(root)
            self.assertNotIn("helper.py", modules)
            self.assertEqual(wheel.select_test_modules(modules, "all"), modules)
            partition = [name for group in ("custom-basis", "providers", "remaining")
                         for name in wheel.select_test_modules(modules, group)]
            self.assertEqual(Counter(partition), Counter(modules))
            self.assertTrue(all(count == 1 for count in Counter(partition).values()))
            (root / "test_runner_new.py").touch()
            self.assertEqual(wheel.select_test_modules(wheel.discover_test_modules(root), "remaining"),
                             ["test_runner_first.py", "test_runner_new.py"])

    def test_missing_named_modules_unknown_groups_and_empty_selections_fail(self):
        named = sorted(set().union(*wheel.NAMED_TEST_GROUPS.values()))
        with self.assertRaisesRegex(RuntimeError, "named test modules are missing"):
            wheel.select_test_modules(named[1:], "all")
        with self.assertRaisesRegex(RuntimeError, "test group is empty"):
            wheel.select_test_modules(named, "remaining")
        with self.assertRaisesRegex(ValueError, "unknown test group"):
            wheel.select_test_modules(named, "typo")
        with self.assertRaisesRegex(RuntimeError, "duplicate test modules"):
            wheel.select_test_modules(named + named, "all")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "no test modules found"):
                wheel.discover_test_modules(Path(directory))
        with self.assertRaisesRegex(RuntimeError, "no test modules selected"):
            wheel.run_test_modules(Path(sys.executable), [])

    def test_inventory_uses_test_id_counts_and_rejects_omitted_or_repeated_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "test_wheel_inventory_duplicate_id.py"
            (root / name).write_text(
                "import unittest\n"
                "class Case(unittest.TestCase):\n"
                "    def test_body_is_not_run(self):\n"
                "        raise AssertionError('discovery must not execute tests')\n"
                "Alias = Case\n", encoding="utf-8")
            # Discovery modifies sys.path and imports modules; keep the test's
            # temporary directory and module state local to this regression.
            with patch.object(sys, "path", sys.path.copy()), patch.dict(sys.modules), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(wheel.verify_test_inventory(root, [name]), {name: 2})
                with self.assertRaisesRegex(RuntimeError, "differs from full discovery"):
                    wheel.verify_test_inventory(root, [])
                with self.assertRaisesRegex(RuntimeError, "differs from full discovery"):
                    wheel.verify_test_inventory(root, [name, name])

    def test_inventory_rejects_import_errors_and_empty_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "test_wheel_inventory_import_error.py"
            (root / name).write_text("raise ImportError('inventory probe')\n", encoding="utf-8")
            with patch.object(sys, "path", sys.path.copy()), patch.dict(sys.modules):
                with self.assertRaisesRegex(RuntimeError, "test discovery failed"):
                    wheel.verify_test_inventory(root, [name])
            (root / name).unlink()
            name = "test_wheel_inventory_empty.py"
            (root / name).touch()
            with patch.object(sys, "path", sys.path.copy()), patch.dict(sys.modules):
                with self.assertRaisesRegex(RuntimeError, "test module is empty"):
                    wheel.verify_test_inventory(root, [name])

    def test_examples_only_run_for_all_or_remaining_and_inventory_always_runs(self):
        modules = sorted(set().union(*wheel.NAMED_TEST_GROUPS.values())) + ["test_extra.py"]
        for group in wheel.TEST_GROUPS:
            with self.subTest(group=group), redirect_stdout(io.StringIO()):
                with patch.object(wheel, "discover_test_modules", return_value=modules), \
                        patch.object(wheel.subprocess, "run") as process, \
                        patch.object(wheel, "run_examples") as examples, \
                        patch.object(wheel, "run_test_modules") as tests:
                    wheel.run_validation(Path("venv/python"), group)
                    process.assert_called_once()
                    command = process.call_args.args[0]
                    self.assertEqual(command[:4], [str(Path("venv/python")), "-X", "faulthandler", "-c"])
                    self.assertEqual(json.loads(command[-2]), modules)
                    self.assertTrue(process.call_args.kwargs["check"])
                    self.assertEqual(examples.call_count, int(group in ("all", "remaining")))
                    tests.assert_called_once_with(Path("venv/python"),
                                                  wheel.select_test_modules(modules, group),
                                                  report_directory=None)

    def test_failed_module_is_nonzero_and_later_module_runs_in_a_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = ["test_wheel_runner_failure.py", "test_wheel_runner_success.py"]
            for name, fail in zip(modules, (True, False)):
                marker = root / (name + ".pid")
                (root / name).write_text(
                    "import os, unittest\nfrom pathlib import Path\n"
                    "class Case(unittest.TestCase):\n"
                    "    def test_run(self):\n"
                    f"        Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                    f"        self.assertFalse({fail!r})\n", encoding="utf-8")
            code = (
                "import json, sys\nfrom pathlib import Path\n"
                "from scripts.smoke_test_wheel import run_test_modules\n"
                "run_test_modules(Path(sys.executable), json.loads(sys.argv[2]), Path(sys.argv[1]))\n"
            )
            result = subprocess.run([sys.executable, "-c", code, str(root), json.dumps(modules)],
                                    cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            pids = [(root / (name + ".pid")).read_text() for name in modules]
            self.assertNotEqual(pids[0], pids[1])
            for name, exit_code in zip(modules, (1, 0)):
                self.assertIn(f"wheel-smoke start module={name}", result.stdout)
                self.assertRegex(result.stdout, f"finish module={name} elapsed=[0-9.]+s exit={exit_code}")
            self.assertIn("wheel test modules failed: test_wheel_runner_failure.py", result.stderr)

    def test_native_signal_is_reported_and_does_not_skip_later_modules(self):
        output = io.StringIO()
        results = [subprocess.CompletedProcess([], -signal.SIGABRT),
                   subprocess.CompletedProcess([], 0)]
        with patch.object(wheel.subprocess, "run", side_effect=results) as process, \
                redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "signal=SIGABRT"):
                wheel.run_test_modules(Path("venv/python"), ["test_crash.py", "test_later.py"])
        self.assertEqual(process.call_count, 2)
        for call, name in zip(process.call_args_list, ("test_crash.py", "test_later.py")):
            self.assertEqual(call.args[0], [str(Path("venv/python")), "-X", "faulthandler", "-m", "unittest",
                                           "discover", "-s", str(wheel.TEST_DIRECTORY), "-p", name, "-v"])
            self.assertFalse(call.kwargs["check"])
        self.assertIn("signal=SIGABRT", output.getvalue())
        self.assertIn("finish module=test_later.py", output.getvalue())

    def test_shard_cli_requires_validation_and_keeps_group_selection_exclusive(self):
        shard = wheel.SHARD_NAMES[0]
        args = wheel.parse_arguments(["--phase", "validation", "--shard", shard,
                                      "--report-directory", "reports"])
        self.assertEqual((args.shard, args.report_directory), (shard, Path("reports")))
        for argv in (["--shard", shard], ["--phase", "package", "--shard", shard],
                     ["--phase", "validation", "--group", "providers", "--shard", shard],
                     ["--phase", "validation", "--shard", "typo"],
                     ["--phase", "package", "--report-directory", "reports"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                wheel.parse_arguments(argv)

    def test_shard_checks_inventory_and_runs_only_selected_process(self):
        modules = sorted(set().union(*wheel.NAMED_TEST_GROUPS.values())) + ["test_extra.py"]
        shard = wheel.SHARD_NAMES[0]
        with patch.object(wheel, "discover_test_modules", return_value=modules), \
                patch.object(wheel.subprocess, "run") as process, \
                patch.object(wheel, "run_examples") as examples, \
                patch.object(wheel, "run_test_modules") as modules_run, \
                patch.object(wheel, "run_test_shard") as shard_run:
            wheel.run_validation(Path("venv/python"), "all", shard)
            process.assert_called_once()
            self.assertIn("verify_partition", process.call_args.args[0][4])
            examples.assert_not_called()
            modules_run.assert_not_called()
            shard_run.assert_called_once_with(Path("venv/python"), shard, None)
        with self.assertRaisesRegex(ValueError, "invalid provider shard"):
            wheel.run_validation(Path("venv/python"), "providers", shard)

    def test_process_records_failure_signal_launch_error_and_incomplete_execution(self):
        for outcome, status, exit_code in (
            (subprocess.CompletedProcess([], 0), "passed", 0),
            (subprocess.CompletedProcess([], -signal.SIGABRT), "failed", -signal.SIGABRT),
            (OSError("cannot launch"), "failed", 1),
            (KeyboardInterrupt(), "running", None),
        ):
            with self.subTest(status=status, exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                def execute(*args, **kwargs):
                    initial = json.loads((root / "probe.json").read_text())
                    self.assertEqual(initial["status"], "running")
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return outcome
                with patch.object(wheel.subprocess, "run", side_effect=execute), redirect_stdout(io.StringIO()):
                    if isinstance(outcome, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            wheel.run_test_process(["python", "probe"], "shard", "probe", root)
                    else:
                        failure = wheel.run_test_process(["python", "probe"], "shard", "probe", root)
                        self.assertEqual(failure is not None, status == "failed")
                record = json.loads((root / "probe.json").read_text())
                self.assertEqual(record["status"], status)
                self.assertEqual(record.get("returncode"), exit_code)
                if exit_code == -signal.SIGABRT:
                    self.assertEqual(record["signal"], "SIGABRT")
                if isinstance(outcome, OSError):
                    self.assertEqual(record["launch_error"], "cannot launch")

    def test_shard_crash_propagates_immediately(self):
        with patch.object(wheel.subprocess, "run", return_value=subprocess.CompletedProcess([], -6)) as run, \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "shard failed.*SIGABRT"):
            wheel.run_test_shard(Path("python"), wheel.SHARD_NAMES[0])
        run.assert_called_once_with(["python", "-X", "faulthandler", "-m",
                                    "scripts.wheel_validation.provider_shards", "--run", wheel.SHARD_NAMES[0]],
                                   check=False)


if __name__ == "__main__":
    unittest.main()
