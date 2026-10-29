"""Wheel scheduling must preserve discovery, process isolation and error handling."""
from collections import Counter
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import smoke_test_wheel as wheel
from scripts.wheel_validation import remaining_shards as shards


class RemainingShardTests(unittest.TestCase):
    def test_every_discovered_module_is_selected_once_including_new_modules(self):
        modules = wheel.discover_test_modules(ROOT / "tests/python")
        modules.append("test_future_numerical_case.py")
        original = wheel.select_test_modules(modules, "remaining")
        groups = [wheel.select_test_modules(modules, group) for group in shards.REMAINING_GROUPS]
        self.assertEqual(Counter(name for group in groups for name in group), Counter(original))
        self.assertTrue(all(groups))
        self.assertNotIn("test_vega_kt_extension_provider.py", original)
        # File-system enumeration order must not change the generated plan.
        self.assertEqual(shards.partition_remaining(original), shards.partition_remaining(original[::-1]))

    def test_hints_are_not_a_coverage_allowlist_and_stale_hints_add_no_tests(self):
        modules = [f"test_{index}.py" for index in range(8)]
        hints = {"examples_seconds": 1, "modules_seconds": {
            "test_0.py": 40, "test_1.py": 20, "test_deleted.py": 10000}}
        plan = shards.partition_remaining(modules, timings=hints)
        self.assertEqual(Counter(name for group in plan.values() for name in group), Counter(modules))
        self.assertFalse(any("test_deleted.py" in group for group in plan.values()))
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            shards.partition_remaining(modules + [modules[0]])
        with self.assertRaisesRegex(RuntimeError, "too few"):
            shards.partition_remaining(modules[:3])

    def test_examples_run_once_and_original_module_runner_is_used(self):
        modules = wheel.discover_test_modules(ROOT / "tests/python")
        for group in shards.REMAINING_GROUPS:
            with self.subTest(group=group), patch.object(wheel, "discover_test_modules", return_value=modules), \
                    patch.object(wheel.subprocess, "run") as discovery, \
                    patch.object(wheel, "run_examples") as examples, \
                    patch.object(wheel, "run_test_modules") as run, redirect_stdout(io.StringIO()):
                wheel.run_validation(Path("venv/python"), group)
                discovery.assert_called_once()
                run.assert_called_once_with(Path("venv/python"), wheel.select_test_modules(modules, group),
                                            report_directory=None, native_debugger=None)
                self.assertEqual(examples.call_count, int(group == shards.EXAMPLES_GROUP))
        for legacy in ("all", "remaining"):
            with patch.object(wheel, "discover_test_modules", return_value=modules), \
                    patch.object(wheel.subprocess, "run"), patch.object(wheel, "run_examples") as examples, \
                    patch.object(wheel, "run_test_modules"), redirect_stdout(io.StringIO()):
                wheel.run_validation(Path("venv/python"), legacy)
                examples.assert_called_once()

    def test_module_failure_is_not_retried_or_suppressed(self):
        modules = wheel.discover_test_modules(ROOT / "tests/python")
        with patch.object(wheel, "discover_test_modules", return_value=modules), \
                patch.object(wheel.subprocess, "run"), patch.object(wheel, "run_examples"), \
                patch.object(wheel, "run_test_modules", side_effect=RuntimeError("native child failed")) as run, \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "native child failed"):
            wheel.run_validation(Path("venv/python"), shards.REMAINING_GROUPS[0])
        run.assert_called_once()

    def test_cli_requires_validation_and_keeps_provider_selection_exclusive(self):
        for group in shards.REMAINING_GROUPS:
            args = wheel.parse_arguments(["--phase", "validation", "--group", group])
            self.assertEqual(args.group, group)
            with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
                wheel.parse_arguments(["--group", group])
            with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
                wheel.parse_arguments(["--phase", "validation", "--group", group,
                                      "--shard", wheel.SHARD_NAMES[0]])

    def test_invalid_timing_numbers_fail_closed(self):
        for value in (-1, float("nan"), float("inf"), True, "10"):
            with self.subTest(value=value), patch.object(shards.json, "loads", return_value={
                    "examples_seconds": value, "modules_seconds": {}}), \
                    self.assertRaisesRegex(ValueError, "finite nonnegative"):
                shards.load_timings()


if __name__ == "__main__":
    unittest.main()
