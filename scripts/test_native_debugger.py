"""A debugger must never turn a crash, stale result or launch failure into a pass."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import venv
from unittest.mock import MagicMock, patch

import smoke_test_wheel as wheel
from scripts.wheel_validation import native_debugger as native


def exited(code=0, **changes):
    return dict(schema_version=1, request_id="current", valid_process=True,
                pid=123, state="exited", exit_code=code) | changes


class NativeDebuggerTests(unittest.TestCase):
    def test_only_current_normal_child_exit_and_successful_debugger_can_pass(self):
        self.assertEqual(native.outcome(exited(), 0, "current"), 0)
        self.assertEqual(native.outcome(exited(7), 0, "current"), 7)
        self.assertEqual(native.outcome(exited(7), 7, "current"), 7)
        for record in (None, [], {}, exited(request_id="stale"), exited(pid=0),
                       exited(pid=True), exited(valid_process=False), exited(schema_version=2),
                       exited(schema_version=True),
                       exited(exit_code=None), exited(exit_code=True), exited(exit_code=-1),
                       exited(exit_code=256), exited(state="stopped"), exited(state="crashed"),
                       exited(state="incomplete")):
            with self.subTest(record=record):
                self.assertNotEqual(native.outcome(record, 0, "current"), 0)
        for code in (1, -signal.SIGABRT):
            self.assertNotEqual(native.outcome(exited(), code, "current"), 0)

    def test_missing_corrupt_stale_and_crashed_records_fail_even_if_lldb_exits_zero(self):
        for content in (None, "not json", json.dumps(exited(request_id="stale")),
                        json.dumps(exited(state="stopped", exit_code=None)), json.dumps(exited())):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = root / "native-debugger/probe.json"
                result.parent.mkdir()
                if content is not None:
                    result.write_text(content)
                child = MagicMock()
                child.stdout = iter(["debugger output\n"])
                child.wait.return_value = 0
                child.__enter__.return_value = child
                with patch.object(native.shutil, "which", return_value="/tool/lldb"), \
                        patch.object(native.uuid, "uuid4", return_value=SimpleNamespace(hex="current")), \
                        patch.object(native.subprocess, "Popen", return_value=child) as launch, \
                        redirect_stdout(io.StringIO()):
                    command = ["python", "-c", "print('literal $HOME')", "a b"]
                    code, metadata = native.run(command, root, "probe")
                self.assertEqual(code == 0, content == json.dumps(exited()))
                self.assertEqual(metadata["debugger_returncode"], 0)
                self.assertEqual(Path(metadata["log"]).read_text(), "debugger output\n")
                args = launch.call_args.args[0]
                self.assertEqual(args[args.index("--") + 1:], command)
                self.assertIn("process launch --disable-aslr false --shell-expand-args false", args)
                self.assertIn("--no-lldbinit", args)
                self.assertIn("-k", args)
                self.assertNotIn("shell", launch.call_args.kwargs)

    def test_debugger_unavailable_and_launch_error_do_not_run_a_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for executable in (None, "/missing/lldb"):
                with self.subTest(executable=executable), \
                        patch.object(native.shutil, "which", return_value=executable), \
                        patch.object(native.subprocess, "Popen", side_effect=OSError("launch failed")) as launch, \
                        self.assertRaises(OSError):
                    native.run(["python", "probe"], root, "probe")
                self.assertEqual(launch.call_count, int(executable is not None))

    def test_capture_records_native_stops_without_resuming_and_keeps_first_observation(self):
        for state in (5, 8, 10):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture.json"
                result = MagicMock()
                result.GetOutput.return_value = "native frames\n"
                result.GetError.return_value = ""
                result.Succeeded.return_value = True
                lldb = SimpleNamespace(eStateStopped=5, eStateCrashed=8, eStateExited=10,
                                       SBCommandReturnObject=lambda: result)
                process = MagicMock()
                process.GetState.return_value = state
                process.IsValid.return_value = True
                process.GetProcessID.return_value = 123
                process.GetExitStatus.return_value = 0
                process.GetExitDescription.return_value = None
                thread = MagicMock()
                thread.GetThreadID.return_value = 456
                thread.GetName.return_value = "pricing-worker-0"
                thread.GetStopReason.return_value = 6
                thread.GetStopDescription.return_value = "EXC_BREAKPOINT"
                thread.GetStopReasonDataCount.return_value = 1
                thread.GetStopReasonDataAtIndex.return_value = 5
                process.__iter__.return_value = iter([thread])
                debugger = MagicMock()
                debugger.GetSelectedTarget.return_value.GetProcess.return_value = process
                with patch.dict(sys.modules, lldb=lldb), redirect_stdout(io.StringIO()):
                    native.capture(debugger, str(root), "current")
                    first = root.read_bytes()
                    process.GetState.return_value = 10
                    native.capture(debugger, str(root), "current")
                self.assertEqual(first, root.read_bytes())
                record = json.loads(first)
                self.assertEqual(native.outcome(record, 0, "current") == 0, state == 10)
                process.Continue.assert_not_called()
                process.Detach.assert_not_called()
                commands = [c.args[0] for c in debugger.GetCommandInterpreter().HandleCommand.call_args_list]
                if state == 10:
                    self.assertEqual(commands, [])
                else:
                    self.assertIn("thread backtrace all", commands)
                    self.assertIn("image list", commands)
                    self.assertEqual(record["threads"][0]["stop_description"], "EXC_BREAKPOINT")
                    self.assertIsNone(record["exit_code"])

    def test_module_failure_remains_failed_and_later_module_still_runs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            outcomes = [(1, {"process": exited(state="crashed", exit_code=None)}),
                        (0, {"process": exited()})]
            with patch.object(wheel.native_runner, "run", side_effect=outcomes) as run, \
                    self.assertRaisesRegex(RuntimeError, "wheel test modules failed: test_crash.py"):
                wheel.run_test_modules(Path("python"), ["test_crash.py", "test_later.py"],
                                       report_directory=root, native_debugger="lldb")
            self.assertEqual(run.call_count, 2)
            for name, status in (("test_crash.py", "failed"), ("test_later.py", "passed")):
                record = json.loads((root / f"{name}.json").read_text())
                self.assertEqual(record["status"], status)
                self.assertIn("native_debugger", record)

    def test_example_failure_is_recorded_and_keeps_existing_fail_fast_order(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            with patch.object(wheel, "EXAMPLES", ("first.py", "later.py")), \
                    patch.object(wheel.native_runner, "run", return_value=(7, {})) as run, \
                    self.assertRaisesRegex(RuntimeError, "wheel example failed"):
                wheel.run_examples(Path("python"), root, "lldb")
            run.assert_called_once_with(["python", "-X", "faulthandler", "first.py"], root, "example-first")
            record = json.loads((root / "example-first.json").read_text())
            self.assertEqual((record["kind"], record["status"], record["returncode"]), ("example", "failed", 7))

    def test_native_shard_failure_is_recorded_and_propagates_without_retry(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            shard = wheel.SHARD_NAMES[0]
            with patch.object(wheel.native_runner, "run", return_value=(1, {})) as run, \
                    self.assertRaisesRegex(RuntimeError, "wheel provider shard failed"):
                wheel.run_test_shard(Path("python"), shard, root, "lldb")
            run.assert_called_once_with(["python", "-X", "faulthandler", "-m",
                                         "scripts.wheel_validation.provider_shards", "--run", shard],
                                        root, shard)
            self.assertEqual(json.loads((root / f"{shard}.json").read_text())["status"], "failed")

    def test_cli_requires_records_and_preserves_validation_selection(self):
        from contextlib import redirect_stderr
        for args in (["--native-debugger", "lldb"],
                     ["--phase", "package", "--native-debugger", "lldb", "--report-directory", "reports"],
                     ["--native-debugger", "unknown", "--report-directory", "reports"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                wheel.parse_arguments(args)
        with patch.object(wheel, "selected_wheel", return_value=Path("wheel.whl")), \
                patch.object(wheel, "prepare_wheel", return_value=Path("python")), \
                patch.object(wheel, "run_validation") as validate:
            wheel.main(["--phase", "validation", "--shard", wheel.SHARD_NAMES[0],
                        "--native-debugger", "lldb", "--report-directory", "reports"])
        validate.assert_called_once_with(Path("python"), "all", wheel.SHARD_NAMES[0], Path("reports"), "lldb")


@unittest.skipUnless(sys.platform == "darwin", "mandatory real LLDB integration on macOS")
class MacNativeDebuggerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(shutil.which("lldb"), "macOS wheel diagnostics require LLDB")

    def test_real_child_exit_and_literal_arguments(self):
        value = "spaces 'quotes' \"double\" $HOME $(exit 9) `exit 9` * 日本語"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "quoted 'venv' path"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            for exit_code in (0, 7):
                with self.subTest(exit_code=exit_code):
                    code = (f"import sys, os; assert sys.argv[1] == {value!r}; "
                            f"assert sys.prefix == {str(environment)!r}; "
                            f"assert os.getcwd() == {str(Path.cwd())!r}; "
                            "assert os.environ['WHEEL_NATIVE_ENV_TEST'] == 'inherited'; "
                            f"sys.exit({exit_code})")
                    with patch.dict(os.environ, WHEEL_NATIVE_ENV_TEST="inherited"):
                        result, metadata = native.run(
                            [str(environment / "bin/python"), "-c", code, value], root, f"exit{exit_code}")
                    self.assertEqual(result, exit_code, metadata)
                    self.assertEqual(metadata["process"]["state"], "exited")

    def test_real_abort_and_trap_capture_native_stack_and_never_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("SIGABRT", "SIGTRAP"):
                with self.subTest(signal=name):
                    marker = root / f"continued-{name}"
                    code = ("import os, resource, signal; from pathlib import Path; "
                            "resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
                            f"os.kill(os.getpid(), signal.{name}); "
                            f"Path({str(marker)!r}).write_text('continued')")
                    result, metadata = native.run([sys.executable, "-c", code], root, name)
                    self.assertNotEqual(result, 0, metadata)
                    self.assertFalse(marker.exists())
                    self.assertIn(metadata["process"]["state"], ("stopped", "crashed"))
                    self.assertTrue(metadata["process"]["threads"])
                    self.assertTrue(metadata["process"]["diagnostics"]["thread backtrace all"])
                    self.assertIn("frame #0", Path(metadata["log"]).read_text())


if __name__ == "__main__":
    unittest.main()
