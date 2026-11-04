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
    def test_stack_snapshot_retains_symbols_addresses_and_missing_debug_info(self):
        frame = MagicMock()
        frame.IsValid.return_value = True
        frame.GetPC.return_value = 0x1020
        frame.GetPCAddress.return_value.GetFileAddress.return_value = 0x20
        frame.GetFunctionName.return_value = None
        frame.GetSymbol.return_value.GetName.return_value = "_dispatch_cache_cleanup"
        module = frame.GetModule.return_value
        module.IsValid.return_value = True
        module.GetFileSpec.return_value.GetDirectory.return_value = "/usr/lib/system"
        module.GetFileSpec.return_value.GetFilename.return_value = "libdispatch.dylib"
        module.GetUUIDString.return_value = "module-uuid"
        frame.GetLineEntry.return_value.IsValid.return_value = False
        unknown = MagicMock()
        unknown.IsValid.return_value = False
        thread = MagicMock()
        thread.GetNumFrames.return_value = 2
        thread.GetFrameAtIndex.side_effect = [frame, unknown]
        with patch.dict(sys.modules, lldb=SimpleNamespace(LLDB_INVALID_ADDRESS=2**64 - 1)):
            record = native.stack_snapshot(thread)
        self.assertEqual(record, {
            "frame_count": 2, "frames_truncated": False,
            "frames": [{"index": 0, "valid": True, "pc": "0x1020", "file_address": "0x20",
                        "function": "_dispatch_cache_cleanup", "module": "libdispatch.dylib",
                        "module_path": str(Path("/usr/lib/system") / "libdispatch.dylib"),
                        "module_uuid": "module-uuid", "source": None},
                       {"index": 1, "valid": False}],
        })

        frame.GetFunctionName.return_value = "pricing::probe"
        frame.GetPC.return_value = 2**64 - 1
        frame.GetPCAddress.return_value.GetFileAddress.return_value = 2**64 - 1
        module.IsValid.return_value = False
        line = frame.GetLineEntry.return_value
        line.IsValid.return_value = True
        line.GetFileSpec.return_value.GetDirectory.return_value = "/source"
        line.GetFileSpec.return_value.GetFilename.return_value = "probe.rs"
        line.GetLine.return_value = 42
        line.GetColumn.return_value = 3
        thread.GetFrameAtIndex.side_effect = None
        thread.GetFrameAtIndex.return_value = frame
        thread.GetNumFrames.return_value = 129
        thread.GetFrameAtIndex.reset_mock()
        with patch.dict(sys.modules, lldb=SimpleNamespace(LLDB_INVALID_ADDRESS=2**64 - 1)):
            record = native.stack_snapshot(thread)
        self.assertEqual(record["frame_count"], 129)
        self.assertTrue(record["frames_truncated"])
        self.assertEqual(len(record["frames"]), 128)
        self.assertEqual(thread.GetFrameAtIndex.call_count, 128)
        self.assertEqual(record["frames"][-1], {
            "index": 127, "valid": True, "pc": None, "file_address": None,
            "function": "pricing::probe", "module": None, "module_path": None,
            "module_uuid": None, "source": {"path": str(Path("/source") / "probe.rs"), "line": 42, "column": 3},
        })

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
                    command = ["quoted `venv`/python", "-c", "print('literal $HOME')",
                               "a b", "", "`exit 9` $(exit 9) * 日本語\nnext\\line"]
                    code, metadata = native.run(command, root, "probe")
                self.assertEqual(code == 0, content == json.dumps(exited()))
                self.assertEqual(metadata["debugger_returncode"], 0)
                self.assertEqual(Path(metadata["log"]).read_text(), "debugger output\n")
                args = launch.call_args.args[0]
                self.assertFalse(any("`" in arg or "$" in arg or "\n" in arg for arg in args))
                self.assertNotIn("--", args)
                self.assertIn("settings set target.process.stop-on-exec false", args)
                self.assertIn("--no-lldbinit", args)
                self.assertNotIn("shell", launch.call_args.kwargs)
                # Execute the bootstrap with a fake loader, checking that its
                # data survives the actual transport through embedded Python.
                callback = MagicMock()
                module = SimpleNamespace(launch=callback)
                spec = SimpleNamespace(loader=MagicMock())
                debugger = object()
                with patch("importlib.util.spec_from_file_location", return_value=spec), \
                        patch("importlib.util.module_from_spec", return_value=module):
                    script = next(arg for arg in args if arg.startswith("script "))
                    exec(script.removeprefix("script "), {"lldb": SimpleNamespace(debugger=debugger)})
                callback.assert_called_once_with(debugger, command, str(result.resolve()), "current")

    def test_api_launch_preserves_venv_arguments_environment_and_child_output(self):
        command = ["relative 'venv'/bin/python", "-c", "print('`literal`')", "", "line\n日本語"]
        info, error, process, target, debugger = (MagicMock() for _ in range(5))
        target.IsValid.return_value = True
        target.Launch.return_value = process
        debugger.CreateTarget.return_value = target
        error.Fail.return_value = False
        process.GetSTDOUT.side_effect = ["child stdout\n", ""]
        process.GetSTDERR.side_effect = ["child stderr\n", ""]
        lldb = SimpleNamespace(SBLaunchInfo=MagicMock(return_value=info),
                               SBFileSpec=lambda value: value, SBError=lambda: error,
                               eLaunchFlagDebug=2)
        output = io.StringIO()
        with patch.dict(sys.modules, lldb=lldb), patch.dict(os.environ, WHEEL_NATIVE_ENV_TEST="inherited"), \
                patch.object(native, "capture") as capture, redirect_stdout(output):
            native.launch(debugger, command, "result.json", "current")
        lldb.SBLaunchInfo.assert_called_once_with(command[1:])
        info.SetExecutableFile.assert_called_once_with(command[0], True)
        info.SetWorkingDirectory.assert_called_once_with(os.getcwd())
        self.assertIn("WHEEL_NATIVE_ENV_TEST=inherited", info.SetEnvironmentEntries.call_args.args[0])
        info.SetLaunchFlags.assert_called_once_with(2)
        info.SetShellExpandArguments.assert_called_once_with(False)
        info.SetDetachOnError.assert_called_once_with(False)
        debugger.SetAsync.assert_called_once_with(False)
        target.Launch.assert_called_once_with(info, error)
        process.Continue.assert_not_called()
        process.Detach.assert_not_called()
        capture.assert_called_once_with(debugger, "result.json", "current")
        self.assertEqual(output.getvalue(), "child stdout\nchild stderr\n")

    def test_api_launch_errors_do_not_record_a_success_or_retry(self):
        for valid_target in (False, True):
            with self.subTest(valid_target=valid_target):
                target, debugger, error, info, process = (MagicMock() for _ in range(5))
                target.IsValid.return_value = valid_target
                debugger.CreateTarget.return_value = target
                target.Launch.return_value = process
                error.Fail.return_value = True
                error.GetCString.return_value = "launch failed"
                process.GetSTDOUT.return_value = process.GetSTDERR.return_value = ""
                lldb = SimpleNamespace(SBLaunchInfo=lambda argv: info, SBFileSpec=lambda value: value,
                                       SBError=lambda: error, eLaunchFlagDebug=2)
                with patch.dict(sys.modules, lldb=lldb), patch.object(native, "capture") as capture, \
                        redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "LLDB could not"):
                    native.launch(debugger, ["python", "probe"], "result.json", "current")
                self.assertEqual(target.Launch.call_count, int(valid_target))
                capture.assert_not_called()
                process.Continue.assert_not_called()

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
        for state, broken_stack in ((5, False), (8, False), (10, False), (5, True)):
            with self.subTest(state=state, broken_stack=broken_stack), tempfile.TemporaryDirectory() as directory:
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
                thread.GetNumFrames.return_value = 0
                process.__iter__.return_value = iter([thread])
                debugger = MagicMock()
                debugger.GetSelectedTarget.return_value.GetProcess.return_value = process
                with patch.dict(sys.modules, lldb=lldb), redirect_stdout(io.StringIO()), \
                        patch.object(native, "stack_snapshot", wraps=native.stack_snapshot,
                                     side_effect=RuntimeError("unwind failed") if broken_stack else None):
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
                    if broken_stack:
                        self.assertEqual(record["threads"][0]["frames_error"], "RuntimeError: unwind failed")
                        self.assertNotIn("frames", record["threads"][0])
                    else:
                        self.assertEqual(record["threads"][0]["frames"], [])
                        self.assertFalse(record["threads"][0]["frames_truncated"])
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


@unittest.skipUnless(sys.platform == "darwin" or
                     (sys.platform == "linux" and shutil.which("lldb")),
                     "real LLDB Python API required; mandatory on macOS")
class LldbLaunchInfoTests(unittest.TestCase):
    def test_real_launch_info_preserves_symlinked_executable_and_argv0(self):
        executable = shutil.which("lldb")
        self.assertIsNotNone(executable)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "quoted 'venv' `literal` path"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            python = environment / "bin/python"
            self.assertTrue(python.is_symlink())
            command = [str(python), "-c", "pass", "", "line\n日本語"]
            observed = root / "launch-info.json"
            # Use the installed LLDB's real SBFileSpec and SBLaunchInfo, with
            # target/process handles substituted; no ptrace permission is needed.
            code = f"""
import importlib.util, json, os
from pathlib import Path
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('wheel_native', {str(Path(native.__file__).resolve())!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
process = SimpleNamespace(GetSTDOUT=lambda size: '', GetSTDERR=lambda size: '')
def observe(info, error):
    file = info.GetExecutableFile()
    Path({str(observed)!r}).write_text(json.dumps({{
        'executable': os.path.join(file.GetDirectory(), file.GetFilename()),
        'argv': [info.GetArgumentAtIndex(i) for i in range(info.GetNumArguments())],
    }}))
    return process
target = SimpleNamespace(IsValid=lambda: True, Launch=observe)
debugger = SimpleNamespace(CreateTarget=lambda path: target,
                          SetSelectedTarget=lambda target: None, SetAsync=lambda value: None)
module.capture = lambda *args: None
module.launch(debugger, {command!r}, 'unused.json', 'current')
"""
            script = f"script exec(bytes.fromhex('{code.encode('utf-8').hex()}').decode('utf-8'), {{}})"
            result = subprocess.run([executable, "--batch", "--no-lldbinit", "-o", script],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(observed.is_file(), result.stdout + result.stderr)
            actual = json.loads(observed.read_text())
            self.assertEqual(actual["executable"], str(python))
            self.assertEqual(actual["argv"], command)


@unittest.skipUnless(sys.platform == "darwin", "mandatory real LLDB integration on macOS")
class MacNativeDebuggerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(shutil.which("lldb"), "macOS wheel diagnostics require LLDB")

    def test_real_child_exit_and_literal_arguments(self):
        values = ["spaces 'quotes' \"double\" $HOME $(exit 9) `exit 9` * 日本語", "", "line\nback\\slash"]
        with tempfile.TemporaryDirectory() as directory:
            # macOS framework Python canonicalizes the launcher's directory.
            # tempfile can return /var while the same directory lives in
            # /private/var. Canonicalize the fixture root, never bin/python:
            # resolving that symlink would select the base interpreter.
            root = Path(directory).resolve()
            environment = root / "quoted 'venv' `literal` path"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            python = str(environment / "bin/python")
            direct_prefix = subprocess.check_output(
                [python, "-c", "import sys; print(sys.prefix)"], text=True).strip()
            self.assertEqual(direct_prefix, str(environment), "direct venv launch precondition")
            for exit_code in (0, 7):
                with self.subTest(exit_code=exit_code):
                    marker = root / f"entered-{exit_code}"
                    code = "\n".join([
                        "import sys, os; from pathlib import Path",
                        f"assert sys.argv[1:] == {values!r}, ('argv', sys.argv[1:])",
                        f"assert sys.prefix == {str(environment)!r}, ('venv_prefix', sys.prefix)",
                        f"assert os.getcwd() == {str(Path.cwd())!r}, ('cwd', os.getcwd())",
                        "assert os.environ['WHEEL_NATIVE_ENV_TEST'] == 'inherited', 'environment'",
                        f"p = Path({str(marker)!r})",
                        "p.write_text(p.read_text() + 'entered' if p.exists() else 'entered')",
                        "print('wheel-child-stdout', flush=True)",
                        "print('wheel-child-stderr', file=sys.stderr, flush=True)",
                        f"sys.exit({exit_code})",
                    ])
                    with patch.dict(os.environ, WHEEL_NATIVE_ENV_TEST="inherited"):
                        result, metadata = native.run(
                            [python, "-c", code, *values], root, f"exit{exit_code}")
                    self.assertEqual(result, exit_code, metadata)
                    self.assertEqual(metadata["process"]["state"], "exited")
                    self.assertEqual(marker.read_text(), "entered")
                    log = Path(metadata["log"]).read_text()
                    self.assertIn("wheel-child-stdout", log)
                    self.assertIn("wheel-child-stderr", log)

    def test_real_exec_handoff_reaches_the_requested_child_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "exec-child"
            child = f"from pathlib import Path; Path({str(marker)!r}).write_text('entered'); raise SystemExit(3)"
            code = f"import os, sys; os.execv(sys.executable, [sys.executable, '-c', {child!r}])"
            result, metadata = native.run([sys.executable, "-c", code], root, "exec-handoff")
            self.assertEqual(result, 3, metadata)
            self.assertEqual(metadata["process"]["state"], "exited")
            self.assertEqual(marker.read_text(), "entered")

    def test_real_abort_and_trap_capture_native_stack_and_never_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("SIGABRT", "SIGTRAP"):
                with self.subTest(signal=name):
                    marker = root / f"continued-{name}"
                    entered = root / f"entered-{name}"
                    code = ("import os, resource, signal; from pathlib import Path; "
                            "resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
                            f"Path({str(entered)!r}).write_text('before-signal'); "
                            f"os.kill(os.getpid(), signal.{name}); "
                            f"Path({str(marker)!r}).write_text('continued')")
                    result, metadata = native.run([sys.executable, "-c", code], root, name)
                    self.assertNotEqual(result, 0, metadata)
                    self.assertTrue(entered.is_file(), metadata)
                    self.assertEqual(entered.read_text(), "before-signal")
                    self.assertFalse(marker.exists())
                    self.assertIn(metadata["process"]["state"], ("stopped", "crashed"))
                    self.assertTrue(metadata["process"]["threads"])
                    descriptions = [thread["stop_description"] or ""
                                    for thread in metadata["process"]["threads"]]
                    self.assertTrue(any(name in description or
                                        (name == "SIGTRAP" and "EXC_BREAKPOINT" in description)
                                        for description in descriptions), descriptions)
                    stopped = [thread for thread in metadata["process"]["threads"]
                               if name in (thread["stop_description"] or "") or
                               (name == "SIGTRAP" and "EXC_BREAKPOINT" in (thread["stop_description"] or ""))]
                    for thread in stopped:
                        self.assertNotIn("frames_error", thread)
                        self.assertTrue(thread["frames"])
                        self.assertGreaterEqual(thread["frame_count"], len(thread["frames"]))
                        self.assertTrue(any(frame.get("function") and frame.get("module_uuid")
                                            and frame.get("pc") for frame in thread["frames"]))
                    self.assertTrue(metadata["process"]["diagnostics"]["thread backtrace all"])
                    self.assertIn("frame #0", Path(metadata["log"]).read_text())


if __name__ == "__main__":
    unittest.main()
