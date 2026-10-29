"""Memory diagnostics must prove instrumentation and preserve failing outcomes."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import native_memory_diagnostic as diagnostic


class NativeMemoryDiagnosticTests(unittest.TestCase):
    def test_framework_interpreter_bypasses_launcher_and_missing_binary_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "Resources/Python.app/Contents/MacOS/Python"
            configuration = {"PYTHONFRAMEWORK": "Python", "prefix": str(root)}
            with patch.object(diagnostic.sysconfig, "get_config_var", configuration.get):
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    diagnostic.interpreter()
                executable.parent.mkdir(parents=True)
                executable.touch()
                self.assertEqual(diagnostic.interpreter(), executable)

    def test_runtime_is_unique_and_comes_from_selected_compiler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(diagnostic.subprocess, "check_output", return_value=str(root)) as compiler:
                with self.assertRaisesRegex(RuntimeError, "expected one"):
                    diagnostic.sanitizer_runtime()
                runtime = root / "librustc-stable_rt.asan.dylib"
                runtime.touch()
                self.assertEqual(diagnostic.sanitizer_runtime(), runtime.resolve())
                compiler.assert_called_with(["rustc", "--print", "target-libdir"], text=True)
                (root / "librustc-nightly_rt.asan.dylib").touch()
                with self.assertRaisesRegex(RuntimeError, "expected one"):
                    diagnostic.sanitizer_runtime()

    def test_preloading_runtime_cannot_disguise_an_uninstrumented_wheel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "native.so").write_bytes(b"diagnostic fixture")
            distribution = SimpleNamespace(version="0.1.0", files=["native.so"], locate_file=lambda p: root / p)
            with patch.object(diagnostic.metadata, "distribution", return_value=distribution), \
                    patch.object(diagnostic.subprocess, "check_output", return_value="U ___asan_init"):
                with self.assertRaisesRegex(RuntimeError, "not ASan-instrumented"):
                    diagnostic.instrumented_extension()
            with patch.object(diagnostic.metadata, "distribution", return_value=distribution), \
                    patch.object(diagnostic.subprocess, "check_output", return_value="U ___asan_report_load1\nU ___asan_report_store8"):
                record = diagnostic.instrumented_extension()
            self.assertEqual(record["path"], str(root / "native.so"))
            self.assertEqual(len(record["sha256"]), 64)

    def test_detector_requires_both_normal_exit_and_a_real_fault_report(self):
        for normal, fault, message, accepted in (
                (0, -6, "AddressSanitizer: heap-use-after-free", True),
                (1, -6, "AddressSanitizer: heap-use-after-free", False),
                (0, 0, "AddressSanitizer: heap-use-after-free", False),
                (0, -6, "interceptors not installed", False),
                (0, 1, "", False)):
            with self.subTest(normal=normal, fault=fault, message=message), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                results = [subprocess.CompletedProcess([], 0),
                           subprocess.CompletedProcess([], normal, stdout="7\n", stderr=""),
                           subprocess.CompletedProcess([], fault, stdout="", stderr=message)]
                with patch.object(diagnostic.subprocess, "run", side_effect=results) as run:
                    if accepted:
                        diagnostic.verify_detector(directory)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "did not distinguish"):
                            diagnostic.verify_detector(directory)
                self.assertEqual(run.call_count, 3)
                self.assertEqual(run.call_args_list[-1].args[0][-1], "--fault")
                self.assertEqual(run.call_args.kwargs["env"]["ASAN_OPTIONS"], diagnostic.ASAN_OPTIONS)
                self.assertEqual(json.loads((directory / "asan-probe.json").read_text())[1]["exit_code"], fault)

    def test_probe_compilation_failure_does_not_launch_tests(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(diagnostic.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["rustc"])) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                diagnostic.verify_detector(Path(temporary))
            self.assertEqual(run.call_count, 1)

    def test_launch_preloads_selected_runtime_and_keeps_virtual_environment(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(diagnostic.platform, "system", return_value="Darwin"), \
                patch.object(diagnostic.platform, "platform", return_value="macOS test fixture"), \
                patch.object(diagnostic, "instrumented_extension", return_value={}), \
                patch.object(diagnostic, "sanitizer_runtime", return_value=Path("/runtime with spaces/asan.dylib")), \
                patch.object(diagnostic, "interpreter", return_value=Path("/actual Python")), \
                patch.object(diagnostic, "verify_detector"), \
                patch.object(diagnostic.subprocess, "check_output", return_value="rustc fixture"), \
                patch.object(diagnostic, "run_test_process", return_value=None) as run, \
                patch.dict(os.environ, {"ASAN_OPTIONS": "halt_on_error=0"}):
            result = diagnostic.prepare_sanitizer(Path(temporary))
            self.assertEqual(result, Path("/actual Python"))
            self.assertEqual(os.environ["DYLD_INSERT_LIBRARIES"], str(Path("/runtime with spaces/asan.dylib")))
            self.assertEqual(os.environ["__PYVENV_LAUNCHER__"], diagnostic.sys.executable)
            self.assertEqual(os.environ["ASAN_OPTIONS"], diagnostic.ASAN_OPTIONS)
            self.assertEqual(run.call_args.args[0][0], str(Path("/actual Python")))
            run.return_value = "import failed"
            with self.assertRaisesRegex(RuntimeError, "import failed"):
                diagnostic.prepare_sanitizer(Path(temporary))

    def test_native_shard_failure_is_propagated_once_with_unchanged_selection(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(diagnostic, "prepare_sanitizer", return_value=Path("literal interpreter")), \
                patch.object(diagnostic, "run_validation", side_effect=RuntimeError("native failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "native failed"):
                diagnostic.main(["--shard", diagnostic.SHARDS[0], "--report-directory", temporary])
            run.assert_called_once_with(Path("literal interpreter"), "all", diagnostic.SHARDS[0], Path(temporary).resolve())

    def test_calibration_isolation_does_not_prepare_or_launch_a_native_wheel(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(diagnostic, "calibration_only") as calibration, \
                patch.object(diagnostic, "prepare_sanitizer") as prepare, \
                patch.object(diagnostic, "run_validation") as run:
            diagnostic.main(["--calibration-only", "--report-directory", temporary])
            calibration.assert_called_once_with(Path(temporary).resolve())
            prepare.assert_not_called()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
