"""A partial diagnostic must fail and leave evidence, without orphan children."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import native_memory_watchdog as watchdog


class NativeMemoryWatchdogTests(unittest.TestCase):
    def test_real_child_results_and_literal_arguments_are_preserved(self):
        literal = "path with spaces 'quotes' `not a shell` $(not a command)"
        for exit_code in (0, 17):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                command = [sys.executable, "-c",
                           "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); "
                           "sys.exit(int(sys.argv[3]))", str(directory / "literal"), literal, str(exit_code)]
                with patch.object(watchdog, "observe", return_value={"status": "observed"}):
                    result = watchdog.supervise(command, directory, 10, .01, 1)
                self.assertEqual(result, exit_code)
                self.assertEqual((directory / "literal").read_text(), literal)
                record = json.loads((directory / "supervisor.json").read_text())
                self.assertEqual(record["child_returncode"], exit_code)
                self.assertEqual(record["status"], "passed" if not exit_code else "failed")

    def test_launch_failure_invalidates_an_old_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "supervisor.json").write_text('{"status":"passed"}')
            with self.assertRaises(OSError):
                watchdog.supervise([str(directory / "absent")], directory)
            record = json.loads((directory / "supervisor.json").read_text())
            self.assertEqual(record["status"], "supervisor_error")
            self.assertIsNone(record["child_returncode"])

    def test_child_signal_remains_a_failed_workload(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(watchdog, "observe", return_value={"status": "observed"}):
            directory = Path(temporary)
            result = watchdog.supervise(
                [sys.executable, "-c", "import os,signal;os.kill(os.getpid(),signal.SIGTERM)"],
                directory, 5, .01, 1)
            record = json.loads((directory / "supervisor.json").read_text())
            self.assertEqual(result, 128 + signal.SIGTERM)
            self.assertEqual(record["child_returncode"], -signal.SIGTERM)
            self.assertEqual(record["status"], "failed")

    def test_observer_failure_does_not_hide_child_failure(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(watchdog, "observe", side_effect=OSError("sampling unavailable")):
            directory = Path(temporary)
            result = watchdog.supervise([sys.executable, "-c", "import time;time.sleep(.1);raise SystemExit(7)"],
                                        directory, 5, .03, 1)
            record = json.loads((directory / "supervisor.json").read_text())
            self.assertEqual(result, 7)
            self.assertEqual(record["status"], "failed")
            self.assertGreater(record["observation_errors"], 0)
            self.assertIn("sampling unavailable", (directory / "observations.jsonl").read_text())

    def assert_process_stopped(self, pid):
        result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
        self.assertTrue(result.returncode != 0 or result.stdout.strip().startswith("Z"),
                        f"child {pid} is still running: {result.stdout}")

    def test_deadline_kills_the_child_group_and_retains_an_incomplete_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            child = ("import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                     f"pathlib.Path({str(directory / 'grandchild')!r}).write_text(str(os.getpid())); time.sleep(60)")
            parent = ("import subprocess,sys,time; "
                      f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)")
            with patch.object(watchdog, "observe", return_value={"status": "observed"}):
                result = watchdog.supervise([sys.executable, "-c", parent], directory, 1, .1, 1)
            record = json.loads((directory / "supervisor.json").read_text())
            self.assertEqual(result, 124)
            self.assertEqual(record["status"], "timed_out")
            self.assertNotEqual(record["child_returncode"], 0)
            self.assert_process_stopped(record["pid"])
            self.assert_process_stopped(int((directory / "grandchild").read_text()))

    def test_cancellation_records_interruption_and_stops_the_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            child = ("import os,pathlib,time; "
                     f"pathlib.Path({str(directory / 'ready')!r}).write_text(str(os.getpid())); time.sleep(60)")
            command = [sys.executable, str(Path(watchdog.__file__)), "--report-directory", str(directory),
                       "--timeout-seconds", "30", "--", sys.executable, "-c", child]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 10
                while not (directory / "ready").exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((directory / "ready").exists())
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 143, stdout + stderr)
                record = json.loads((directory / "supervisor.json").read_text())
                self.assertEqual(record["status"], "interrupted")
                self.assertIn("SIGTERM", record["error"])
                self.assert_process_stopped(int((directory / "ready").read_text()))
            finally:
                if (directory / "ready").exists():
                    try:
                        os.killpg(int((directory / "ready").read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)

    def test_cancellation_during_observation_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(watchdog, "observe", side_effect=watchdog.DiagnosticCancelled("SIGTERM")):
            directory = Path(temporary)
            with self.assertRaises(watchdog.DiagnosticCancelled):
                watchdog.supervise([sys.executable, "-c", "import time;time.sleep(60)"], directory, 10)
            record = json.loads((directory / "supervisor.json").read_text())
            self.assertEqual(record["status"], "interrupted")
            self.assertEqual(record["observation_errors"], 0)
            self.assert_process_stopped(record["pid"])

    def test_observation_limits_capture_to_the_owned_process_tree(self):
        rows = "11 1 0 12 00:02 Python\n12 11 90 100 00:01 /path with space/Python\n99 1 80 200 00:05 private\n"
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(watchdog.subprocess, "check_output", return_value=rows), \
                patch.object(watchdog.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "exited")) as sample, \
                patch.dict(os.environ, {"DYLD_INSERT_LIBRARIES": "/asan", "ASAN_OPTIONS": "fixture"}):
            record = watchdog.observe(11, Path(temporary), 0, True)
            self.assertEqual([p["pid"] for p in record["processes"]], [11, 12])
            self.assertEqual(record["sample"]["pid"], 12)
            self.assertEqual(record["sample"]["returncode"], 1)
            self.assertEqual(record["sample"]["status"], "unavailable")
            self.assertNotIn("DYLD_INSERT_LIBRARIES", sample.call_args.kwargs["env"])
            self.assertEqual(os.environ["DYLD_INSERT_LIBRARIES"], "/asan")

    def test_zero_sampling_exit_without_an_artifact_is_not_a_capture(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(watchdog.subprocess, "check_output", return_value="11 1 0 12 00:02 Python\n"), \
                patch.object(watchdog.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "target exited")):
            record = watchdog.observe(11, Path(temporary), 0, True)
            self.assertEqual(record["sample"]["status"], "unavailable")
            self.assertIsNone(record["sample"]["file"])

    def test_invalid_intervals_never_launch(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(watchdog.subprocess, "Popen") as launch:
            for value in (0, -1, math.nan, math.inf):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    watchdog.supervise(["unused"], Path(temporary), timeout_seconds=value)
            launch.assert_not_called()

    def test_signal_denial_is_ignored_only_for_a_dead_group(self):
        with patch.object(watchdog.os, "killpg", side_effect=PermissionError("denied")):
            for rows in ("42 Z\n", "99 S\n"):
                with patch.object(watchdog.subprocess, "check_output", return_value=rows):
                    watchdog.signal_group(42, signal.SIGKILL)
            for rows in ("42 S\n", "42 Z\n42 R\n"):
                with patch.object(watchdog.subprocess, "check_output", return_value=rows), \
                        self.assertRaises(PermissionError):
                    watchdog.signal_group(42, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
