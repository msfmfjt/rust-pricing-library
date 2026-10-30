"""Incomplete allocator stress must never become successful evidence."""
import copy
import json
from pathlib import Path
import re
import shlex
import tempfile
import unittest
from unittest.mock import patch

import system_allocator_diagnostic as diagnostic
from test_ci_failure_propagation import run_block


class SystemAllocatorDiagnosticTests(unittest.TestCase):
    def events(self):
        return [
            {"event": "start", "workers": 2, "rounds": 1, "seconds": 1,
             "nano_mode": 2, "secure_allocator": False},
            {"event": "case-start", "round": 0, "bytes": 8},
            {"event": "case", "round": 0, "bytes": 8, "elapsed_seconds": 1.01, "iterations": [100, 90]},
            {"event": "case-start", "round": 0, "bytes": 64},
            {"event": "case", "round": 0, "bytes": 64, "elapsed_seconds": 1.02, "iterations": [80, 70]},
            {"event": "completed", "cases": 2},
        ]

    def test_requires_complete_ordered_execution_by_every_worker(self):
        mutations = [
            lambda e: e.pop(),
            lambda e: e[0].update(workers=3),
            lambda e: e[1].update(bytes=64),
            lambda e: e[2].update(round=1),
            lambda e: e[4].update(bytes=8),
            lambda e: e[4].update(iterations=[80, 0]),
            lambda e: e[4].update(iterations=[80]),
            lambda e: e[4].update(iterations=[80, True]),
            lambda e: e[4].update(elapsed_seconds=0.2),
            lambda e: e[4].update(elapsed_seconds=float("nan")),
            lambda e: e[-1].update(cases=1),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            original = self.events()
            path.write_text("\n".join(map(json.dumps, original)))
            result = diagnostic.validate_events(path, 2, 1, 1)
            self.assertEqual(result["allocation_free_pairs"], 340)
            self.assertEqual(result["completed_cases"], 2)
            for index, mutate in enumerate(mutations):
                with self.subTest(mutation=index):
                    events = copy.deepcopy(original)
                    mutate(events)
                    path.write_text("\n".join(map(json.dumps, events)))
                    with self.assertRaises(ValueError):
                        diagnostic.validate_events(path, 2, 1, 1)

    def test_rejects_allocator_overrides_and_invalidates_previous_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary) / "system-allocator.json"
            record.write_text('{"status":"completed_without_reproduction"}')
            with patch.object(diagnostic.platform, "system", return_value="Darwin"), \
                    patch.object(diagnostic.platform, "platform", return_value="macOS test fixture"), \
                    patch.dict(diagnostic.os.environ, {"MallocNanoZone": "0"}, clear=True), \
                    patch.object(diagnostic, "run_test_process") as run:
                with self.assertRaisesRegex(RuntimeError, "no allocator overrides"):
                    diagnostic.main(["--report-directory", temporary])
                run.assert_not_called()
            self.assertEqual(json.loads(record.read_text())["status"], "failed_or_incomplete")

    def test_abnormal_child_never_validates_or_retries(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(diagnostic.platform, "system", return_value="Darwin"), \
                patch.object(diagnostic.platform, "platform", return_value="macOS test fixture"), \
                patch.dict(diagnostic.os.environ, {}, clear=True), \
                patch.object(diagnostic.subprocess, "check_output", return_value="fixture"), \
                patch.object(diagnostic.subprocess, "run") as compile_probe, \
                patch.object(diagnostic, "run_test_process", return_value="native signal") as run, \
                patch.object(diagnostic, "validate_events") as validate:
            (Path(temporary) / "system-allocator-probe").write_bytes(b"fixture")
            with self.assertRaisesRegex(RuntimeError, "native signal"):
                diagnostic.main(["--report-directory", temporary, "--rounds", "1", "--seconds", "1"])
            compile_probe.assert_called_once()
            run.assert_called_once()
            validate.assert_not_called()
            self.assertEqual(json.loads((Path(temporary) / "system-allocator.json").read_text())["status"],
                             "failed_or_incomplete")

    def test_workflow_log_pipeline_preserves_probe_failure(self):
        workflow = (diagnostic.ROOT / ".github/workflows/native-memory-diagnostic.yml").read_text()
        command = re.search(r"^        run: (python scripts/system_allocator_diagnostic.py[^\n]+)$",
                            workflow, re.MULTILINE)
        self.assertIsNotNone(command)
        for code in (0, 23):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                result = run_block(f"cd {shlex.quote(temporary)}\n"
                                   f"python() {{ printf '%s\\n' \"$@\"; return {code}; }}\n"
                                   + command.group(1) + "\nprintf 'continued\\n'\n")
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(result.stdout.splitlines(),
                                 ["scripts/system_allocator_diagnostic.py", "--report-directory",
                                  "system-allocator-reports"] + (["continued"] if code == 0 else []))


if __name__ == "__main__":
    unittest.main()
