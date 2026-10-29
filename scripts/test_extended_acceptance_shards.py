"""Exercise exact inventory, real libtest execution and fail-closed aggregation."""

from collections import Counter
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import extended_acceptance_shards as shards
from check_source_archive import ci_job_text
from test_ci_failure_propagation import validate_archive, WORKFLOW, REUSABLE_WORKFLOW


class PartitionTests(unittest.TestCase):
    def test_source_fingerprint_tracks_dirty_new_and_runtime_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            (root / "Cargo.toml").write_text("original\n", encoding="utf-8")
            subprocess.run(["git", "add", "Cargo.toml"], cwd=root, check=True)
            with patch.object(shards, "ROOT", root):
                original = shards.source_fingerprint()
                (root / "Cargo.toml").write_text("dirty\n", encoding="utf-8")
                dirty = shards.source_fingerprint()
                (root / "crates").mkdir()
                (root / "crates/new.rs").write_text("new source\n", encoding="utf-8")
                added = shards.source_fingerprint()
                (root / "fixtures").mkdir()
                (root / "fixtures/runtime.json").write_text("{}\n", encoding="utf-8")
                fixture = shards.source_fingerprint()
                self.assertEqual(len({original, dirty, added, fixture}), 4)
                (root / "README.md").write_text("unrelated docs\n", encoding="utf-8")
                self.assertEqual(shards.source_fingerprint(), fixture)

    def test_workflow_cannot_drop_a_platform_or_failed_dependency(self):
        validate_archive(WORKFLOW)
        mutations = {
            "extended-acceptance-run": (
                ('os: [ubuntu-24.04, macos-15, windows-2025]', 'os: [ubuntu-24.04]'),
                ('    strategy:', '    continue-on-error: true\n    strategy:'),
                ('    strategy:', '    needs: all-platform-builds\n    strategy:'),
                ('    strategy:', '    if: always()\n    strategy:'),
            ),
            "extended-acceptance": (
                ('runs-on: ubuntu-24.04', 'runs-on: ${{ matrix.os }}'),
                ('os: [ubuntu-24.04, macos-15, windows-2025]', 'os: [ubuntu-24.04]'),
                ('    if: always()', "    if: success()"),
                ('    needs: extended-acceptance-run', '    needs: other-job'),
                ('ACCEPTANCE_RESULT: ${{ needs.extended-acceptance-run.result }}', 'ACCEPTANCE_RESULT: success'),
                ('if result != "success":', 'if result == "failure":'),
            ),
        }
        for job_name, replacements in mutations.items():
            original = ci_job_text(WORKFLOW, job_name)
            for before, after in replacements:
                self.assertIn(before, original)
                with self.subTest(job=job_name, mutation=before), self.assertRaises(SystemExit):
                    validate_archive(WORKFLOW.replace(original, original.replace(before, after)))

    def test_reusable_workflow_cannot_drop_shards_or_add_platform_barriers(self):
        for name in ("build", "risk", "accuracy"):
            original = ci_job_text(REUSABLE_WORKFLOW, name)
            replacements = [
                ('runs-on: ${{ inputs.os }}', 'runs-on: ubuntu-24.04'),
                ('        run: python', '        continue-on-error: true\n        run: python'),
                ('        run: python', '        if: false\n        run: python'),
            ]
            if name == "build":
                replacements.extend([
                    ('    steps:', '    needs: all-platform-builds\n    steps:'),
                    ('    steps:', '    strategy:\n      matrix:\n        os: [ubuntu-24.04, macos-15, windows-2025]\n    steps:'),
                ])
            else:
                replacements.extend([
                    ('    needs: build', '    needs: all-platform-builds'),
                    ('    needs: build', '    needs: [build, other-platform-build]'),
                    ('name: extended-acceptance-binaries-${{ inputs.os }}', 'name: extended-acceptance-binaries-ubuntu-24.04'),
                ])
            if name == "accuracy":
                replacements.extend([
                    ('shard: [0, 1, 2, 3]', 'shard: [0, 1, 2]'),
                    ('        shard:', '        os: [ubuntu-24.04, macos-15, windows-2025]\n        shard:'),
                ])
            else:
                replacements.append(('    steps:', '    strategy:\n      matrix:\n        os: [ubuntu-24.04]\n    steps:'))
            command = re.search(r'^        run: python [^\n]+', original, re.MULTILINE).group(0)
            replacements.append((command, command + ' || true'))
            for before, after in replacements:
                self.assertIn(before, original)
                changed = REUSABLE_WORKFLOW.replace(original, original.replace(before, after))
                with self.subTest(job=name, mutation=after), self.assertRaises(SystemExit):
                    validate_archive(WORKFLOW, changed)
            with self.subTest(missing=name), self.assertRaises(SystemExit):
                validate_archive(WORKFLOW, REUSABLE_WORKFLOW.replace(original, ""))

    def test_actual_aggregate_guard_rejects_failure_cancellation_and_skips(self):
        gate = ci_job_text(WORKFLOW, "extended-acceptance")
        match = re.search(r'^        shell: python\n        run: \|\n((?:          [^\n]*\n)+)', gate, re.MULTILINE)
        self.assertIsNotNone(match)
        command = textwrap.dedent(match.group(1))
        for result in ("success", "failure", "cancelled", "skipped", ""):
            for options in ([], ["-O"]):
                with self.subTest(result=result, optimized=bool(options)):
                    completed = subprocess.run(
                        [sys.executable, *options, "-c", command],
                        env={**os.environ, "ACCEPTANCE_RESULT": result},
                        text=True, capture_output=True, check=False, timeout=30,
                    )
                    self.assertEqual(completed.returncode == 0, result == "success")
                    if result != "success":
                        self.assertIn("incomplete acceptance jobs:", completed.stderr)

    def test_historical_duration_partition_is_complete_and_balanced(self):
        weights = shards.weights()
        groups = shards.partition(list(weights))
        self.assertEqual(len(groups), 4)
        self.assertEqual(Counter(n for group in groups for n in group), Counter(weights.keys()))
        self.assertEqual(groups, shards.partition(list(reversed(weights))))
        longest = max(sum(weights[n] for n in group) for group in groups)
        self.assertLess(longest, sum(weights.values()) * 0.26)

    def test_new_missing_duplicate_and_empty_cases_cannot_silently_disappear(self):
        weights = {"a": 1, "b": 2, "c": 3, "d": 4}
        for cases in ([], ["a", "a", "b", "c", "d"], ["a", "b", "c"],
                      ["a", "b", "c", "d", "new_case"]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                shards.partition(cases, weights)
        with self.assertRaises(ValueError):
            shards.partition(list(weights), weights, count=5)


class LibtestIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.binary = cls.root / ("acceptance.exe" if os.name == "nt" else "acceptance")
        cls.names = [f"case_{index}" for index in range(8)]
        cls.weights = {name: index + 1 for index, name in enumerate(cls.names)}
        source = cls.root / "probe.rs"
        source.write_text("\n".join(
            f'#[test] #[ignore] fn {name}() {{ println!("executed={name}"); '
            f'if std::env::var("ACCEPTANCE_PROBE_FAIL").as_deref() == Ok("{name}") '
            '{ std::process::exit(23); } }' for name in cls.names)
            + '\n#[test] fn ordinary() { panic!("non-ignored test must not run"); }\n',
            encoding="utf-8")
        subprocess.run(["rustc", "--test", str(source), "-o", str(cls.binary)], check=True,
                       capture_output=True, text=True)
        cls.head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=shards.ROOT, text=True).strip()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "binaries"
        self.bundle.mkdir()
        shutil.copy2(self.binary, self.bundle / self.binary.name)
        self.manifest = {"source_head": self.head,
                         "source_fingerprint": shards.source_fingerprint(),
                         "platform": {"system": platform.system(), "machine": platform.machine()},
                         "compiler": "fixture", "binaries": {
            suite: {"file": self.binary.name, "sha256": shards.sha256(self.binary), "cases": self.names}
            for suite in shards.TARGETS}}
        self.save_manifest()
        self.reports = self.root / "reports"

    def save_manifest(self):
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def execute(self, selection):
        with patch.object(shards, "weights", return_value=self.weights), redirect_stdout(io.StringIO()):
            return shards.run(selection, self.bundle, self.reports)

    def collect(self):
        with patch.object(shards, "weights", return_value=self.weights):
            return shards.combine(self.reports, self.root / "combined.log")

    def execute_all(self):
        return [self.execute(f"accuracy-{i}") for i in range(4)]

    def test_real_libtest_exact_filters_run_every_ignored_case_once(self):
        records = self.execute_all()
        report = self.collect()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["test_count"], 8)
        self.assertEqual(Counter(n for r in records for n in r["selected"]), Counter(self.names))
        combined = (self.root / "combined.log").read_text(encoding="utf-8")
        for name in self.names:
            self.assertEqual(combined.count(f"executed={name}"), 1)
        self.assertTrue((self.root / "combined.json").is_file())
        risk = self.execute("risk")
        self.assertEqual(risk["selected"], self.names)

    def test_native_failure_is_retained_and_rejected_by_aggregation(self):
        selected = shards.partition(self.names, self.weights)[0][0]
        with patch.dict(os.environ, {"ACCEPTANCE_PROBE_FAIL": selected}), self.assertRaises(RuntimeError):
            self.execute("accuracy-0")
        record = json.loads((self.reports / "extended-model-accuracy-0.json").read_text())
        self.assertEqual(record["returncode"], 23)
        self.assertEqual(record["status"], "failed")
        for i in range(1, 4):
            self.execute(f"accuracy-{i}")
        with self.assertRaisesRegex(ValueError, "failed, incomplete"):
            self.collect()

    def test_missing_duplicate_stale_and_tampered_evidence_is_rejected(self):
        self.execute_all()
        path = self.reports / "extended-model-accuracy-0.json"
        original = path.read_text()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            self.collect()
        path.write_text(original)
        duplicate = self.reports / "extended-model-accuracy-copy.json"
        duplicate.write_text(original)
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            self.collect()
        duplicate.unlink()
        record = json.loads(original)
        for key, value in (("source_head", "stale"), ("source_fingerprint", "different"), ("status", "running"),
                           ("selected", []), ("binary_sha256", "other")):
            record = json.loads(original)
            record[key] = value
            path.write_text(json.dumps(record))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.collect()
        path.write_text(original)
        log = self.reports / "extended-model-accuracy-0.log"
        log.write_text(log.read_text() + "tampered\n")
        with self.assertRaisesRegex(ValueError, "log differs"):
            self.collect()

    def test_bundle_checksum_head_and_inventory_must_match(self):
        for mutate in (lambda d: d.update(source_head="old"),
                       lambda d: d.update(source_fingerprint="dirty-source-change"),
                       lambda d: d.update(platform={"system": "other", "machine": "other"}),
                       lambda d: d["binaries"]["accuracy"].update(cases=["missing"]),
                       lambda d: d["binaries"]["accuracy"].update(sha256="bad"),
                       lambda d: d["binaries"]["accuracy"].update(file="../other")):
            original = json.loads(json.dumps(self.manifest))
            mutate(self.manifest)
            self.save_manifest()
            with self.assertRaises(ValueError):
                self.execute("accuracy-0")
            self.manifest = original


if __name__ == "__main__":
    unittest.main()
