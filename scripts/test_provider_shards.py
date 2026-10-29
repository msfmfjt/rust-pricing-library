"""Coverage and failure guards for provider CI partitioning (no wheel needed)."""
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from contextlib import nullcontext
from dataclasses import replace
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from wheel_validation import provider_shards as shards
from check_source_archive import check_ci_workflow


class ProviderShardTests(unittest.TestCase):
    def test_combined_reproduction_keeps_original_evidence_and_restores_environment(self):
        seen = []

        class Probe(unittest.TestCase):
            def test_observation_mode(self):
                seen.append(shards.os.environ.get("PRICING_TEST_ORIGINAL_EVIDENCE"))

        for selected, expected in ((shards.COMBINED_SHARDS[0], "1"), (shards.SHARDS[0], "0")):
            with self.subTest(selection=selected.name), \
                    patch.dict(shards.os.environ, {"PRICING_TEST_ORIGINAL_EVIDENCE": "0"}), \
                    patch.object(shards, "load_suite", return_value=(None, Probe)), \
                    patch.object(shards, "configured", return_value=nullcontext()), \
                    patch.object(shards, "selected_methods", return_value=["test_observation_mode"]), \
                    redirect_stderr(io.StringIO()):
                self.assertTrue(shards.run_shard(selected.name))
                self.assertEqual(seen[-1], expected)
                self.assertEqual(shards.os.environ["PRICING_TEST_ORIGINAL_EVIDENCE"], "0")

    def test_archive_workflow_gate_requires_complete_plan_and_producer_wheel_reuse(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
        reusable = (Path(__file__).resolve().parents[1] / ".github/workflows/extended-acceptance.yml").read_text()

        def validate(source):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                for name, body in (("ci.yml", source), ("extended-acceptance.yml", reusable)):
                    content = body.encode()
                    member = tarfile.TarInfo(".github/workflows/" + name)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
            buffer.seek(0)
            with tarfile.open(fileobj=buffer, mode="r") as archive:
                check_ci_workflow(archive, "probe")

        validate(workflow)
        for before, after in (
            ("needs: [python-wheel-linux, supported-wheels, wheel-validation-plan]", "needs: [wheel-validation-plan]"),
            ("matrix: ${{ fromJSON(needs.wheel-validation-plan.outputs.matrix) }}", "matrix: {os: [ubuntu-24.04]}"),
            ("--github-matrix", "--run extension-contracts"),
            ("--report-directory wheel-validation-reports", ""),
            ("path: wheel-validation-reports/", "path: dist/*.whl"),
            ("timeout-minutes: 330\n    strategy:", "timeout-minutes: 330\n    continue-on-error: true\n    strategy:"),
            ("'test_provider_shards.py'", "'missing.py'"),
            ("'test_native_debugger.py'", "'missing-debugger.py'"),
            ("${{ runner.os == 'macOS' && '--native-debugger lldb' || '' }}", ""),
            ("name: Verify native debugger on this macOS runner", "name: missing debugger preflight"),
        ):
            with self.subTest(before=before), self.assertRaises(SystemExit):
                validate(workflow.replace(before, after))

    def test_ci_matrix_runs_every_selection_once_per_platform(self):
        matrix = shards.ci_matrix()["include"]
        self.assertEqual(len(matrix), 120)
        self.assertLessEqual(len(matrix), 256)
        self.assertEqual(len(shards.SHARDS), 36)
        self.assertEqual(len(shards.COMBINED_SHARDS), 4)
        self.assertEqual(len(shards.SHARD_NAMES), 40)
        self.assertEqual(len(set(shards.SHARD_NAMES)), 40)
        ci_names = tuple(shard.name for shard in shards.SHARDS)
        for platform in ("ubuntu-24.04", "macos-15", "windows-2025"):
            rows = [row for row in matrix if row["os"] == platform]
            self.assertEqual(Counter(row["selection"] for row in rows),
                             Counter((*ci_names, *shards.REMAINING_GROUPS)))
            for row in rows:
                self.assertEqual(row["option"], "--group" if row["selection"] in shards.REMAINING_GROUPS else "--shard")

    def test_split_preserves_case_multiplicity_and_rejects_missing_or_overlapping_ledgers(self):
        source = '''import unittest
from itertools import product
class Case(unittest.TestCase):
    policies = ('first',)
    intervals = (4, 6)
    @classmethod
    def wheel_cases(cls):
        return {
            'test_a_native_panels': list(product(cls.policies, cls.intervals, ('call', 'put', 'put'))),
            'test_b_dated_ledgers': list(product(cls.policies, cls.intervals,
                ('call', 'put'), ('arithmetic_asian', 'fixed_lookback'))),
        }
    @classmethod
    def setUpClass(cls):
        raise AssertionError('inventory must not construct providers')
    def test_a_native_panels(self):
        raise AssertionError('inventory must not run test bodies')
    def test_b_dated_ledgers(self):
        raise AssertionError('inventory must not run test bodies')
    def test_control(self):
        raise AssertionError('inventory must not run test bodies')
'''
        spec = shards.ProviderSuite("probe", "test_ledger_inventory_probe", "Case", "policies", ("first",))
        original = (shards.ProviderShard("contracts", spec),
                    *(shards.ProviderShard(f"first-{n}", spec, "first", n) for n in (4, 6)))
        plan = tuple(part for shard in original for part in shards.split_shard(shard))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / (spec.module + ".py")).write_text(source, encoding="utf-8")
            with patch.object(shards, "SUITES", (spec,)), patch.object(sys, "path", sys.path.copy()), \
                    patch.dict(sys.modules), redirect_stdout(io.StringIO()):
                report = shards.verify_partition(root, plan)
                old = shards.verify_partition(root, original)
                total = lambda rows: sum(c["count"] for row in rows.values() for c in row["cases"])
                self.assertEqual(total(report), 15)
                self.assertEqual(total(report), total(old))
                self.assertEqual(len(report), 7)
                self.assertEqual(sum(c["count"] for c in report["first-4-panels"]["cases"]), 3)
                for part in plan[2:6]:
                    self.assertEqual(report[part.name]["methods"], ["test_b_dated_ledgers"])
                    self.assertEqual(len(report[part.name]["cases"]), 1)
                    self.assertEqual(report[part.name]["cases"][0]["parameters"], list(part.parameters))
                for broken in (plan[:5] + plan[6:], plan + (replace(plan[5], name="repeat"),)):
                    with self.assertRaisesRegex(RuntimeError, "case inventory differs"):
                        shards.verify_partition(root, broken)
                _, cls = shards.load_suite(spec, root)
                with self.assertRaisesRegex(RuntimeError, "unknown provider case method"):
                    with shards.configured(cls, replace(plan[2], method="test_typo")):
                        shards.selected_methods(cls, plan[2])
                with self.assertRaisesRegex(RuntimeError, "empty provider shard"):
                    with shards.configured(cls, replace(plan[2], parameters=("typo",))):
                        shards.selected_methods(cls, plan[2])
                with self.assertRaisesRegex(RuntimeError, "requires a method"):
                    with shards.configured(cls, replace(plan[2], method=None)):
                        pass
                self.assertEqual(cls.intervals, (4, 6))
                self.assertEqual(len(cls.wheel_cases()["test_b_dated_ledgers"]), 8)

    def test_ledger_runtime_and_combined_alias_keep_order_setup_and_failure(self):
        events = []
        class Parent(unittest.TestCase):
            policies = ("first",)
            intervals = (4, 6)
            @classmethod
            def wheel_cases(cls):
                return {
                    "test_a_native_panels": [(p, n, "call") for p in cls.policies for n in cls.intervals],
                    "test_b_dated_ledgers": [(p, n, side, kind)
                        for p in cls.policies for n in cls.intervals for side in ("call", "put")
                        for kind in ("arithmetic_asian", "fixed_lookback")],
                }
            @classmethod
            def setUpClass(cls):
                events.append("setup")
            @classmethod
            def tearDownClass(cls):
                events.append("teardown")
            def test_a_native_panels(self):
                events.extend(shards.iter_cases(self))
            def test_b_dated_ledgers(self):
                for values in shards.iter_cases(self):
                    events.append(values)
                    if self.fail_ledger:
                        self.fail("selected ledger failed")
        class Child(Parent):
            fail_ledger = False
        spec = shards.ProviderSuite("probe", "probe", "Child", "policies", ("first",))
        original = shards.ProviderShard("first-4", spec, "first", 4)
        plan = shards.split_shard(original)
        with patch.object(shards, "SHARDS", plan), patch.object(shards, "COMBINED_SHARDS", (original,)), \
                patch.object(shards, "load_suite", return_value=(None, Child)), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for part in plan:
                self.assertTrue(shards.run_shard(part.name))
            split_events = events.copy()
            self.assertEqual(split_events.count("setup"), 5)
            self.assertEqual(split_events.count("teardown"), 5)
            events.clear()
            self.assertTrue(shards.run_shard(original.name))
            self.assertEqual([x for x in split_events if isinstance(x, tuple)], events[1:-1])
            self.assertEqual((events[0], events[-1]), ("setup", "teardown"))
            events.clear()
            Child.fail_ledger = True
            self.assertFalse(shards.run_shard(plan[-1].name))
            self.assertEqual(events, ["setup", ("first", 4, "put", "fixed_lookback"), "teardown"])
            with self.assertRaisesRegex(RuntimeError, "probe error"):
                with shards.configured(Child, plan[-1]):
                    raise RuntimeError("probe error")
        for name in ("wheel_cases", "policies", "intervals"):
            self.assertNotIn(name, Child.__dict__)
        self.assertEqual(Child.wheel_cases(), Parent.wheel_cases())

    def test_partition_rejects_lost_extra_duplicate_cases_and_retains_new_controls(self):
        source = '''import unittest
from itertools import product
class Case(unittest.TestCase):
    policies = ('first', 'second')
    intervals = (4, 6)
    @classmethod
    def wheel_cases(cls):
        return {'test_panels': list(product(cls.policies, cls.intervals, ('call', 'put')))}
    @classmethod
    def setUpClass(cls):
        raise AssertionError('inventory must not construct providers')
    def test_panels(self):
        raise AssertionError('inventory must not run test bodies')
    def test_control(self):
        raise AssertionError('inventory must not run test bodies')
    def test_new_control(self):
        raise AssertionError('inventory must not run test bodies')
'''
        spec = shards.ProviderSuite("probe", "test_shard_inventory_probe", "Case", "policies", ("first", "second"))
        plan = (shards.ProviderShard("contracts", spec), *(shards.ProviderShard(f"{p}-{n}", spec, p, n)
                 for p in spec.policies for n in (4, 6)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / (spec.module + ".py")).write_text(source, encoding="utf-8")
            with patch.object(shards, "SUITES", (spec,)), patch.object(sys, "path", sys.path.copy()), \
                    patch.dict(sys.modules), redirect_stdout(io.StringIO()):
                report = shards.verify_partition(root, plan)
                self.assertEqual(sum(c["count"] for row in report.values() for c in row["cases"]), 10)
                self.assertEqual(report["contracts"]["methods"], ["test_control", "test_new_control"])
                _, cls = shards.load_suite(spec, root)
                self.assertEqual(cls.policies, ("first", "second"))
                self.assertEqual(cls.intervals, (4, 6))
                for broken in (plan[:-1], plan + (replace(plan[-1], name="repeat"),),
                               plan[:-1] + (replace(plan[-1], policy="typo"),)):
                    with self.assertRaisesRegex(RuntimeError, "case inventory differs"):
                        shards.verify_partition(root, broken)
                with self.assertRaisesRegex(RuntimeError, "duplicate provider shard"):
                    shards.verify_partition(root, plan + (plan[-1],))
                with self.assertRaisesRegex(RuntimeError, "unknown provider suite"):
                    shards.verify_partition(root, (replace(plan[0], suite=replace(spec, name="typo")),))
                # unittest discovers aliases twice; a set comparison would miss this.
                sys.modules[spec.module].Alias = cls
                with self.assertRaisesRegex(RuntimeError, "case inventory differs"):
                    shards.verify_partition(root, plan)

    def test_configuration_restores_inherited_attributes_even_on_error(self):
        class Parent:
            policies = ("first", "second")
            intervals = (4, 6)
        class Child(Parent):
            pass
        spec = shards.ProviderSuite("probe", "probe", "Child", "policies", ("first", "second"))
        shard = shards.ProviderShard("probe-first-4", spec, "first", 4)
        with self.assertRaisesRegex(RuntimeError, "probe error"):
            with shards.configured(Child, shard):
                self.assertEqual(Child.policies, ("first",))
                self.assertEqual(Child.intervals, (4,))
                raise RuntimeError("probe error")
        self.assertNotIn("policies", Child.__dict__)
        self.assertNotIn("intervals", Child.__dict__)
        self.assertEqual(Child.policies, Parent.policies)
        with self.assertRaisesRegex(RuntimeError, "unknown axis"):
            with shards.configured(Child, replace(shard, suite=replace(spec, axis="typo"))):
                pass
        self.assertNotIn("typo", Child.__dict__)

    def test_empty_methods_missing_methods_and_empty_shards_fail_closed(self):
        class Case(unittest.TestCase):
            @classmethod
            def wheel_cases(cls):
                return {"test_panel": []}
            def test_panel(self):
                pass
        spec = shards.ProviderSuite("probe", "probe", "Case", "policies", ("first",))
        shard = shards.ProviderShard("probe", spec, "first", 4)
        with self.assertRaisesRegex(RuntimeError, "empty provider shard"):
            shards.selected_methods(Case, shard)
        with self.assertRaisesRegex(RuntimeError, "empty provider method"):
            shards.inventory([Case("test_panel")])
        with self.assertRaisesRegex(RuntimeError, "no provider cases"):
            list(shards.iter_cases(Case("test_panel")))
        with patch.object(Case, "wheel_cases", return_value={"test_missing": [(1,)]}):
            with self.assertRaisesRegex(RuntimeError, "missing test methods"):
                shards.selected_methods(Case, shard)

    def test_runtime_uses_declared_cases_and_preserves_class_setup(self):
        events = []
        class Case(unittest.TestCase):
            policies = ("first", "second")
            intervals = (4, 6)
            @classmethod
            def wheel_cases(cls):
                return {"test_panel": [(p, n) for p in cls.policies for n in cls.intervals]}
            @classmethod
            def setUpClass(cls):
                events.append("setup")
            @classmethod
            def tearDownClass(cls):
                events.append("teardown")
            def test_panel(self):
                events.extend(shards.iter_cases(self))
            def test_control(self):
                raise AssertionError("control belongs to another shard")
        spec = shards.ProviderSuite("probe", "probe", "Case", "policies", ("first", "second"))
        shard = shards.ProviderShard("probe", spec, "second", 6)
        with patch.object(shards, "SHARDS", (shard,)), \
                patch.object(shards, "load_suite", return_value=(None, Case)), redirect_stdout(io.StringIO()):
            self.assertTrue(shards.run_shard("probe"))
        self.assertEqual(events, ["setup", ("second", 6), "teardown"])
        self.assertEqual(Case.policies, ("first", "second"))

    def test_github_output_contains_a_single_complete_json_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.dict(shards.os.environ, {"GITHUB_OUTPUT": str(output)}), \
                    patch.object(sys, "argv", ["provider_shards.py", "--github-matrix"]), \
                    patch.object(shards.faulthandler, "register", create=True) as register:
                shards.main()
            register.assert_not_called()
            lines = output.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            key, value = lines[0].split("=", 1)
            self.assertEqual(key, "matrix")
            self.assertEqual(json.loads(value), shards.ci_matrix())

    def test_windows_cli_keeps_builtin_fault_handler_and_propagates_failure(self):
        # faulthandler.register is not available on Windows. The interpreter's
        # -X faulthandler remains enabled there, including Windows exceptions.
        with patch.object(sys, "platform", "win32"), \
                patch.object(sys, "argv", ["provider_shards.py", "--run", shards.SHARD_NAMES[0]]), \
                patch.object(shards.faulthandler, "register", create=True) as register, \
                patch.object(shards, "run_shard", return_value=False) as run, \
                self.assertRaises(SystemExit) as error:
            shards.main()
        self.assertEqual(error.exception.code, 1)
        register.assert_not_called()
        run.assert_called_once_with(shards.SHARD_NAMES[0])

    @unittest.skipUnless(sys.platform != "win32" and hasattr(signal, "SIGTRAP"),
                         "SIGTRAP and user fault handlers require POSIX")
    def test_real_trap_dumps_stack_and_remains_a_failed_process(self):
        # Run the actual shard CLI, loader, runner and parent recorder with a
        # synthetic provider. Never load the native pricing module or write a
        # core file, and only signal the child process which owns this test.
        code = '''
import os, resource, signal, sys, unittest
from scripts.wheel_validation import provider_shards as shards
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
class Probe(unittest.TestCase):
    @classmethod
    def wheel_cases(cls):
        return {}
    def test_native_trap_probe(self):
        print("before native trap", flush=True)
        os.kill(os.getpid(), signal.SIGTRAP)
        print("after native trap", flush=True)
shards.load_suite = lambda *args: (None, Probe)
sys.argv = ["provider_shards.py", "--run", shards.SHARD_NAMES[0]]
shards.main()
'''
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            parent = '''
import sys
from pathlib import Path
from scripts.smoke_test_wheel import run_test_process
failure = run_test_process([sys.executable, "-X", "faulthandler", "-c", sys.argv[1]],
                           "shard", "trap-probe", Path(sys.argv[2]))
raise SystemExit(1 if failure else 0)
'''
            result = subprocess.run([sys.executable, "-c", parent, code, str(reports)],
                                    cwd=Path(__file__).resolve().parents[1],
                                    text=True, capture_output=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("before native trap", result.stdout)
            self.assertNotIn("after native trap", result.stdout)
            self.assertIn("signal=SIGTRAP", result.stdout)
            self.assertIn("test_native_trap_probe", result.stderr)
            self.assertIn('File "<string>"', result.stderr)
            record = json.loads((reports / "trap-probe.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["returncode"], -signal.SIGTRAP)
            self.assertEqual(record["signal"], "SIGTRAP")


if __name__ == "__main__":
    unittest.main()
