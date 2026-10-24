"""Coverage and failure guards for provider CI partitioning (no wheel needed)."""
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from wheel_validation import provider_shards as shards
from check_source_archive import check_ci_workflow


class ProviderShardTests(unittest.TestCase):
    def test_archive_workflow_gate_requires_complete_plan_and_producer_wheel_reuse(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()

        def validate(source):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                content = source.encode()
                member = tarfile.TarInfo(".github/workflows/ci.yml")
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
        ):
            with self.subTest(before=before), self.assertRaises(SystemExit):
                validate(workflow.replace(before, after))

    def test_ci_matrix_runs_every_selection_once_per_platform(self):
        matrix = shards.ci_matrix()["include"]
        self.assertEqual(len(matrix), 63)
        self.assertLessEqual(len(matrix), 256)
        self.assertEqual(len(shards.SHARD_NAMES), 20)
        self.assertEqual(len(set(shards.SHARD_NAMES)), 20)
        for platform in ("ubuntu-24.04", "macos-15", "windows-2025"):
            rows = [row for row in matrix if row["os"] == platform]
            self.assertEqual(Counter(row["selection"] for row in rows),
                             Counter((*shards.SHARD_NAMES, "remaining")))
            for row in rows:
                self.assertEqual(row["option"], "--group" if row["selection"] == "remaining" else "--shard")

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
                    patch.object(sys, "argv", ["provider_shards.py", "--github-matrix"]):
                shards.main()
            lines = output.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            key, value = lines[0].split("=", 1)
            self.assertEqual(key, "matrix")
            self.assertEqual(json.loads(value), shards.ci_matrix())


if __name__ == "__main__":
    unittest.main()
