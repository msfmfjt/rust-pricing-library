"""Oversized execution metadata must produce catchable native errors, not panics."""
import copy
import json
from pathlib import Path
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def task_document(name):
    return json.loads((ROOT / f"fixtures/task-v1/{name}.golden.json").read_text())


def execution_source(document):
    source = document["source"]
    return source.get("specification", source.get("equity", source))


class ExecutionResourceTests(unittest.TestCase):
    def test_impossible_time_grids_fail_at_compilation_across_task_adapters(self):
        for name in ("multi_asset_bs", "stochastic_dividend_bs", "stochastic_dividend_hw"):
            original = task_document(name)
            healthy = rp.PricingTask.from_json(json.dumps(original)).compile()
            baseline = healthy.evaluate().report.to_json()
            for operation in ({"type": "evaluate"}, original["operation"]):
                for step in (2.0**-62, 2.0**-64, float.fromhex("0x0.0000000000001p-1022")):
                    with self.subTest(source=name, operation=operation, step=step):
                        value = copy.deepcopy(original)
                        value["operation"] = operation
                        source = value["source"]
                        if name == "multi_asset_bs":
                            source["numerics"]["maximum_step"] = step
                        else:
                            source["maximum_step"] = step
                            if source["equity"]["numerics"].get("maximum_step") is not None:
                                source["equity"]["numerics"]["maximum_step"] = step
                        task = rp.PricingTask.from_json(json.dumps(value))
                        for _ in range(2):
                            with self.assertRaises(rp.PricingError) as caught:
                                task.compile()
                            self.assertIsInstance(caught.exception, RuntimeError)
                            self.assertEqual(caught.exception.stage, "compile")
                            self.assertEqual(caught.exception.code, "resource_limit")
                            self.assertEqual(caught.exception.field, "time_grid")
                            self.assertTrue(caught.exception.causes)
            self.assertEqual(healthy.evaluate().report.to_json(), baseline)

    def test_impossible_time_grids_preserve_single_asset_error_contracts(self):
        # The rough-family guard predates the common allocator check and retains
        # its own classification. Specification also keeps its ValueError API.
        for name, code, field in (
            ("bergomi_hull_white", "resource_limit", "time_grid"),
            ("rough_heston", "invalid_hull_white_input", "rough_time_grid_resource_limit"),
        ):
            original = json.loads((ROOT / f"fixtures/specification-v1/{name}.golden.json").read_text())
            healthy = rp.PricingSpecification.from_json(json.dumps(original)).compile()
            baseline = healthy.evaluate().value
            for step in (2.0**-62, 2.0**-64):
                with self.subTest(source=name, step=step):
                    value = copy.deepcopy(original)
                    value["numerics"]["maximum_step"] = step
                    source = rp.PricingSpecification.from_json(json.dumps(value))
                    with self.assertRaises(rp.ValidationError) as caught:
                        source.compile()
                    self.assertIsInstance(caught.exception, ValueError)
                    self.assertEqual(caught.exception.stage, "compile")
                    self.assertEqual(caught.exception.code, code)
                    self.assertEqual(caught.exception.field, field)
                    self.assertEqual(caught.exception.issues[0].code, "plan_compile_error")
                    self.assertEqual(caught.exception.issues[0].document_kind, "pricing_specification")
                    self.assertEqual(caught.exception.issues[0].schema_version, 2)
                    task = rp.PricingTask.from_specification(source, operation="evaluate")
                    with self.assertRaises(rp.PricingError) as caught:
                        task.compile()
                    self.assertIsInstance(caught.exception, RuntimeError)
                    self.assertEqual(caught.exception.stage, "compile")
                    self.assertEqual(caught.exception.code, code)
                    self.assertEqual(caught.exception.field, field)
                    self.assertTrue(caught.exception.causes)
            self.assertEqual(healthy.evaluate().value, baseline)

    def test_oversized_reduction_tables_return_pricing_errors_across_models(self):
        for name in ("standard_risk", "multi_asset_bs", "stochastic_dividend_bs"):
            original = task_document(name)
            valid = rp.PricingTask.from_json(json.dumps(original)).compile()
            baseline = valid.evaluate().report.to_json()
            operations = [{"type": "evaluate"}]
            if original["operation"] != operations[0]:
                operations.append(original["operation"])
            for operation in operations:
                with self.subTest(source=name, operation=operation):
                    value = copy.deepcopy(original)
                    value["operation"] = operation
                    source = execution_source(value)
                    source["engine"] = {
                        "type": "pseudo_monte_carlo", "master_seed": 7,
                        "independent_sampling_units": 2**64 - 1,
                        "variance_reduction": {"antithetic": False, "brownian_bridge": False},
                    }
                    source["numerics"]["reduction_block_size"] = 1
                    plan = rp.PricingTask.from_json(json.dumps(value)).compile()
                    for _ in range(2):
                        with self.assertRaises(rp.PricingError) as caught:
                            plan.evaluate()
                        self.assertIsInstance(caught.exception, RuntimeError)
                        self.assertEqual(caught.exception.stage, "evaluate")
                        self.assertEqual(caught.exception.code, "resource_limit")
                        self.assertEqual(caught.exception.field, "reduction_blocks")
                    self.assertEqual(valid.evaluate().report.to_json(), baseline)

    def test_maximum_block_size_is_valid_for_a_small_workload(self):
        document = task_document("standard_risk")
        source = execution_source(document)
        source["numerics"]["reduction_block_size"] = 2**64 - 1
        task = rp.PricingTask.from_json(json.dumps(document))
        result = task.compile().evaluate().report
        restored = rp.PricingTask.from_json(task.to_json())
        self.assertEqual(result.to_json(), restored.compile().evaluate().report.to_json())
        # Both configurations have one block and therefore identical arithmetic.
        source["numerics"]["reduction_block_size"] = source["engine"]["independent_sampling_units"]
        expected = rp.PricingTask.from_json(json.dumps(document)).compile().evaluate().report
        self.assertEqual(result.value, expected.value)
        self.assertEqual(result.standard_error, expected.standard_error)
        self.assertEqual([r.values for r in result.risks], [r.values for r in expected.risks])


if __name__ == "__main__":
    unittest.main()
