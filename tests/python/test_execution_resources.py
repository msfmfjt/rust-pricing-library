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
                        if name != "multi_asset_bs":
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
