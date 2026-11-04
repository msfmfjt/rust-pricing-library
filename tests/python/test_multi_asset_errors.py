"""Typed causes survive both native basket and common-task boundaries."""
import json
import unittest

import rust_pricing as rp

from test_typed_pricing_sources import correlation, engine, fixture, market, model


def graph_product(overflow=False):
    builder = rp.SourceGraphBuilder()
    total = builder.add(builder.spot(1, "2027-01-01"), builder.spot(2, "2027-01-01"))
    amount = builder.multiply(total, builder.literal(1e308 if overflow else 0.5))
    return builder.finish([amount], ["2027-01-01"], currency_id=1, underlying_ids=[1, 2])


def native_plan(document, product):
    source = document["source"]
    return rp.MultiAssetPlan.compile(
        source["valuation_date"], rp.MultiAssetProduct.from_graph(product),
        [market(m) for m in source["markets"]], [model(m) for m in source["models"]],
        correlation(source["correlation"]), engine(source["engine"]), **source["numerics"])


def graph_task(overflow=False):
    document = fixture("multi_asset_bs")
    product = graph_product(overflow)
    document["source"]["product"] = json.loads(product.to_json())["product"]
    return document, product


class MultiAssetErrorTests(unittest.TestCase):
    def assert_failure(self, call, code, stage, field, operation=None, *, error_type=rp.PricingError):
        with self.assertRaises(error_type) as caught:
            call()
        error = caught.exception
        self.assertIsInstance(error, ValueError if error_type is rp.ValidationError else RuntimeError)
        self.assertEqual(error.code, code)
        self.assertEqual(error.stage, stage)
        self.assertEqual(error.field, field)
        self.assertEqual(error.operation, operation)
        self.assertTrue(error.causes)
        self.assertEqual(error.args, (str(error),))
        return error

    def test_rqmc_dimension_failure_keeps_its_resource_classification(self):
        document, product = graph_task()
        document["operation"] = {"type": "evaluate"}
        document["source"]["engine"] = {
            "type": "randomized_quasi_monte_carlo", "points_per_scramble": 2,
            "master_scramble_seed": 7, "scramble_count": 2,
            "variance_reduction": {"antithetic": False, "brownian_bridge": False},
        }
        # Two drivers exceed the Sobol dimension limit before allocating its table.
        document["source"]["numerics"]["maximum_step"] = 1 / 11000
        native = self.assert_failure(lambda: native_plan(document, product),
                                     "resource_limit", "compile", "effective_dimension",
                                     error_type=rp.ValidationError)
        self.assertEqual(native.issues[0].pointer, "/multi_asset")
        self.assertEqual(native.issues[0].code, "invalid_multi_asset_configuration")
        self.assertEqual(native.issues[0].phase, "domain")
        self.assertEqual(native.issues[0].message, str(native))
        task = rp.PricingTask.from_json(json.dumps(document))
        common = self.assert_failure(task.compile, "resource_limit", "compile",
                                     "effective_dimension", "evaluate")
        self.assertEqual(str(native), str(common))
        self.assertEqual(native.causes, common.causes[1:])

    def test_runtime_graph_failures_keep_sampling_context_for_price_and_aad(self):
        document, product = graph_task(overflow=True)
        native = native_plan(document, product)
        for operation in ("evaluate", "aad"):
            with self.subTest(operation=operation):
                document["operation"] = {"type": operation}
                task = rp.PricingTask.from_json(json.dumps(document)).compile()
                call = native.evaluate if operation == "evaluate" else native.evaluate_aad
                direct = self.assert_failure(call, "graph_error", "evaluate", "product")
                common = self.assert_failure(task.evaluate, "graph_error", "evaluate",
                                             "product", operation)
                self.assertEqual(str(direct), str(common))
                self.assertEqual(direct.causes, common.causes[1:])
                self.assertTrue(any("sampling unit 0" in cause for cause in direct.causes))


if __name__ == "__main__":
    unittest.main()
