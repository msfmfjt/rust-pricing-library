"""Stable failure metadata without changing historical exception inheritance."""
import json
from pathlib import Path
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def task_document(name):
    return json.loads((ROOT / f"fixtures/task-v1/{name}.golden.json").read_text())


class StructuredErrorTests(unittest.TestCase):
    def assert_metadata(self, error, code, stage, field=None, operation=None, reason=None):
        self.assertEqual(error.code, code)
        self.assertEqual(error.stage, stage)
        self.assertEqual(error.field, field)
        self.assertEqual(error.operation, operation)
        self.assertEqual(error.reason, reason)
        self.assertIsInstance(error.causes, list)
        self.assertTrue(all(isinstance(cause, str) for cause in error.causes))
        self.assertEqual(str(error), error.args[0])

    def test_parsing_limits_and_schema_errors_keep_validation_issues(self):
        for cls in (rp.PricingRequest, rp.PricingSpecification, rp.PricingTask, rp.ValuationReport, rp.GraphProduct):
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(rp.ValidationError) as caught:
                    cls.from_json("{")
                error = caught.exception
                self.assertIsInstance(error, ValueError)
                self.assert_metadata(error, "invalid_json", "parse")
                self.assertEqual(error.issues[0].code, "invalid_json")
                with self.assertRaises(rp.ValidationError) as caught:
                    cls.from_json("[" * 257 + "]" * 257)
                error = caught.exception
                self.assertEqual((error.code, error.stage), ("resource_limit", "parse"))
                self.assertTrue(error.field)
                self.assertEqual(error.issues[0].phase, "syntax_and_limits")
        document = task_document("standard_risk")
        document["schema_version"] = 99
        with self.assertRaises(rp.ValidationError) as caught:
            rp.PricingTask.from_json(json.dumps(document))
        self.assert_metadata(caught.exception, "unsupported_schema_version", "parse", "/schema_version")
        self.assertEqual(caught.exception.issues[0].schema_version, 99)

    def test_unsupported_operation_retains_the_capability_reason(self):
        basket = rp.PricingTask.from_json(json.dumps(task_document("multi_asset_bs")))
        with self.assertRaises(rp.PricingError) as caught:
            basket.with_operation("aad", gamma_relative_bump=0.)
        self.assertIsInstance(caught.exception, RuntimeError)
        self.assert_metadata(caught.exception, "unsupported_operation", "validation", "operation", "aad", "invalid_gamma_bump")
        with self.assertRaises(ValueError) as caught:
            basket.capability("evaluate", gamma_relative_bump=.01)
        self.assert_metadata(caught.exception, "invalid_operation", "validation", "operation", "evaluate")

        document = task_document("stochastic_dividend_lsv")
        document["operation"] = {"type": "evaluate"}
        document["source"]["equity"]["model"]["equity"]["particles"]["retain_reverse_trace"] = False
        task = rp.PricingTask.from_json(json.dumps(document))
        with self.assertRaises(rp.PricingError) as caught:
            task.with_operation("local_variance_risk")
        self.assert_metadata(caught.exception, "unsupported_operation", "validation", "operation", "local_variance_risk", "reverse_trace_required")

    def test_native_preflight_failure_has_compile_stage(self):
        document = task_document("multi_asset_joint")
        document["operation"] = {"type": "evaluate"}
        document["source"]["driver_correlations"][0][1][2] = -.2
        task = rp.PricingTask.from_json(json.dumps(document))
        self.assertEqual(task.capability("evaluate").status, "conditional")
        with self.assertRaises(rp.PricingError) as caught:
            task.compile()
        self.assert_metadata(caught.exception, "invalid_correlation", "compile", "correlation", operation="evaluate")
        self.assertTrue(caught.exception.causes)

    def test_task_domain_errors_locate_the_nested_source(self):
        cases = [
            ('standard_risk', '/source/specification/numerics/worker_threads', 0,
             '/source/specification/numerics'),
            ('stochastic_dividend_bs', '/source/equity/market/spot', -1.,
             '/source/equity/market/spot'),
            ('multi_asset_bs', '/source/models/1/volatility', -.2,
             '/source/models/1/volatility'),
            ('multi_asset_bs', '/source/markets/1/spot', -1.,
             '/source/markets/1/spot'),
            ('multi_asset_joint', '/source/lsv/0/value/particles/particle_count', 0,
             '/source/lsv/0/value/particles'),
            ('multi_asset_joint', '/source/hull_white/lsv_targets/0/value/log_densities', [],
             '/source/hull_white/lsv_targets/0/value'),
        ]

        def locate(document, keys):
            for key in keys:
                document = document[int(key)] if isinstance(document, list) else document[key]
            return document

        for name, path, bad, expected in cases:
            with self.subTest(source=name, path=path):
                document = task_document(name)
                keys = path.strip('/').split('/')
                parent = locate(document, keys[:-1])
                parent[int(keys[-1]) if isinstance(parent, list) else keys[-1]] = bad
                with self.assertRaises(rp.ValidationError) as caught:
                    rp.PricingTask.from_json(json.dumps(document))
                error = caught.exception
                self.assert_metadata(error, 'invalid_domain_value', 'parse', expected)
                self.assertEqual(error.issues[0].pointer, expected)
                self.assertIsNotNone(locate(document, expected.strip('/').split('/')))

        # The standalone source keeps its own document-relative location.
        source = task_document('standard_risk')['source']['specification']
        source['numerics']['worker_threads'] = 0
        with self.assertRaises(rp.ValidationError) as caught:
            rp.PricingSpecification.from_json(json.dumps(source))
        self.assert_metadata(caught.exception, 'invalid_domain_value', 'parse', '/numerics')

    def test_legacy_compile_exception_and_validation_issue_contract_remain(self):
        source = rp.PricingSpecification.from_json(json.dumps(task_document("standard_risk")["source"]["specification"]))
        # Obtain a legacy request with the same inputs, then reject the execution policy.
        document = task_document("standard_risk")["source"]["specification"]
        document["document_kind"] = "pricing_request"
        document["schema_version"] = 3
        document["model"] = document["model"]["equity"]["model"]
        del document["numerics"]
        request = rp.PricingRequest.from_json(json.dumps(document))
        with self.assertRaises(rp.ValidationError) as caught:
            rp.PricingPlan.compile(request, worker_threads=0)
        self.assert_metadata(caught.exception, "invalid_execution_policy", "compile", "worker_threads")
        self.assertEqual(caught.exception.issues[0].code, "plan_compile_error")
        with self.assertRaises(rp.ValidationError) as caught:
            rp.PricingSpecification.from_request(request, worker_threads=0)
        self.assert_metadata(caught.exception, "invalid_execution_policy", "validation", "worker_threads")
        with self.assertRaises(rp.PricingError) as caught:
            rp.PricingTask.from_specification(source, operation="aad")
        self.assert_metadata(caught.exception, "unsupported_operation", "validation", "operation", "aad", "dedicated_risk_api")

    def test_numerical_evaluation_failure_is_shared_across_entry_points(self):
        builder = rp.SourceGraphBuilder()
        amount = builder.multiply(builder.spot(1, "2027-09-04"), builder.literal(1e308))
        graph = builder.finish([amount], ["2027-09-04"], currency_id=2, underlying_ids=[1])
        document = task_document("standard_risk")["source"]["specification"]
        document["risk"] = {"delta": False, "vega": False, "smile_dynamics": {"type": "sticky_log_moneyness"}}
        document["product"] = {"type": "graph", "graph": json.loads(graph.to_json())["product"]}
        source = rp.PricingSpecification.from_json(json.dumps(document))
        task = rp.PricingTask.from_specification(source)
        messages = []
        for plan, operation in [(source.compile(), None), (task.compile(), "evaluate")]:
            with self.subTest(operation=operation), self.assertRaises(rp.PricingError) as caught:
                plan.evaluate()
            self.assert_metadata(caught.exception, "graph_error", "evaluate", "product", operation)
            self.assertTrue(caught.exception.causes)
            messages.append(str(caught.exception))
        self.assertEqual(messages[0], messages[1])
        with self.assertRaises(rp.PricingError) as caught:
            rp.PricingRequest("2026-09-04", rp.Product.from_graph(graph),
                rp.Market.equity(2, 1, 100., rp.DiscountCurve(10, [0., 1.], [1., .95]),
                    rp.DiscountCurve(11, [0., 1.], [1., .98])),
                rp.Model.black_scholes(.2), rp.Engine.pseudo_monte_carlo(7, 4), rp.RiskRequest()).to_json()
        self.assertEqual((caught.exception.code, caught.exception.stage), ("unsupported_schema_feature", "serialize"))


if __name__ == "__main__":
    unittest.main()
