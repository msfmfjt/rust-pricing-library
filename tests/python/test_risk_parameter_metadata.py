"""Native coordinate metadata stays aligned across every AAD scope."""
import json
from pathlib import Path
import unittest

import rust_pricing as rp
from test_stochastic_dividends import make_request, compile_plan as bs
from test_bergomi_dividends import compile_plan as bergomi
from test_rough_dividends import compile_plan as rough
from test_stochastic_dividend_hull_white import compile_plan as hw

ROOT = Path(__file__).resolve().parents[2]


class RiskParameterMetadataTests(unittest.TestCase):
    def test_dividend_scopes_keep_explicit_units_and_immutable_snapshots(self):
        request = make_request(points=16, dividends=((.5, 4.), (1.4, 8.)))
        cases = [
            (bs(request), ("evaluate_aad", "evaluate_correlation_aad")),
            (bergomi(False, request), ("evaluate_aad", "evaluate_bergomi_aad", "evaluate_correlation_aad")),
            (bergomi(True, request), ("evaluate_aad", "evaluate_bergomi_aad", "evaluate_correlation_aad")),
            (rough(request), ("evaluate_aad", "evaluate_rough_aad", "evaluate_correlation_aad")),
            (hw(request, rate_volatility_times=[0., .8], rate_volatilities=[.04, .05]),
                ("evaluate_aad", "evaluate_hull_white_aad", "evaluate_correlation_aad")),
        ]
        for plan, operations in cases:
            for operation in operations:
                with self.subTest(plan=plan.scheme, operation=operation):
                    risk = getattr(plan, operation)()
                    prefix = 5 + len(risk.cash_times) + len(risk.discount_times) + len(risk.repo_spread_times)
                    kinds = ["spot", "volatility", "model_parameter", "model_parameter", "volatility"]
                    kinds += ["cash_mean"] * len(risk.cash_times)
                    kinds += ["log_discount_factor"] * (len(risk.discount_times) + len(risk.repo_spread_times))
                    units = ["currency_per_spot", "currency_per_unit_volatility", "currency_per_unit_parameter",
                             "currency_per_unit_parameter", "currency_per_unit_volatility"]
                    units += ["currency_per_unit_parameter"] * len(risk.cash_times)
                    units += ["currency_per_log_discount_factor"] * (len(risk.discount_times) + len(risk.repo_spread_times))
                    if operation in ("evaluate_hull_white_aad", "evaluate_correlation_aad") and isinstance(plan, rp.StochasticDividendHullWhitePlan):
                        kinds += ["model_parameter", "volatility", "volatility"]
                        units += ["currency_per_unit_parameter", "currency_per_unit_volatility", "currency_per_unit_volatility"]
                    remainder = len(risk.derivatives) - len(kinds)
                    kinds += ["model_parameter"] * remainder
                    units += ["currency_per_unit_parameter"] * remainder
                    self.assertEqual(len(risk.parameter_labels), len(risk.derivatives))
                    self.assertEqual(risk.parameter_kinds, kinds)
                    self.assertEqual(risk.parameter_units, units)
                    self.assertGreaterEqual(len(kinds), prefix)
                    snapshot = risk.parameter_units
                    snapshot[0] = "changed"
                    self.assertEqual(risk.parameter_units[0], "currency_per_spot")
                    with self.assertRaises(AttributeError):
                        risk.parameter_kinds = []

    def test_hybrid_and_common_reports_copy_native_units(self):
        source = rp.PricingSpecification.from_json((ROOT / "fixtures/specification-v1/bergomi_hull_white.golden.json").read_text())
        tasks = [rp.PricingTask.from_specification(source, operation="aad")]
        tasks += [rp.PricingTask.from_json((ROOT / f"fixtures/task-v1/{name}.golden.json").read_text())
                  for name in ("stochastic_dividend_bs", "stochastic_dividend_hw")]
        for task in tasks:
            result = task.compile().evaluate()
            native = result.native_result
            self.assertEqual(native.parameter_kinds[0], "spot")
            self.assertEqual(native.parameter_units[0], "currency_per_spot")
            self.assertEqual(len(native.parameter_units), len(native.derivatives))
            self.assertEqual([r.name for r in result.report.risks], native.parameter_labels)
            self.assertEqual([r.unit for r in result.report.risks], native.parameter_units)
            self.assertEqual([r.values[0] for r in result.report.risks], native.derivatives)
            self.assertEqual(json.loads(result.report.to_json()), json.loads(rp.ValuationReport.from_json(result.report.to_json()).to_json()))


if __name__ == "__main__":
    unittest.main()
