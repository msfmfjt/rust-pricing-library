"""Fail-closed report conversion without requiring the native extension."""

from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from rust_pricing_backtest import BacktestError, Contract
from rust_pricing_backtest.task_adapter import evaluate_request, ordinary_greeks


class TaskAdapterTests(unittest.TestCase):
    def setUp(self):
        self.contract = Contract('target', date(2027, 1, 1), 100.)
        self.delta = SimpleNamespace(name='delta.raw', underlying_ids=[1], axes=[], values=[.5],
                                     unit='currency_per_spot', calibration='not_applicable', method='pathwise')
        self.gamma = SimpleNamespace(name='gamma.raw', underlying_ids=[1], axes=[], values=[.03],
                                     unit='currency_per_spot_squared', calibration='not_applicable', method='central_bump')
        self.report = SimpleNamespace(currency_id=2, uncertainty_scope='pricing_sampling', risks=[self.delta, self.gamma])

    def test_selects_raw_scalars_and_preserves_optional_gamma(self):
        self.report.risks.insert(0, SimpleNamespace(name='delta.market_scaled', values=[50.]))
        self.assertEqual(ordinary_greeks(self.report, self.contract), (self.delta, self.gamma))
        self.report.risks.remove(self.gamma)
        self.assertEqual(ordinary_greeks(self.report, self.contract), (self.delta, None))

    def test_missing_raw_delta_cannot_use_market_scaled_delta(self):
        self.delta.name = 'delta.market_scaled'
        with self.assertRaisesRegex(BacktestError, 'missing_risk'):
            ordinary_greeks(self.report, self.contract)

    def test_duplicate_and_non_scalar_blocks_are_rejected(self):
        for block in (self.delta, self.gamma):
            with self.subTest(block=block.name, duplicate=True):
                self.report.risks.append(block)
                with self.assertRaisesRegex(BacktestError, 'task_risk_shape'):
                    ordinary_greeks(self.report, self.contract)
                self.report.risks.pop()
            for field, bad in (('underlying_ids', [9]), ('underlying_ids', [1, 2]),
                               ('axes', ['maturity']), ('values', []), ('values', [1., 2.])):
                with self.subTest(block=block.name, field=field, bad=bad):
                    before = getattr(block, field)
                    setattr(block, field, bad)
                    with self.assertRaisesRegex(BacktestError, 'task_risk_shape'):
                        ordinary_greeks(self.report, self.contract)
                    setattr(block, field, before)

    def test_units_are_not_rescaled_or_inferred(self):
        for block in (self.delta, self.gamma):
            with self.subTest(block=block.name):
                before = block.unit
                block.unit = 'currency_per_percent_spot'
                with self.assertRaisesRegex(BacktestError, 'unit_mismatch'):
                    ordinary_greeks(self.report, self.contract)
                block.unit = before

    def test_currency_and_uncertainty_cannot_change_ledger_meaning(self):
        self.report.currency_id = 9
        with self.assertRaisesRegex(BacktestError, 'currency_mismatch'):
            ordinary_greeks(self.report, self.contract)
        self.report.currency_id = 2
        self.report.uncertainty_scope = 'conditional_on_calibration'
        with self.assertRaisesRegex(BacktestError, 'task_risk_convention'):
            ordinary_greeks(self.report, self.contract)

    def test_calibrated_or_exercise_policy_risks_are_not_fixed_grid_greeks(self):
        for calibration in ('fixed_exercise_policy', 'recalibrated'):
            for block in (self.delta, self.gamma):
                with self.subTest(calibration=calibration, block=block.name):
                    block.calibration = calibration
                    with self.assertRaisesRegex(BacktestError, 'task_risk_convention'):
                        ordinary_greeks(self.report, self.contract)
                    block.calibration = 'not_applicable'

    def test_other_native_result_families_are_rejected(self):
        class Specification:
            pass
        for source in (object(), Specification()):
            with self.subTest(source=type(source).__name__):
                if isinstance(source, Specification):
                    source.native_result = object()
                task = Mock()
                task.compile.return_value.evaluate.return_value = SimpleNamespace(native_result=source)
                rp = SimpleNamespace(PricingSpecification=Mock(), PricingTask=Mock(),
                                     SpecificationResult=Specification, PricingResult=str)
                rp.PricingTask.from_specification.return_value = task
                with self.assertRaisesRegex(BacktestError, 'unsupported_task_result'):
                    evaluate_request(rp, object(), worker_threads=2, reduction_block_size=256)


if __name__ == '__main__':
    unittest.main()
