"""Provider startup contract against the installed native wheel."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backtesting'))
from rust_pricing_backtest import BacktestError, RustPricingProvider


class NativeProviderContractTests(unittest.TestCase):
    def test_installed_wheel_supports_every_provider_path(self):
        modes = [dict(diagnostic_only=True), dict(numerical_scope='p0_terminal_replication'),
                 dict(numerical_scope='multi_step_gamma'), dict(numerical_scope='multi_step_path_gamma')]
        for scope in ('multi_step_extension_gamma', 'multi_step_extension_put_gamma'):
            for extension in ('all_rows', 'linear_time'):
                modes.append(dict(numerical_scope=scope, variance_extension=extension))
        for basis in ('row_hats', 'weighted_time_hats', 'coupled_row_hats'):
            modes.append(dict(numerical_scope='multi_step_custom_basis_gamma', variance_basis_policy=basis))
        for sampling in ('rqmc_256x16', 'rqmc_path_16_scrambles', 'rqmc_quote_control_512x24'):
            modes.append(dict(numerical_scope='multi_step_extension_gamma', variance_extension='all_rows',
                              sampling_policy=sampling))
        for mode in modes:
            with self.subTest(mode=mode):
                provider = RustPricingProvider(**mode)
                self.assertIs(provider.rp, rp)
                self.assertFalse(provider._cache.results)

    def test_same_version_missing_task_is_rejected_on_each_construction(self):
        native = SimpleNamespace(**vars(rp))
        with patch.dict(sys.modules, rust_pricing=native):
            RustPricingProvider(diagnostic_only=True)
            del native.PricingTask
            with self.assertRaisesRegex(BacktestError, 'unsupported_native_version.*PricingTask'):
                RustPricingProvider(diagnostic_only=True)
        self.assertEqual(native.__version__, rp.__version__)

    def test_old_specification_execution_signature_fails_before_pricing(self):
        class OldSpecification:
            @staticmethod
            def from_request(request, *, worker_threads):
                raise AssertionError('must not run')
        native = SimpleNamespace(**vars(rp)); native.PricingSpecification = OldSpecification
        with patch.dict(sys.modules, rust_pricing=native):
            with self.assertRaisesRegex(BacktestError, 'PricingSpecification.from_request.*reduction_block_size'):
                RustPricingProvider(diagnostic_only=True)

    def test_old_discount_curve_and_report_shapes_fail_at_startup(self):
        for name, replacement, missing in (
                ('DiscountCurve', type('OldCurve', (), {}), 'DiscountCurve.discount'),
                ('ValuationReport', type('OldReport', (), {}), 'ValuationReport.uncertainty_scope'),
                ('ReportedRisk', type('OldRisk', (), {}), 'ReportedRisk.calibration'),
                ('VegaKtProjection', type('OldProjection', (), {}), 'VegaKtProjection.reporting_stats')):
            with self.subTest(name=name):
                native = SimpleNamespace(**vars(rp)); setattr(native, name, replacement)
                with patch.dict(sys.modules, rust_pricing=native):
                    with self.assertRaisesRegex(BacktestError, missing):
                        RustPricingProvider(diagnostic_only=True)

    def test_ordinary_provider_does_not_require_multi_step_gamma_api(self):
        native = SimpleNamespace(**vars(rp)); del native.PricingPlan
        with patch.dict(sys.modules, rust_pricing=native):
            RustPricingProvider(diagnostic_only=True)
            with self.assertRaisesRegex(BacktestError, 'PricingPlan'):
                RustPricingProvider(numerical_scope='multi_step_gamma')

    def test_joint_scopes_require_paired_samples_but_basic_multi_step_does_not(self):
        class OldPlan:
            compile = staticmethod(rp.PricingPlan.compile)
            evaluate = rp.PricingPlan.evaluate

            def evaluate_vega_kt_multi_step_gamma_chain_audit(self, rows, boundaries, *,
                    variance_risk_directions, subdivisions=64, relative_tolerance=1e-6,
                    gamma_relative_standard_error_limit=.25):
                raise AssertionError('must not run')
        native = SimpleNamespace(**vars(rp)); native.PricingPlan = OldPlan
        with patch.dict(sys.modules, rust_pricing=native):
            RustPricingProvider(numerical_scope='multi_step_gamma')
            for mode in (dict(numerical_scope='multi_step_extension_gamma', variance_extension='all_rows'),
                         dict(numerical_scope='multi_step_custom_basis_gamma', variance_basis_policy='row_hats')):
                with self.subTest(mode=mode), self.assertRaisesRegex(BacktestError, 'retain_samples'):
                    RustPricingProvider(**mode)


if __name__ == '__main__':
    unittest.main()
