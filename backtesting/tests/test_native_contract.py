"""Startup failures and scope isolation without an installed native wheel."""

import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rust_pricing_backtest import BacktestError, RustPricingProvider
from rust_pricing_backtest import native_contract


CONTROL = dict(numerical_scope='multi_step_extension_gamma', variance_extension='all_rows',
               sampling_policy='rqmc_quote_control_512x24')


class Request:
    @staticmethod
    def from_json(json):
        raise AssertionError('startup must not parse a request')


class Plan:
    @staticmethod
    def compile(request, *, worker_threads, reduction_block_size=None):
        raise AssertionError('startup must not compile')

    def evaluate(self):
        raise AssertionError('startup must not price')

    def evaluate_vega_kt_multi_step_gamma_chain_audit(self, rows, boundaries, *,
            variance_risk_directions, subdivisions=64, relative_tolerance=1e-6,
            gamma_relative_standard_error_limit=.25, retain_samples=False):
        raise AssertionError('startup must not estimate Gamma')


class Result:
    value = delta_raw = gamma_raw = diagnostics = property(lambda self: None)


class Diagnostics:
    gamma_method = gamma_spot_bump = property(lambda self: None)


def control_native():
    # Independent minimal dedicated-path fixture: deliberately no common tasks,
    # reports, DiscountCurve or VegaKT allocation API. Methods must never run.
    return SimpleNamespace(__version__='0.1.0', __file__=__file__, PricingError=ValueError,
                           PricingRequest=Request, PricingPlan=Plan, PricingResult=Result,
                           Diagnostics=Diagnostics)


class NativeContractTests(unittest.TestCase):
    def test_quote_control_needs_no_unused_common_task_api_or_pricing_at_startup(self):
        native = control_native()
        with patch.dict(sys.modules, rust_pricing=native):
            provider = RustPricingProvider(**CONTROL)
        self.assertIs(provider.rp, native)
        self.assertFalse(provider._cache.results)
        self.assertFalse(provider._cache.native)

    def test_same_version_without_tasks_is_rejected_before_identity_or_cache(self):
        provider = RustPricingProvider.__new__(RustPricingProvider)
        with patch.dict(sys.modules, rust_pricing=control_native()), \
                patch('rust_pricing_backtest.rust_adapter.hashlib.sha256', side_effect=AssertionError('too late')):
            with self.assertRaises(BacktestError) as caught:
                provider.__init__(diagnostic_only=True)
        self.assertEqual(caught.exception.code, 'unsupported_native_version')
        for text in ('0.1.0', 'PricingTask', 'DiscountCurve', 'same source revision'):
            self.assertIn(text, caught.exception.message)
        self.assertFalse(hasattr(provider, '_configuration'))
        self.assertFalse(hasattr(provider, '_cache'))

    def test_old_paired_sample_signature_is_rejected(self):
        class OldPlan(Plan):
            def evaluate_vega_kt_multi_step_gamma_chain_audit(self, rows, boundaries, *,
                    variance_risk_directions, subdivisions=64, relative_tolerance=1e-6,
                    gamma_relative_standard_error_limit=.25):
                raise AssertionError('must not run')
        native = control_native(); native.PricingPlan = OldPlan
        with patch.dict(sys.modules, rust_pricing=native):
            with self.assertRaisesRegex(BacktestError, 'PricingPlan.*retain_samples'):
                RustPricingProvider(**CONTROL)

    def test_missing_callable_result_member_and_exception_are_reported_together(self):
        class BrokenPlan(Plan):
            evaluate = None
        native = control_native()
        native.PricingPlan, native.PricingResult, native.PricingError = BrokenPlan, type('Empty', (), {}), str
        with patch.dict(sys.modules, rust_pricing=native):
            with self.assertRaises(BacktestError) as caught:
                RustPricingProvider(**CONTROL)
        for text in ('PricingError', 'PricingPlan.evaluate', 'PricingResult.delta_raw'):
            self.assertIn(text, caught.exception.message)

    def test_compatible_optional_argument_and_version_change_are_accepted(self):
        class NewPlan(Plan):
            @staticmethod
            def compile(request, *, worker_threads, reduction_block_size=None, optional_future_setting=None):
                raise AssertionError('must not run')
        native = control_native(); native.PricingPlan = NewPlan; native.__version__ = '0.1.1'
        with patch.dict(sys.modules, rust_pricing=native):
            RustPricingProvider(**CONTROL)

    def test_missing_binary_provenance_is_a_clear_startup_failure(self):
        native = control_native(); del native.__file__
        with patch.dict(sys.modules, rust_pricing=native):
            with self.assertRaisesRegex(BacktestError, 'unsupported_native_version.*binary provenance'):
                RustPricingProvider(**CONTROL)

    def test_import_and_loader_errors_are_distinct_from_programming_errors(self):
        for error in (ModuleNotFoundError('rust_pricing'), ImportError('incompatible ABI'), OSError('missing shared library')):
            with self.subTest(error=error), patch.object(native_contract, 'import_module', side_effect=error):
                with self.assertRaisesRegex(BacktestError, 'native_import_failure.*Python ABI') as caught:
                    RustPricingProvider(diagnostic_only=True)
                self.assertIs(caught.exception.__cause__, error)
        with patch.object(native_contract, 'import_module', side_effect=RuntimeError('unexpected bug')):
            with self.assertRaisesRegex(RuntimeError, 'unexpected bug'):
                RustPricingProvider(diagnostic_only=True)

    def test_invalid_configuration_is_rejected_before_import(self):
        with patch.object(native_contract, 'import_module', side_effect=AssertionError('must not import')):
            with self.assertRaisesRegex(BacktestError, 'rqmc_provider_policy'):
                RustPricingProvider(numerical_scope='multi_step_gamma', sampling_policy='rqmc_quote_control_512x24')
            with self.assertRaisesRegex(BacktestError, 'invalid_provider'):
                RustPricingProvider(diagnostic_only=True, worker_threads=0)

    def test_companion_import_keeps_native_dependency_optional(self):
        code = '''import sys
sys.modules['rust_pricing'] = None
from rust_pricing_backtest import RustPricingProvider, BacktestError
try:
    RustPricingProvider(diagnostic_only=True)
except BacktestError as error:
    assert error.code == 'native_import_failure', error
else:
    raise AssertionError('missing dependency accepted')
'''
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
