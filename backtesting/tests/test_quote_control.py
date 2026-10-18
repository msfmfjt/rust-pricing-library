"""Contract exclusion, covariance and fail-closed control boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from multi_step_cases import model
from rust_pricing_backtest import BacktestError, RustPricingProvider
from rust_pricing_backtest.domain import fingerprint
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.multi_step_provider import check_sampling, validation
from rust_pricing_backtest.portfolio_market_risk import sampling_covariance, evaluate_portfolio
from rust_pricing_backtest.quote_control import POLICY, CONTROL_ID, virtual_request, adjust_report


class QuoteControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = model(6)
        c = MultiStepExtensionCalibrator(cls.base, extension='all_rows')
        cls.tangent = c.all_maturity_iv_tangent()[0]; cls.q = c.iv.size

    def fixture(self):
        ids = [CONTROL_ID, 'lower', 'upper']
        requests = {cid: deepcopy(self.base) for cid in ids}
        for cid, strike in zip(ids, (100., 95., 105.)): requests[cid]['product']['strike'] = strike
        unit = np.array([-3., -1., 1., 3.]); load = np.arange(1, self.q+1)*.1
        cube = np.zeros((4, 3, self.q+1)); cube[:, :, 0] = 10.+unit[:, None]
        cube[:, :, 1:] = unit[:, None, None]*load[None, None, :]+np.arange(3)[None, :, None]
        r = dict(status='passed', implementation_identity='synthetic', instrument_order=ids,
            request_hashes={cid:fingerprint(p) for cid,p in requests.items()}, extension='all_rows', variance_basis=None,
            instrument_engines={cid:deepcopy(p['engine']) for cid,p in requests.items()},
            precision_policy=dict(subdivisions=512), independent_sampling_units=4,
            calibration_tangent=self.tangent.tolist(), sampling_unit_values=cube.tolist(),
            price_and_market_sample_covariance=sampling_covariance(cube.reshape(4, -1)).tolist(),
            audits={cid:dict(status='path_gamma_chain_passed', recovery_passed=True, validation_passed=True) for cid in ids},
            instruments={cid:dict(price=10., buckets=cube[:, i, 1:].mean(axis=0).tolist()) for i,cid in enumerate(ids)},
            quantities={CONTROL_ID:0., 'lower':-1., 'upper':-1.})
        r['report_hash'] = fingerprint(r)
        return r, requests, cube

    def setUp(self):
        p = patch('rust_pricing_backtest.portfolio_market_risk.implementation_identity', return_value='synthetic')
        p.start(); self.addCleanup(p.stop)

    def test_virtual_contract_is_excluded_and_signed_covariance_uses_paired_samples(self):
        native, requests, raw = self.fixture(); before = deepcopy(native)
        report = adjust_report(native, requests); cube = np.array(report['sampling_unit_values'])
        self.assertEqual(report['instrument_order'], ['lower', 'upper'])
        for field in ('audits', 'instruments', 'request_hashes', 'instrument_engines', 'quantities'):
            self.assertNotIn(CONTROL_ID, report[field])
        self.assertEqual(native, before)
        np.testing.assert_array_equal(cube[:, :, 0], raw[:, 1:, 0])
        self.assertLess(max(report['net_bucket_standard_errors']), 1e-13)
        self.assertEqual(report['audits']['lower'], native['audits']['lower'])
        net = -cube.sum(axis=1)
        np.testing.assert_allclose(report['net_price_and_market_mean_covariance'], sampling_covariance(net)/4, atol=1e-13)

    def test_missing_cross_covariance_cannot_claim_precision(self):
        native, requests, _ = self.fixture(); cov = np.array(native['price_and_market_sample_covariance'])
        width = self.q+1
        cov[:width, width:] = 0; cov[width:, :width] = 0
        native['price_and_market_sample_covariance'] = cov.tolist()
        native.pop('report_hash'); native['report_hash'] = fingerprint(native)
        with self.assertRaisesRegex(BacktestError, 'quote_control_covariance'): adjust_report(native, requests)

    def test_failed_control_and_failed_target_both_block_release(self):
        for cid in (CONTROL_ID, 'lower'):
            native, requests, _ = self.fixture(); native['audits'][cid]['recovery_passed'] = False
            native.pop('report_hash'); native['report_hash'] = fingerprint(native)
            with self.assertRaisesRegex(BacktestError, 'quote_control_native'): adjust_report(native, requests)

    def test_cached_rejected_native_audit_still_blocks_release(self):
        key = fingerprint(dict(request=self.base, tangent=self.tangent.tolist(), subdivisions=512,
            relative_tolerance=1e-6, gamma_relative_standard_error_limit=.25, workers=1))
        cache = {key: dict(audit=dict(status='validation_failed', estimate=None,
                                     recovery_passed=True, validation_passed=False), risk=None)}
        native = SimpleNamespace(PricingPlan=Mock(), PricingRequest=Mock())
        with patch.dict(sys.modules, {'rust_pricing': native}):
            with self.assertRaisesRegex(BacktestError, 'portfolio_gamma_failure'):
                evaluate_portfolio({'target':self.base}, {'target':-1.}, _native_cache=cache)
            native.PricingPlan.compile.assert_not_called()
            native.PricingRequest.from_json.assert_not_called()

    def test_control_rolls_with_current_spot_and_is_detached(self):
        before = deepcopy(self.base); moved = deepcopy(self.base); moved['market']['spot'] = 103.
        self.assertEqual(virtual_request(moved)['product']['strike'], 103.)
        self.assertEqual(virtual_request(self.base)['product']['strike'], 100.)
        self.assertEqual(before, self.base)
        for change in ('put', 'rate', 'no_atm'):
            p = deepcopy(self.base)
            if change == 'put': p['product']['side'] = {'type':'put'}
            elif change == 'rate': p['market']['discount_curve']['discount_factors'][-1] = .99
            else: p['model']['reporting_iv_basis']['log_forward_moneyness_nodes'] = [-.2, -.1, .001, .1, .2]
            with self.assertRaisesRegex(BacktestError, 'quote_control_scope'): virtual_request(p)

    def test_exact_policy_and_declared_scope(self):
        engine = dict(type='randomized_quasi_monte_carlo', points_per_scramble=512, scramble_count=24,
                      variance_reduction=dict(antithetic=True, brownian_bridge=False))
        check_sampling(engine, 6, True, POLICY, 'european_vanilla')
        for count in (12, 24., True):
            with self.assertRaises(BacktestError): check_sampling(engine|{'scramble_count':count}, 6, True, POLICY, 'european_vanilla')
        for intervals, kind in ((4, 'european_vanilla'), (6, 'arithmetic_asian')):
            with self.assertRaisesRegex(BacktestError, 'quote_control_scope'): check_sampling(engine, intervals, True, POLICY, kind)
        for scope, extension in (('multi_step_extension_put_gamma', 'all_rows'), ('multi_step_extension_gamma', 'linear_time')):
            with self.assertRaisesRegex(BacktestError, 'rqmc_provider_policy'):
                RustPricingProvider(numerical_scope=scope, variance_extension=extension, sampling_policy=POLICY)
        metadata = validation('multi_step_extension_gamma', 'all_rows', None, POLICY)
        self.assertEqual(metadata['products'], ['european_call'])
        self.assertEqual(metadata['intervals'], [6])
        self.assertEqual(metadata['scramble_count'], 24)


if __name__ == '__main__': unittest.main()
