import unittest
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
from rust_pricing_backtest.domain import BacktestError
from rust_pricing_backtest.named_variance_basis import named_variance_basis, POLICIES
from rust_pricing_backtest.multi_step_provider import validation, implementation_identity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from custom_basis_cases import basis_for
from multi_step_cases import model


class NamedVarianceBasisTests(unittest.TestCase):
    def test_named_constructions_equal_independently_accepted_definitions(self):
        for n in (4, 6):
            for policy in ('row_hats', 'weighted_time_hats'):
                p = model(n); before = deepcopy(p)
                b, audit = named_variance_basis(p, policy)
                ref, expected = basis_for(p, policy)
                np.testing.assert_array_equal(b, ref)
                self.assertEqual(audit, expected)
                self.assertEqual(p, before)
                with self.assertRaises(ValueError): b[0, 0] = 0.

    def test_policy_and_shape_are_explicit(self):
        for policy in (None, 'custom', 'all_rows', 'linear_time', 'weighted'):
            with self.assertRaisesRegex(BacktestError, 'custom_basis_provider_policy'):
                named_variance_basis(model(4), policy)
        p = model(4); p['model']['local_variance_grid']['log_forward_moneyness_nodes'] = [0., 1.]
        with self.assertRaisesRegex(BacktestError, 'custom_basis_provider_scope'):
            named_variance_basis(p, 'row_hats')
        p = model(4); p['model']['reporting_iv_basis']['maturity_nodes'][0] += .01
        with self.assertRaisesRegex(BacktestError, 'custom_basis_provider_scope'):
            named_variance_basis(p, 'weighted_time_hats')

    def test_metadata_declares_limited_acceptance_and_implementation(self):
        for policy in POLICIES:
            a = validation('multi_step_custom_basis_gamma', basis_policy=policy)
            self.assertFalse(a['diagnostic_only'])
            self.assertEqual(a['variance_basis_policy'], policy)
            self.assertEqual(a['arbitrary_variance_basis'], 'not_accepted')
            self.assertEqual(a['full_market_iv_bucket_acceptance'], 'not_established')
            self.assertNotEqual(implementation_identity(basis_policy=policy), implementation_identity('all_rows'))


if __name__ == '__main__': unittest.main()
