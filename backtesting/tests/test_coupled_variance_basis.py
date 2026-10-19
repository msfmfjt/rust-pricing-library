import json
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from rust_pricing_backtest.domain import BacktestError, fingerprint
from rust_pricing_backtest.named_variance_basis import named_variance_basis, COUPLED_BASIS_HASHES
from rust_pricing_backtest.variance_basis import _portable_fma
from test_named_variance_basis import model

ROOT = Path(__file__).resolve().parents[2]


class CoupledVarianceBasisTests(unittest.TestCase):
    def test_python_312_fused_fallback_preserves_every_accepted_matrix_bit(self):
        with patch('rust_pricing_backtest.named_variance_basis._fma', _portable_fma), \
             patch('rust_pricing_backtest.variance_basis._fma', _portable_fma):
            for n in (4, 6):
                basis, _ = named_variance_basis(model(n), 'coupled_row_hats')
                doc = json.loads((ROOT/f'fixtures/backtesting/raw-coupled-variance-basis-{n}-v0.1.json').read_text())
                np.testing.assert_array_equal(basis, doc['values'])
                self.assertEqual(fingerprint(basis.tolist()), COUPLED_BASIS_HASHES[n])

    def test_both_matrices_equal_independent_accepted_inputs(self):
        for n in (4, 6):
            payload = model(n); before = deepcopy(payload)
            basis, audit = named_variance_basis(payload, 'coupled_row_hats')
            doc = json.loads((ROOT/f'fixtures/backtesting/raw-coupled-variance-basis-{n}-v0.1.json').read_text())
            np.testing.assert_array_equal(basis, doc['values'])
            self.assertEqual(audit['basis_hash'], COUPLED_BASIS_HASHES[n])
            self.assertEqual(audit['accepted_matrix_hash'], fingerprint(doc['values']))
            self.assertEqual(audit['parameter_metric'], doc['parameter_metric'])
            self.assertLess(basis.min(), 0.)
            self.assertEqual(payload, before)
            with self.assertRaises(ValueError): basis[0, 0] = 9.
            audit['coupling_coefficients']['adjacent_node'] = 99.
            self.assertEqual(named_variance_basis(payload, 'coupled_row_hats')[1]['coupling_coefficients']['adjacent_node'], -.12)

    def test_changed_matrix_is_rejected(self):
        p = model(4)
        p['model']['local_variance_grid']['log_forward_moneyness_nodes'][1] += .001
        with self.assertRaisesRegex(BacktestError, 'custom_basis_provider_scope'):
            named_variance_basis(p, 'coupled_row_hats')
        with patch.dict(COUPLED_BASIS_HASHES, {4: 'different'}):
            with self.assertRaisesRegex(BacktestError, 'custom_basis_provider_scope'):
                named_variance_basis(model(4), 'coupled_row_hats')

    def test_matrix_is_unchanged_when_dates_roll(self):
        p = model(4)
        for times in ([0., .1, .2, .3, .4], [0., .05, .2, .21, .4]):
            p['model']['local_variance_grid']['time_nodes'] = times
            p['model']['reporting_iv_basis']['maturity_nodes'] = times[2::2]
            basis, audit = named_variance_basis(p, 'coupled_row_hats')
            self.assertEqual(fingerprint(basis.tolist()), COUPLED_BASIS_HASHES[4])
            self.assertEqual(audit['time_knots'], ((np.array(times[:-1])+times[1:])/2).tolist())

    def test_old_bases_keep_distinct_parameter_metrics(self):
        for n in (4, 6):
            p = model(n); coupled, audit = named_variance_basis(p, 'coupled_row_hats')
            row, old = named_variance_basis(p, 'row_hats')
            self.assertEqual(audit['uncoupled_basis_hash'], old['basis_hash'])
            self.assertFalse(np.array_equal(coupled @ coupled.T, row @ row.T))
            self.assertNotEqual(audit['basis_hash'], old['basis_hash'])


if __name__ == '__main__': unittest.main()
