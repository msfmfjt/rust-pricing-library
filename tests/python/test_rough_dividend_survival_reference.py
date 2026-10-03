"""Independent survival-reference probability, tangent and uncertainty controls."""
import json
from datetime import date
from pathlib import Path
import unittest

import numpy as np

from rough_dividend_barrier_reference import cdf
from rough_dividend_survival_reference import batch_means, hybrid_weights, inverse_cdf, path_values


class SurvivalBarrierReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parents[2] / 'fixtures/stochastic-dividends'
        cls.fixture = json.loads((directory / 'rough-survival-barrier-reference.json').read_text())
        cls.market = json.loads((directory / cls.fixture['market_contract_fixture']).read_text())
        cls.two_step = json.loads((directory / cls.fixture['calibration_fixture']).read_text())

    def test_normal_transport_and_hybrid_cell_covariance(self):
        p = np.array([1e-12, 1e-6, 0.1, 0.5, 0.9, 1 - 1e-12])
        np.testing.assert_allclose(cdf(inverse_cdf(p)), p, rtol=1e-13, atol=1e-16)
        times = np.array([0., 0.125, 0.5, 1.])
        for h in (0.01, 0.1, 0.3, 0.5):
            w, residual, var = hybrid_weights(times, h)
            self.assertAlmostEqual(var[1], times[1] ** (2 * h), delta=1e-14)
            self.assertTrue(np.all(np.triu(w, 0) == 0))
            if h == 0.5:
                np.testing.assert_allclose(var, times, rtol=0, atol=1e-14)
                np.testing.assert_equal(residual, 0.)

    def test_two_step_estimates_against_independent_quadrature(self):
        for case in self.two_step['cases']:
            case = dict(case, times=[0., self.market['fixing_time'], self.market['expiry_time']],
                        monitoring_indices=[1, 2])
            means = batch_means(case, self.market, batches=32, pairs=8192)
            se = means.std(axis=0, ddof=1) / np.sqrt(len(means))
            error = np.abs(means.mean(axis=0) - [case['price'], case['delta']])
            self.assertTrue(np.all(error < 4 * se + 2e-7))
            self.assertLess(se[0], 0.01)
            self.assertLess(se[1], 0.0004)

    def test_multistep_pathwise_tangents_and_no_monitoring(self):
        for case in self.fixture['cases']:
            steps = len(case['times']) - 1
            for index, day in zip(case['monitoring_indices'], case['monitoring_dates']):
                expected = (date.fromisoformat(day) - date(2026, 9, 4)).days / 365
                self.assertAlmostEqual(case['times'][index], expected, delta=1e-15)
            rng = np.random.Generator(np.random.PCG64(701))
            z, u = rng.standard_normal((64, steps, 3)), rng.random((64, steps))
            for spot in (95., 100., 105.):
                analytic = path_values(case, self.market, z, u, spot=spot)[:, 1]
                for bump in (1e-4, 5e-5):
                    fd = (path_values(case, self.market, z, u, spot=spot + bump)[:, 0]
                          - path_values(case, self.market, z, u, spot=spot - bump)[:, 0]) / (2 * bump)
                    np.testing.assert_allclose(analytic, fd, rtol=0, atol=2e-6)
            empty = dict(case, monitoring_indices=[])
            np.testing.assert_equal(path_values(empty, self.market, z, u), 0.)

    def test_more_monitoring_dates_increase_knock_in_price(self):
        cases = {case['id']: case for case in self.fixture['cases']}
        two, four = cases['h01_n8_two'], cases['h01_n8_four']
        self.assertEqual(two['times'], four['times'])
        self.assertEqual(two['squared_leverage'], four['squared_leverage'])
        paired = np.asarray(four['batch_means'])[:, 0] - np.asarray(two['batch_means'])[:, 0]
        se = paired.std(ddof=1) / np.sqrt(len(paired))
        self.assertGreater(paired.mean() - 4 * se, 0.)

    def test_retained_batch_means_and_standard_errors(self):
        for case in self.fixture['cases']:
            means = np.asarray(case['batch_means'])
            self.assertEqual(means.shape, (self.fixture['sampling']['batches'], 2))
            np.testing.assert_allclose(means.mean(axis=0), [case['price'], case['delta']], rtol=0, atol=1e-13)
            np.testing.assert_allclose(means.std(axis=0, ddof=1) / np.sqrt(len(means)),
                                       [case['price_se'], case['delta_se']], rtol=0, atol=1e-13)
            for q in ('price', 'delta'):
                self.assertLess(case[q + '_se'], self.fixture['acceptance']['reference_' + q + '_se'])


if __name__ == '__main__':
    unittest.main()
