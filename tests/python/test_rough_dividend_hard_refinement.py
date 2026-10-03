"""Controls for fixed-surface hard-Barrier refinement and its Gaussian coupling."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from rough_dividend_hard_refinement import (coarsen, coupled_batch_means, cross_kernel,
    deterministic_normals, primal_checkpoints, refine_case)
from rough_dividend_survival_reference import path_values


class HardBarrierRefinement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parents[2] / 'fixtures/stochastic-dividends'
        cls.paths = json.loads((directory / 'rough-hard-barrier-paths.json').read_text())
        cls.source = json.loads((directory / 'rough-survival-barrier-reference.json').read_text())
        cls.market = json.loads((directory / cls.source['market_contract_fixture']).read_text())
        cls.two_step = json.loads((directory / cls.source['calibration_fixture']).read_text())

    def test_kernel_quadrature_against_retained_independent_constants(self):
        constants = {0.1: [0.2972848796035765, 0.20535642686670175, 0.14995909763209497],
                     0.3: [0.7004045779860347, 0.5861688788927809, 0.5020851560707129]}
        for h, values in constants.items():
            for offset, expected in zip((1, 3, 7), values):
                for order in (128, 256):
                    self.assertAlmostEqual(cross_kernel(h, offset, order), expected, delta=3e-12)

    def test_coarse_gaussian_law_and_cross_covariances(self):
        rho_dv = 0.15
        loading = np.array([rho_dv, math.sqrt(1 - rho_dv ** 2)])
        for h in (0.01, 0.1, 0.3, 0.49, 0.5):
            for block in (1, 2, 4, 8):
                dim = 4 * block + 1
                basis = np.eye(dim)
                fine = basis[:, :-1].reshape(dim, block, 4)
                mapping = coarsen(fine, basis[:, -1:], h, block, rho_dv)[:, 0].T
                np.testing.assert_allclose(mapping @ mapping.T, np.eye(4), rtol=0, atol=3e-11)
                p = h + 0.5
                a, r = math.sqrt(2 * h) / p, (0.5 - h) / p
                coarse_near = a * block ** h * (loading @ mapping[:2]) + r * block ** h * mapping[2]
                self.assertAlmostEqual(coarse_near @ coarse_near, block ** (2 * h), delta=3e-11)
                for j in range(block):
                    distance = block - j
                    dw_cov = a * (distance ** p - (distance - 1) ** p)
                    np.testing.assert_allclose(coarse_near[4*j:4*j+2], dw_cov * loading, rtol=0, atol=3e-11)
                    near = np.zeros(dim)
                    near[4*j:4*j+2], near[4*j+2] = a * loading, r
                    self.assertAlmostEqual(coarse_near @ near, cross_kernel(h, distance - 1), delta=3e-11)

    def test_surface_knots_observations_and_fine_grid_tangents(self):
        for base in self.source['cases'][2:]:
            original = np.asarray(base['squared_leverage']).reshape(9, 3)
            for steps in (16, 32, 64, 128):
                case = refine_case(base, steps)
                ratio = steps // 8
                np.testing.assert_equal(np.asarray(case['times'])[::ratio], base['times'])
                rows = np.concatenate((np.repeat(original[:-1], ratio, axis=0), original[-1:]))
                np.testing.assert_equal(np.asarray(case['squared_leverage']).reshape(-1, 3), rows)
                self.assertEqual([case['times'][i] for i in case['monitoring_indices']],
                                 [base['times'][i] for i in base['monitoring_indices']])
            rng = np.random.Generator(np.random.PCG64(910))
            z, u = rng.standard_normal((32, 128, 3)), rng.random((32, 128))
            analytic = path_values(case, self.market, z, u)[:, 1]
            bump = 5e-5
            fd = (path_values(case, self.market, z, u, spot=100+bump)[:, 0]
                  - path_values(case, self.market, z, u, spot=100-bump)[:, 0]) / (2*bump)
            np.testing.assert_allclose(analytic, fd, rtol=0, atol=2e-6)

    def test_nonuniform_coupling_is_rejected(self):
        base = dict(self.source['cases'][2])
        base['times'] = list(base['times'])
        base['times'][1] *= 0.9
        with self.assertRaisesRegex(ValueError, 'uniform'):
            coupled_batch_means(base, self.market, batches=2, pairs=2)

    def test_retained_primal_checkpoints(self):
        bases = {case['id']: case for case in self.source['cases']}
        for record in self.paths['records']:
            case = refine_case(bases[record['case']], record['steps'])
            result = primal_checkpoints(case, self.market, deterministic_normals(record['steps'], record['pattern']))
            np.testing.assert_allclose(result, record['checkpoints'], rtol=0, atol=2e-12)

    def test_refined_exact_limit_against_bivariate_reference(self):
        base = dict(self.two_step['cases'][0], eta=0., kappa=0.,
                    times=[0., self.market['fixing_time'], self.market['expiry_time']],
                    squared_leverage=[0.04]*9, monitoring_indices=[1, 2])
        means = coupled_batch_means(base, self.market, levels=(2, 4, 8), batches=32, pairs=512)
        se = means.std(axis=0, ddof=1) / np.sqrt(len(means))
        error = np.abs(means.mean(axis=0) - [self.market['price'], self.market['delta']])
        self.assertTrue(np.all(error < 4 * se + 2e-8))


if __name__ == '__main__':
    unittest.main()
