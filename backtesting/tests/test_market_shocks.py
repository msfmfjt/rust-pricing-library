"""Independent shock/price and covariance identities, including lost-risk witnesses."""

from dataclasses import replace
from datetime import date, timedelta
import unittest

import numpy as np

from rust_pricing_backtest.domain import BacktestError, RiskBasis
from rust_pricing_backtest.market_shocks import MarketShockMap, solve_market_hedge


def basis(nodes):
    at = date(2026, 1, 5)
    return RiskBasis(at, (at+timedelta(days=64), at+timedelta(days=128)), tuple(nodes))


class MarketShockTests(unittest.TestCase):
    def setUp(self):
        self.coarse = basis([-1., 0., 1.])
        self.fine = basis([-1., -.5, 0., .5, 1.])
        self.mapping = MarketShockMap(self.fine, self.coarse)

    def test_chain_rule_preserves_price_change_and_parallel_shock(self):
        risk = np.array([2., -3., 5., 7., -11., 13., -17., 19., 23., -29.])
        shock = np.array([.01, -.02, .03, -.04, .05, -.06])
        # Hand-built linear interpolation, independently of the projection.
        lifted = [.01, -.005, -.02, .005, .03, -.04, .005, .05, -.005, -.06]
        np.testing.assert_allclose(self.mapping.lift_shock(shock), lifted, atol=1e-16)
        self.assertAlmostEqual(risk@lifted, self.mapping.project_risk(risk)@shock, places=14)
        self.assertAlmostEqual(sum(self.mapping.project_risk(risk)), sum(risk), places=14)
        np.testing.assert_array_equal(self.mapping.lift_shock(np.ones(6)), np.ones(10))

    def test_covariance_matches_transformed_actual_samples(self):
        # Deliberately correlated columns; summing only marginal variances fails.
        x = np.arange(50, dtype=float).reshape(5, 10)**2
        x[:, 3] = -2*x[:, 4]
        samples = np.array([self.mapping.project_risk(r) for r in x])
        actual = self.mapping.project_sample_covariance(np.cov(x, rowvar=False, ddof=1))
        expected = np.cov(samples, rowvar=False, ddof=1)
        np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=2e-9)
        price = np.array([1., -2., 4., 8., -16.])
        cov_price = np.cov(np.column_stack([price, x]), rowvar=False)[0, 1:]
        expected_price = np.cov(np.column_stack([price, samples]), rowvar=False)[0, 1:]
        np.testing.assert_allclose(self.mapping.project_risk(cov_price), expected_price, atol=1e-11)

    def test_omitted_direction_has_zero_chosen_risk_but_nonzero_price_risk(self):
        # This fine-grid shock is orthogonal to all coarse interpolated shocks.
        residual = np.array([-.5, 1., -.5, 0., 0., 0., 0., 0., 0., 0.])
        np.testing.assert_array_equal(self.mapping.project_risk(residual), np.zeros(6))
        a = self.mapping.complement_audit(residual)
        self.assertGreater(a['complement_l2'], 1.)
        np.testing.assert_allclose(a['complement_buckets'], residual, atol=1e-15)
        witness = np.asarray(a['unit_complement_shock'])
        np.testing.assert_allclose(self.mapping.project_risk(witness), 0., atol=1e-15)
        self.assertAlmostEqual(.01*residual@witness, a['worst_complement_one_point_pnl'], places=15)
        self.assertEqual(a['market_hedge_effectiveness'], 'not_established')

    def test_pythagorean_audit_and_identity_mapping(self):
        residual = np.arange(10)-4.
        a = self.mapping.complement_audit(residual)
        self.assertAlmostEqual(a['reporting_l2']**2, a['covered_l2']**2+a['complement_l2']**2, places=12)
        identity = MarketShockMap(self.fine, self.fine)
        np.testing.assert_array_equal(identity.project_risk(residual), residual)
        self.assertLess(identity.complement_audit(residual)['complement_l2'], 1e-14)

    def test_solver_uses_fixed_market_shocks_and_keeps_constraints(self):
        a = np.zeros((10, 1))
        a[1, 0] = 2.
        v = -3*a[:, 0]
        solution = solve_market_hedge(self.mapping, v, a, [0.], [-10.], [2.], [0.], ridge=0.)
        self.assertEqual(solution.solution.quantities, (2.,))
        self.assertGreater(solution.solution.objective_before, solution.solution.objective_after)
        self.assertEqual(solution.market_basis_id, self.coarse.basis_id)
        np.testing.assert_allclose(solution.reporting_residual, -a[:, 0])
        self.assertGreater(solution.complement['complement_l2'], 0.)

    def test_rejects_non_nested_incompatible_and_nonfinite_inputs(self):
        for changed in [basis([-1., -.2, 1.]), replace(self.coarse, coordinate='other'),
                        replace(self.coarse, delta_convention='other'),
                        replace(self.coarse, maturity_dates=(date(2027, 1, 1), date(2028, 1, 1))),
                        basis([-.9, 0., .9])]:
            with self.assertRaises(BacktestError):
                MarketShockMap(self.fine, changed)
        for bad in [np.ones(9), [float('nan')]*10]:
            with self.assertRaises(BacktestError):
                self.mapping.project_risk(bad)
        with self.assertRaises(BacktestError):
            self.mapping.lift_shock(np.ones(5))

    def test_covariance_rejects_marginals_asymmetry_and_negative_modes(self):
        for bad in [np.ones(10), -np.eye(10), np.tril(np.ones((10, 10))),
                    np.full((10, 10), float('nan'))]:
            with self.assertRaises(BacktestError):
                self.mapping.project_sample_covariance(bad)
        bad = np.eye(10)
        bad[0, 1] = bad[1, 0] = 2.
        with self.assertRaises(BacktestError):
            self.mapping.project_sample_covariance(bad)

    def test_identity_binds_basis_and_returns_no_mutable_mapping(self):
        original = self.mapping.map_id
        copied = self.mapping.matrix
        copied[:] = 99.
        self.assertEqual(self.mapping.map_id, original)
        self.assertFalse(np.all(self.mapping.matrix == 99.))
        self.assertNotEqual(original, MarketShockMap(self.fine, self.fine).map_id)

    def test_tiny_complement_witness_still_excludes_chosen_market_shocks(self):
        r = self.mapping.lift_shock([1., 2., 3., -2., 1., 3.])
        a = self.mapping.complement_audit(r)
        self.assertLess(a['complement_l2'], 1e-14)
        np.testing.assert_allclose(self.mapping.project_risk(a['unit_complement_shock']), 0., atol=1e-14)

    def test_unrepresentable_risk_norm_is_an_error(self):
        with self.assertRaisesRegex(BacktestError, 'market_shock_nonfinite'):
            MarketShockMap(self.fine, self.fine).complement_audit(np.full(10, 1e308))


if __name__ == '__main__':
    unittest.main()
