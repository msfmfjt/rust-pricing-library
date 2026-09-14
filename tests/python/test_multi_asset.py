"""Native multi-asset API: numerical contracts, immutable inputs and validation."""
from datetime import date
import math
import unittest
import rust_pricing as rp

TOLERANCES = dict(symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12,
                  psd_abs_tol=1e-12, psd_rel_tol=1e-12,
                  zero_pivot_abs_tol=1e-12, zero_pivot_rel_tol=1e-12)

class MultiAssetTests(unittest.TestCase):
    def markets(self, spots=(100., 90.)):
        r = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.04)])
        q = rp.DiscountCurve(2, [0., 2.], [1., 1.])
        return [rp.Market.equity(1, i + 1, s, r, q) for i, s in enumerate(spots)]

    def correlation(self, matrix=None):
        return rp.CorrelationSchedule([1, 2], [date(2026, 1, 1)],
                                      [matrix or [[1., .4], [.4, 1.]]], **TOLERANCES)

    def plan(self, product=None, *, spots=(100., 90.), models=None, workers=1):
        product = product or rp.MultiAssetProduct.basket(
            [1, 2], [.5, .5], [1., 1.], "call", 95., "2027-01-01", "2027-01-01",
            smoothing_half_width=2.)
        return rp.MultiAssetPlan.compile(
            "2026-01-01", product, self.markets(spots), models or [rp.Model.black_scholes(.2), rp.Model.black_scholes(.3)],
            self.correlation(), rp.Engine.randomized_quasi_monte_carlo(256, 702, scramble_count=8, antithetic=True),
            maximum_step=.25, worker_threads=workers, reduction_block_size=128)

    def test_basket_risk_and_worker_replay(self):
        plan = self.plan()
        result = plan.evaluate_aad(gamma_relative_bump=.001)
        self.assertEqual(plan.evaluate().value, result.value)
        self.assertEqual(result.underlying_ids, [1, 2])
        self.assertEqual(len(result.gamma), 2)
        self.assertGreater(result.risks[0].bs_vega.value, 0.)
        self.assertEqual(result.risks[0].local_variance, [])
        self.assertEqual(result.evaluated_paths, 4096)
        parallel = self.plan(workers=3).evaluate_aad(gamma_relative_bump=.001)
        self.assertEqual(result.value, parallel.value)
        self.assertEqual(result.fingerprint, parallel.fingerprint)
        self.assertEqual(result.standard_error, parallel.standard_error)
        for j, s in enumerate((100., 90.)):
            up, down = [100., 90.], [100., 90.]
            up[j], down[j] = s + .001, s - .001
            fd = (self.plan(spots=up).evaluate().value - self.plan(spots=down).evaluate().value) / .002
            self.assertAlmostEqual(result.risks[j].delta.value, fd, places=5)
            self.assertAlmostEqual(result.risks[j].bs_vega_per_vol_point.value, .01 * result.risks[j].bs_vega.value)
        with self.assertRaises(AttributeError):
            result.value = 0.
        copied = result.underlying_ids
        copied[0] = 100
        self.assertEqual(result.underlying_ids, [1, 2])

    def test_local_variance_risk_axes_and_bs_limit(self):
        local = rp.Model.local_volatility_from_grid([0., .5, 1.], [-.8, 0., .8], [.04]*9, floor=1e-5, cap=2.)
        result = self.plan(models=[local, rp.Model.black_scholes(.3)]).evaluate_aad()
        bs = self.plan().evaluate_aad()
        self.assertAlmostEqual(result.value, bs.value, places=11)
        risk = result.risks[0]
        self.assertIsNone(risk.bs_vega)
        self.assertEqual(risk.local_variance_time_nodes, [0., .5, 1.])
        self.assertEqual(risk.local_variance_log_moneyness_nodes, [-.8, 0., .8])
        self.assertEqual(len(risk.local_variance), 9)
        self.assertAlmostEqual(sum(.4*x.value for x in risk.local_variance), bs.risks[0].bs_vega.value, places=9)

    def test_worst_of_and_autocall_smoothing_contract(self):
        worst = rp.MultiAssetProduct.worst_of([1, 2], [100., 90.], "put", 1., "2027-01-01", "2027-01-01", notional=100., smoothing_half_width=.05)
        self.assertGreater(self.plan(worst).evaluate_aad().value, 0.)
        observations = [rp.AutocallObservation(t, t, coupon_amount=5., coupon_level=.9, call_level=1.05)
                        for t in ["2026-07-01", "2027-01-01"]]
        def product(width):
            return rp.MultiAssetProduct.autocallable([1, 2], [100., 90.], observations,
                "2027-01-01", "2027-01-01", notional=100., final_barrier=.7, memory=True,
                on_autocall="pay", on_maturity="forfeit", smoothing_half_width=width)
        exact = self.plan(product(None))
        self.assertGreater(exact.evaluate().value, 0.)
        with self.assertRaises(rp.PricingError):
            exact.evaluate_aad()
        smoothed = self.plan(product(.05))
        self.assertEqual(smoothed.evaluate().value, smoothed.evaluate_aad().value)

    def test_correlation_copy_rank_and_order_validation(self):
        matrix = [[1., 1.], [1., 1.]]
        c = self.correlation(matrix)
        matrix[0][1] = .2
        self.assertEqual(c.ranks, [1])
        self.assertEqual(c.zero_pivots, [[1]])
        self.assertEqual(c.matrices[0][0][1], 1.)
        with self.assertRaises(rp.ValidationError):
            self.correlation([[1., 1.01], [1.01, 1.]])
        with self.assertRaises(rp.ValidationError):
            rp.CorrelationSchedule([1, 1], ["2026-01-01"], [[[1., 0.], [0., 1.]]], **TOLERANCES)
        with self.assertRaises(rp.ValidationError):
            rp.MultiAssetProduct.basket([1, 2], [.5], [1., 1.], "call", 95., "2027-01-01", "2027-01-01")
        with self.assertRaises(rp.ValidationError):
            self.plan(models=[rp.Model.black_scholes(.2)])
        with self.assertRaises(rp.PricingError):
            self.plan().evaluate_aad(gamma_relative_bump=0.)

if __name__ == "__main__":
    unittest.main()
