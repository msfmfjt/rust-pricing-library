"""Joint Bergomi LSV: public wheel API, calibrated risk and independent pricing."""
import math
import unittest

import numpy as np
import rust_pricing as rp

TOL = dict(symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12,
           psd_abs_tol=1e-12, psd_rel_tol=1e-12,
           zero_pivot_abs_tol=1e-12, zero_pivot_rel_tol=1e-12)
FULL = [[1., .4, -.65, .1], [.4, 1., -.15, -.35],
        [-.65, -.15, 1., .3], [.1, -.35, .3, 1.]]
VALUES = [[.048, .04, .035, .06, .048, .039, .065, .05, .041],
          [.1, .09, .08, .11, .095, .075, .12, .10, .075]]


def configs(trace=True, nu=.3, particles=512):
    return [rp.MultiAssetLsvConfig(mean_reversion=k, vol_of_vol=v, correlation=rho,
                particle_count=particles, calibration_seed=401+i, log_bandwidth=.8,
                minimum_effective_samples=2., retain_reverse_trace=trace)
            for i, (k, v, rho) in enumerate(((.8, nu, -.65), (2.2, .8*nu, -.35)))]


def compile_plan(*, product=None, values=VALUES, spots=(100., 90.), cfg=None,
                 full=True, engine=None, workers=1, maximum_step=.25):
    curve = rp.DiscountCurve(1, [0., 2.], [1., 1.])
    markets = [rp.Market.equity(1, i+1, s, curve, curve) for i, s in enumerate(spots)]
    models = [rp.Model.local_volatility_from_grid([0., .5, 1.], [-.8, 0., .8], v,
              floor=1e-5, cap=2.) for v in values]
    product = product or rp.MultiAssetProduct.basket([1, 2], [.6, .4], [1., 1.],
              "call", 96., "2027-01-01", "2027-01-01", smoothing_half_width=3.)
    return rp.MultiAssetPlan.compile("2026-01-01", product, markets, models,
        rp.CorrelationSchedule([1, 2], ["2026-01-01"], [[[1., .4], [.4, 1.]]], **TOL),
        engine or rp.Engine.randomized_quasi_monte_carlo(128, 702, scramble_count=8,
            antithetic=True, brownian_bridge=True),
        maximum_step=maximum_step, worker_threads=workers, reduction_block_size=128,
        lsv_configs=configs() if cfg is None else cfg,
        driver_correlations=[FULL] if full else None)


class MultiAssetLsvTests(unittest.TestCase):
    def test_calibration_snapshots_risk_and_worker_replay(self):
        plan = compile_plan()
        result = plan.evaluate_aad(gamma_relative_bump=.001)
        self.assertEqual(result.value, plan.evaluate().value)
        other = compile_plan(workers=3).evaluate_aad(gamma_relative_bump=.001)
        self.assertEqual(result.value, other.value)
        self.assertEqual(result.fingerprint, other.fingerprint)
        self.assertEqual(plan.random_factor_count, 4)
        self.assertEqual(plan.lsv_driver_correlations, [FULL])
        self.assertEqual(len(plan.lsv_transition_covariances), 4)
        cal = plan.lsv_calibrations[0]
        self.assertEqual(cal.calibration_seed, 401)
        self.assertEqual(cal.time_nodes, [0., .25, .5, .75, 1.])
        self.assertEqual(len(cal.squared_leverage), 15)
        self.assertEqual(len(cal.effective_samples), 15)
        self.assertEqual(cal.particle_mean_normalized_f[0], 1.)
        copied = cal.squared_leverage
        copied[0] = -1.
        self.assertGreater(cal.squared_leverage[0], 0.)
        with self.assertRaises(AttributeError):
            cal.calibration_seed = 0
        for i in range(2):
            risk = result.risks[i].lsv_local_variance
            self.assertEqual(risk.node_adjoints, other.risks[i].lsv_local_variance.node_adjoints)
            self.assertEqual(risk.standard_errors, other.risks[i].lsv_local_variance.standard_errors)
            self.assertEqual(risk.time_nodes, [0., .5, 1.])
            self.assertEqual(len(risk.node_adjoints), 9)
            self.assertTrue(all(math.isfinite(v) and v >= 0 for v in risk.standard_errors))
            self.assertEqual(result.risks[i].local_variance, [])
            self.assertIsNone(result.risks[i].bs_vega)
            up, down = [v[:] for v in VALUES], [v[:] for v in VALUES]
            up[i][4] += 1e-6
            down[i][4] -= 1e-6
            fd = (compile_plan(values=up).evaluate().value-compile_plan(values=down).evaluate().value)/2e-6
            self.assertAlmostEqual(risk.node_adjoints[4], fd, delta=2e-4)

    def test_trace_mc_and_driver_validation(self):
        plan = compile_plan(cfg=configs(False))
        self.assertGreater(plan.evaluate().value, 0.)
        with self.assertRaises(rp.PricingError):
            plan.evaluate_aad()
        mc = compile_plan(engine=rp.Engine.pseudo_monte_carlo(128, 702, antithetic=True))
        self.assertIsNone(mc.evaluate_aad().risks[0].lsv_local_variance.standard_errors)
        for cfg, full in [([None], False), ([None, None], True),
                          ([configs()[0], None], True)]:
            with self.assertRaises(rp.ValidationError):
                compile_plan(cfg=cfg, full=full)
        mixed = compile_plan(cfg=[configs()[0], None], full=False)
        self.assertEqual(mixed.random_factor_count, 3)
        self.assertIsNone(mixed.lsv_calibrations[1])
        self.assertIsNone(mixed.evaluate_aad().risks[1].lsv_local_variance)
        with self.assertRaises(AttributeError):
            configs()[0].vol_of_vol = 0.

    def test_worst_of_and_memory_autocall_use_lsv_aad(self):
        worst = rp.MultiAssetProduct.worst_of([1, 2], [100., 90.], "put", 1.,
                "2027-01-01", "2027-01-01", notional=100., smoothing_half_width=.05)
        self.assertGreater(compile_plan(product=worst).evaluate_aad().value, 0.)
        obs = [rp.AutocallObservation(t, t, coupon_amount=5., coupon_level=.9, call_level=1.05)
               for t in ["2026-07-01", "2027-01-01"]]
        for width in (None, .05):
            product = rp.MultiAssetProduct.autocallable([1, 2], [100., 90.], obs,
                "2027-01-01", "2027-01-01", notional=100., final_barrier=.7, memory=True,
                on_autocall="pay", on_maturity="forfeit", smoothing_half_width=width)
            plan = compile_plan(product=product)
            if width is None:
                self.assertGreater(plan.evaluate().value, 0.)
                with self.assertRaises(rp.PricingError):
                    plan.evaluate_aad()
            else:
                risk = plan.evaluate_aad()
                self.assertEqual(plan.evaluate().value, risk.value)
                fd = (compile_plan(product=product, spots=(100.001, 90.)).evaluate().value
                      - compile_plan(product=product, spots=(99.999, 90.)).evaluate().value)/.002
                self.assertAlmostEqual(risk.risks[0].delta.value, fd, delta=2e-6)

    def test_two_step_lsv_exchange_price_against_four_dimensional_quadrature(self):
        # Integrate the final pair of stock increments analytically (conditional
        # Margrabe), and the first stock/OU innovations by independent 4-D GH.
        product = rp.MultiAssetProduct.basket([1, 2], [1., -1.], [1., 1.], "call", 0.,
                   "2027-01-01", "2027-01-01")
        plan = compile_plan(product=product, maximum_step=.5, cfg=configs(False, particles=2048),
                engine=rp.Engine.randomized_quasi_monte_carlo(8192, 702,
                    scramble_count=8, antithetic=True, brownian_bridge=True))
        self.assertEqual(plan.time_nodes, [0., .5, 1.])
        ks, nus = [0., 0., .8, 2.2], [.3, .24]
        cov = np.array([[FULL[i][j]*(.5 if ks[i]+ks[j] == 0 else
                -math.expm1(-(ks[i]+ks[j])*.5)/(ks[i]+ks[j])) for j in range(4)] for i in range(4)])
        lower = np.linalg.cholesky(cov)
        calibrations = plan.lsv_calibrations
        def quadrature(order):
            nodes, weights = np.polynomial.hermite.hermgauss(order)
            nodes, weights = nodes*math.sqrt(2.), weights/math.sqrt(math.pi)
            index = np.indices((order,)*4).reshape(4, -1)
            innovations = lower @ nodes[index]
            joint_weights = np.prod(weights[index], axis=0)
            ms, variances = [], []
            for i in range(2):
                c = calibrations[i]
                rows = np.array(c.squared_leverage).reshape(3, 3)
                v0 = np.interp(0., c.log_moneyness_nodes, rows[0])
                m = np.exp(-.5*v0*.5 + math.sqrt(v0)*innovations[i])
                v1 = np.interp(np.log(m), c.log_moneyness_nodes, rows[1])*np.exp(2*nus[i]*innovations[2+i])
                ms.append(m)
                variances.append(v1)
            s1, s2 = 100.*ms[0], 90.*ms[1]
            sd = np.sqrt(.5*(variances[0]+variances[1]-2*.4*np.sqrt(variances[0]*variances[1])))
            d1 = np.log(s1/s2)/sd + .5*sd
            cdf = lambda x: .5*(1+np.vectorize(math.erf)(x/math.sqrt(2.)))
            return float(joint_weights @ (s1*cdf(d1)-s2*cdf(d1-sd)))
        coarse, fine = quadrature(10), quadrature(16)
        self.assertLess(abs(fine-coarse), .01)
        result = plan.evaluate()
        self.assertAlmostEqual(result.value, fine,
            delta=7*result.standard_error + 2*abs(fine-coarse) + .001)


if __name__ == "__main__":
    unittest.main()
