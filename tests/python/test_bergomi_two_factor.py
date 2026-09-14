"""Two-factor LSV, including an independent mixed-factor exchange-price check."""
import math
import unittest

import numpy as np
import rust_pricing as rp
import test_lsv
from test_multi_asset_lsv import TOL, VALUES, configs

FULL = [[1., .4, -.65, -.25, .1, -.05],
        [.4, 1., -.15, -.05, -.35, -.1],
        [-.65, -.15, 1., .5, .2, .04],
        [-.25, -.05, .5, 1., .03, .15],
        [.1, -.35, .2, .03, 1., .25],
        [-.05, -.1, .04, .15, .25, 1.]]


def two_configs(trace=True, particles=512, nu=.3):
    return [rp.MultiAssetLsv2FactorConfig(mean_reversions=k, vol_of_vol=v,
        mixing_weight=theta, spot_correlations=r, factor_correlation=r12,
        particle_count=particles, calibration_seed=401+i, log_bandwidth=.8,
        minimum_effective_samples=2., retain_reverse_trace=trace)
        for i, (k, v, theta, r, r12) in enumerate([
            ([4., .35], nu, .3, [-.65, -.25], .5),
            ([2.2, .12], .8*nu, .65, [-.35, -.1], .25)])]


def build(*, cfg=None, full=FULL, points=128, workers=1, step=.25, product=None):
    curve = rp.DiscountCurve(1, [0., 2.], [1., 1.])
    markets = [rp.Market.equity(1, i+1, s, curve, curve) for i, s in enumerate([100., 90.])]
    models = [rp.Model.local_volatility_from_grid([0., .5, 1.], [-.8, 0., .8], v,
              floor=1e-5, cap=2.) for v in VALUES]
    product = product or rp.MultiAssetProduct.basket([1, 2], [.6, .4], [1., 1.],
        "call", 96., "2027-01-01", "2027-01-01", smoothing_half_width=3.)
    return rp.MultiAssetPlan.compile("2026-01-01", product, markets, models,
        rp.CorrelationSchedule([1, 2], ["2026-01-01"], [[[1., .4], [.4, 1.]]], **TOL),
        rp.Engine.randomized_quasi_monte_carlo(points, 702, scramble_count=8,
            antithetic=True, brownian_bridge=True), maximum_step=step,
        worker_threads=workers, reduction_block_size=128,
        lsv_configs=two_configs() if cfg is None else cfg,
        driver_correlations=None if full is None else [full])


class TwoFactorBergomiTests(unittest.TestCase):
    def test_six_driver_snapshots_replay_and_explicit_validation(self):
        plan = build()
        self.assertEqual(plan.random_factor_count, 6)
        self.assertEqual(plan.lsv_driver_correlations, [FULL])
        self.assertEqual([c.volatility_factor_count for c in plan.lsv_calibrations], [2, 2])
        result = plan.evaluate_aad(gamma_relative_bump=.001)
        other = build(workers=3).evaluate_aad(gamma_relative_bump=.001)
        self.assertEqual(result.value, other.value)
        self.assertEqual(result.value, plan.evaluate().value)
        for a, b in zip(result.risks, other.risks):
            self.assertEqual(a.lsv_local_variance.node_adjoints, b.lsv_local_variance.node_adjoints)
            self.assertEqual(a.lsv_local_variance.standard_errors, b.lsv_local_variance.standard_errors)
        c = two_configs()[0]
        self.assertEqual(c.mean_reversions, [4., .35])
        self.assertEqual(c.spot_correlations, [-.65, -.25])
        w = c.normalized_weights
        self.assertAlmostEqual(w[0]**2+w[1]**2+2*c.factor_correlation*w[0]*w[1], 1.)
        with self.assertRaises(AttributeError):
            c.mixing_weight = .8
        with self.assertRaises(AttributeError):
            plan.lsv_calibrations[0].volatility_factor_count = 1
        with self.assertRaises(rp.PricingError):
            build(cfg=two_configs(False)).evaluate_aad()
        with self.assertRaises(rp.ValidationError):
            build(cfg=[object(), None], full=None)
        bad = [row[:] for row in FULL]
        bad[2][3] = bad[3][2] = .45
        with self.assertRaises(rp.ValidationError):
            build(full=bad)
        with self.assertRaises(rp.ValidationError):
            rp.MultiAssetLsv2FactorConfig(mean_reversions=[4., .35], vol_of_vol=.3,
                mixing_weight=.3, spot_correlations=[.9, .9], factor_correlation=-.9,
                particle_count=512, calibration_seed=401, log_bandwidth=.8,
                minimum_effective_samples=2.)

    def test_mixed_two_factor_bs_one_factor_lv_and_zero_nu(self):
        curve = rp.DiscountCurve(1, [0., 2.], [1., 1.])
        ids = [1, 2, 3, 4]
        markets = [rp.Market.equity(1, i, 100., curve, curve) for i in ids]
        lv = rp.Model.local_volatility_from_grid([0., .5, 1.], [-.8, 0., .8], VALUES[0], floor=1e-5, cap=2.)
        models = [lv, rp.Model.black_scholes(.2), lv, lv]
        product = rp.MultiAssetProduct.basket(ids, [.25]*4, [1.]*4, "call", 100.,
            "2027-01-01", "2027-01-01", smoothing_half_width=3.)
        corr = rp.CorrelationSchedule(ids, ["2026-01-01"],
            [[[1. if i == j else .3 for j in range(4)] for i in range(4)]], **TOL)
        engine = rp.Engine.pseudo_monte_carlo(256, 702, antithetic=True, brownian_bridge=True)
        def compile(cfg):
            return rp.MultiAssetPlan.compile("2026-01-01", product, markets, models, corr, engine,
                maximum_step=.25, lsv_configs=cfg)
        p = compile([two_configs()[0], None, configs()[1], None])
        self.assertEqual(p.random_factor_count, 7)
        self.assertEqual([None if c is None else c.volatility_factor_count for c in p.lsv_calibrations], [2, None, 1, None])
        r = p.evaluate_aad()
        self.assertIsNotNone(r.risks[1].bs_vega)
        self.assertEqual(len(r.risks[3].local_variance), 9)
        for i in (0, 2):
            self.assertIsNone(r.risks[i].lsv_local_variance.standard_errors)
        zero = compile([two_configs(nu=0.)[0], None, configs(nu=0.)[1], None]).evaluate_aad()
        plain = compile(None).evaluate_aad()
        self.assertAlmostEqual(zero.value, plain.value, delta=1e-12)
        for i in (0, 2):
            for a, b in zip(zero.risks[i].lsv_local_variance.node_adjoints, plain.risks[i].local_variance):
                self.assertAlmostEqual(a, b.value, delta=1e-10)

    def test_single_asset_two_factor_public_plan(self):
        request = test_lsv.BergomiLsvSmokeTest().request()
        def compile(trace):
            return rp.Bergomi2FactorLsvPlan.compile(request, mean_reversions=[4., .3],
                vol_of_vol=.6, mixing_weight=.4, spot_correlations=[-.65, -.25],
                factor_correlation=.5, particle_count=256, calibration_seed=42,
                log_bandwidth=.35, minimum_effective_samples=5., retain_reverse_trace=trace,
                worker_threads=2, reduction_block_size=32)
        p = compile(True)
        r = p.evaluate_local_variance_risk()
        self.assertEqual(r.price.value, p.evaluate().value)
        self.assertEqual(r.price.scheme, "bergomi-two-factor-lsv-log-euler-exact-ou-v1")
        self.assertEqual(len(r.node_adjoints), 9)
        self.assertEqual(len(r.standard_errors), 9)
        self.assertEqual(p.plan_fingerprint, compile(True).plan_fingerprint)
        with self.assertRaises(rp.PricingError):
            compile(False).evaluate_local_variance_risk()

    def test_mixed_factor_exchange_price_against_independent_five_dimensional_quadrature(self):
        # Integrate the final stock increments analytically (Margrabe) and all
        # first-step spot/OU innovations by independent 5-D Gauss-Hermite.
        full = [row[:5] for row in FULL[:5]]
        cfg = [two_configs(False, 2048)[0], configs(False, particles=2048)[1]]
        product = rp.MultiAssetProduct.basket([1, 2], [1., -1.], [1., 1.], "call", 0.,
            "2027-01-01", "2027-01-01")
        p = build(cfg=cfg, full=full, points=8192, step=.5, product=product)
        self.assertEqual(p.random_factor_count, 5)
        self.assertEqual(p.time_nodes, [0., .5, 1.])
        k = [0., 0., 4., .35, 2.2]
        cov = np.array([[full[i][j]*(.5 if k[i]+k[j] == 0 else
            -math.expm1(-(k[i]+k[j])*.5)/(k[i]+k[j])) for j in range(5)] for i in range(5)])
        lower = np.linalg.cholesky(cov)
        # Derive normalization independently of the implementation's getter.
        theta, r12 = .3, .5
        norm = math.sqrt((1-theta)**2+theta**2+2*r12*theta*(1-theta))
        def quadrature(order):
            nodes, weights = np.polynomial.hermite.hermgauss(order)
            nodes, weights = nodes*math.sqrt(2.), weights/math.sqrt(math.pi)
            index = np.indices((order,)*5).reshape(5, -1)
            innovations = lower @ nodes[index]
            joint_weights = np.prod(weights[index], axis=0)
            exponent = [.3*((1-theta)*innovations[2]+theta*innovations[3])/norm, .24*innovations[4]]
            ms, variances = [], []
            for i, c in enumerate(p.lsv_calibrations):
                rows = np.array(c.squared_leverage).reshape(3, 3)
                v0 = np.interp(0., c.log_moneyness_nodes, rows[0])
                m = np.exp(-.25*v0+math.sqrt(v0)*innovations[i])
                v1 = np.interp(np.log(m), c.log_moneyness_nodes, rows[1])*np.exp(2*exponent[i])
                ms.append(m)
                variances.append(v1)
            s1, s2 = 100.*ms[0], 90.*ms[1]
            sd = np.sqrt(.5*(variances[0]+variances[1]-.8*np.sqrt(variances[0]*variances[1])))
            d1 = np.log(s1/s2)/sd+.5*sd
            cdf = lambda x: .5*(1+np.vectorize(math.erf)(x/math.sqrt(2.)))
            return float(joint_weights @ (s1*cdf(d1)-s2*cdf(d1-sd)))
        coarse, fine = quadrature(10), quadrature(16)
        self.assertLess(abs(fine-coarse), .01)
        r = p.evaluate()
        self.assertAlmostEqual(r.value, fine, delta=7*r.standard_error+2*abs(fine-coarse)+.001)


if __name__ == "__main__":
    unittest.main()
