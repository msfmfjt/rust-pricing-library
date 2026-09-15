"""Installed multi-asset rough-LSV API and independent non-Markov pricing."""
import math
import unittest

import numpy as np
import rust_pricing as rp
from test_bergomi_hull_white import build, configs as markov_configs, target
from test_multi_asset_lsv import TOL


def configs(trace=True):
    return [rp.MultiAssetRoughLsvConfig(hurst=h, vol_of_vol=eta, correlation=rho,
        particle_count=512, calibration_seed=401+i, log_bandwidth=.8,
        minimum_effective_samples=2., retain_reverse_trace=trace)
        for i, (h, eta, rho) in enumerate([(.08, .5, -.65), (.33, .6, -.35)])]


FULL = [[1., .4, -.65, .1, .2], [.4, 1., -.15, -.35, -.1],
        [-.65, -.15, 1., .3, -.05], [.1, -.35, .3, 1., .03],
        [.2, -.1, -.05, .03, 1.]]
RATES = [.2, -.1, -.05, .03]


def rough_plan(**kwargs):
    options = dict(cfg=configs(), rate_corr=RATES, full=[FULL])
    options.update(kwargs)
    return build(**options)


class MultiAssetRoughTests(unittest.TestCase):
    def test_config_driver_order_quote_risk_and_replay(self):
        c = configs()[0]
        self.assertEqual((c.hurst, c.vol_of_vol, c.correlation, c.particle_count,
            c.calibration_seed, c.log_bandwidth, c.minimum_effective_samples,
            c.retain_reverse_trace), (.08, .5, -.65, 512, 401, .8, 2., True))
        with self.assertRaises(AttributeError):
            c.hurst = .2
        arguments = dict(hurst=.1, vol_of_vol=.5, correlation=-.6,
            particle_count=512, calibration_seed=1, log_bandwidth=.5,
            minimum_effective_samples=2.)
        for key, value in [('hurst', 0.), ('hurst', .51), ('hurst', math.nan),
                           ('vol_of_vol', -.1), ('correlation', 1.1),
                           ('particle_count', 0), ('log_bandwidth', 0.)]:
            with self.subTest(key=key, value=value), self.assertRaises(rp.ValidationError):
                rp.MultiAssetRoughLsvConfig(**{**arguments, key: value})
        p = rough_plan()
        self.assertEqual(p.random_factor_count, 8)
        self.assertEqual(len(p.lsv_driver_correlations[0]), 5)
        self.assertEqual(len(p.lsv_transition_covariances[0]), 8)
        self.assertEqual([c.volatility_factor_count for c in p.hull_white_calibrations], [1, 1])
        r = p.evaluate_aad(gamma_relative_bump=.001)
        replay = rough_plan(workers=3).evaluate_aad(gamma_relative_bump=.001)
        self.assertEqual(r.value, p.evaluate().value)
        self.assertEqual(r.fingerprint, replay.fingerprint)
        self.assertEqual(r.value, replay.value)
        for i, risk in enumerate(r.risks):
            hw = risk.hull_white_lsv
            self.assertEqual(hw.vega_kt_raw, replay.risks[i].hull_white_lsv.vega_kt_raw)
            self.assertEqual(hw.vega_kt_standard_errors,
                replay.risks[i].hull_white_lsv.vega_kt_standard_errors)
            self.assertEqual(hw.forward_log_density_adjoints[:3], [0.]*3)
            self.assertAlmostEqual(hw.parallel_vega, sum(hw.vega_kt_raw), places=12)
            bumped = []
            for bump in (1e-7, -1e-7):
                ts = [target(), target()]
                quotes = [.28]*6
                quotes[1] += bump
                ts[i] = target(quotes)
                bumped.append(rough_plan(targets=ts).evaluate().value)
            self.assertAlmostEqual(hw.vega_kt_raw[1], (bumped[0]-bumped[1])/2e-7, delta=3e-5)
        with self.assertRaises(rp.ValidationError):
            rough_plan(full=[np.eye(8).tolist()])  # Gaussian count is not Brownian count.
        with self.assertRaises(rp.ValidationError):
            rough_plan(targets=[rp.HullWhiteLsvTarget.flat(.28, [0., .5, 1.], [-.35, 0., .35])]*2)

    def test_products_mixed_models_and_trace_contract(self):
        observations = [rp.AutocallObservation(t, pay, coupon_amount=5.,
            coupon_level=.9, call_level=1.05) for t, pay in
            [('2026-07-02', '2026-07-16'), ('2027-01-01', '2027-01-15')]]
        products = [rp.MultiAssetProduct.worst_of([1, 2], [100., 90.], 'put', 1.,
            '2027-01-01', '2027-02-01', notional=100., smoothing_half_width=.05),
            rp.MultiAssetProduct.autocallable([1, 2], [100., 90.], observations,
            '2027-01-01', '2027-01-15', notional=100., final_barrier=.7,
            memory=True, on_autocall='pay', on_maturity='forfeit', smoothing_half_width=.05)]
        for product in products:
            r = rough_plan(product=product).evaluate_aad()
            fd = (rough_plan(product=product, spots=(100.001, 90.)).evaluate().value
                  - rough_plan(product=product, spots=(99.999, 90.)).evaluate().value)/.002
            self.assertAlmostEqual(r.risks[0].delta.value, fd, delta=3e-6)
        for second, count, rates in [(markov_configs()[0], 8, RATES+[.02]),
                                    (markov_configs()[1], 7, RATES),
                                    (None, 6, RATES[:3])]:
            p = rough_plan(cfg=[configs()[0], second], full=None, rate_corr=rates,
                targets=[target(), target() if second is not None else None])
            self.assertEqual(p.random_factor_count, count)
            self.assertTrue(math.isfinite(p.evaluate_aad().value))
        mc = rough_plan(engine=rp.Engine.pseudo_monte_carlo(128, 511,
            antithetic=True)).evaluate_aad()
        self.assertIsNone(mc.risks[0].hull_white_lsv.vega_kt_standard_errors)
        p = rough_plan(cfg=configs(False))
        self.assertTrue(math.isfinite(p.evaluate().value))
        with self.assertRaises(rp.PricingError):
            p.evaluate_aad()

    def test_two_rough_exchange_against_independent_five_dimensional_quadrature(self):
        # Integrate first-step [W_A,W_B,I_r,J_A,J_B] independently, then the
        # final exchange payoff analytically. Common future rate factors cancel
        # against discounting; the first rate integral still drives leverage.
        xs = [-.6, -.3, 0., .3, .6]
        targets = [rp.HullWhiteLsvTarget.flat(v, [0., .5, 1.], xs) for v in [.25, .3]]
        curve = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.05)])
        markets = [rp.Market.equity(1, i+1, spot, curve,
            rp.DiscountCurve(100+i, [0., 2.], [1., math.exp(-.02*(i+1))]))
            for i, spot in enumerate([100., 90.])]
        product = rp.MultiAssetProduct.basket([1, 2], [1., -1.], [1., 1.], 'call', 0.,
            '2027-01-01', '2027-01-01')
        hs, etas = [.12, .32], [.4, .35]
        cfg = [rp.MultiAssetRoughLsvConfig(hurst=hs[i], vol_of_vol=etas[i],
            correlation=[-.65, -.35][i], particle_count=2048, calibration_seed=401+i,
            log_bandwidth=.35, minimum_effective_samples=10.) for i in range(2)]
        schedule = rp.CorrelationSchedule([1, 2], ['2026-01-01'], [[[1., .4], [.4, 1.]]], **TOL)
        engine = rp.Engine.randomized_quasi_monte_carlo(8192, 732, scramble_count=8,
            antithetic=True, brownian_bridge=True)
        arguments = dict(maximum_step=.5, worker_threads=2, reduction_block_size=128,
            lsv_configs=cfg, driver_correlations=[FULL],
            rate_model=rp.HullWhiteModel(.13, [0.], [.004]),
            rate_correlations=RATES, lsv_targets=targets)
        def compile(options):
            return rp.MultiAssetPlan.compile('2026-01-01', product, markets,
                [t.model for t in targets], schedule, engine, **options)
        p = compile(arguments)
        # A rough config needs the paired-HW adapter, even with deterministic rates.
        without_hw = {k: v for k, v in arguments.items()
            if k not in ('rate_model', 'rate_correlations', 'lsv_targets', 'driver_correlations')}
        with self.assertRaises(rp.ValidationError):
            compile(without_hw)
        nodes, weights = np.polynomial.legendre.leggauss(128)
        y, w = (nodes+1.)*.5, weights*.5
        powers = np.array([0., 0., 0., hs[0]-.5, hs[1]-.5])
        coeffs = np.array([1., 1., 1., math.sqrt(2.*hs[0]), math.sqrt(2.*hs[1])])
        drivers = [0, 1, 4, 2, 3]
        cov = np.empty((5, 5))
        for i in range(5):
            for j in range(5):
                power = 1.+powers[i]+powers[j]
                # u=.5*y^(1/power) removes the endpoint singularity.
                u = .5*y**(1./power)
                kernel = np.ones_like(u)
                for k in (i, j):
                    if k == 2:
                        kernel *= .004*(-np.expm1(-.13*u))/.13
                cov[i, j] = (FULL[drivers[i]][drivers[j]]*coeffs[i]*coeffs[j]
                    *.5**power/power*float(w @ kernel))
        lower = np.linalg.cholesky(cov)
        lev = [np.array(c.squared_leverage).reshape(3, len(xs)) for c in p.hull_white_calibrations]
        cdf = np.vectorize(lambda x: .5*math.erfc(-x/math.sqrt(2.)), otypes=[float])

        def quadrature(order):
            z, w = np.polynomial.hermite.hermgauss(order)
            z, w = z*math.sqrt(2.), w/math.sqrt(math.pi)
            indices = np.indices((order,)*5).reshape(5, -1)
            inc = lower @ z[indices]
            weight = np.prod(w[indices], axis=0)
            units, vols = [], []
            for i in range(2):
                l0 = float(np.interp(0., xs, lev[i][0]))
                unit = np.exp(inc[2]+.5*cov[2, 2]-.25*l0+math.sqrt(l0)*inc[i])
                centered = inc[3+i]-.5*etas[i]*.5**(2.*hs[i])
                vol = np.sqrt(np.interp(np.log(unit), xs, lev[i][1]))*np.exp(.5*etas[i]*centered)
                units.append(unit)
                vols.append(vol)
            ca, cb = 100.*math.exp(.015)*units[0], 90.*math.exp(.005)*units[1]
            variance = .5*(vols[0]**2+vols[1]**2-.8*vols[0]*vols[1])
            root = np.sqrt(variance)
            d1 = (np.log(ca/cb)+.5*variance)/root
            price = math.exp(-.025)*np.exp(-inc[2]-.5*cov[2, 2])*(ca*cdf(d1)-cb*cdf(d1-root))
            return float(weight @ price)

        low, high = quadrature(10), quadrature(14)
        self.assertLess(abs(high-low), .025)
        result = p.evaluate()
        self.assertAlmostEqual(result.value, high,
            delta=6.*result.standard_error+2.*abs(high-low)+.003)


if __name__ == '__main__':
    unittest.main()
