"""Particle Local Correlation: installed API and independent numerical checks."""
import math
import unittest

import numpy as np
import rust_pricing as rp
from test_multi_asset_lsv import TOL, configs as lsv_configs

TIMES, XS = [0., .5, 1.], [-.6, 0., .6]
ASSET = [.082, .070, .065, .086, .074, .068, .090, .078, .072]
TARGET = [.059, .053, .049, .061, .055, .050, .063, .057, .052]


def correlation(rho):
    return rp.CorrelationSchedule([1, 2], ['2026-01-01'],
        [[[1., rho], [rho, 1.]]], **TOL)


def lv(values, times=TIMES, xs=XS):
    return rp.Model.local_volatility_from_grid(times, xs, values, floor=1e-6, cap=1.)


def config(**kwargs):
    options = dict(basket_weights=[.6, .4], target_model=lv(TARGET),
        second_correlations=correlation(.95), particle_count=512,
        calibration_seed=8401, log_bandwidth=.7, minimum_effective_samples=2.,
        feasibility='project_and_report', retain_reverse_trace=True)
    return rp.LocalCorrelationConfig(**{**options, **kwargs})


def build(*, product=None, cfg=None, spots=(100., 90.), workers=1,
          engine=None, step=.25, cash=False, **kwargs):
    curve = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.05)])
    markets = [rp.Market.equity(1, i+1, s, curve,
        rp.DiscountCurve(100+i, [0., 2.], [1., math.exp(-.02*(i+1))]),
        discrete_dividends=[rp.DividendEvent.fixed_cash_and_proportional(1, .5, 2., .03)]
        if cash and i == 0 else None) for i, s in enumerate(spots)]
    product = product or rp.MultiAssetProduct.basket([1, 2], [.6, .4], [1., 1.],
        'call', 96., '2027-01-01', '2027-01-01', smoothing_half_width=3.)
    return rp.MultiAssetPlan.compile('2026-01-01', product, markets,
        [lv(ASSET), rp.Model.black_scholes(.3)], correlation(-.3),
        engine or rp.Engine.randomized_quasi_monte_carlo(128, 702, scramble_count=8,
            antithetic=True, brownian_bridge=True), maximum_step=step,
        worker_threads=workers, reduction_block_size=128,
        local_correlation=cfg or config(), **kwargs)


def black_call(forward, strike, sigma):
    d1 = math.log(forward/strike)/sigma+.5*sigma
    cdf = lambda x: .5*math.erfc(-x/math.sqrt(2.))
    return forward*cdf(d1)-strike*cdf(d1-sigma)


class LocalCorrelationTests(unittest.TestCase):
    def test_config_diagnostics_recalibrated_risk_and_replay(self):
        c = config()
        self.assertEqual((c.basket_weights, c.particle_count, c.calibration_seed,
            c.log_bandwidth, c.minimum_effective_samples, c.feasibility,
            c.minimum_variance_span, c.retain_reverse_trace),
            ([.6, .4], 512, 8401, .7, 2., 'project_and_report', 1e-12, True))
        with self.assertRaises(AttributeError):
            c.log_bandwidth = 1.
        for key, value in [('basket_weights', [0., 1.]), ('basket_weights', [.5, .6]),
                           ('particle_count', 0), ('log_bandwidth', 0.),
                           ('minimum_variance_span', math.nan), ('feasibility', 'silent'),
                           ('target_model', rp.Model.black_scholes(.2))]:
            with self.subTest(key=key), self.assertRaises(rp.ValidationError):
                config(**{key: value})
        for options in [dict(lsv_configs=lsv_configs()),
                        dict(rate_model=rp.HullWhiteModel(.1, [0.], [.01])),
                        dict(driver_correlations=[np.eye(4).tolist()])]:
            with self.assertRaises(rp.ValidationError):
                build(**options)
        p = build()
        self.assertEqual(p.random_factor_count, 4)
        cal = p.local_correlation_calibration
        self.assertEqual(cal.time_nodes, [0., .25, .5, .75, 1.])
        self.assertEqual(cal.log_nodes, XS)
        self.assertEqual(cal.particle_means[0], [1., 1.])
        original = cal.mixing_coefficients
        original[0] = -5.
        self.assertGreaterEqual(cal.mixing_coefficients[0], 0.)
        self.assertTrue(cal.retains_reverse_trace)
        result = p.evaluate_aad(gamma_relative_bump=.001)
        other = build(workers=3).evaluate_aad(gamma_relative_bump=.001)
        risk = result.local_correlation_risk
        self.assertEqual(result.value, p.evaluate().value)
        self.assertEqual((result.value, result.fingerprint), (other.value, other.fingerprint))
        for attr in ['basket_variance_adjoints', 'basket_standard_errors',
                     'asset_adjoints', 'asset_standard_errors']:
            self.assertEqual(getattr(risk, attr), getattr(other.local_correlation_risk, attr))
        self.assertEqual(risk.basket_time_nodes, TIMES)
        self.assertEqual(risk.asset_time_nodes, [TIMES, []])
        self.assertEqual(risk.asset_log_nodes, [XS, []])
        self.assertTrue(all(v.bs_vega is None and v.local_variance == [] for v in result.risks))
        values = []
        for h in [1e-7, -1e-7]:
            target = TARGET.copy()
            target[4] += h
            values.append(build(cfg=config(target_model=lv(target))).evaluate().value)
        self.assertAlmostEqual(risk.basket_variance_adjoints[4],
            (values[0]-values[1])/2e-7, delta=3e-5)
        traceless = build(cfg=config(retain_reverse_trace=False))
        self.assertTrue(math.isfinite(traceless.evaluate().value))
        with self.assertRaises(rp.PricingError):
            traceless.evaluate_aad()
        mc = build(engine=rp.Engine.pseudo_monte_carlo(128, 511,
            antithetic=True)).evaluate_aad().local_correlation_risk
        self.assertIsNone(mc.basket_standard_errors)
        self.assertIsNone(mc.asset_standard_errors)

    def test_worst_of_autocall_dividends_and_payment_lags(self):
        observations = [rp.AutocallObservation(t, pay, coupon_amount=5.,
            coupon_level=.9, call_level=1.05) for t, pay in
            [('2026-07-02', '2026-07-16'), ('2027-01-01', '2027-01-15')]]
        products = [rp.MultiAssetProduct.worst_of([1, 2], [100., 90.], 'put', 1.,
            '2027-01-01', '2027-02-01', notional=100., smoothing_half_width=.05),
            rp.MultiAssetProduct.autocallable([1, 2], [100., 90.], observations,
            '2027-01-01', '2027-01-15', notional=100., final_barrier=.7,
            memory=True, on_autocall='pay', on_maturity='forfeit', smoothing_half_width=.05)]
        for product in products:
            r = build(product=product, cash=True).evaluate_aad()
            values = [build(product=product, cash=True, spots=(100.+h, 90.)).evaluate().value
                      for h in [1e-4, -1e-4]]
            self.assertAlmostEqual(r.risks[0].delta.value, (values[0]-values[1])/2e-4, delta=3e-6)

    def test_exchange_against_independent_two_dimensional_quadrature(self):
        product = rp.MultiAssetProduct.basket([1, 2], [1., -1.], [1., 1.],
            'call', 0., '2027-01-01', '2027-01-01')
        p = build(product=product, step=.5, cfg=config(particle_count=2048),
            engine=rp.Engine.randomized_quasi_monte_carlo(8192, 732,
                scramble_count=8, antithetic=True, brownian_bridge=True))
        # Use the published scalar surface, but independently construct the
        # state covariance and integrate the final exchange payoff analytically.
        mixing = np.array(p.local_correlation_calibration.mixing_coefficients).reshape(3, 3)
        rho0 = -.3+1.25*np.interp(0., XS, mixing[0])
        lower = np.linalg.cholesky([[1., rho0], [rho0, 1.]])
        cdf = np.vectorize(lambda x: .5*math.erfc(-x/math.sqrt(2.)), otypes=[float])

        def quadrature(order):
            z, w = np.polynomial.hermite.hermgauss(order)
            indices = np.indices((order, order)).reshape(2, -1)
            inc = lower @ (math.sqrt(2.)*z[indices])
            weights = np.prod(w[indices]/math.sqrt(math.pi), axis=0)
            m = np.exp(-.25*np.array([.07, .09])[:, None]
                +math.sqrt(.5)*np.array([math.sqrt(.07), .3])[:, None]*inc)
            sigma_a = np.sqrt(np.interp(np.log(m[0]), XS, ASSET[3:6]))
            rho = -.3+1.25*np.interp(np.log(.6*m[0]+.4*m[1]), XS, mixing[1])
            variance = .5*(sigma_a*sigma_a+.09-.6*rho*sigma_a)
            ca, cb = 100.*math.exp(.015)*m[0], 90.*math.exp(.005)*m[1]
            root = np.sqrt(variance)
            d1 = (np.log(ca/cb)+.5*variance)/root
            value = math.exp(-.025)*(ca*cdf(d1)-cb*cdf(d1-root))
            return float(weights @ value)

        low, high = quadrature(48), quadrature(64)
        self.assertLess(abs(high-low), .004)
        result = p.evaluate()
        self.assertAlmostEqual(result.value, high,
            delta=6.*result.standard_error+2.*abs(high-low)+.002)

    def test_normalized_index_target_refinement_and_constituent_marginals(self):
        curve = rp.DiscountCurve(1, [0., 2.], [1., 1.])
        markets = [rp.Market.equity(1, i, 100., curve, curve) for i in [1, 2]]
        times, xs = [0., .25, .5, .75, 1.], np.linspace(-.7, .7, 15).tolist()
        target = lv([.27**2]*(len(times)*len(xs)), times, xs)
        engine = rp.Engine.randomized_quasi_monte_carlo(4096, 995,
            scramble_count=8, antithetic=True, brownian_bridge=True)

        def value(particles, bandwidth, step, strike, weights):
            cfg = config(basket_weights=[.5, .5], target_model=target,
                second_correlations=correlation(1.), particle_count=particles,
                log_bandwidth=bandwidth, minimum_effective_samples=16.,
                feasibility='reject', retain_reverse_trace=False)
            product = rp.MultiAssetProduct.basket([1, 2], weights, [1., 1.],
                'call', strike, '2027-01-01', '2027-01-01')
            p = rp.MultiAssetPlan.compile('2026-01-01', product, markets,
                [rp.Model.black_scholes(.3)]*2, correlation(-.4), engine,
                maximum_step=step, worker_threads=2, reduction_block_size=128,
                local_correlation=cfg)
            self.assertFalse(any(p.local_correlation_calibration.projected_nodes))
            return p.evaluate()

        coarse, fine = [], []
        for strike in [90., 100., 110.]:
            reference = black_call(100., strike, .27)
            a = value(1024, .4, .125, strike, [.5, .5])
            b = value(8192, .15, 1./32., strike, [.5, .5])
            coarse.append(abs(a.value-reference))
            fine.append(abs(b.value-reference))
            # Separate finite calibration/grid bias from valuation sampling SE.
            self.assertAlmostEqual(b.value, reference, delta=6.*b.standard_error+.025)
        self.assertLess(sum(fine), sum(coarse))
        for weights in [[1., 0.], [0., 1.]]:
            result = value(8192, .15, 1./32., 100., weights)
            self.assertAlmostEqual(result.value, black_call(100., 100., .3),
                delta=6.*result.standard_error+.003)


if __name__ == '__main__':
    unittest.main()
