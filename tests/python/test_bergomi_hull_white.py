"""Installed-wheel two-factor/common-rate contracts and independent pricing."""
import math
import unittest

import numpy as np
import rust_pricing as rp
from test_multi_asset_lsv import TOL


def target(quotes=None):
    return rp.HullWhiteLsvTarget.from_market_iv([.5, 1.], [-.5, 0., .5],
        [0.28]*6 if quotes is None else quotes, [0., .5, 1.], [-.35, 0., .35])


def configs(trace=True):
    shared = dict(particle_count=512, log_bandwidth=.8,
        minimum_effective_samples=2., retain_reverse_trace=trace)
    return [rp.MultiAssetLsv2FactorConfig(mean_reversions=[4., .35], vol_of_vol=.3,
                mixing_weight=.3, spot_correlations=[-.65, -.25], factor_correlation=.5,
                calibration_seed=401, **shared),
            rp.MultiAssetLsvConfig(mean_reversion=1.3, vol_of_vol=.25, correlation=-.35,
                calibration_seed=402, **shared)]


def build(*, product=None, targets=None, cfg=None, spots=(100., 90.), workers=1,
          engine=None, rate_corr=None, full=None):
    targets = [target(), target()] if targets is None else targets
    curve = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.05)])
    markets = [rp.Market.equity(1, i+1, spot, curve,
        rp.DiscountCurve(100+i, [0., 2.], [1., math.exp(-.02*(i+1))]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(i+1, .5, 2.-i)])
        for i, spot in enumerate(spots)]
    product = product or rp.MultiAssetProduct.basket([1, 2], [.6, .4], [1., 1.],
        "call", 96., "2027-01-01", "2027-01-01", smoothing_half_width=3.)
    return rp.MultiAssetPlan.compile("2026-01-01", product, markets,
        [t.model if t is not None else rp.Model.black_scholes(.3) for t in targets],
        rp.CorrelationSchedule([1, 2], ["2026-01-01"], [[[1., .4], [.4, 1.]]], **TOL),
        engine or rp.Engine.randomized_quasi_monte_carlo(128, 702, scramble_count=8,
            antithetic=True, brownian_bridge=True), maximum_step=.25,
        worker_threads=workers, reduction_block_size=128,
        lsv_configs=configs() if cfg is None else cfg, driver_correlations=full,
        rate_model=rp.HullWhiteModel(.13, [0., .37], [.004, .006]),
        rate_correlations=[.2, -.1, -.05, .02, .03] if rate_corr is None else rate_corr,
        lsv_targets=targets)


class BergomiHullWhiteTests(unittest.TestCase):
    def test_multi_asset_typed_snapshots_quotes_replay_and_curve_uncertainty(self):
        plan = build()
        self.assertTrue(plan.has_hull_white)
        self.assertEqual(plan.random_factor_count, 7)
        self.assertEqual([c.volatility_factor_count for c in plan.hull_white_calibrations], [2, 1])
        self.assertEqual(plan.lsv_calibrations, [None, None])
        self.assertEqual(len(plan.lsv_driver_correlations[0]), 6)
        self.assertEqual(len(plan.lsv_transition_covariances[0]), 7)
        risk = plan.evaluate_aad(gamma_relative_bump=.001)
        self.assertEqual(risk.value, plan.evaluate().value)
        replay = build(workers=3).evaluate_aad(gamma_relative_bump=.001)
        self.assertEqual(risk.value, replay.value)
        self.assertEqual(risk.fingerprint, replay.fingerprint)
        for i, r in enumerate(risk.risks):
            hw = r.hull_white_lsv
            self.assertEqual(r.lsv_local_variance.time_nodes, plan.time_nodes)
            self.assertEqual(hw.vega_kt_raw, replay.risks[i].hull_white_lsv.vega_kt_raw)
            self.assertEqual(hw.vega_kt_standard_errors, replay.risks[i].hull_white_lsv.vega_kt_standard_errors)
            self.assertEqual(hw.vega_kt_market_scaled, [v*.01 for v in hw.vega_kt_raw])
            self.assertAlmostEqual(hw.parallel_vega, sum(hw.vega_kt_raw), places=12)
            self.assertTrue(math.isfinite(hw.parallel_vega_standard_error))
            self.assertEqual(hw.forward_log_density_adjoints[:3], [0.]*3)
            quotes = [.28]*6
            quotes[1] += 1e-7
            up = [target(), target()]
            up[i] = target(quotes)
            quotes[1] -= 2e-7
            down = [target(), target()]
            down[i] = target(quotes)
            fd = (build(targets=up).evaluate().value-build(targets=down).evaluate().value)/2e-7
            self.assertAlmostEqual(hw.vega_kt_raw[1], fd, delta=3e-5)
        curves = risk.hull_white_curve_risk
        self.assertEqual(curves.discount_time_nodes, [0., 2.])
        for raw, scaled, time in zip(curves.discount_log_df_adjoints, curves.discount_node_dv01, curves.discount_time_nodes):
            self.assertAlmostEqual(scaled.value, -1e-4*time*raw.value)
            self.assertAlmostEqual(scaled.standard_error, 1e-4*time*raw.standard_error)
        cal = plan.hull_white_calibrations[0]
        self.assertEqual(cal.mean_relative_discount[0], 1.)
        self.assertEqual(cal.mean_discounted_normalized_equity[0], 1.)
        self.assertTrue(cal.retains_reverse_trace)
        copied = cal.squared_leverage
        copied[0] = -1.
        self.assertGreater(cal.squared_leverage[0], 0.)
        with self.assertRaises(AttributeError):
            cal.squared_leverage = []
        with self.assertRaises(AttributeError):
            risk.risks[0].hull_white_lsv.vega_kt_raw = []

    def test_multi_asset_cashflows_bs_mixture_and_invalid_marginals(self):
        observations = [rp.AutocallObservation("2026-07-02", "2026-07-16",
            coupon_amount=4., coupon_level=.9, call_level=1.05),
            rp.AutocallObservation("2027-01-01", "2027-01-15",
            coupon_amount=5., coupon_level=.9, call_level=1.05)]
        autocall = rp.MultiAssetProduct.autocallable([1, 2], [100., 90.], observations,
            "2027-01-01", "2027-01-15", notional=100., final_barrier=.7,
            memory=True, on_autocall="pay", on_maturity="forfeit", smoothing_half_width=.05)
        worst = rp.MultiAssetProduct.worst_of([1, 2], [100., 90.], "put", 1.,
            "2027-01-01", "2027-02-01", notional=100., smoothing_half_width=.05)
        for product in (autocall, worst):
            p = build(product=product)
            r = p.evaluate_aad()
            self.assertEqual(r.value, p.evaluate().value)
            fd = (build(product=product, spots=(100.001, 90.)).evaluate().value
                  - build(product=product, spots=(99.999, 90.)).evaluate().value)/.002
            self.assertAlmostEqual(r.risks[0].delta.value, fd, delta=3e-6)
        mixed = build(targets=[target(), None], cfg=[configs()[0], None],
            rate_corr=[.2, -.1, -.05, .02]).evaluate_aad()
        self.assertIsNotNone(mixed.risks[1].bs_vega)
        self.assertIsNone(mixed.risks[1].hull_white_lsv)
        mc = build(engine=rp.Engine.pseudo_monte_carlo(128, 511, antithetic=True)).evaluate_aad()
        self.assertIsNone(mc.risks[0].hull_white_lsv.vega_kt_standard_errors)
        self.assertIsNone(mc.risks[0].lsv_local_variance.standard_errors)
        with self.assertRaises(rp.PricingError):
            build(cfg=configs(False)).evaluate_aad()
        for rates in ([.2], [.99]*5, [math.nan]*5):
            with self.assertRaises(rp.ValidationError):
                build(rate_corr=rates)
        with self.assertRaises(rp.ValidationError):
            build(full=[[[1., 0.], [0., 1.]]])
        with self.assertRaises(rp.ValidationError):
            build(targets=[rp.HullWhiteLsvTarget.flat(.28, [0., .5, 1.], [-.35, 0., .35])]*2)

    def test_single_asset_two_factor_entry_point_and_vega_kt(self):
        def compile(quotes, workers=1, trace=True):
            t = target(quotes)
            curve = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.05)])
            request = rp.PricingRequest("2026-01-01",
                rp.Product.european_vanilla(1, 1, "2027-01-01", 100., 1., "call"),
                rp.Market.equity(1, 1, 100., curve, curve,
                    discrete_dividends=[rp.DividendEvent.fixed_cash(1, .5, 2.)]),
                t.model, rp.Engine.randomized_quasi_monte_carlo(128, 713,
                    scramble_count=4, antithetic=True, brownian_bridge=True), rp.RiskRequest())
            return rp.HullWhiteEquityPlan.compile_lsv_two_factor(request, t,
                rp.HullWhiteModel(.13, [0., .37], [.004, .006]),
                mean_reversions=[4., .35], vol_of_vol=.3, mixing_weight=.3,
                spot_correlations=[-.65, -.25], factor_correlation=.5,
                equity_rate_correlation=.2, vol_rate_correlations=[-.05, .02],
                particle_count=512, calibration_seed=401, log_bandwidth=.8,
                minimum_effective_samples=2., worker_threads=workers,
                reduction_block_size=128, retain_reverse_trace=trace)
        p = compile([.28]*6)
        self.assertEqual(p.random_factor_count, 5)
        r = p.evaluate_aad()
        self.assertEqual(r.price.value, p.evaluate().value)
        self.assertEqual(r.derivatives, compile([.28]*6, 3).evaluate_aad().derivatives)
        up, down = [.28]*6, [.28]*6
        up[1] += 1e-7
        down[1] -= 1e-7
        fd = (compile(up).evaluate().value-compile(down).evaluate().value)/2e-7
        self.assertAlmostEqual(r.vega_kt_raw[1], fd, delta=3e-5)
        with self.assertRaises(rp.PricingError):
            compile([.28]*6, trace=False).evaluate_aad()

    def test_two_step_price_against_independent_five_dimensional_quadrature(self):
        # Conditional on the compiled leverage, integrate the first joint step
        # by Gauss-Hermite and the last stochastic-rate stock step analytically.
        xs = [-.6, -.3, 0., .3, .6]
        t = rp.HullWhiteLsvTarget.flat(.25, [0., .5, 1.], xs)
        curve = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.05)])
        q = rp.DiscountCurve(2, [0., 2.], [1., math.exp(-.02)])
        request = rp.PricingRequest("2026-01-01",
            rp.Product.european_vanilla(1, 1, "2027-01-01", 100., 1., "call"),
            rp.Market.equity(1, 1, 100., curve, q), t.model,
            rp.Engine.randomized_quasi_monte_carlo(8192, 712, scramble_count=8,
                antithetic=True, brownian_bridge=True), rp.RiskRequest())
        p = rp.HullWhiteEquityPlan.compile_lsv_two_factor(request, t,
            rp.HullWhiteModel(.2, [0.], [.007]), mean_reversions=[4., .35],
            vol_of_vol=.3, mixing_weight=.3, spot_correlations=[-.6, -.2],
            factor_correlation=.4, equity_rate_correlation=.25,
            vol_rate_correlations=[-.05, .08], particle_count=2048,
            calibration_seed=411, log_bandwidth=.35, minimum_effective_samples=10.,
            worker_threads=2, reduction_block_size=128)
        corr = np.array([[1., -.6, .25, .25, -.2], [-.6, 1., -.05, -.05, .4],
            [.25, -.05, 1., 1., .08], [.25, -.05, 1., 1., .08],
            [-.2, .4, .08, .08, 1.]])
        nodes, weights = np.polynomial.legendre.leggauss(96)
        z = (nodes+1.)*.25
        kernels = np.array([np.ones_like(z), np.exp(-4.*z), .007*np.exp(-.2*z),
            .007*(-np.expm1(-.2*z))/.2, np.exp(-.35*z)])
        cov = corr * ((kernels*(weights*.25)) @ kernels.T)
        lower = np.linalg.cholesky(cov)
        z1 = (nodes+1.)*.5
        vi = float(np.sum(weights*.5*(.007*(-np.expm1(-.2*z1))/.2)**2))
        bh = -math.expm1(-.1)/.2
        shift = .5*(vi-cov[3, 3])
        mixing = np.array([.7, .3])/math.sqrt(.7**2+.3**2+2.*.4*.7*.3)
        lev = np.array(p.squared_leverage).reshape(3, len(xs))
        normal_cdf = np.vectorize(lambda x: .5*math.erfc(-x/math.sqrt(2.)), otypes=[float])

        def quadrature(order):
            z, w = np.polynomial.hermite.hermgauss(order)
            z, w = z*math.sqrt(2.), w/math.sqrt(math.pi)
            indices = np.indices((order,)*5).reshape(5, -1)
            inc = lower @ z[indices]
            weight = np.prod(w[indices], axis=0)
            l0 = float(np.interp(0., xs, lev[0]))
            unit = np.exp(inc[3]+.5*cov[3, 3]-.25*l0+math.sqrt(l0)*inc[0])
            vol = np.sqrt(np.interp(np.log(unit), xs, lev[1]))*np.exp(.3*(mixing[0]*inc[1]+mixing[1]*inc[4]))
            variance = .5*vol**2+2.*vol*cov[0, 3]+cov[3, 3]
            forward = 100.*math.exp(.015)*unit*np.exp(bh*inc[2]+shift-.5*cov[3, 3])
            discount = np.exp(-.025-inc[3]-bh*inc[2]-.5*vi+.5*cov[3, 3])
            d1 = (np.log(forward/100.)+.5*variance)/np.sqrt(variance)
            calls = discount*(forward*normal_cdf(d1)-100.*normal_cdf(d1-np.sqrt(variance)))
            return float(weight @ calls)

        low, high = quadrature(10), quadrature(14)
        self.assertLess(abs(high-low), .025)
        result = p.evaluate()
        self.assertAlmostEqual(result.value, high,
            delta=6.*result.standard_error+2.*abs(high-low)+.003)


if __name__ == "__main__":
    unittest.main()
