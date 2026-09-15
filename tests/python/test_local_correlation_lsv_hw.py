"""Joint particle calibration: quote transpose and marginal-law acceptance."""
import math
import unittest
import numpy as np
import rust_pricing as rp
from test_local_correlation import correlation


def paired(sigma, quotes=None):
    return rp.HullWhiteLsvTarget.from_market_iv([.5, 1.], [-.5, 0., .5],
        [sigma]*6 if quotes is None else quotes, [0., .5, 1.], [-.35, 0., .35])


def build(mode='two', *, basket_quotes=None, asset_quotes=None, workers=1,
          product=None, engine=None, trace=True, rate_vol=.012, **kwargs):
    shared = dict(vol_of_vol=.4, particle_count=512, calibration_seed=318,
        log_bandwidth=.8, minimum_effective_samples=2., retain_reverse_trace=trace)
    if mode == 'two':
        cfg = rp.MultiAssetLsv2FactorConfig(mean_reversions=[.4, 2.], mixing_weight=.35,
            spot_correlations=[-.35, -.2], factor_correlation=.25, **shared)
    elif mode == 'one':
        cfg = rp.MultiAssetLsvConfig(mean_reversion=.7, correlation=-.35, **shared)
    elif mode == 'rough':
        cfg = rp.MultiAssetRoughLsvConfig(hurst=.12, correlation=-.35, **shared)
    else:
        cfg = None
    asset, basket = paired(.28, asset_quotes), paired(.235, basket_quotes)
    local = rp.LocalCorrelationConfig(basket_weights=[.6, .4], target_model=basket.model,
        second_correlations=correlation(.95), particle_count=512, calibration_seed=8401,
        log_bandwidth=.7, minimum_effective_samples=2., feasibility='project_and_report',
        retain_reverse_trace=True, hull_white_target=basket)
    curve = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.05)])
    markets = [rp.Market.equity(1, i+1, s, curve,
        rp.DiscountCurve(100+i, [0., 2.], [1., math.exp(-.02*(i+1))]))
        for i, s in enumerate((100., 90.))]
    product = product or rp.MultiAssetProduct.basket([1, 2], [.6, .4], [1., 1.],
        'call', 96., '2027-01-01', '2027-02-01', smoothing_half_width=3.)
    options = dict(maximum_step=.25, worker_threads=workers, reduction_block_size=128,
        lsv_configs=[cfg, None], rate_model=rp.HullWhiteModel(.13, [0., .37], [rate_vol, rate_vol*1.2]),
        rate_correlations=[.1, -.05] + [.02]*(2 if mode == 'two' else int(cfg is not None)),
        lsv_targets=[asset if cfg else None, None], local_correlation=local)
    options.update(kwargs)
    return rp.MultiAssetPlan.compile('2026-01-01', product, markets,
        [asset.model if cfg else rp.Model.black_scholes(.28), rp.Model.black_scholes(.3)],
        correlation(-.3), engine or rp.Engine.randomized_quasi_monte_carlo(
            128, 702, scramble_count=8, antithetic=True, brownian_bridge=True), **options)


class JointLocalCorrelationTests(unittest.TestCase):
    def test_joint_market_quote_transpose_and_replay(self):
        for mode in ('one', 'two', 'rough'):
            with self.subTest(mode=mode):
                plan = build(mode)
                result = plan.evaluate_aad(gamma_relative_bump=.001)
                risk = result.local_correlation_risk
                replay = build(mode, workers=3).evaluate_aad(gamma_relative_bump=.001)
                self.assertEqual(result.value, plan.evaluate().value)
                self.assertEqual(result.fingerprint, replay.fingerprint)
                self.assertEqual(risk.asset_adjoints, replay.local_correlation_risk.asset_adjoints)
                self.assertEqual(risk.basket_time_nodes, plan.time_nodes)
                self.assertIsNone(result.risks[0].lsv_local_variance)
                self.assertEqual(risk.asset_hull_white[1], None)
                for label, hw, sigma in [('basket', risk.basket_hull_white, .235),
                                          ('asset', risk.asset_hull_white[0], .28)]:
                    values = []
                    for h in [1e-7, -1e-7]:
                        quotes = [sigma]*6
                        quotes[1] += h
                        values.append(build(mode, **{label+'_quotes': quotes}).evaluate().value)
                    self.assertAlmostEqual(hw.vega_kt_raw[1], (values[0]-values[1])/2e-7, delta=8e-5)
                    self.assertEqual(hw.forward_log_density_adjoints[:3], [0.]*3)
                    self.assertEqual(hw.vega_kt_market_scaled, [.01*v for v in hw.vega_kt_raw])
                    self.assertAlmostEqual(hw.parallel_vega, sum(hw.vega_kt_raw), places=11)
                    self.assertTrue(all(math.isfinite(v) and v >= 0 for v in hw.vega_kt_standard_errors))
                cal = plan.local_correlation_calibration
                self.assertTrue(any(abs(v)>1e-8 for v in cal.rate_corrections))
                self.assertEqual(cal.rate_corrections[:3], [0.]*3)
                for t in (0., .5, 1.):
                    matrix = np.array(cal.driver_correlation_at(t, 0.))
                    self.assertGreater(np.linalg.eigvalsh(matrix).min(), -1e-12)
                    self.assertAlmostEqual(matrix[0, 2], -.35)
                    self.assertAlmostEqual(matrix[0, -1], .1)
                    self.assertAlmostEqual(matrix[1, -1], -.05)
                with self.assertRaises(rp.PricingError):
                    build(mode, trace=False).evaluate_aad()
                with self.assertRaises(rp.ValidationError):
                    build(mode, lsv_targets=[None, None])
                with self.assertRaises(AttributeError):
                    risk.basket_hull_white = None
        pseudo = build(engine=rp.Engine.pseudo_monte_carlo(256, 702, antithetic=True)).evaluate_aad().local_correlation_risk
        self.assertIsNone(pseudo.basket_hull_white.vega_kt_standard_errors)
        self.assertIsNone(pseudo.asset_hull_white[0].density_standard_errors)

    def test_bs_hw_marginal_matches_independent_bond_numeraire_formula(self):
        product = rp.MultiAssetProduct.basket([1, 2], [1., 0.], [1., 1.],
            'call', 100., '2027-01-01', '2027-01-01')
        plan = build('bs', product=product,
            engine=rp.Engine.randomized_quasi_monte_carlo(4096, 916, scramble_count=16,
                antithetic=True, brownian_bridge=True))
        result = plan.evaluate()
        # Independent time integration of HW's integrated-rate kernel.
        u = (np.arange(200000)+.5)/200000
        vol = np.where(u < .37, .012, .0144)
        kernel = vol*(-np.expm1(-.13*(1-u)))/.13
        variance = .28**2 + np.mean(kernel**2) + 2*.28*.1*np.mean(kernel)
        sigma = math.sqrt(variance)
        forward = 100*math.exp(.025-.01)
        d1 = math.log(forward/100)/sigma+.5*sigma
        cdf = lambda x: .5*math.erfc(-x/math.sqrt(2))
        expected = math.exp(-.025)*(forward*cdf(d1)-100*cdf(d1-sigma))
        self.assertAlmostEqual(result.value, expected, delta=6*result.standard_error+.006)
        self.assertEqual(plan.random_factor_count, 8)
        zero = build('bs', rate_vol=0.).local_correlation_calibration
        self.assertEqual(zero.rate_corrections, [0.]*len(zero.rate_corrections))
