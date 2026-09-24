"""Explicit common-noise Gamma and independent recompiled Delta bumps."""
import math
import unittest
import rust_pricing as rp
from test_stochastic_dividends import compile_plan as bs
from test_bergomi_dividends import compile_plan as bergomi


def request(spot=100.):
    return rp.PricingRequest(
        "2026-09-04", rp.Product.european_vanilla(1, 2, "2027-09-04", 100., 1., "call"),
        rp.Market.equity(2, 1, spot, rp.DiscountCurve(10, [0.,1.], [1.,.95]),
            rp.DiscountCurve(11, [0.,1.], [1.,.98]),
            discrete_dividends=[rp.DividendEvent.fixed_cash(1,1.,3.),
                                rp.DividendEvent.fixed_cash(2,1.4,25.)]),
        rp.Model.black_scholes(.2), rp.Engine.randomized_quasi_monte_carlo(
            128, 612, scramble_count=8, antithetic=True, brownian_bridge=True), rp.RiskRequest())


def plan(family, spot=100., workers=1):
    settings=dict(maximum_step=.125, worker_threads=workers)
    return bs(request(spot), mean_reversion=.7, **settings) if family==0 else bergomi(family==2, request(spot), **settings)


class StochasticDividendGammaTest(unittest.TestCase):
    def test_full_recompile_delta_bumps_and_basic_results(self):
        for family in range(3):
            p=plan(family); r=p.evaluate_gamma(gamma_absolute_bump=1.)
            self.assertIsInstance(r, rp.StochasticDividendGammaRisk)
            self.assertEqual(r.price.value, p.evaluate().value)
            self.assertEqual(r.price.standard_error, p.evaluate().standard_error)
            basic=p.evaluate_aad()
            self.assertEqual(r.delta, basic.delta)
            self.assertEqual(r.delta_standard_error, basic.standard_errors[0])
            for h,g in zip(r.spot_bumps,r.gamma_estimates):
                fd=(plan(family,100.+h).evaluate_aad().delta-plan(family,100.-h).evaluate_aad().delta)/(2*h)
                self.assertAlmostEqual(g,fd,delta=2e-12)
            self.assertEqual(r.gamma,r.gamma_estimates[1])
            self.assertEqual(r.standard_error,r.gamma_standard_errors[1])
            self.assertEqual(r.delta_change_per_one_percent_spot,.01*r.spot*r.gamma)
            self.assertEqual(r.payoff_evaluations,7*r.price.evaluated_paths)
            self.assertEqual(r.price.independent_sampling_units,8)
            self.assertEqual(r.spot_bumps,[.5,1.,2.])
            self.assertEqual(r.method,'buehler-common-noise-aad-delta-gamma-v1')
            self.assertEqual(r.uncertainty_scope,'sampling_only_fixed_bump_grid_and_smoothing')
            self.assertTrue(all(math.isfinite(x) and x>=0 for x in r.gamma_standard_errors+r.bump_difference_standard_errors))

    def test_replay_copy_semantics_and_bump_identity(self):
        for family in range(3):
            a=plan(family).evaluate_gamma(gamma_relative_bump=.01)
            b=plan(family,workers=3).evaluate_gamma(gamma_relative_bump=.01)
            for field in ['gamma_estimates','gamma_standard_errors','bump_differences','bump_difference_standard_errors','risk_fingerprint','delta']:
                self.assertEqual(getattr(a,field),getattr(b,field))
            absolute=plan(family).evaluate_gamma(gamma_absolute_bump=1.)
            self.assertEqual(a.gamma_estimates,absolute.gamma_estimates)
            self.assertNotEqual(a.risk_fingerprint,absolute.risk_fingerprint)
            with self.assertRaises(AttributeError): a.gamma=0.
            copy=a.gamma_estimates;copy[0]=1e99
            self.assertNotEqual(a.gamma_estimates[0],1e99)

    def test_invalid_bump_conventions_and_funding_reject(self):
        p=plan(0)
        for args in [{},dict(gamma_absolute_bump=1.,gamma_relative_bump=.01),
                     dict(gamma_absolute_bump=0.),dict(gamma_absolute_bump=-1.),
                     dict(gamma_relative_bump=math.nan),dict(gamma_absolute_bump=math.inf)]:
            with self.assertRaises(rp.ValidationError): p.evaluate_gamma(**args)
        for h in [1e-300,60.,1e308]:
            with self.assertRaises(rp.PricingError): p.evaluate_gamma(gamma_absolute_bump=h)
        self.assertTrue(math.isfinite(p.evaluate_aad().delta))

if __name__=='__main__': unittest.main()
