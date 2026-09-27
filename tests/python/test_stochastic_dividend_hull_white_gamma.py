"""HW Gamma: recompiled AAD Delta, paired diagnostics and a Gaussian reference."""
import math
import unittest
from datetime import date
import rust_pricing as rp
from test_stochastic_dividend_hull_white import compile_plan
from test_stochastic_dividend_hull_white_risk import request
from hw_dividend_reference import b, j, vi


def plan(spot=100., workers=1):
    product=rp.Product.arithmetic_asian(1,2,100.,1.,'call',[
        rp.AsianObservation.unknown('2027-03-05',.4),
        rp.AsianObservation.unknown('2027-09-04',.6)],'2027-12-04')
    return compile_plan(request(spot=spot,product=product),maximum_step=.125,
        rate_volatility_times=[0.,.8,1.15],rate_volatilities=[.04,.06,.09],
        worker_threads=workers)


class StochasticDividendHullWhiteGammaTest(unittest.TestCase):
    def test_recompiled_delta_ladder_identity_replay_and_metadata(self):
        p=plan();r=p.evaluate_gamma(gamma_absolute_bump=1.)
        self.assertIsInstance(r,rp.StochasticDividendGammaRisk)
        self.assertEqual(r.price.value,p.evaluate().value)
        self.assertEqual(r.price.standard_error,p.evaluate().standard_error)
        aad=p.evaluate_aad()
        self.assertEqual(r.delta,aad.delta)
        self.assertEqual(r.delta_standard_error,aad.standard_errors[0])
        for h,g in zip(r.spot_bumps,r.gamma_estimates):
            fd=(plan(100.+h).evaluate_aad().delta-plan(100.-h).evaluate_aad().delta)/(2*h)
            self.assertAlmostEqual(g,fd,delta=2e-12)
        for k in range(2):
            self.assertAlmostEqual(r.bump_differences[k],
                r.gamma_estimates[k]-r.gamma_estimates[k+1],delta=2e-12)
        self.assertEqual(r.spot_bumps,[.5,1.,2.])
        self.assertEqual(r.gamma,r.gamma_estimates[1])
        self.assertEqual(r.standard_error,r.gamma_standard_errors[1])
        self.assertEqual(r.delta_change_per_one_percent_spot,.01*r.spot*r.gamma)
        self.assertEqual(r.payoff_evaluations,7*r.price.evaluated_paths)
        self.assertEqual(r.price.independent_sampling_units,8)
        self.assertEqual(r.method,'buehler-bs-hw-common-noise-aad-delta-gamma-v1')
        self.assertEqual(r.uncertainty_scope,'sampling_only_fixed_bump_grid_and_smoothing')
        self.assertTrue(all(math.isfinite(x) and x>=0 for x in
            r.gamma_standard_errors+r.bump_difference_standard_errors))
        replay=plan(workers=3).evaluate_gamma(gamma_absolute_bump=1.)
        for field in ['gamma_estimates','gamma_standard_errors','bump_differences',
                      'bump_difference_standard_errors','delta','delta_standard_error']:
            self.assertEqual(getattr(r,field),getattr(replay,field))
        self.assertNotEqual(r.risk_fingerprint,replay.risk_fingerprint)
        relative=p.evaluate_gamma(gamma_relative_bump=.01)
        self.assertEqual(r.gamma_estimates,relative.gamma_estimates)
        self.assertEqual(r.gamma_standard_errors,relative.gamma_standard_errors)
        self.assertNotEqual(r.risk_fingerprint,relative.risk_fingerprint)
        copy=r.gamma_estimates;copy[0]=1e99
        self.assertNotEqual(r.gamma_estimates[0],1e99)
        with self.assertRaises(AttributeError): r.gamma=0.

    def test_invalid_conventions_and_funding_leave_basic_risk_unchanged(self):
        p=plan();delta=p.evaluate_aad().delta
        for kwargs in [{},dict(gamma_absolute_bump=1.,gamma_relative_bump=.01),
                       dict(gamma_absolute_bump=0.),dict(gamma_absolute_bump=-1.),
                       dict(gamma_absolute_bump=math.inf),dict(gamma_relative_bump=math.nan)]:
            with self.assertRaises(rp.ValidationError): p.evaluate_gamma(**kwargs)
        for h in [1e-300,60.,1e308]:
            with self.assertRaises(rp.PricingError): p.evaluate_gamma(gamma_absolute_bump=h)
        self.assertEqual(delta,p.evaluate_aad().delta)
        singular=compile_plan(request(points=32),equity_dividend_correlation=1.,
            equity_rate_correlation=.2,dividend_rate_correlation=.2,rate_volatilities=[0.])
        self.assertTrue(math.isfinite(singular.evaluate_gamma(gamma_absolute_bump=1.).gamma))
        with self.assertRaises(rp.PricingError): singular.evaluate_correlation_aad()

    def test_no_cash_delayed_gamma_matches_independent_gaussian_delta_bumps(self):
        T=1.;U=(date(2027,12,4)-date(2026,9,4)).days/365
        a,eta,sigma,rho=.4,.04,.2,.25
        variance=vi(a,eta,T)+sigma*sigma*T+2*sigma*rho*eta*j(a,T)
        root=math.sqrt(variance)
        forward_factor=.98/.95*math.exp(-b(a,U-T)*(
            .5*eta*eta*b(a,T)**2+sigma*rho*eta*b(a,T)))
        cdf=lambda z:.5*math.erfc(-z/math.sqrt(2))
        delta=lambda spot:.95**U*forward_factor*cdf(
            math.log(spot*forward_factor/100.)/root+.5*root)
        req=rp.PricingRequest('2026-09-04',rp.Product.arithmetic_asian(1,2,100.,1.,'call',
            [rp.AsianObservation.unknown('2027-09-04',1.)],'2027-12-04'),
            rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.],[1.,.95]),
                rp.DiscountCurve(11,[0.,1.],[1.,.98])),rp.Model.black_scholes(sigma),
            rp.Engine.randomized_quasi_monte_carlo(8192,1801,scramble_count=8,
                antithetic=True,brownian_bridge=True),rp.RiskRequest())
        r=compile_plan(req).evaluate_gamma(gamma_absolute_bump=1.)
        for h,g,se in zip(r.spot_bumps,r.gamma_estimates,r.gamma_standard_errors):
            expected=(delta(100.+h)-delta(100.-h))/(2*h)
            self.assertLess(abs(g-expected),6*se+3e-5,(h,g,expected,se))


if __name__=='__main__': unittest.main()
