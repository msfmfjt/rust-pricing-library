"""Finite-hybrid risk at fixed correlations; full-recompile FD is independent."""
import math
import unittest
import rust_pricing as rp
from test_stochastic_dividends import make_request, compile_plan as bs
from test_rough_dividends import compile_plan


class RoughDividendRiskTest(unittest.TestCase):
    def test_hurst_eta_full_recompile_and_exact_basic_prefix(self):
        request = make_request(points=64, dividends=((.5,4.),(1.,3.),(1.4,8.)))
        for h in (.1,.49):
            plan = compile_plan(request, hurst=h, maximum_step=.125)
            basic, risk = plan.evaluate_aad(), plan.evaluate_rough_aad()
            n = len(basic.derivatives)
            self.assertEqual(risk.parameter_labels[:n],basic.parameter_labels)
            self.assertEqual(risk.derivatives[:n],basic.derivatives)
            self.assertEqual(risk.standard_errors[:n],basic.standard_errors)
            self.assertEqual(risk.price.value,plan.evaluate().value)
            self.assertEqual(risk.price.standard_error,basic.price.standard_error)
            self.assertEqual(risk.cash_mean_adjoints,basic.cash_mean_adjoints)
            self.assertEqual(risk.discount_node_dv01,basic.discount_node_dv01)
            self.assertEqual(risk.repo_spread_node_dv01,basic.repo_spread_node_dv01)
            self.assertEqual(risk.parameter_labels[n:],['rough_hurst','rough_vol_of_vol'])
            for j,(key,value) in enumerate((('hurst',h),('vol_of_vol',.6))):
                for eps in (1e-5,1e-6):
                    settings = dict(hurst=h,maximum_step=.125)
                    plus=compile_plan(request,**(settings | {key:value+eps})).evaluate().value
                    minus=compile_plan(request,**(settings | {key:value-eps})).evaluate().value
                    fd=(plus-minus)/(2*eps)
                    aad=risk.derivatives[n+j]
                    self.assertLessEqual(abs(aad-fd),3e-5+2e-5*max(abs(aad),abs(fd)))
            replay=compile_plan(request,hurst=h,maximum_step=.125,worker_threads=3).evaluate_rough_aad()
            self.assertEqual(risk.derivatives,replay.derivatives)
            self.assertEqual(risk.standard_errors,replay.standard_errors)
            with self.assertRaises(AttributeError):
                risk.method='changed'
            copy=risk.derivatives
            copy[-1]=math.inf
            self.assertTrue(math.isfinite(risk.derivatives[-1]))

    def test_zero_limits_and_remaining_unsupported_scopes(self):
        request=make_request(points=32,strike=60.)
        for h,eta in ((.5,.6),(.1,0.),(.5,0.)):
            p=compile_plan(request,hurst=h,vol_of_vol=eta,maximum_step=.125)
            r=p.evaluate_rough_aad()
            for j,(key,value) in enumerate((('hurst',h),('vol_of_vol',eta))):
                eps=-1e-7 if key=='hurst' and h==.5 else 1e-7
                settings=dict(hurst=h,vol_of_vol=eta,maximum_step=.125)
                bumped=compile_plan(request,**(settings | {key:value+eps})).evaluate().value
                self.assertLess(abs((bumped-r.price.value)/eps-r.derivatives[-2+j]),3e-4)
            with self.assertRaises(rp.PricingError): p.evaluate_bergomi_aad()
            with self.assertRaises(rp.PricingError): p.evaluate_correlation_aad()
            self.assertEqual(r.price.value,p.evaluate().value)
        p=compile_plan(make_request(sigma=0.,points=32),maximum_step=.125)
        r=p.evaluate_rough_aad()
        self.assertEqual(r.derivatives[-2:],[0.,0.])
        self.assertEqual(r.standard_errors[-2:],[0.,0.])
        with self.assertRaises(rp.PricingError): bs(request).evaluate_rough_aad()
        # Fixed singular correlations need no Cholesky derivative for H/eta.
        p=compile_plan(request,correlation=1.,equity_dividend_correlation=1.,
                       dividend_volatility_correlation=1.)
        p.evaluate_aad();p.evaluate_rough_aad();p.evaluate_gamma(gamma_absolute_bump=1.)

    def test_gamma_recompiles_physical_spot_and_pairs_errors(self):
        def request(spot):
            return rp.PricingRequest('2026-09-04',
                rp.Product.european_vanilla(1,2,'2027-09-04',100.,1.,'call'),
                rp.Market.equity(2,1,spot,rp.DiscountCurve(10,[0.,1.],[1.,.95]),
                    rp.DiscountCurve(11,[0.,1.],[1.,.98]),
                    discrete_dividends=[rp.DividendEvent.fixed_cash(1,1.,3.),
                                        rp.DividendEvent.fixed_cash(2,1.4,8.)]),
                rp.Model.black_scholes(.2),
                rp.Engine.randomized_quasi_monte_carlo(64,884,scramble_count=4,
                    antithetic=True,brownian_bridge=True),rp.RiskRequest())
        p=compile_plan(request(100.),maximum_step=.125)
        g=p.evaluate_gamma(gamma_relative_bump=.01)
        self.assertEqual(g.price.value,p.evaluate().value)
        self.assertEqual(g.delta,p.evaluate_aad().delta)
        self.assertEqual(g.spot_bumps,[.5,1.,2.])
        for bump,gamma in zip(g.spot_bumps,g.gamma_estimates):
            up=compile_plan(request(100.+bump),maximum_step=.125).evaluate_aad().delta
            down=compile_plan(request(100.-bump),maximum_step=.125).evaluate_aad().delta
            self.assertAlmostEqual(gamma,(up-down)/(2*bump),delta=2e-12)
        replay=compile_plan(request(100.),maximum_step=.125,worker_threads=3).evaluate_gamma(gamma_absolute_bump=1.)
        self.assertEqual(g.gamma_estimates,replay.gamma_estimates)
        self.assertEqual(g.gamma_standard_errors,replay.gamma_standard_errors)
        self.assertEqual(g.bump_difference_standard_errors,replay.bump_difference_standard_errors)
        self.assertTrue(all(math.isfinite(x) and x>=0 for x in g.gamma_standard_errors))
        with self.assertRaises(rp.PricingError): p.evaluate_gamma(gamma_absolute_bump=60.)


if __name__=='__main__': unittest.main()
