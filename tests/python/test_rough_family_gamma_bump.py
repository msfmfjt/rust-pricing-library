"""Explicit finite-bump Gamma, independent payoff/SE and public contract checks."""
import math
import unittest
import rust_pricing as rp
from test_rough_volatility_families import models
from test_rough_family_lsv_spot_delta import compile_plan, asian_product, asian_frozen_reference, mean_se


def pure(model, spot=100.0, product=None, smooth=None, qmc=True):
    product = product or asian_product()
    market = rp.Market.equity(2, 1, spot,
        rp.DiscountCurve(10,[0.,1.,2.],[1.,.95,.95**2]),
        rp.DiscountCurve(11,[0.,1.,2.],[1.,.98,.98**2]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1,.25,3.),
                           rp.DividendEvent.fixed_cash(2,1.5,4.)])
    engine = (rp.Engine.randomized_quasi_monte_carlo(64,819,scramble_count=4,
                antithetic=True,brownian_bridge=True) if qmc else
              rp.Engine.pseudo_monte_carlo(819,128,antithetic=True,brownian_bridge=False))
    req=rp.PricingRequest('2026-09-04',product,market,rp.Model.black_scholes(.2),engine,
                        rp.RiskRequest(payoff_smoothing_half_width=smooth))
    return rp.RoughVolatilityPlan.compile(req,model,maximum_step=.25,
                                         worker_threads=1,reduction_block_size=64)

class RoughGammaTest(unittest.TestCase):
    def test_asian_nonflat_carry_delayed_payment_full_recalibration(self):
        for model in models():
            for qmc in [False,True]:
                for lsv in [False,True]:
                    build=(lambda spot: compile_plan(model,spot,product=asian_product(),qmc=qmc)) if lsv else (lambda spot:pure(model,spot,qmc=qmc))
                    plan=build(100.)
                    g=plan.evaluate_sticky_moneyness_gamma_bump(1.) if lsv else plan.evaluate_gamma_bump(1.)
                    self.assertEqual(g.price.value,plan.evaluate().value)
                    for h,gamma in [(1.,g.gamma),(.5,g.half_bump_gamma)]:
                        expected=(build(100+h).evaluate().value+build(100-h).evaluate().value-2*g.price.value)/h**2
                        self.assertAlmostEqual(gamma,expected,delta=2e-9)
                    self.assertEqual(g.payoff_evaluations,5*g.price.evaluated_paths)
                    self.assertAlmostEqual(g.bump_difference,g.half_bump_gamma-g.gamma,delta=1e-13)
                    for field in ['gamma','half_bump_gamma','bump_difference','spot_bump','risk_fingerprint']:
                        with self.assertRaises(AttributeError):setattr(g,field,0)

    def test_frozen_asian_independent_paths_and_paired_standard_errors(self):
        for model in models():
            p=compile_plan(model,product=asian_product(),qmc=False,flat=True)
            g=p.evaluate_frozen_leverage_gamma_bump(1.)
            # Independent recurrence/payoff; public variance history and RNG are shared.
            values=[asian_frozen_reference(p,model,h) for h in [0.,1.,-1.,.5,-.5]]
            # Parent helper returns one value per antithetic sampling unit.
            full=[u+d-2*b for b,u,d in zip(values[0],values[1],values[2])]
            half=[4*(u+d-2*b) for b,u,d in zip(values[0],values[3],values[4])]
            for x,m,s in [(full,g.gamma,g.gamma_standard_error),(half,g.half_bump_gamma,g.half_bump_standard_error),
                          ([b-a for a,b in zip(full,half)],g.bump_difference,g.bump_difference_standard_error)]:
                mean,se=mean_se(x)
                self.assertAlmostEqual(m,mean,delta=2e-9)
                self.assertAlmostEqual(s,se,delta=2e-9)

    def test_explicit_width_and_unsupported_hard_risk(self):
        model=models()[3]
        digital=rp.Product.digital(1,2,'2027-09-04',100.,1.,'call','cash')
        for p,method in [(pure(model),'evaluate_gamma_bump'),(compile_plan(model),'evaluate_sticky_moneyness_gamma_bump')]:
            for h in [0.,-1.,float('nan'),float('inf'),1e-8,100.]:
                with self.assertRaises(rp.PricingError):getattr(p,method)(h)
            with self.assertRaises(TypeError):getattr(p,method)()
            a=getattr(p,method)(1.);b=getattr(p,method)(.5)
            self.assertNotEqual(a.risk_fingerprint,b.risk_fingerprint)
            self.assertAlmostEqual(a.half_bump_gamma,b.gamma,delta=1e-13)
        for smooth in [None,2.]:
            p=pure(model,product=digital,smooth=smooth)
            l=compile_plan(model,product=digital,smooth=smooth)
            if smooth is None:
                with self.assertRaises(rp.PricingError):p.evaluate_gamma_bump(1.)
                with self.assertRaises(rp.PricingError):l.evaluate_sticky_moneyness_gamma_bump(1.)
            else:
                self.assertTrue(math.isfinite(p.evaluate_gamma_bump(1.).gamma))
                self.assertTrue(math.isfinite(l.evaluate_sticky_moneyness_gamma_bump(1.).gamma))

    def test_independent_black_reference_is_not_a_pathwise_zero_gamma(self):
        N=lambda x:math.erfc(-x/math.sqrt(2))/2
        def price(s):
            d1=math.log(s/100)/.2+.1
            return s*N(d1)-100*N(d1-.2)
        gamma=math.exp(-.005)/math.sqrt(2*math.pi)/20
        fd=[(price(100+h)+price(100-h)-2*price(100))/h**2 for h in [1.,.5]]
        self.assertAlmostEqual(gamma,.01984762737385059,delta=1e-16)
        self.assertAlmostEqual(fd[0],.019844113046403322,delta=2e-12)
        self.assertAlmostEqual(fd[1],.019846748725001362,delta=2e-12)
        self.assertLess(abs(fd[1]-gamma),abs(fd[0]-gamma))

if __name__=='__main__':unittest.main()
