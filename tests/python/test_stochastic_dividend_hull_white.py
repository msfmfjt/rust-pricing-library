"""HW cash means, independent Gaussian option prices, payment and API limits."""
import math
import unittest
from datetime import date
import rust_pricing as rp
from test_stochastic_dividends import make_request
from hw_dividend_reference import conditional_price, b, j, vi


def compile_plan(request=None, **kwargs):
    return rp.StochasticDividendHullWhitePlan.compile_bs(
        request or make_request(dividends=((1.,3.),(1.4,8.)),points=4096),
        **(dict(dividend_mean_reversion=.7,equity_linkage=.6,dividend_volatility=.35,
                equity_dividend_correlation=-.25,rate_mean_reversion=.4,
                rate_volatility_times=[0.],rate_volatilities=[.04],
                equity_rate_correlation=.25,dividend_rate_correlation=-.2,
                maximum_step=1.,worker_threads=1,reduction_block_size=64)|kwargs))


class StochasticDividendHullWhiteTest(unittest.TestCase):
    def test_independent_price_oracle_and_cash_forward_metadata(self):
        for k in [0.,.7]:
            expected=conditional_price(32,k)
            self.assertAlmostEqual(expected,conditional_price(40,k),delta=2e-10)
            plan=compile_plan(dividend_mean_reversion=k)
            result=plan.evaluate()
            self.assertLess(abs(result.value-expected),6*result.standard_error+.002)
            self.assertEqual(result.plan_fingerprint,plan.plan_fingerprint)
            self.assertEqual(result.scheme,'buehler-bs-hw-conditional-cash-v1')
            self.assertEqual(plan.random_factor_count,4)
            self.assertEqual(result.independent_sampling_units,8)
            self.assertEqual(plan.cash_times,[1.,1.4])
            if k==0:
                for i,(t,mean) in enumerate([(1.,3.),(1.4,8.)]):
                    forward=mean*math.exp(-.35*(-.2)*.04*j(.4,t))
                    self.assertAlmostEqual(plan.initial_dividend_forwards[i],forward,delta=3e-13)
                    self.assertAlmostEqual(plan.initial_dividend_claim_values[i],.95**t*forward,delta=3e-13)

    def test_worker_replay_copy_immutability_and_future_rate_knot(self):
        request=make_request(dividends=((.5,4.),(1.4,8.)),points=128)
        a=compile_plan(request,maximum_step=.125)
        result=a.evaluate()
        other=compile_plan(request,maximum_step=.125,worker_threads=3).evaluate()
        self.assertEqual(result.value,other.value)
        self.assertEqual(result.standard_error,other.standard_error)
        self.assertNotEqual(result.plan_fingerprint,other.plan_fingerprint)
        p=compile_plan(request,maximum_step=.125,rate_volatility_times=[0.,1.1],rate_volatilities=[.04,.09])
        self.assertNotEqual(p.initial_dividend_claim_values,a.initial_dividend_claim_values)
        self.assertNotEqual(p.plan_fingerprint,a.plan_fingerprint)
        copy=a.initial_dividend_claim_values
        copy[0]=-100
        self.assertGreater(a.initial_dividend_claim_values[0],0)
        with self.assertRaises(AttributeError): a.risky_spot=0
        self.assertTrue(hasattr(a, 'evaluate_aad'))
        for name in ['evaluate_rough_aad','evaluate_correlation_aad','evaluate_gamma']:
            self.assertFalse(hasattr(a,name))

    def test_deterministic_fixed_cash_limit_and_explicit_validation(self):
        request=make_request(sigma=0.,dividends=((1.,10.),),strike=80.,points=32)
        result=compile_plan(request,rate_volatilities=[0.],equity_linkage=0.,dividend_volatility=0.).evaluate()
        self.assertAlmostEqual(result.value,.95*(100*.98/.95-10-80),delta=3e-12)
        self.assertAlmostEqual(result.standard_error,0.,delta=1e-13)
        for kwargs in [dict(rate_mean_reversion=-1),dict(rate_volatilities=[math.nan]),
                       dict(rate_volatility_times=[.1]),dict(dividend_mean_reversion=-1)]:
            with self.assertRaises(rp.ValidationError): compile_plan(**kwargs)
        for kwargs in [dict(equity_rate_correlation=.99,dividend_rate_correlation=.99),
                       dict(equity_rate_correlation=math.nan),dict(maximum_step=0.)]:
            with self.assertRaises(rp.PricingError): compile_plan(**kwargs)
        with self.assertRaises(rp.PricingError):
            compile_plan(make_request(risk=rp.RiskRequest(delta=True)))

    def test_delayed_one_fixing_asian_matches_independent_gaussian_black(self):
        # No cash: closed-form Q^payment lognormal law tests BOTH the rate
        # covariance in the stock and the conditional payment-date discount.
        T=1.;U=(date(2027,12,4)-date(2026,9,4)).days/365
        a,sr,sf,rho=.4,.04,.2,.25
        integral_var=vi(a,sr,T)
        variance=integral_var+sf*sf*T+2*sf*rho*sr*j(a,T)
        mean_forward=100*.98/.95*math.exp(-b(a,U-T)*(.5*sr*sr*b(a,T)**2+sf*rho*sr*b(a,T)))
        root=math.sqrt(variance)
        d1=math.log(mean_forward/100)/root+.5*root
        cdf=lambda x:.5*math.erfc(-x/math.sqrt(2))
        expected=.95**U*(mean_forward*cdf(d1)-100*cdf(d1-root))
        request=rp.PricingRequest('2026-09-04',rp.Product.arithmetic_asian(1,2,100.,1.,'call',
            [rp.AsianObservation.unknown('2027-09-04',1.)],'2027-12-04'),
            rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.],[1.,.95]),
                rp.DiscountCurve(11,[0.,1.],[1.,.98])),rp.Model.black_scholes(sf),
            rp.Engine.randomized_quasi_monte_carlo(4096,1451,scramble_count=8,
                antithetic=True,brownian_bridge=True),rp.RiskRequest())
        plan=compile_plan(request)
        self.assertEqual(plan.time_nodes,[0.,1.])  # no artificial post-expiry simulation
        result=plan.evaluate()
        self.assertLess(abs(result.value-expected),6*result.standard_error+.002)

if __name__=='__main__':
    unittest.main()
