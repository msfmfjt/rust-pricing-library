"""HW basic AAD: full recompiles, boundaries and an independent Gaussian risk law."""
import math
import unittest
import rust_pricing as rp
from test_stochastic_dividend_hull_white import compile_plan
from hw_dividend_reference import b, j, vi


def request(spot=100., sigma=.2, cash=8., log_p=0., log_q=0., points=128, product=None):
    return rp.PricingRequest('2026-09-04', product or
        rp.Product.european_vanilla(1, 2, '2027-09-04', 100., 1., 'call'),
        rp.Market.equity(2, 1, spot,
            rp.DiscountCurve(10, [0., 1.], [1., .95*math.exp(log_p)]),
            rp.DiscountCurve(11, [0., 1.], [1., .98*math.exp(log_q)]),
            discrete_dividends=[rp.DividendEvent.fixed_cash(1,.5,4.),
                               rp.DividendEvent.fixed_cash(2,1.4,cash)]),
        rp.Model.black_scholes(sigma), rp.Engine.randomized_quasi_monte_carlo(
            points,1859,scramble_count=8,antithetic=True,brownian_bridge=True),rp.RiskRequest())


def plan(**kwargs):
    constructor={key: kwargs.pop(key) for key in list(kwargs) if key in
                 ['dividend_mean_reversion','equity_linkage','dividend_volatility','worker_threads']}
    return compile_plan(request(**kwargs), maximum_step=.125,
        rate_volatility_times=[0.,.8,1.15],rate_volatilities=[.04,.06,.09],**constructor)


class StochasticDividendHullWhiteRiskTest(unittest.TestCase):
    def test_basic_risk_full_recompile_two_widths(self):
        risk=plan().evaluate_aad()
        for key,value,label in [
            ('spot',100.,'spot'),('sigma',.2,'initial_volatility'),
            ('cash',8.,'cash_mean[2]'),('log_p',0.,'discount_log_df[1]'),
            ('log_q',0.,'repo_spread_log_df[1]'),
            ('dividend_mean_reversion',.7,'dividend_mean_reversion'),
            ('equity_linkage',.6,'equity_linkage'),
            ('dividend_volatility',.35,'dividend_volatility')]:
            aad=risk.derivatives[risk.parameter_labels.index(label)]
            for h in [1e-5,1e-6]:
                fd=(plan(**{key:value+h}).evaluate().value-plan(**{key:value-h}).evaluate().value)/(2*h)
                self.assertLessEqual(abs(aad-fd),3e-5+2e-5*max(abs(aad),abs(fd)),(key,h,aad,fd))

    def test_result_identity_immutability_worker_replay_and_risk_scope(self):
        p=plan();r=p.evaluate_aad();price=p.evaluate()
        self.assertIsInstance(r,rp.StochasticDividendAadRisk)
        self.assertEqual(r.price.value,price.value)
        self.assertEqual(r.price.standard_error,price.standard_error)
        self.assertEqual(r.price.plan_fingerprint,p.plan_fingerprint)
        self.assertEqual(r.price.independent_sampling_units,8)
        self.assertEqual(r.method,'buehler-bs-hw-cash-payoff-reverse-fixed-rates-correlation-v1')
        self.assertEqual(r.cash_times,[.5,1.4])
        self.assertEqual(r.discount_times,[0.,1.]);self.assertEqual(r.repo_spread_times,[0.,1.])
        self.assertEqual(r.cash_mean_adjoints,r.derivatives[5:7])
        self.assertEqual(r.discount_node_dv01,[0.,-1e-4*r.derivatives[8]])
        self.assertEqual(r.repo_spread_node_dv01,[0.,-1e-4*r.derivatives[10]])
        self.assertEqual(r.initial_volatility_vega_per_vol_point,.01*r.derivatives[1])
        self.assertTrue(all(math.isfinite(x) and x>=0 for x in r.standard_errors))
        self.assertTrue(hasattr(p,'evaluate_hull_white_aad'))
        other=plan(worker_threads=3).evaluate_aad()
        self.assertEqual(r.derivatives,other.derivatives);self.assertEqual(r.standard_errors,other.standard_errors)
        self.assertNotEqual(r.price.plan_fingerprint,other.price.plan_fingerprint)
        copy=r.derivatives;copy[0]=1e9;self.assertNotEqual(r.delta,1e9)
        with self.assertRaises(AttributeError):r.delta=0.
        for name in ['evaluate_gamma','evaluate_rough_aad','evaluate_correlation_aad']:
            self.assertFalse(hasattr(p,name))

    def test_hull_white_parameter_aad_appends_piecewise_rate_risk(self):
        request=rp.PricingRequest('2026-09-04',
            rp.Product.european_vanilla(1,2,'2027-09-04',20.,1.,'call'),
            rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.],[1.,.95]),
                rp.DiscountCurve(11,[0.,1.],[1.,.98]),
                discrete_dividends=[rp.DividendEvent.fixed_cash(1,.5,4.),
                                    rp.DividendEvent.fixed_cash(2,1.4,8.)]),
            rp.Model.black_scholes(.2),
            rp.Engine.randomized_quasi_monte_carlo(128,2207,scramble_count=4,
                antithetic=True,brownian_bridge=True),rp.RiskRequest())
        p=compile_plan(request,maximum_step=.125,rate_volatility_times=[0.,.8,1.15],
            rate_volatilities=[.04,.06,.09])
        risk=p.evaluate_hull_white_aad()
        self.assertEqual(risk.price.value,p.evaluate().value)
        self.assertEqual(risk.price.standard_error,p.evaluate().standard_error)
        self.assertEqual(risk.parameter_labels[-4:],['rate_mean_reversion',
            'rate_volatility[0]','rate_volatility[1]','rate_volatility[2]'])
        self.assertEqual(risk.method,'buehler-bs-hw-cash-payoff-forward-rate-parameter-adjoint-v1')
        self.assertTrue(all(math.isfinite(x) for x in risk.derivatives[-4:]))

    def test_zero_cash_and_zero_diffusion_boundaries(self):
        product=rp.Product.european_vanilla(1,2,'2027-09-04',20.,1.,'call')
        args=dict(sigma=0.,cash=0.,equity_linkage=0.,dividend_volatility=0.,dividend_mean_reversion=0.,product=product)
        r=plan(**args).evaluate_aad()
        for key,label in [('sigma','initial_volatility'),('cash','cash_mean[2]'),
                          ('equity_linkage','equity_linkage'),('dividend_volatility','dividend_volatility'),
                          ('dividend_mean_reversion','dividend_mean_reversion')]:
            h=1e-7;fd=(plan(**(args|{key:h})).evaluate().value-r.price.value)/h
            self.assertAlmostEqual(fd,r.derivatives[r.parameter_labels.index(label)],delta=3e-4)

    def test_no_cash_delayed_payment_delta_vega_against_gaussian_black(self):
        # Independent payment-measure Gaussian law, not bumped production prices.
        from datetime import date
        T=1.;U=(date(2027,12,4)-date(2026,9,4)).days/365
        a,sr,sf,rho=.4,.04,.2,.25
        variance=vi(a,sr,T)+sf*sf*T+2*sf*rho*sr*j(a,T)
        duration=b(a,U-T)
        F=100*.98/.95*math.exp(-duration*(.5*sr*sr*b(a,T)**2+sf*rho*sr*b(a,T)))
        root=math.sqrt(variance);d1=math.log(F/100)/root+.5*root
        cdf=lambda x:.5*math.erfc(-x/math.sqrt(2))
        density=math.exp(-.5*d1*d1)/math.sqrt(2*math.pi)
        delta=.95**U*F/100*cdf(d1)
        vega=.95**U*F*(-duration*rho*sr*b(a,T)*cdf(d1)+density*(sf*T+rho*sr*j(a,T))/root)
        req=rp.PricingRequest('2026-09-04',rp.Product.arithmetic_asian(1,2,100.,1.,'call',
            [rp.AsianObservation.unknown('2027-09-04',1.)],'2027-12-04'),
            rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.],[1.,.95]),rp.DiscountCurve(11,[0.,1.],[1.,.98])),
            rp.Model.black_scholes(sf),rp.Engine.randomized_quasi_monte_carlo(8192,1801,scramble_count=8,
                antithetic=True,brownian_bridge=True),rp.RiskRequest())
        p=compile_plan(req);r=p.evaluate_aad()
        self.assertEqual(p.time_nodes,[0.,1.])
        for idx,expected in [(0,delta),(1,vega)]:
            self.assertLess(abs(r.derivatives[idx]-expected),6*r.standard_errors[idx]+2e-5)


if __name__=='__main__':unittest.main()
