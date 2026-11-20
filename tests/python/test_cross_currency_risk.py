"""Independent bond, share, Black and path-pair checks for settlement-currency risk."""
import math
import unittest

import rust_pricing as rp

from test_cross_currency_basket import TODAY, PAY, IDS, curve, years
from test_cross_currency_mixed_basket import C, QUOTES, USD, EUR, JPY


class CrossCurrencyRiskTests(unittest.TestCase):
    def setUp(self):
        self.jpy = curve(JPY,.012)
        self.markets = [rp.Market.equity(USD,10,100.,curve(USD,.05),curve(10,.016)),
                        rp.Market.equity(EUR,20,80.,curve(EUR,.031),curve(11,.023))]
        self.base = dict(settlement_currency_id=JPY, equity_models=[rp.Model.black_scholes(.2),
                         rp.Model.black_scholes(.28)], fx_quotes=QUOTES, correlations=C,
                         rates=[rp.HullWhiteModel(.2,[0.],[v]) for v in [.012,.018,.025]], maximum_step=.25)
        self.engine = rp.Engine.randomized_quasi_monte_carlo(1024,231,scramble_count=8,
                                                            antithetic=True,brownian_bridge=True)

    def product(self, ccy, *, equity=False, day=PAY):
        b = rp.SourceGraphBuilder()
        v = b.spot(10,day) if equity else b.literal(9.)
        return b.finish([v],[day],currency_id=ccy,underlying_ids=IDS)

    def basket(self, products, **settings):
        return rp.CrossCurrencyEquityBasket.from_legs(TODAY,products,self.markets,self.jpy,
                                                      **(self.base|settings))

    def test_currency_bond_fx_delta_foreign_and_domestic_rate_sensitivities(self):
        opt = self.basket([self.product(USD),self.product(JPY)])
        t, h = years(PAY), .0001
        usd = 9.*150.*math.exp(-.05*t)
        jpy = 9.*math.exp(-.012*t)
        for factor, identifier, bump, ref in [
            ("fx_spot",USD,.1,usd/150.),("discount_rate",USD,h,-usd*math.sinh(h*t)/h),
            ("discount_rate",JPY,h,-jpy*math.sinh(h*t)/h),("fx_spot",EUR,.1,0.)]:
            with self.subTest(factor=factor, identifier=identifier):
                v = opt.sensitivity(factor,identifier,bump,self.engine)
                self.assertAlmostEqual(v.value,ref,delta=6*v.standard_error+1.e-7)

    def test_composite_equity_delta_dividend_and_model_vega(self):
        opt = self.basket([self.product(JPY,equity=True)],performance="composite")
        t = years(PAY)
        reference = 100.*150.*math.exp(-.016*t)
        for factor,i,h,ref in [("equity_spot",10,.1,reference/100.),
                               ("dividend_yield",10,.0001,-reference*math.sinh(.0001*t)/.0001),
                               ("equity_volatility",10,.003,0.),("fx_volatility",USD,.003,0.)]:
            with self.subTest(factor=factor):
                v = opt.sensitivity(factor,i,h,self.engine)
                self.assertAlmostEqual(v.value,ref,delta=6*v.standard_error+1.e-6)

    def test_quanto_vegas_include_drift_adjustment(self):
        opt = self.basket([self.product(JPY,equity=True)],rates=None,
                          correlations=[row[:4] for row in C[:4]])
        rho, sigma, vol = C[0][2], .2, QUOTES[0][2]
        t = years(PAY)
        pv = 100.*math.exp((.05-.016-.012-rho*sigma*vol)*t)
        for factor, i, coef in [("fx_volatility",USD,-rho*sigma*t),("equity_volatility",10,-rho*vol*t)]:
            v = opt.sensitivity(factor,i,.005,self.engine)
            self.assertAlmostEqual(v.value,pv*math.sinh(.005*coef)/.005,delta=6*v.standard_error+1.e-8)

    def test_paired_scenario_errors_fixed_fx_and_boundary_inputs(self):
        opt = self.basket([self.product(USD)])
        no_change = opt.scenario_pnl(opt,self.engine)
        self.assertEqual((no_change.value,no_change.standard_error),(0.,0.))
        fixed = opt.with_leg_fx(0,fixed_fx_rate=130.)
        v = fixed.sensitivity("fx_spot",USD,.1,self.engine)
        self.assertEqual((v.value,v.standard_error),(0.,0.))
        immediate = self.basket([self.product(USD, day=TODAY)])
        exact = immediate.scenario_pnl(immediate.with_market_shift("fx_spot",USD,1.),self.engine)
        self.assertEqual((exact.value,exact.standard_error),(9.,0.))
        for factor, i, h in [("fx_spot",JPY,.1),("equity_spot",200,.1),
                             ("fx_volatility",USD,2.),("discount_rate",USD,0.)]:
            with self.assertRaises((ValueError,rp.PricingError)):
                opt.sensitivity(factor,i,h,self.engine)
        with self.assertRaises(rp.PricingError):
            opt.scenario_pnl(immediate,self.engine)

    def test_quadratic_basket_cashflow_has_exact_gamma(self):
        builder = rp.SourceGraphBuilder()
        spot = builder.spot(10,TODAY)
        graph = builder.finish([builder.multiply(spot,spot)],[TODAY],currency_id=USD,underlying_ids=IDS)
        opt = self.basket([graph])
        result = opt.sensitivity("equity_spot",10,1.,self.engine,order=2)
        self.assertEqual((result.value,result.standard_error),(300.,0.))
        with self.assertRaises(ValueError):
            opt.sensitivity("equity_spot",10,1.,self.engine,order=3)

    def native(self, product, mode="floating_fx"):
        return rp.CrossCurrencyEquityOption(TODAY,product,self.markets[0],self.jpy,
            settlement=mode,equity_volatility=.2,fx_spot=150.,fx_volatility=.16,correlation=-.32,
            fixed_fx_rate=145. if mode=="quanto" else None)

    def test_native_vanilla_gamma_and_fx_delta_against_black(self):
        t = years(PAY)
        opt = self.native(rp.Product.european_vanilla(10,JPY,PAY,100.,1.,"call"))
        d1 = (.05-.016+.5*.2**2)*math.sqrt(t)/.2
        gamma = 150.*math.exp(-.016*t-d1*d1/2)/(100.*.2*math.sqrt(2*math.pi*t))
        result = opt.sensitivity("equity_spot",10,.02,self.engine,order=2)
        self.assertAlmostEqual(result.value,gamma,delta=gamma*1.e-6)
        self.assertEqual(result.standard_error,0.)
        delta = opt.sensitivity("fx_spot",USD,.1,self.engine)
        self.assertAlmostEqual(delta.value,opt.analytical_price()/150.,places=8)
        self.assertEqual(opt.scenario_pnl(opt,self.engine).standard_error,0.)

    def test_independent_paired_bermudan_policies_and_continuous_barrier_risk(self):
        bermudan = rp.Product.american_vanilla(10,JPY,PAY,110.,1.,"put",["2026-07-01",PAY])
        lsm = rp.LsmConfig(rp.Engine.pseudo_monte_carlo(487,256,antithetic=True))
        barrier = rp.Product.barrier(10,JPY,PAY,100.,130.,1.,"call","up","knock_out","continuous",
                                    [TODAY,PAY],PAY)
        mc = rp.Engine.pseudo_monte_carlo(391,256,antithetic=True)
        for p, training in [(bermudan,lsm),(barrier,None)]:
            opt = self.native(p)
            delta = opt.sensitivity("fx_spot",USD,.5,mc,lsm=training,replications=4)
            self.assertGreater(delta.value,0.)
            self.assertGreater(delta.standard_error,0.)
            pnl = opt.scenario_pnl(opt.with_market_shift("fx_spot",USD,1.),mc,
                                   lsm=training,replications=4)
            self.assertAlmostEqual(pnl.value,delta.value,places=9)
            self.assertAlmostEqual(pnl.standard_error,delta.standard_error,places=9)
            matched = opt.scenario_pnl(opt,mc,lsm=training,replications=3)
            self.assertEqual((matched.value,matched.standard_error),(0.,0.))
            with self.assertRaises(rp.PricingError):
                opt.scenario_pnl(opt,mc,lsm=training,replications=1)
