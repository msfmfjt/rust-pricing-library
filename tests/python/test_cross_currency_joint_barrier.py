"""Continuous monitoring on stock/FX paths with domestic and foreign short rates."""
import math
import unittest

import rust_pricing as rp
import test_cross_currency_rates as rates
import test_cross_currency_dividends as dividends
from test_cross_currency_models import models
from test_cross_currency_rates import TODAY, MID, END, C


class JointBarrierTests(unittest.TestCase):
    def setUp(self):
        self.inputs = rates.CrossCurrencyRateTests()
        self.inputs.setUp()
        self.mc = rp.Engine.pseudo_monte_carlo(481, 1024, antithetic=True, brownian_bridge=True)

    def barrier(self, mode, style="knock_out", *, dates=(TODAY, END), hit=None, pay=END):
        x = 150. if mode == "composite" else 1.
        return rp.Product.barrier(10, 2, END, 98.*x, 123.*x, 1., "call", "up", style,
                                  "continuous", dates, pay, historical_hit=hit)

    def test_knock_in_out_parity_under_all_equity_models_and_both_rates(self):
        # Dated vanilla and barrier share grid and innovations, making parity pathwise.
        for name, model in models(True):
            mode = "composite"
            def price(product):
                return self.inputs.option(product, mode, model=model).with_hull_white_rates(
                    self.inputs.d, self.inputs.f, correlations=C).price_mc(self.mc).value
            with self.subTest(model=name):
                ki = price(self.barrier(mode, "knock_in"))
                ko = price(self.barrier(mode))
                vanilla = price(rp.Product.european_vanilla(10, 2, END, 98.*150., 1., "call"))
                self.assertGreater(min(ki, ko), 0.)
                self.assertAlmostEqual(ki+ko, vanilla, places=9)

    def test_flat_model_limits_match_native_continuous_prices(self):
        flat = dict(models(False))["LV"]
        tiny = rp.HullWhiteModel(.2, [0.], [1.e-8])
        for mode in ["quanto", "floating_fx", "composite"]:
            p = self.barrier(mode)
            reference = self.inputs.option(p, mode).price_mc(self.mc)
            for model in [flat, rp.Model.black_scholes(.2)]:
                got = self.inputs.option(p, mode, model=model).with_hull_white_rates(
                    tiny, tiny, correlations=C).price_mc(self.mc)
                self.assertAlmostEqual(got.value, reference.value,
                                       delta=5*math.hypot(got.standard_error, reference.standard_error))

    def test_absorbing_hits_ended_monitoring_and_validation(self):
        for mode in ["quanto", "floating_fx", "composite"]:
            def price(p):
                return self.inputs.option(p, mode).with_hull_white_rates(
                    self.inputs.d, self.inputs.f, correlations=C).price_mc(self.mc).value
            for style, hit in [("knock_out", True), ("knock_in", False)]:
                self.assertEqual(price(self.barrier(mode, style, dates=["2025-12-20"], hit=hit)), 0.)
            self.assertEqual(price(self.barrier(mode, dates=["2025-12-20", END], hit=True)), 0.)
            with self.assertRaises(rp.PricingError):
                price(self.barrier(mode, dates=["2025-12-20", END]))
            with self.assertRaises(rp.PricingError):
                price(self.barrier(mode, hit=False))

    def test_dividend_jump_at_end_of_monitoring_and_payment_lag(self):
        zero = rp.DiscountCurve(2,[0.,2.],[1.,1.])
        market = rp.Market.equity(1,10,100.,zero,zero,discrete_dividends=[
            rp.DividendEvent.fixed_cash(1,rates.time(MID),25.)])
        # Cash dividend takes spot from 100 to 75 exactly at the last monitoring time.
        p = rp.Product.barrier(10,2,END,60.,80.,1.,"call","down","knock_out",
                "continuous",[TODAY,MID],rates.PAY,rebate=3.)
        dividend = rp.BuehlerDividendModel(mean_reversion=0.,equity_linkage=0.,
                                          dividend_volatility=0.,equity_dividend_correlation=0.)
        opt = rp.CrossCurrencyEquityOption(TODAY,p,market,zero,equity_model=rp.Model.black_scholes(0.),
            dividend_model=dividend,settlement="floating_fx",fx_spot=150.,fx_volatility=0.,
            correlation=0.,maximum_step=.2)
        self.assertEqual(opt.price_mc(self.mc).value,450.)

    def test_stochastic_physical_dividends_in_composite_monitoring(self):
        inputs = dividends.CrossCurrencyDividendTests()
        inputs.setUp()
        for name, model in models(True):
            if name not in ["BS","Bergomi","Bergomi2 LSV","roughBergomi LSV"]:
                continue
            foreign = inputs.foreign if name in ("BS", "Bergomi") else inputs.zero
            def price(p):
                return inputs.price(p,"composite",model,foreign).price_mc(self.mc).value
            ki = price(self.barrier("composite","knock_in"))
            ko = price(self.barrier("composite"))
            vanilla = price(rp.Product.european_vanilla(10,2,END,14700.,1.,"call"))
            self.assertGreater(min(ki,ko),0.)
            self.assertAlmostEqual(ki+ko,vanilla,places=9)
