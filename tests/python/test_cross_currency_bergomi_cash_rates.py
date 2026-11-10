"""Stochastic residual volatility, discounted cash reserves and JPY autocall replay."""
from pathlib import Path
import math
import sys
import unittest

import rust_pricing as rp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/python"))
import cross_currency_stochastic_dividend_autocall as example
from cross_currency_usd_eur_autocall import USD, EUR, JPY, curve, contract_terms, OBSERVATIONS, PAYMENTS


class HybridCashTests(unittest.TestCase):
    def setUp(self):
        self.engine = rp.Engine.randomized_quasi_monte_carlo(256, 4187, scramble_count=8,
                                                             antithetic=True, brownian_bridge=True)
        self.as_of = "2026-04-10"
        self.pay = "2027-04-10"
        self.mid = "2026-10-10"
        self.corr = [[1., -.3, .2, .5], [-.3, 1., .1, -.35], [.2, .1, 1., .25], [.5, -.35, .25, 1.]]
        self.rates = [rp.HullWhiteModel(.25,[0.],[.026]), rp.HullWhiteModel(.4,[0.,.7],[.06,.035])]
        self.div = rp.BuehlerDividendModel(mean_reversion=1.3, equity_linkage=.8,
                                         dividend_volatility=.4, equity_dividend_correlation=.55)
        self.equity = rp.EquityModelSpec.stochastic_volatility(.25,rp.VolatilityFactorSpec.bergomi(.8,.75,-.55))

    def basket(self, graph, equity=None, correlations=None):
        events = [rp.DividendEvent.fixed_cash(i+1, t, amount) for i,(t,amount) in enumerate([
            (example.years(self.as_of,self.mid),9.), (1.,8.), (1.5,11.)])]
        market = rp.Market.equity(USD, 10, 100., curve(1,.04), curve(4,0.), discrete_dividends=events)
        return rp.CrossCurrencyEquityBasket.from_legs(self.as_of,[graph],[market],curve(JPY,.015),
            settlement_currency_id=JPY, equity_models=[self.equity if equity is None else equity],
            dividend_models=[self.div], rates=self.rates, fx_quotes=[(USD,150.,.17)],
            correlations=self.corr if correlations is None else correlations, maximum_step=1/24)

    def test_gain_process_including_dividends_replicates_cash_equity(self):
        b = rp.SourceGraphBuilder()
        # First dividend plus terminal cum-dividend stock. The reserve also funds
        # a dividend after expiry. All flows convert at their own payment date.
        cash = b.subtract(b.spot(10,self.mid,pre_dividend=True), b.spot(10,self.mid))
        product = b.finish([cash,b.spot(10,self.pay,pre_dividend=True)],
                           [self.mid,self.pay], currency_id=USD, underlying_ids=[10])
        base = self.basket(product)
        for opt in [base, base.with_dividend_state(10,model_age=.6,volatility_factor=.45,
                                                  equity_factor=.91,dividend_factor=1.17)]:
            v = opt.price_mc(self.engine)
            self.assertAlmostEqual(v.value,150.*100.,delta=5*v.standard_error+1.)
            delta = opt.sensitivity("fx_spot", USD, .5, self.engine)
            self.assertAlmostEqual(delta.value,100.,delta=5*delta.standard_error+.01)

    def test_midlife_forecast_is_the_mean_of_the_actual_cash_jump(self):
        b = rp.SourceGraphBuilder()
        cash = b.subtract(b.spot(10,self.mid,pre_dividend=True), b.spot(10,self.mid))
        graph = b.finish([cash],[self.mid],currency_id=USD,underlying_ids=[10])
        # The foreign short-rate driver is independent of the cash/stock factors.
        # The price of the actual USD 9 expected dividend is its discounted FX value,
        # also when today's cash factor is 1.6 and today's residual-share factor is 1.25.
        corr = [r[:] for r in self.corr]
        corr[0][3] = corr[3][0] = 0.
        opt = self.basket(graph,correlations=corr).with_dividend_state(
            10,model_age=.7,volatility_factor=-.4,equity_factor=1.25,dividend_factor=1.6)
        price = opt.price_mc(self.engine)
        expected = 9.*150.*math.exp(-.04*example.years(self.as_of,self.mid))
        self.assertAlmostEqual(price.value,expected,delta=5*price.standard_error+.03)

    def test_constant_volatility_limit_and_historical_factor_validation(self):
        b = rp.SourceGraphBuilder()
        call = b.maximum(b.subtract(b.spot(10,self.pay), b.literal(100.)),b.literal(0.))
        graph = b.finish([call],[self.pay],currency_id=JPY,underlying_ids=[10])
        black = self.basket(graph,rp.Model.black_scholes(.25))
        limit = self.basket(graph,rp.EquityModelSpec.stochastic_volatility(.25,
                             rp.VolatilityFactorSpec.bergomi(.8,0.,-.55)))
        # Sobol coordinates differ between two- and three-driver equity layouts.
        a, c = black.price_mc(self.engine), limit.price_mc(self.engine)
        self.assertAlmostEqual(c.value,a.value,delta=5*math.hypot(c.standard_error,a.standard_error)+.01)
        live = self.basket(graph)
        for stock,age,x,f,y in [(88,.3,.1,1.,1.),(10,0.,.1,1.,1.),(10,-1.,0.,1.,1.),
                                (10,.3,math.nan,1.,1.),(10,.3,.1,0.,1.),(10,.3,.1,1.,-2.)]:
            with self.assertRaises(ValueError):
                live.with_dividend_state(stock,model_age=age,volatility_factor=x,equity_factor=f,dividend_factor=y)
        with self.assertRaises(ValueError):
            black.with_dividend_state(10,model_age=1.,volatility_factor=0.,equity_factor=1.,dividend_factor=1.)
        cold = live.with_dividend_state(10,model_age=.4,volatility_factor=-.6,equity_factor=.96,dividend_factor=1.1)
        hot = live.with_dividend_state(10,model_age=.4,volatility_factor=.6,equity_factor=.96,dividend_factor=1.1)
        diff = cold.scenario_pnl(hot,self.engine)
        self.assertGreater(diff.value,5*diff.standard_error)

    def test_split_note_revalues_and_bumped_inputs_match_full_rebuild(self):
        issue = example.note(example.VALUATION)
        live = example.note(example.REVALUATION)
        for note in (issue,live):
            pv = note.price_mc(self.engine)
            self.assertGreater(pv.value,0.)
            self.assertGreater(pv.standard_error,0.)
        for factor, ident, h, lo, hi in [
            ("equity_vol_of_vol",10,.05,dict(vol_of_vol=(.50,.70)),dict(vol_of_vol=(.60,.70))),
            ("dividend_volatility",20,.04,dict(dividend_vols=(.30,.34)),dict(dividend_vols=(.30,.42))),
        ]:
            v=live.sensitivity(factor,ident,h,self.engine)
            down=example.note(example.REVALUATION,**lo)
            up=example.note(example.REVALUATION,**hi)
            pnl=down.scenario_pnl(up,self.engine)
            self.assertAlmostEqual(v.value,pnl.value/(2*h),places=7)
            self.assertAlmostEqual(v.standard_error,pnl.standard_error/(2*h),places=7)
        self.assertEqual(live.sensitivity("fx_spot",EUR,.1,self.engine).value,0.)
        self.assertGreater(live.sensitivity("fx_spot",USD,.1,self.engine).value,0.)

    def test_confirmed_call_leaves_a_yen_bond_with_no_equity_or_fx_risk(self):
        original = rp.MultiAssetProduct.autocallable_legs(**contract_terms("floating_fx",JPY,USD),
                                                         coupon_currency_id=USD, principal_currency_id=JPY)
        first = OBSERVATIONS[0]
        history = [(10,first,110.),(20,first,90.)]
        residual = [leg.remaining_cashflows(self.as_of,fixings=history) for leg in original]
        self.assertEqual([leg.payment_dates for leg in residual],[[PAYMENTS[0]],[PAYMENTS[0]]])
        from cross_currency_usd_eur_autocall import equity_markets, model_inputs, FX_QUOTES
        settings = model_inputs(FX_QUOTES) | dict(equity_models=[self.equity]*2,dividend_models=[self.div]*2)
        note = rp.CrossCurrencyEquityBasket.from_legs(self.as_of,residual,equity_markets(99.,77.),
                    curve(JPY,.01),settlement_currency_id=JPY,**settings)
        note = note.with_leg_fx(0,schedule=[(PAYMENTS[0],first)],known_fixings=[(first,147.)])
        pv = note.price_mc(self.engine)
        expected = (150000.+20.*147.)*math.exp(-.01*example.years(self.as_of,PAYMENTS[0]))
        self.assertAlmostEqual(pv.value,expected,delta=6*pv.standard_error+.001)
        for factor, identifier in [("fx_spot",USD),("equity_volatility",10),("equity_vol_of_vol",20),
                                   ("dividend_volatility",10)]:
            risk=note.sensitivity(factor,identifier,.01,self.engine)
            self.assertEqual((risk.value,risk.standard_error),(0.,0.))
