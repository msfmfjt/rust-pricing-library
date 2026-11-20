"""Settlement-currency shares and payout currencies without an underlying share."""
import math
import unittest

import rust_pricing as rp

from test_cross_currency_basket import TODAY, PAY, IDS, curve, years
from test_cross_currency_mixed_basket import USD, JPY, EUR


class CurrencyUniverseTests(unittest.TestCase):
    def setUp(self):
        self.jpy, self.usd, self.eur = curve(JPY,.012), curve(USD,.05), curve(EUR,.035)
        self.stocks = [rp.Market.equity(JPY,10,90.,self.jpy,curve(10,.016)),
                       rp.Market.equity(USD,20,70.,self.usd,curve(11,.025))]
        self.quotes = [(EUR,170.,.18),(USD,145.,.14)]
        self.hw = [rp.HullWhiteModel(.2,[0.],[s]) for s in [.01,.015,.02]]
        # A positive-definite correlation matrix with nonzero cross-currency terms.
        loads = [.6,-.31,.4,.65,.5,-.35,.72]
        self.corr = [[1. if i == j else x*y for j,y in enumerate(loads)] for i,x in enumerate(loads)]
        self.config = dict(equity_models=[rp.Model.black_scholes(.22),rp.Model.black_scholes(.28)],
                           fx_quotes=self.quotes, currency_curves={EUR:self.eur},
                           correlations=self.corr, rates=self.hw, maximum_step=.25)
        self.engine = rp.Engine.randomized_quasi_monte_carlo(2048,1937,scramble_count=10,
                                                             antithetic=True,brownian_bridge=True)

    def graph(self, currency, asset=None, ids=IDS, fixing=PAY):
        b = rp.SourceGraphBuilder()
        value = b.literal(9.) if asset is None else b.spot(asset,fixing)
        return b.finish([value],[fixing],currency_id=currency,underlying_ids=ids)

    def test_jpy_usd_equities_and_eur_cash_against_replication(self):
        # PV of JPY share, USD share in JPY, and EUR zero coupon are additive.
        products = [self.graph(JPY,10),self.graph(JPY,20),self.graph(EUR)]
        opt = rp.CrossCurrencyEquityBasket.from_legs(TODAY,products,self.stocks,self.jpy,
            settlement_currency_id=JPY,performance="composite",**self.config)
        result = opt.price_mc(self.engine)
        t = years(PAY)
        expected = 90.*math.exp(-.016*t)+145.*70.*math.exp(-.025*t)+170.*9.*math.exp(-.035*t)
        self.assertAlmostEqual(result.value,expected,delta=6*result.standard_error)

    def test_same_day_native_currency_and_cash_only_currency(self):
        products = [self.graph(JPY,10,fixing=TODAY), self.graph(EUR,fixing=TODAY)]
        for mode in ["local","composite"]:
            opt = rp.CrossCurrencyEquityBasket.from_legs(TODAY,products,self.stocks,self.jpy,
                settlement_currency_id=JPY,performance=mode,**self.config)
            result = opt.price_mc(self.engine)
            self.assertEqual((result.value,result.standard_error),(90.+9.*170.,0.))

    def test_domestic_only_with_one_hw_rate(self):
        opt = rp.CrossCurrencyEquityBasket.from_currencies(TODAY,self.graph(JPY,10,[10]),
            self.stocks[:1],self.jpy,settlement="composite",equity_models=self.config['equity_models'][:1],
            fx_quotes=[],correlations=[[1.,.55],[.55,1.]],rates=self.hw[:1],maximum_step=.25)
        v = opt.price_mc(self.engine)
        self.assertAlmostEqual(v.value,90.*math.exp(-.016*years(PAY)),delta=6*v.standard_error)

    def test_cash_only_payout_on_single_graph_and_curve_conflicts(self):
        graph = self.graph(JPY)
        def make(**changes):
            return rp.CrossCurrencyEquityBasket.from_currencies(TODAY,graph,self.stocks,self.jpy,
                settlement="floating_fx",payout_currency_id=EUR,**(self.config|changes))
        v = make().price_mc(self.engine)
        self.assertAlmostEqual(v.value,9.*170.*math.exp(-.035*years(PAY)),delta=6*v.standard_error)
        for curves in [{}, {EUR:self.eur,USD:self.jpy}, {EUR:self.eur,JPY:self.jpy}]:
            with self.assertRaises(ValueError):
                make(currency_curves=curves)
        with self.assertRaises(ValueError):
            rp.CrossCurrencyEquityBasket.from_legs(TODAY,[graph],self.stocks,self.eur,
                settlement_currency_id=JPY,**self.config)
