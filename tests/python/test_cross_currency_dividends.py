"""Cash-dividend claims and collateral-currency equivalence."""
import math
import unittest

import rust_pricing as rp

from test_cross_currency_models import models
from test_cross_currency_rates import C, END, MID, TODAY, time


class CrossCurrencyDividendTests(unittest.TestCase):
    def setUp(self):
        def curve(i, r):
            return rp.DiscountCurve(i, [0., 1.], [1., math.exp(-r)])
        self.jpy = curve(2, .01)
        self.market = rp.Market.equity(1, 10, 100., curve(1, .04), curve(3, .015),
            discrete_dividends=[rp.DividendEvent.fixed_cash(i, t, v)
                                for i, (t, v) in enumerate([(time(MID), 8.), (1., 6.), (1.25, 10.)], 1)])
        self.dividends = rp.BuehlerDividendModel(mean_reversion=0., equity_linkage=.5,
            dividend_volatility=.4, equity_dividend_correlation=.6)
        self.engine = rp.Engine.randomized_quasi_monte_carlo(512, 572,
            scramble_count=12, antithetic=True, brownian_bridge=True)
        self.domestic = rp.HullWhiteModel(.3, [0.], [.035])
        self.foreign = rp.HullWhiteModel(.2, [0.], [.05])
        self.zero = rp.HullWhiteModel(.2, [0.], [0.])

    def price(self, product, mode, model, foreign=None):
        option = rp.CrossCurrencyEquityOption(TODAY, product, self.market, self.jpy,
            equity_model=model, dividend_model=self.dividends, maximum_step=.25,
            settlement=mode, fx_spot=150., fx_volatility=.18, correlation=C[0][1],
            fixed_fx_rate=140. if mode == "quanto" else None)
        if foreign is not None:
            option = option.with_hull_white_rates(self.domestic, foreign, correlations=C)
        return option

    def test_claim_payments_equal_foreign_dividends_and_quanto_moment(self):
        b = rp.SourceGraphBuilder()
        dates = [MID, END]
        amounts = [b.subtract(b.spot(10, d, pre_dividend=True), b.spot(10, d)) for d in dates]
        product = rp.Product.from_graph(b.finish(amounts, dates, currency_id=2, underlying_ids=[10]))
        stock = rp.Model.black_scholes(.22)
        for mode in ["quanto", "floating_fx", "composite"]:
            with self.subTest(mode=mode):
                result = self.price(product, mode, stock).price_mc(self.engine)
                # Composite pre/post difference is X(t) times the cash dividend.
                rate = .01 + .4 * .6 * C[0][1] * .18 if mode == "quanto" else .04
                fx = 140. if mode == "quanto" else 150.
                expected = sum(a * fx * math.exp(-rate*time(d)) for a, d in zip([8., 6.], dates))
                self.assertAlmostEqual(result.value, expected, delta=6*result.standard_error+1.e-8)

        # Under the foreign measure, E[D(T)Y(T)] uses the covariance between
        # the foreign integrated short rate and the cash-dividend Brownian.
        expected = 0.
        for a, d in zip([8., 6.], dates):
            t = time(d)
            integral = (t + math.expm1(-.2*t)/.2)/.2
            expected += 150.*a*math.exp(-.04*t-.4*.6*C[0][3]*.05*integral)
        for mode in ["floating_fx", "composite"]:
            result = self.price(product, mode, stock, self.foreign).price_mc(self.engine)
            self.assertAlmostEqual(result.value, expected, delta=6*result.standard_error+1.e-8)

    def test_floating_price_matches_native_cash_dividend_engines(self):
        product = rp.Product.european_vanilla(10, 2, END, 93., 1., "call")
        original = rp.Product.european_vanilla(10, 1, END, 93., 1., "call")
        for name, model in models(True):
            if name != "BS" and not name.startswith(("Bergomi", "roughBergomi")):
                continue
            for hw in ([False, True] if name == "BS" else [False]):
                with self.subTest(model=name, stochastic_foreign_rate=hw):
                    rate = rp.RateModelSpec.hull_white(self.foreign, equity_rate_correlation=C[0][3]) if hw else None
                    spec = rp.EquityModelSpec.standard(model) if name == "BS" else model
                    native_source = rp.PricingSpecification(TODAY, original, self.market,
                        rp.PricingModelSpec(spec, rates=rate), self.engine, rp.RiskRequest(),
                        rp.PricingNumerics(worker_threads=1, maximum_step=(.25 if hw or
                            (name != "BS" and not name.endswith(" LSV")) else None)))
                    vol_correlations = ([-.24, -.12] if "Bergomi2" in name else [-.24]) if name != "BS" else None
                    source = rp.StochasticDividendSpecification(native_source, self.dividends,
                        maximum_step=.25, volatility_correlations=vol_correlations,
                        rate_correlation=.6*C[0][3] if hw else None)
                    native = rp.PricingTask.from_source(source).compile().evaluate().native_result
                    actual = self.price(product, "floating_fx", model,
                                        self.foreign if hw else self.zero).price_mc(self.engine)
                    self.assertAlmostEqual(actual.value/150., native.value,
                        delta=6*math.hypot(actual.standard_error/150., native.standard_error)+.005)

    def test_autocall_with_dividends_and_observation_day_fx(self):
        contract = rp.ContractBuilder("coupons", 2, 10, "close").autocall(
            100., 100., 60., [MID, END], ["2026-07-15", "2027-01-15"],
            [104., 104.], [80., 80.], [.03, .04])
        graph, = contract.to_graphs(TODAY, [10])
        option = self.price(rp.Product.from_graph(graph), "floating_fx",
                            rp.Model.black_scholes(.22), self.foreign)
        actual = option.with_payment_fx_fixings([("2026-07-15", MID),
            ("2027-01-15", END)]).price_mc(self.engine)
        self.assertGreater(actual.value, 10000.)
        self.assertLess(actual.value, 17000.)
        self.assertGreater(actual.standard_error, 0.)

    def test_reject_incompatible_combinations(self):
        product = rp.Product.european_vanilla(10, 2, END, 100., 1., "call")
        candidates = dict(models(True))
        with self.assertRaisesRegex(ValueError, "Buehler"):
            self.price(product, "quanto", candidates["roughHeston"])
        with self.assertRaisesRegex(ValueError, "one-factor Bergomi"):
            self.price(product, "quanto", candidates["roughBergomi"], self.foreign)
