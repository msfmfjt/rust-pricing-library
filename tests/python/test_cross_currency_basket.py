"""Shared FX, correlated foreign stocks and two collateral-currency short rates."""
import math
import unittest
from datetime import date

import rust_pricing as rp

from test_cross_currency_models import models

TODAY, OBS, FIX, PAY = "2026-01-01", "2026-10-01", "2026-11-01", "2027-01-15"
IDS = [10, 20]
C = [[1., .45, -.28, .18, .32], [.45, 1., .22, -.18, .14],
     [-.28, .22, 1., .3, -.24], [.18, -.18, .3, 1., .27], [.32, .14, -.24, .27, 1.]]
TOL = dict(symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12, psd_abs_tol=1e-12,
           psd_rel_tol=1e-12, zero_pivot_abs_tol=1e-14, zero_pivot_rel_tol=1e-14)


def years(day):
    return (date.fromisoformat(day) - date.fromisoformat(TODAY)).days / 365.


def curve(i, rate):
    return rp.DiscountCurve(i, [0., 2.], [1., math.exp(-2*rate)])


def graph_of_product():
    b = rp.SourceGraphBuilder()
    product = b.multiply(b.spot(10, OBS), b.spot(20, OBS))
    return b.finish([b.divide(product, b.literal(8000.))], [PAY], currency_id=2, underlying_ids=IDS)


def product_moment(mode):
    # Expectation of exp(log discounted S1*S2 times contractual FX).
    # Compute the joint normal moment directly from continuous Brownian kernels.
    t, p = years(OBS), years(PAY)
    f = years(FIX) if mode == "floating_fx" else t
    k = {"quanto": 0, "floating_fx": 1, "composite": 2}[mode]
    mean = math.log(145. if mode == "quanto" else 150.**k)
    s1, s2, sx = .2, .30, .16
    mean -= ((.01 + .025) + .5*(s1*s1+s2*s2) + sx*(s1*C[0][2]+s2*C[1][2]))*t
    mean -= .5*k*sx*sx*f
    # (rate index, horizon, coefficient) in the log payment
    terms = [(3, p, -1.), (4, t, 2.), (3, f, k), (4, f, -k)]
    mean += sum(weight*(.015 if i == 3 else .045)*h for i, h, weight in terms)
    var = 0.
    breaks = sorted(set([0., .38, t, f, p]))
    for left, right in zip(breaks, breaks[1:]):
        step = (right-left)/128
        for j in range(128):
            u = left+(j+.5)*step
            load = [s1*(u<t), s2*(u<t), k*sx*(u<f), 0., 0.]
            for i, h, w in terms:
                if h <= u:
                    continue
                a = .35 if i == 3 else .12
                vol = (.035 if u < .38 else .02) if i == 3 else .05
                kernel = -vol*math.expm1(-a*(h-u))/a
                load[i] += w*kernel
                mean += step*w*(.5*kernel*kernel - (kernel*sx*C[2][4] if i == 4 else 0.))
            var += step*sum(x*y*C[i][j] for i,x in enumerate(load) for j,y in enumerate(load))
    return math.exp(mean + .5*var)


class BasketTests(unittest.TestCase):
    def setUp(self):
        self.usd, self.jpy = curve(1, .045), curve(2, .015)
        self.markets = [rp.Market.equity(1, 10, 100., self.usd, curve(3, .01)),
                        rp.Market.equity(1, 20, 80., self.usd, curve(4, .025))]
        self.models = [rp.Model.black_scholes(.2), rp.Model.black_scholes(.3)]
        self.rates = (rp.HullWhiteModel(.35, [0., .38], [.035, .02]),
                      rp.HullWhiteModel(.12, [0.], [.05]))
        self.engine = rp.Engine.randomized_quasi_monte_carlo(2048, 1703, scramble_count=12,
                                                              antithetic=True, brownian_bridge=True)

    def option(self, graph, mode, **kwargs):
        args = dict(equity_models=self.models, fx_spot=150., fx_volatility=.16, settlement=mode,
                    rates=self.rates, correlations=C, maximum_step=.25,
                    fixed_fx_rate=145. if mode == "quanto" else None)
        markets = kwargs.pop("markets", self.markets)
        return rp.CrossCurrencyEquityBasket(TODAY, graph, markets, self.jpy, **(args | kwargs))

    def test_joint_moment_in_all_settlements_and_delayed_fx(self):
        for mode in ["quanto", "floating_fx", "composite"]:
            with self.subTest(mode=mode):
                option = self.option(graph_of_product(), mode)
                if mode == "floating_fx":
                    option = option.with_payment_fx_fixings([(PAY, FIX)])
                v = option.price_mc(self.engine)
                self.assertAlmostEqual(v.value, product_moment(mode), delta=6*v.standard_error+1e-5)

    def test_multi_stock_memory_note_matches_independent_foreign_engine(self):
        obs = [rp.AutocallObservation("2026-04-01", "2026-04-15", coupon_amount=3., coupon_level=1.04),
               rp.AutocallObservation(OBS, FIX, coupon_amount=4., coupon_level=.88, call_level=1.02)]
        def note(currency):
            return rp.MultiAssetProduct.autocallable(IDS, [100., 80.], obs, "2027-01-01", PAY,
                currency_id=currency, notional=100., final_barrier=.7, memory=True,
                on_autocall="pay", on_maturity="forfeit").to_graph()
        corr = rp.CorrelationSchedule(IDS, [TODAY], [[r[:2] for r in C[:2]]], **TOL)
        foreign = rp.MultiAssetModelSpec(self.models, corr, hull_white=rp.MultiAssetHullWhiteConfig(
            self.rates[1], [C[0][4], C[1][4]]))
        spec = rp.MultiAssetSpecification(TODAY, note(1), self.markets, foreign, self.engine,
                                         maximum_step=.25, worker_threads=1)
        reference = rp.PricingTask.from_source(spec).compile().evaluate().native_result
        pv = self.option(note(2), "floating_fx").price_mc(self.engine)
        self.assertAlmostEqual(pv.value/150., reference.value,
                               delta=6*math.hypot(pv.standard_error/150., reference.standard_error))

    def test_identical_stocks_reduce_to_one_stock_with_perfect_correlation(self):
        b = rp.SourceGraphBuilder()
        worst = b.minimum(b.spot(10, OBS), b.spot(20, OBS))
        graph = b.finish([worst], [PAY], currency_id=2, underlying_ids=IDS)
        b = rp.SourceGraphBuilder()
        graph1 = b.finish([b.spot(10, OBS)], [PAY], currency_id=2, underlying_ids=[10])
        markets = [self.markets[0], rp.Market.equity(1, 20, 100., self.usd, curve(3, .01))]
        c = [[1., -.28, .18, .32], [-.28, 1., .3, -.24], [.18, .3, 1., .27], [.32, -.24, .27, 1.]]
        order = [0, 0, 1, 2, 3]
        basket = self.option(graph, "composite", markets=markets, equity_models=[self.models[0]]*2,
                             correlations=[[c[i][j] for j in order] for i in order]).price_mc(self.engine)
        single = rp.CrossCurrencyEquityOption(TODAY, rp.Product.from_graph(graph1), markets[0], self.jpy,
            settlement="composite", fx_spot=150., fx_volatility=.16, equity_volatility=.2,
            correlation=-.28, maximum_step=.25).with_hull_white_rates(*self.rates, correlations=c)
        reference = single.price_mc(self.engine)
        self.assertAlmostEqual(basket.value, reference.value, delta=6*math.hypot(
            basket.standard_error, reference.standard_error))

    def test_worst_name_controls_memory_early_call_and_downside(self):
        flat = curve(90, 0.)
        for mode in ["quanto", "floating_fx", "composite"]:
            unit = 150. if mode == "composite" else 1.
            obs = [rp.AutocallObservation("2026-04-01", "2026-04-15", coupon_amount=2.*unit,
                                         coupon_level=1.05, call_level=1.10),
                   rp.AutocallObservation(OBS, FIX, coupon_amount=3.*unit,
                                         coupon_level=.90, call_level=1.10)]
            note = rp.MultiAssetProduct.autocallable(IDS, [100.*unit]*2, obs, "2027-01-01", PAY,
                currency_id=2, notional=100.*unit, final_barrier=.70, memory=True,
                on_autocall="pay", on_maturity="forfeit").to_graph()
            for last_spot, cash in [(112., 102.), (95., 105.), (55., 55.)]:
                with self.subTest(mode=mode, spot=last_spot):
                    markets = [rp.Market.equity(1, i, v, flat, flat) for i,v in zip(IDS, [120., last_spot])]
                    pricer = rp.CrossCurrencyEquityBasket(TODAY, note, markets, flat,
                        equity_models=[rp.Model.black_scholes(0.)]*2, settlement=mode,
                        fixed_fx_rate=145. if mode == "quanto" else None, fx_spot=150., fx_volatility=0.,
                        correlations=[row[:3] for row in C[:3]], maximum_step=.25)
                    pv = pricer.price_mc(rp.Engine.pseudo_monte_carlo(65, 8))
                    self.assertAlmostEqual(pv.value, cash*(145. if mode=="quanto" else 150.), places=8)
                    self.assertEqual(pv.standard_error, 0.)

    def test_single_name_dividend_observed_in_a_multi_stock_contract(self):
        b = rp.SourceGraphBuilder()
        dividend = b.subtract(b.spot(20, OBS, pre_dividend=True), b.spot(20, OBS))
        graph = b.finish([dividend], [OBS], currency_id=2, underlying_ids=IDS)
        t = years(OBS)
        markets = [self.markets[0], rp.Market.equity(1, 20, 80., self.usd, curve(4, .025),
                   discrete_dividends=[rp.DividendEvent.fixed_cash(1, t, 5.)])]
        div_model = rp.BuehlerDividendModel(mean_reversion=0., equity_linkage=.4,
                                            dividend_volatility=.3, equity_dividend_correlation=.6)
        option = self.option(graph, "floating_fx", markets=markets, dividend_models=[None, div_model])
        actual = option.price_mc(self.engine)
        integral = (t + math.expm1(-.12*t)/.12)/.12
        expected = 150.*5.*math.exp(-.045*t-.3*.6*C[1][4]*.05*integral)
        self.assertAlmostEqual(actual.value, expected, delta=6*actual.standard_error)

    def test_same_day_and_cash_only_graphs(self):
        b = rp.SourceGraphBuilder()
        now = b.finish([b.minimum(b.spot(10, TODAY), b.spot(20, TODAY))], [TODAY],
                       currency_id=2, underlying_ids=IDS)
        value = self.option(now, "composite").price_mc(self.engine)
        self.assertEqual((value.value, value.standard_error), (80.*150., 0.))
        b = rp.SourceGraphBuilder()
        fixed = b.finish([b.literal(-2.), b.literal(5.)], [OBS, PAY], currency_id=2, underlying_ids=IDS)
        value = self.option(fixed, "floating_fx").price_mc(self.engine)
        expected = 150.*(-2.*math.exp(-.045*years(OBS))+5.*math.exp(-.045*years(PAY)))
        self.assertAlmostEqual(value.value, expected, delta=6*value.standard_error)

    def test_model_selection_and_distinct_underlying_horizons(self):
        b = rp.SourceGraphBuilder()
        cash = b.add(b.spot(10, "2026-04-01"), b.spot(20, "2027-01-01"))
        graph = b.finish([cash], [PAY], currency_id=2, underlying_ids=IDS)
        engine = rp.Engine.pseudo_monte_carlo(410, 64, antithetic=True)
        for name, model in models(True):
            with self.subTest(model=name):
                v = self.option(graph, "quanto", equity_models=[model, self.models[1]]).price_mc(engine)
                self.assertGreater(v.value, 20000.)
                self.assertLess(v.value, 32000.)
        early = rp.Model.local_volatility_from_grid([0., .25], [-1., 1.], [.04]*4, 1e-9, 2.)
        v = self.option(graph, "floating_fx", equity_models=[early, self.models[1]]).price_mc(engine)
        self.assertGreater(v.value, 20000.)

    def test_zero_rates_and_worker_reproducibility(self):
        graph = graph_of_product()
        c = [r[:3] for r in C[:3]]
        zero = rp.HullWhiteModel(.2, [0.], [0.])
        engine = rp.Engine.pseudo_monte_carlo(893, 128, antithetic=True)
        a = self.option(graph, "floating_fx", rates=None, correlations=c).price_mc(engine)
        b = self.option(graph, "floating_fx", rates=(zero, zero)).price_mc(engine, worker_threads=2)
        self.assertEqual((a.value, a.standard_error), (b.value, b.standard_error))
        rate = self.option(graph, "quanto")
        a, b = (rate.price_mc(engine, worker_threads=k) for k in [1, 2])
        self.assertEqual((a.value, a.standard_error), (b.value, b.standard_error))

    def test_input_validation_and_fixing_dependencies(self):
        graph = graph_of_product()
        invalid = [dict(markets=self.markets[::-1]), dict(equity_models=self.models[:1]),
                   dict(correlations=[[1.]]), dict(markets=[self.markets[0],
                       rp.Market.equity(3, 20, 80., self.usd, curve(4, .025))]),
                   dict(markets=[self.markets[0], rp.Market.equity(1, 20, 80., curve(5, .05), curve(4, .025))])]
        indefinite = [row[:] for row in C]
        indefinite[0][1] = indefinite[1][0] = -.999
        for kwargs in invalid + [dict(correlations=indefinite)]:
            with self.subTest(keys=list(kwargs)), self.assertRaises(ValueError):
                self.option(graph, "quanto", **kwargs)
        option = self.option(graph, "floating_fx")
        for schedule in [[(PAY, TODAY)], [(PAY, FIX), (PAY, FIX)], [(FIX, OBS)]]:
            with self.assertRaises(ValueError):
                option.with_payment_fx_fixings(schedule)
        with self.assertRaises(ValueError):
            self.option(graph, "composite").with_payment_fx_fixings([(PAY, FIX)])
