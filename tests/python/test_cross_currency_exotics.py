"""Foreign equity exotics priced through the installed extension."""
import math
import unittest

import rust_pricing as rp


TODAY, MID, EXPIRY, PAY = "2026-01-01", "2026-07-01", "2027-01-01", "2027-02-01"
T_MID, T_PAY = 181 / 365, 396 / 365


def curve(id, rate):
    return rp.DiscountCurve(id, [0., 1.], [1., math.exp(-rate)])


class CrossCurrencyExotics(unittest.TestCase):
    def setUp(self):
        self.usd, self.jpy, self.dividend = curve(1, .04), curve(2, .01), curve(3, .02)
        self.market = rp.Market.equity(1, 10, 100., self.usd, self.dividend)
        self.engine = rp.Engine.pseudo_monte_carlo(387, 2048, antithetic=True)

    def option(self, product, settlement, **inputs):
        kwargs = dict(fx_spot=150., equity_volatility=.20, fx_volatility=.10, correlation=.35)
        if settlement == "quanto":
            kwargs["fixed_fx_rate"] = 145.
        kwargs.update(inputs)
        return rp.CrossCurrencyEquityOption(TODAY, product, self.market, self.jpy,
                                           settlement=settlement, **kwargs)

    def test_asian_digital_and_lookback_deterministic_paths(self):
        obs = [rp.AsianObservation.unknown(MID, .4), rp.AsianObservation.unknown(EXPIRY, .6)]
        for settlement in ("quanto", "floating_fx", "composite"):
            with self.subTest(settlement=settlement):
                composite = settlement == "composite"
                s0, carry, strike = (15_000., -.01, 14_500.) if composite else (100., .02, 97.)
                s1, s2 = [s0 * math.exp(carry * t) for t in (T_MID, 1.)]
                fx = {"quanto": 145., "floating_fx": 150. * math.exp(-.03), "composite": 1.}[settlement]
                for product, payoff in [
                    (rp.Product.arithmetic_asian(10, 2, strike, 2., "call", obs, PAY),
                     2. * max(.4 * s1 + .6 * s2 - strike, 0.)),
                    (rp.Product.fixed_lookback(10, 2, strike, 2., "call", [MID, EXPIRY], PAY),
                     2. * max(max(s1, s2) - strike, 0.)),
                    (rp.Product.digital(10, 2, EXPIRY, strike, 3., "call", "asset", payment_date=PAY),
                     3. * s2 if s2 > strike else 0.),
                ]:
                    estimate = self.option(product, settlement, equity_volatility=0., fx_volatility=0.).price_mc(self.engine)
                    self.assertAlmostEqual(estimate.value, math.exp(-.01 * T_PAY) * fx * payoff, places=8)
                    self.assertEqual(estimate.standard_error, 0.)

    def test_stochastic_one_fixing_reduces_to_vanilla(self):
        for settlement, strike in [("quanto", 100.), ("floating_fx", 100.), ("composite", 15000.)]:
            vanilla = rp.Product.european_vanilla(10, 2, EXPIRY, strike, 1., "put")
            expected = self.option(vanilla, settlement, payment_date=PAY).analytical_price()
            asian = rp.Product.arithmetic_asian(10, 2, strike, 1., "put",
                                                [rp.AsianObservation.unknown(EXPIRY, 1.)], PAY)
            rqmc = rp.Engine.randomized_quasi_monte_carlo(1024, 297, scramble_count=8)
            estimate = self.option(asian, settlement).price_mc(rqmc)
            self.assertLess(abs(estimate.value - expected), 6. * estimate.standard_error)

    def test_two_date_geometric_asian_against_its_lognormal_law(self):
        # The two log spots share the first interval's Brownian increment.
        rqmc = rp.Engine.randomized_quasi_monte_carlo(2048, 128, scramble_count=12, antithetic=True)
        equity_vol, fx_vol, rho = .20, .10, -.6
        composite_vol = math.sqrt(equity_vol**2 + fx_vol**2 + 2*rho*equity_vol*fx_vol)
        def cdf(x):
            return .5 * math.erfc(-x / math.sqrt(2.))
        for settlement, spot, carry, vol, factor in [
            ("quanto", 100., .04-.02-rho*equity_vol*fx_vol, equity_vol, 145.*math.exp(-.01)),
            ("floating_fx", 100., .04-.02, equity_vol, 150.*math.exp(-.04)),
            ("composite", 15_000., .01-.02, composite_vol, math.exp(-.01)),
        ]:
            with self.subTest(settlement=settlement):
                strike = .99 * spot
                contract = rp.ContractBuilder("geo", 2, 10, "close").asian(
                    "geometric_average", "call", strike, 1., [MID, EXPIRY], EXPIRY)
                graph, = contract.to_graphs(TODAY, [10])
                result = self.option(rp.Product.from_graph(graph), settlement, correlation=rho).price_mc(rqmc)
                variance = vol**2 * (3*T_MID + 1.)/4
                mean = math.log(spot) + (carry - vol**2/2) * (1.+T_MID)/2
                d2 = (mean - math.log(strike))/math.sqrt(variance)
                reference = factor * (math.exp(mean+variance/2)*cdf(d2+math.sqrt(variance)) - strike*cdf(d2))
                self.assertAlmostEqual(result.value, reference, delta=6*result.standard_error)

    def test_continuous_barrier_captures_carry_reversal_between_dates(self):
        # FX is deterministic. The foreign stock rises to 125 at the curve pillar
        # and finishes at 100, so only continuous monitoring sees the upper hit.
        curve_with_peak = rp.DiscountCurve(1, [0., .5, 1.], [1., .8, 1.])
        self.market = rp.Market.equity(1, 10, 100., curve_with_peak, curve(3, 0.))
        for monitoring, expected in [("discrete", 10.), ("continuous", 3.)]:
            barrier = rp.Product.barrier(10, 2, EXPIRY, 90., 120., 1., "call", "up",
                                         "knock_out", monitoring, [TODAY, EXPIRY], PAY, rebate=3.)
            result = self.option(barrier, "quanto", fixed_fx_rate=2., equity_volatility=0.,
                                 fx_volatility=0.).price_mc(self.engine)
            self.assertAlmostEqual(result.value, 2. * expected * math.exp(-.01*T_PAY), places=10)

    def test_seasoned_composite_uses_known_domestic_fixing(self):
        obs = [rp.AsianObservation.known("2025-12-01", .5, 12_000.),
               rp.AsianObservation.unknown(EXPIRY, .5)]
        asian = rp.Product.arithmetic_asian(10, 2, 10_000., 1., "call", obs, PAY)
        result = self.option(asian, "composite", equity_volatility=0., fx_volatility=0.).price_mc(self.engine)
        expected = (.5*(12_000.+15_000.*math.exp(-.01))-10_000.) * math.exp(-.01*T_PAY)
        self.assertAlmostEqual(result.value, expected, places=9)

    def test_autocall_cancels_later_cashflows_and_uses_each_payment_fx(self):
        note = rp.ContractBuilder("note", 2, 10, "close").autocall(
            100., 100., 60., [MID, EXPIRY], ["2026-07-15", PAY],
            [200., 200.], [80., 80.], [.03, .05])
        def price(note):
            graph, = note.to_graphs(TODAY, [10])
            return self.option(rp.Product.from_graph(graph), "floating_fx",
                               equity_volatility=0., fx_volatility=0.).price_mc(self.engine).value
        df1, df2 = math.exp(-.04 * 195/365), math.exp(-.04*T_PAY)
        self.assertAlmostEqual(price(note), 150.*(3.*df1 + 105.*df2), places=8)
        called = rp.ContractBuilder("call", 2, 10, "close").autocall(
            100., 100., 60., [MID, EXPIRY], ["2026-07-15", PAY],
            [90., 200.], [80., 80.], [.03, .05])
        self.assertAlmostEqual(price(called), 150.*103.*df1, places=8)

    def test_floating_bermudan_matches_foreign_exercise_policy(self):
        def put(currency):
            return rp.Product.american_vanilla(10, currency, EXPIRY, 110., 1., "put", [MID, EXPIRY])
        lsm = rp.LsmConfig(rp.Engine.pseudo_monte_carlo(39, 1024, antithetic=True))
        native = rp.PricingRequest(TODAY, put(1), self.market, rp.Model.black_scholes(.20),
                                   self.engine, rp.RiskRequest(), lsm=lsm)
        ref = rp.PricingPlan.compile(native, worker_threads=1).evaluate()
        estimate = self.option(put(2), "floating_fx").price_mc(self.engine, lsm=lsm)
        self.assertAlmostEqual(estimate.value, 150.*ref.value, places=9)
        self.assertAlmostEqual(estimate.standard_error, 150.*ref.standard_error, places=10)

    def test_validation(self):
        product = rp.Product.digital(10, 2, EXPIRY, 100., 1., "call", "cash", payment_date=PAY)
        with self.assertRaises(ValueError):
            self.option(product, "quanto", payment_date=EXPIRY)
        with self.assertRaises(rp.PricingError):
            self.option(product, "quanto").analytical_price()
        builder = rp.SourceGraphBuilder()
        graph = builder.finish([builder.literal(1.)], [EXPIRY], currency_id=2, underlying_ids=[10, 11])
        with self.assertRaises(ValueError):
            self.option(rp.Product.from_graph(graph), "composite")
