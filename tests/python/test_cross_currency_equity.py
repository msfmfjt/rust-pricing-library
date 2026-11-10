"""Valuation identities and joint path pricing through the installed Python API."""
from math import exp, sqrt
import unittest

import rust_pricing as rp


def curve(curve_id, rate):
    return rp.DiscountCurve(curve_id, [0., 1.], [1., exp(-rate)])


class CrossCurrencyEquityTests(unittest.TestCase):
    def setUp(self):
        self.usd_curve = curve(1, .04)
        self.jpy_curve = curve(2, .01)
        self.dividend = curve(3, .02)
        self.market = rp.Market.equity(1, 10, 100., self.usd_curve, self.dividend)

    def option(self, settlement, side="call", *, strike=None, expiry="2027-01-01",
               valuation="2026-01-01", market=None, currency=2, stock=10, **kwargs):
        if strike is None:
            strike = 15_000. if settlement == "composite" else 100.
        params = dict(fx_spot=150., equity_volatility=.20, fx_volatility=.10, correlation=.35)
        if settlement == "quanto":
            params["fixed_fx_rate"] = 145.
        params.update(kwargs)
        product = rp.Product.european_vanilla(stock, currency, expiry, strike, 1., side)
        return rp.CrossCurrencyEquityOption(
            valuation, product, market or self.market, self.jpy_curve, settlement=settlement, **params)

    def test_quanto_reference_call_and_put(self):
        # Haug's half-year test reproduced in QuantLib test-suite/quantooption.cpp.
        # Convert annual inputs from ACT/360 to ACT/365F for the same 180 days.
        scale = 365 / 360
        foreign = rp.Market.equity(1, 10, 100., curve(1, .05 * scale), curve(3, .04 * scale))
        domestic_curve = curve(2, .08 * scale)
        for side, conversion, expected in [("call", 1.5, 5.3280), ("put", 1., 8.1636)]:
            product = rp.Product.european_vanilla(10, 2, "2026-06-30", 105., 1., side)
            option = rp.CrossCurrencyEquityOption(
                "2026-01-01", product, foreign, domestic_curve, settlement="quanto",
                fx_spot=2., equity_volatility=.2 * sqrt(scale), fx_volatility=.1 * sqrt(scale),
                correlation=.3, fixed_fx_rate=conversion)
            self.assertAlmostEqual(option.analytical_price(), expected, delta=1.e-4)

    def test_call_put_parity_for_each_settlement_with_payment_lag(self):
        fx_forward = 150. * exp(.01 - .04)
        equity_forward = 100. * exp(.04 - .02)
        payment_df = exp(-.01 * (1. + 30 / 365))
        parity = {
            "quanto": 145. * (equity_forward * exp(-.35 * .20 * .10) - 100.),
            "floating_fx": fx_forward * (equity_forward - 100.),
            "composite": 150. * 100. * exp(.01 - .02) - 15_000.,
        }
        for settlement, spread in parity.items():
            with self.subTest(settlement=settlement):
                call, put = [self.option(settlement, side, payment_date="2027-01-31").analytical_price()
                             for side in ("call", "put")]
                self.assertAlmostEqual(call - put, payment_df * spread, places=9)

    def test_conversion_and_correlation(self):
        for settlement in ("quanto", "floating_fx", "composite"):
            low, high = [self.option(settlement, correlation=rho).analytical_price() for rho in (-.8, .8)]
            if settlement == "quanto":
                self.assertGreater(low, high)
                self.assertEqual(high, self.option(settlement, correlation=.8, fx_spot=190.).analytical_price())
            elif settlement == "floating_fx":
                self.assertEqual(low, high)
                self.assertEqual(low, self.option(settlement, fx_volatility=0.).analytical_price())
            else:
                self.assertLess(low, high)
        # Deterministic FX makes the three payoffs equivalent at appropriately scaled strikes.
        forward = 150. * exp(.01 - .04)
        floating = self.option("floating_fx", fx_volatility=0.).analytical_price()
        quanto = self.option("quanto", fx_volatility=0., fixed_fx_rate=forward).analytical_price()
        composite = self.option("composite", fx_volatility=0., strike=100. * forward).analytical_price()
        self.assertAlmostEqual(quanto, floating, places=9)
        self.assertAlmostEqual(composite, floating, places=9)

    def test_joint_path_prices_and_uncertainty(self):
        engines = [rp.Engine.pseudo_monte_carlo(387, 8192, antithetic=True),
                   rp.Engine.randomized_quasi_monte_carlo(1024, 954, scramble_count=8, antithetic=True)]
        for engine in engines:
            for settlement in ("quanto", "floating_fx", "composite"):
                for side in ("call", "put"):
                    with self.subTest(engine=engine, settlement=settlement, side=side):
                        option = self.option(settlement, side, payment_date="2027-01-31")
                        result = option.price_mc(engine)
                        self.assertGreater(result.standard_error, 0.)
                        self.assertAlmostEqual(result.value, option.analytical_price(),
                                               delta=5. * result.standard_error)

    def test_expiry_and_singular_correlation(self):
        engine = rp.Engine.pseudo_monte_carlo(387, 128)
        for settlement, strike, amount in [("quanto", 90., 1450.), ("floating_fx", 90., 1500.),
                                           ("composite", 14000., 1000.)]:
            option = self.option(settlement, strike=strike, expiry="2026-01-01", payment_date="2027-01-01")
            result = option.price_mc(engine)
            self.assertAlmostEqual(result.value, amount * exp(-.01), places=10)
            self.assertEqual(result.standard_error, 0.)
        option = self.option("composite", strike=14_000., correlation=-1., fx_volatility=.20)
        expected = 15_000. * exp(-.02) - 14_000. * exp(-.01)
        self.assertAlmostEqual(option.analytical_price(), expected, places=9)
        self.assertAlmostEqual(option.price_mc(engine).value, expected, places=9)

    def test_reject_inconsistent_inputs(self):
        for kwargs in [dict(fixed_fx_rate=None), dict(fixed_fx_rate=0.), dict(correlation=1.001),
                       dict(correlation=float("nan")), dict(fx_spot=0.), dict(fx_volatility=-.1),
                       dict(stock=11), dict(currency=1), dict(payment_date="2026-12-01"),
                       dict(valuation="2027-01-02")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.option("quanto", **kwargs)
        with self.assertRaises(ValueError):
            self.option("floating_fx", fixed_fx_rate=145.)
        with self.assertRaises(ValueError):
            self.option("unknown")
        discrete = rp.Market.equity(1, 10, 100., self.usd_curve, self.dividend,
                                    discrete_dividends=[rp.DividendEvent.fixed_cash(1, .25, 1.)])
        with self.assertRaises(ValueError):
            self.option("composite", market=discrete)
