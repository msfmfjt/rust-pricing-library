"""Coupons and principal in contractual units on a shared USD/EUR/JPY path."""
import math
import unittest

import rust_pricing as rp

from test_cross_currency_basket import TODAY, OBS, FIX, PAY, IDS, curve, years
from test_cross_currency_mixed_basket import C, QUOTES, JPY, USD, EUR


FIRST, FIRST_PAY = "2026-04-01", "2026-04-08"
END = "2027-01-01"
NOTIONAL = 150_000.


def note_terms(observations, **changes):
    return dict(underlying_ids=IDS, reference_levels=[100., 80.], observations=observations,
                maturity=END, maturity_payment=PAY, notional=NOTIONAL, final_barrier=.70,
                memory=True, on_autocall="pay", on_maturity="pay") | changes


def legs(terms, coupon=USD, principal=JPY):
    return rp.MultiAssetProduct.autocallable_legs(
        **terms, coupon_currency_id=coupon, principal_currency_id=principal)


class AutocallLegTests(unittest.TestCase):
    def setUp(self):
        self.domestic_rate = .02
        self.foreign_rates = {USD: .09, EUR: .04}
        self.domestic = curve(2, self.domestic_rate)
        self.markets = []
        for i, (currency, spot) in enumerate([(USD, 100.), (EUR, 80.)]):
            foreign = curve(3+i, self.foreign_rates[currency])
            # Flat local equity paths in the zero-volatility tests.
            self.markets.append(rp.Market.equity(currency, IDS[i], spot, foreign, foreign))
        self.config = dict(equity_models=[rp.Model.black_scholes(0.)]*2,
                           fx_quotes=[(c, x, 0.) for c, x, _ in QUOTES],
                           correlations=[row[:4] for row in C[:4]], maximum_step=.25)
        self.engine = rp.Engine.pseudo_monte_carlo(107, 16)

    def basket(self, products, **kwargs):
        return rp.CrossCurrencyEquityBasket.from_legs(
            TODAY, products, self.markets, self.domestic, settlement_currency_id=JPY,
            **(self.config | kwargs))

    def fx(self, currency, fixing):
        spot = dict((c, x) for c, x, _ in QUOTES)[currency]
        return spot * math.exp((self.domestic_rate-self.foreign_rates[currency])*years(fixing))

    def discount(self, day):
        return math.exp(-self.domestic_rate*years(day))

    def test_call_bonus_recovery_and_forfeit_on_same_day_as_jpy_principal(self):
        # With the lower coupon trigger the second observation releases
        # both coupons before the call, even under the forfeit policy.
        for policy, trigger, usd in [("pay", 1.1, 39.), ("forfeit", 1.1, 7.),
                                     ("forfeit", .9, 39.)]:
            with self.subTest(policy=policy, trigger=trigger):
                obs = [rp.AutocallObservation(FIRST, FIRST_PAY, coupon_amount=13., coupon_level=1.1),
                       rp.AutocallObservation(OBS, FIX, coupon_amount=19., coupon_level=trigger,
                                              call_level=.95, call_coupon_amount=7.),
                       rp.AutocallObservation(END, PAY, coupon_amount=500., coupon_level=.8)]
                products = legs(note_terms(obs, on_autocall=policy))
                self.assertEqual([p.currency_id for p in products], [USD, JPY])
                basket = self.basket(products).with_payment_fx_fixings([(FIX, OBS)])
                expected = self.discount(FIX)*(NOTIONAL + usd*self.fx(USD, OBS))
                self.assertAlmostEqual(basket.price_mc(self.engine).value, expected, places=7)

    def test_ordinary_coupon_and_later_memory_follow_coupon_currency_through_maturity(self):
        # First coupon pays; subsequent missed amount is either recovered at
        # maturity or forfeited. The worst local return is 50% of reference.
        obs = [rp.AutocallObservation(FIRST,FIRST_PAY,coupon_amount=3.,coupon_level=.4),
               rp.AutocallObservation(OBS,FIX,coupon_amount=9.,coupon_level=.8)]
        for policy, recovered in [("pay", 9.), ("forfeit", 0.)]:
            terms = note_terms(obs, reference_levels=[100., 160.], on_maturity=policy)
            products = legs(terms)
            schedule = [(FIRST_PAY, FIRST)] + ([(PAY, END)] if recovered else [])
            basket = self.basket(products).with_payment_fx_fixings(schedule)
            expected = self.discount(FIRST_PAY)*3.*self.fx(USD,FIRST)
            expected += self.discount(PAY)*(.5*NOTIONAL + recovered*self.fx(USD,END))
            self.assertAlmostEqual(basket.price_mc(self.engine).value, expected, places=7)

    def test_coupon_fixing_can_precede_same_payment_domestic_principal_observation(self):
        obs = [rp.AutocallObservation(OBS, PAY, coupon_amount=12., coupon_level=.8)]
        products = legs(note_terms(obs, on_maturity="forfeit"))
        # USD coupon depends on OBS only. JPY principal depends on the later END.
        basket = self.basket(products)
        for fixing in [OBS, PAY]:
            schedule = [] if fixing == PAY else [(PAY, fixing)]
            pv = basket.with_payment_fx_fixings(schedule).price_mc(self.engine)
            expected = self.discount(PAY)*(NOTIONAL + 12.*self.fx(USD, fixing))
            self.assertAlmostEqual(pv.value, expected, places=7)
        with self.assertRaises(ValueError):
            basket.with_payment_fx_fixings([(PAY, FIRST)])

    def test_today_and_composite_references_with_euro_principal(self):
        obs = [rp.AutocallObservation(TODAY, TODAY, coupon_amount=11., coupon_level=.9,
                                      call_level=.95, call_coupon_amount=3.)]
        terms = note_terms(obs, maturity=TODAY, maturity_payment=TODAY, notional=900.,
                           reference_levels=[100.*150., 80.*173.])
        result = self.basket(legs(terms, USD, EUR), performance="composite").price_mc(self.engine)
        self.assertEqual((result.value, result.standard_error), (14.*150.+900.*173., 0.))

    def test_same_currency_legs_match_full_note_value_and_sampling_error(self):
        obs = [rp.AutocallObservation(FIRST, FIRST_PAY, coupon_amount=11., coupon_level=1.1,
                                      call_level=1.02, call_coupon_amount=8.),
               rp.AutocallObservation(OBS, FIX, coupon_amount=15., coupon_level=.9, call_level=.98)]
        config = self.config | dict(equity_models=[rp.Model.black_scholes(v) for v in [.22,.31]],
                                    fx_quotes=QUOTES, correlations=C,
                                    rates=[rp.HullWhiteModel(.2,[0.],[v]) for v in [.01,.02,.03]])
        engine = rp.Engine.pseudo_monte_carlo(989, 1024, antithetic=True)
        for currency, mode in [(USD,"floating_fx"), (JPY,"quanto")]:
            terms = note_terms(obs, notional=1000.)
            whole = rp.MultiAssetProduct.autocallable(**terms, currency_id=JPY).to_graph()
            old = rp.CrossCurrencyEquityBasket.from_currencies(
                TODAY, whole, self.markets, self.domestic, **config,
                settlement=mode, payout_currency_id=currency)
            new = self.basket(legs(terms, currency, currency), **config)
            if currency == USD:
                schedule = [(FIRST_PAY,FIRST), (FIX,OBS), (PAY,END)]
                old, new = (b.with_payment_fx_fixings(schedule) for b in (old, new))
            v, w = old.price_mc(engine), new.price_mc(engine)
            self.assertAlmostEqual(w.value, v.value, places=8)
            self.assertAlmostEqual(w.standard_error, v.standard_error, places=9)

    def test_three_rate_mixed_note_against_usd_and_jpy_bonds(self):
        # Valuation-day exercise makes both amounts certain; each pays on PAY.
        obs = [rp.AutocallObservation(TODAY, PAY, coupon_amount=20., coupon_level=.8,
                                      call_level=.9)]
        products = legs(note_terms(obs, maturity=TODAY))
        config = dict(fx_quotes=QUOTES, correlations=C,
                      rates=[rp.HullWhiteModel(.2,[0.],[v]) for v in [.01,.02,.03]])
        engine = rp.Engine.randomized_quasi_monte_carlo(1024, 837, scramble_count=8,
                                                        antithetic=True, brownian_bridge=True)
        result = self.basket(products, **config).price_mc(engine)
        expected = NOTIONAL*self.discount(PAY) + 20.*150.*math.exp(-.09*years(PAY))
        self.assertAlmostEqual(result.value, expected, delta=6.*result.standard_error+1.e-8)

    def test_invalid_currency_underlyings_and_fx_fixings(self):
        obs = [rp.AutocallObservation(OBS,FIX,coupon_amount=1.,coupon_level=.8)]
        terms = note_terms(obs, on_maturity="forfeit")
        products = legs(terms)
        bad_ids = legs(terms | dict(underlying_ids=[20,10]))[1]
        for invalid in [[], legs(terms, coupon=5), [products[0], bad_ids]]:
            with self.assertRaises(ValueError):
                self.basket(invalid)
        with self.assertRaises(ValueError):
            self.basket(products, performance="floating_fx")
        with self.assertRaises(ValueError):
            self.basket(products).with_payment_fx_fixings([(PAY, END)])  # JPY only
