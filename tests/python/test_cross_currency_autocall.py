"""Multi-payment USD equity notes with JPY settlement."""
import math
import unittest
from datetime import date

import rust_pricing as rp


AS_OF = "2026-01-01"
OBS = ["2026-04-01", "2026-07-01", "2027-01-01"]
PAY = ["2026-04-15", "2026-07-15", "2027-02-01"]
SETTLEMENTS = ("quanto", "floating_fx", "composite")


def years(day):
    return (date.fromisoformat(day) - date.fromisoformat(AS_OF)).days / 365


def curve(identifier, rate):
    return rp.DiscountCurve(identifier, [0., 1.], [1., math.exp(-rate)])


def note(settlement, *, calls, coupons, spot=100., protection=.70,
         observations=OBS, payments=PAY, rates=(.02, .03, .05), vols=(0., 0.)):
    unit = 150. if settlement == "composite" else 1.
    initial = 100. * unit
    contract = rp.ContractBuilder("note", 2, 10, "close").autocall(
        notional=100. * unit, initial_spot=initial, protection_level=protection * initial,
        observation_dates=observations, payment_dates=payments,
        call_levels=[v * initial for v in calls], coupon_levels=[v * initial for v in coupons],
        coupon_rates=rates)
    graph, = contract.to_graphs(AS_OF, [10])
    return equity_option(settlement, graph, spot, vols)


def equity_option(settlement, graph, spot=100., vols=(0., 0.)):
    return rp.CrossCurrencyEquityOption(
        AS_OF, rp.Product.from_graph(graph), rp.Market.equity(1, 10, spot, curve(1, .04), curve(3, .02)),
        curve(2, .01), settlement=settlement, fixed_fx_rate=145. if settlement == "quanto" else None,
        fx_spot=150., equity_volatility=vols[0], fx_volatility=vols[1], correlation=.35)


def pv(settlement, fractions, payments=PAY, fixings=None):
    # Build discounted cash in JPY directly from each date's deterministic FX forward.
    fixings = payments if fixings is None else fixings
    value = 0.
    for amount, payment, fixing in zip(fractions, payments, fixings, strict=True):
        conversion = 150. if settlement == "composite" else 145.
        if settlement == "floating_fx":
            conversion = 150. * math.exp((.01 - .04) * years(fixing))
        value += 100. * amount * conversion * math.exp(-.01 * years(payment))
    return value


class CrossCurrencyAutocalls(unittest.TestCase):
    def setUp(self):
        self.engine = rp.Engine.pseudo_monte_carlo(391, 32, antithetic=True)

    def test_coupon_call_and_downside_events_in_all_three_settlement_conventions(self):
        for settlement in SETTLEMENTS:
            composite = settlement == "composite"
            downside = .65 * math.exp(-.01 if composite else .02)
            cases = [
                ("miss middle coupon", 100., [1.2]*3, [.8, 1.2, .8], [.02, 0., 1.05]),
                ("called after first coupon", 100., [1.2, .95, 1.2], [.8]*3, [.02, 1.03, 0.]),
                ("USD and JPY triggers diverge", 100., [1.2, 1.005, 1.2], [.8]*3,
                 [.02, .03, 1.05] if composite else [.02, 1.03, 0.]),
                ("unprotected redemption", 65., [1.2]*3, [.6, .8, .8], [.02, 0., downside]),
            ]
            for label, spot, calls, coupons, amounts in cases:
                with self.subTest(settlement=settlement, label=label):
                    result = note(settlement, spot=spot, calls=calls, coupons=coupons).price_mc(self.engine)
                    self.assertAlmostEqual(result.value, pv(settlement, amounts), places=8)
                    self.assertEqual(result.standard_error, 0.)

    def test_partial_fixings_shared_by_coupon_and_redemption(self):
        base = note("floating_fx", calls=[1.2]*3, coupons=[.8]*3)
        amounts = [.02, .03, 1.05]
        pairs = [(date.fromisoformat(PAY[2]), date.fromisoformat(OBS[2])), (PAY[0], OBS[0])]
        custom = base.with_payment_fx_fixings(pairs)
        expected = pv("floating_fx", amounts, fixings=[OBS[0], PAY[1], OBS[2]])
        self.assertAlmostEqual(custom.price_mc(self.engine).value, expected, places=8)
        for original in (base, custom.with_payment_fx_fixings([])):
            self.assertAlmostEqual(original.price_mc(self.engine).value, pv("floating_fx", amounts), places=8)

    def test_memory_coupon_converts_when_recovered(self):
        for settlement in SETTLEMENTS:
            unit = 150. if settlement == "composite" else 1.
            observations = [
                rp.AutocallObservation(OBS[0], PAY[0], coupon_amount=2.*unit, coupon_level=1.20),
                rp.AutocallObservation(OBS[1], PAY[1], coupon_amount=3.*unit, coupon_level=.90, call_level=.95),
            ]
            for memory in (False, True):
                with self.subTest(settlement=settlement, memory=memory):
                    product = rp.MultiAssetProduct.autocallable(
                        [10], [100.*unit], observations, OBS[2], PAY[2], notional=100.*unit,
                        final_barrier=.70, memory=memory, on_autocall="pay", on_maturity="forfeit", currency_id=2)
                    option = equity_option(settlement, product.to_graph())
                    if settlement == "floating_fx":
                        option = option.with_payment_fx_fixings(list(zip(PAY[:2], OBS[:2])))
                    expected = pv(settlement, [0., 1.05 if memory else 1.03, 0.], fixings=OBS)
                    self.assertAlmostEqual(option.price_mc(self.engine).value, expected, places=8)

    def test_two_redemption_dates_against_digital_probability(self):
        obs, pay = OBS[1:], PAY[1:]
        engine = rp.Engine.randomized_quasi_monte_carlo(2048, 177, scramble_count=16, antithetic=True)
        for settlement, carry, vol in [
            ("quanto", .04 - .02 - .35*.20*.10, .20),
            ("floating_fx", .04 - .02, .20),
            ("composite", .01 - .02, math.sqrt(.20**2 + .10**2 + 2*.35*.20*.10)),
        ]:
            t = years(obs[0])
            z = ((carry - vol**2/2)*t - math.log(1.02)) / (vol*math.sqrt(t))
            early = .5 * math.erfc(-z/math.sqrt(2.))
            base = note(settlement, calls=[1.02, .01], coupons=[1.02, .01], protection=0.,
                        observations=obs, payments=pay, rates=[.03, .12], vols=(.20, .10))
            variants = [(base, pay)]
            if settlement == "floating_fx":
                variants.append((base.with_payment_fx_fixings(list(zip(pay, obs))), obs))
            for option, fixings in variants:
                with self.subTest(settlement=settlement, fixings=fixings):
                    result = option.price_mc(engine)
                    expected = pv(settlement, [1.03*early, 1.12*(1.-early)], pay, fixings)
                    self.assertAlmostEqual(result.value, expected, delta=6*result.standard_error)

    def test_invalid_fixings(self):
        args = dict(calls=[1.2]*3, coupons=[.8]*3)
        base = note("floating_fx", **args)
        for schedule in [
            [(OBS[0], OBS[0])], [(PAY[0], "2025-12-31")], [(PAY[0], PAY[1])],
            [(PAY[0], OBS[0]), (PAY[0], PAY[0])], [(PAY[1], OBS[0])],
        ]:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                base.with_payment_fx_fixings(schedule)
        for settlement in ("quanto", "composite"):
            with self.assertRaises(ValueError):
                note(settlement, **args).with_payment_fx_fixings([(PAY[0], OBS[0])])

    def test_known_payments_with_nonflat_curves(self):
        # The first FX fixes today. The second fixes after the curve's middle pillar.
        fixing = "2026-10-01"
        times = [0., years(PAY[0]), years(fixing), years(PAY[2])]
        usd_dfs, jpy_dfs = [1., .99, .96, .925], [1., 1.002, .995, .987]
        usd = rp.DiscountCurve(1, times, usd_dfs)
        jpy = rp.DiscountCurve(2, times, jpy_dfs)
        builder = rp.SourceGraphBuilder()
        graph = builder.finish([builder.literal(40.), builder.literal(-10.)],
                               [PAY[0], PAY[2]], currency_id=2, underlying_ids=[10])
        base = rp.CrossCurrencyEquityOption(
            AS_OF, rp.Product.from_graph(graph), rp.Market.equity(1, 10, 100., usd, curve(3, 0.)),
            jpy, settlement="floating_fx", fx_spot=150., equity_volatility=.20, fx_volatility=.10,
            correlation=.35)
        result = base.with_payment_fx_fixings([(PAY[0], AS_OF), (PAY[2], fixing)]).price_mc(self.engine)
        expected = 150.*40.*jpy_dfs[1] - 150.*10.*usd_dfs[2]/jpy_dfs[2]*jpy_dfs[3]
        self.assertAlmostEqual(result.value, expected, places=9)
        self.assertEqual(result.standard_error, 0.)
