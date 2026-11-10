"""Quarterly coupons on a USD stock, with redemption and settlement in JPY."""
from math import exp

import rust_pricing as rp


valuation = "2026-01-05"
observations = ["2026-04-07", "2026-07-07", "2026-10-07", "2027-01-05"]
payments = ["2026-04-14", "2026-07-14", "2026-10-14", "2027-01-12"]
stock, usd, jpy = 10, 1, 2
spot, fx = 100., 150.


def curve(identifier, rate):
    return rp.DiscountCurve(identifier, [0., 1.], [1., exp(-rate)])


market = rp.Market.equity(usd, stock, spot, curve(1, .04), curve(3, .02))
jpy_curve = curve(2, .01)
engine = rp.Engine.randomized_quasi_monte_carlo(4096, 271, scramble_count=16, antithetic=True)


def autocall(settlement):
    # Composite principal and reference levels are in JPY. The other contracts
    # specify USD principal and USD levels before converting the resulting cash.
    unit = fx if settlement == "composite" else 1.
    reference = spot * unit
    note = rp.ContractBuilder("quarterly-note", jpy, stock, "close").autocall(
        notional=1_000. * unit,
        initial_spot=reference,
        protection_level=.70 * reference,
        observation_dates=observations,
        payment_dates=payments,
        call_levels=[ratio * reference for ratio in (1.05, 1.025, 1., 1.)],
        coupon_levels=[.80 * reference] * 4,
        coupon_rates=[.02] * 4,
    )
    graph, = note.to_graphs(valuation, [stock])
    return rp.CrossCurrencyEquityOption(
        valuation, rp.Product.from_graph(graph), market, jpy_curve, settlement=settlement,
        fx_spot=fx, equity_volatility=.20, fx_volatility=.10, correlation=.35,
        fixed_fx_rate=fx if settlement == "quanto" else None,
    )


if __name__ == "__main__":
    floating = autocall("floating_fx")
    # Short-rate volatility is in absolute annual rate units. Matrix order is
    # USD equity, JPY per USD, JPY short rate, USD short rate.
    jpy_rates = rp.HullWhiteModel(.25, [0.], [.008])
    usd_rates = rp.HullWhiteModel(.15, [0., .5], [.014, .017])
    floating_hw = floating.with_hull_white_rates(jpy_rates, usd_rates, correlations=[
        [1., .35, .1, .2], [.35, 1., -.2, .15], [.1, -.2, 1., .3], [.2, .15, .3, 1.],
    ]).with_payment_fx_fixings(list(zip(payments, observations)))
    for label, note in [
        ("Quanto, fixed at 150 JPY/USD", autocall("quanto")),
        ("Floating FX, fixed on each payment", floating),
        ("Floating FX, fixed on each observation", floating.with_payment_fx_fixings(
            list(zip(payments, observations)))),
        ("Composite, observed in JPY", autocall("composite")),
        ("Floating FX on observation, JPY/USD HW", floating_hw),
    ]:
        estimate = note.price_mc(engine)
        print(f"{label:40} PV {estimate.value:12,.2f} JPY   SE {estimate.standard_error:6.2f} JPY")
