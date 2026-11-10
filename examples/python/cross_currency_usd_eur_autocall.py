"""A US and a euro-area stock, with quarterly memory coupons settled in yen."""
from math import exp

import rust_pricing as rp


USD, JPY, EUR = 1, 2, 3
VALUATION = "2026-01-05"
STOCK_IDS = [10, 20]
OBSERVATIONS = ["2026-04-07", "2026-07-07", "2026-10-07", "2027-01-05"]
PAYMENTS = ["2026-04-14", "2026-07-14", "2026-10-14", "2027-01-12"]
FX_QUOTES = [(USD, 150., .12), (EUR, 173., .14)]  # currency, JPY per unit, annual volatility
YEN_PER_UNIT = {JPY: 1., USD: 150., EUR: 173.}

# US stock, European stock, JPY per USD, JPY per EUR, JPY rate, USD rate, EUR rate
CORRELATIONS = [
    [1., .42, -.32, .17, .10, .26, -.16],
    [.42, 1., .12, .31, -.12, -.14, .28],
    [-.32, .12, 1., .38, .16, -.18, .10],
    [.17, .31, .38, 1., -.20, .12, .25],
    [.10, -.12, .16, -.20, 1., .22, .17],
    [.26, -.14, -.18, .12, .22, 1., -.15],
    [-.16, .28, .10, .25, .17, -.15, 1.],
]


def curve(identifier, rate):
    return rp.DiscountCurve(identifier, [0., 2.], [1., exp(-2*rate)])


def note(settlement, payout_currency):
    # Each contract starts with the equivalent of JPY 150,000. Coupon and
    # redemption amounts are expressed in this one contractual cash currency.
    notional = 150_000. / YEN_PER_UNIT[payout_currency]
    references = [100., 80.]
    if settlement == "composite":
        references = [100.*YEN_PER_UNIT[USD], 80.*YEN_PER_UNIT[EUR]]
    observations = [
        rp.AutocallObservation(obs, pay, coupon_amount=.02*notional, coupon_level=.80,
                               call_level=trigger)
        for obs, pay, trigger in zip(OBSERVATIONS, PAYMENTS, [1.05, 1.025, 1., None])
    ]
    graph = rp.MultiAssetProduct.autocallable(
        STOCK_IDS, references, observations, OBSERVATIONS[-1], PAYMENTS[-1], currency_id=JPY,
        notional=notional, final_barrier=.70, memory=True, on_autocall="pay", on_maturity="forfeit",
    ).to_graph()
    markets = [
        rp.Market.equity(USD, 10, 100., curve(1, .04), curve(4, .015)),
        rp.Market.equity(EUR, 20, 80., curve(3, .025), curve(5, .030)),
    ]
    basket = rp.CrossCurrencyEquityBasket.from_currencies(
        VALUATION, graph, markets, curve(2, .01),
        equity_models=[rp.Model.black_scholes(.20), rp.Model.black_scholes(.28)],
        settlement=settlement, payout_currency_id=payout_currency, fx_quotes=FX_QUOTES,
        correlations=CORRELATIONS, maximum_step=.25,
        rates=[rp.HullWhiteModel(.25, [0.], [.008]), rp.HullWhiteModel(.15, [0., .5], [.014, .017]),
               rp.HullWhiteModel(.30, [0.], [.012])],
    )
    if settlement == "floating_fx":
        return basket.with_payment_fx_fixings(list(zip(PAYMENTS, OBSERVATIONS)))
    return basket


if __name__ == "__main__":
    engine = rp.Engine.randomized_quasi_monte_carlo(2048, 8237, scramble_count=12,
                                                    antithetic=True, brownian_bridge=True)
    for label, mode, currency in [
        ("Local performance, JPY cash", "quanto", JPY),
        ("Local performance, USD cash", "floating_fx", USD),
        ("Local performance, EUR cash", "floating_fx", EUR),
        ("JPY performance, JPY cash", "composite", JPY),
    ]:
        pv = note(mode, currency).price_mc(engine)
        print(f"{label:30} {pv.value:12,.2f} JPY   SE {pv.standard_error:6.2f} JPY")
