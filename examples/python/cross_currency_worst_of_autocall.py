"""Two USD stocks, quarterly memory coupons, early redemption and JPY settlement."""
from math import exp

import rust_pricing as rp


VALUATION = "2026-01-05"
OBSERVATIONS = ["2026-04-07", "2026-07-07", "2026-10-07", "2027-01-05"]
PAYMENTS = ["2026-04-14", "2026-07-14", "2026-10-14", "2027-01-12"]
STOCKS = [10, 20]
FX = 150.  # JPY per USD
# Stock A, stock B, JPY/USD, JPY rate, USD rate.
CORRELATIONS = [
    [1., .45, -.25, .10, .20],
    [.45, 1., .15, -.10, .25],
    [-.25, .15, 1., -.20, .15],
    [.10, -.10, -.20, 1., .30],
    [.20, .25, .15, .30, 1.],
]


def curve(identifier, rate):
    return rp.DiscountCurve(identifier, [0., 2.], [1., exp(-2*rate)])


def note(settlement):
    # Composite references, coupons and principal are denominated in JPY.
    unit = FX if settlement == "composite" else 1.
    notional = 1000.*unit
    obs = [
        rp.AutocallObservation(d, p, coupon_amount=.02*notional, coupon_level=.80,
                              call_level=trigger)
        for d, p, trigger in zip(OBSERVATIONS, PAYMENTS, [1.05, 1.025, 1., None])
    ]
    graph = rp.MultiAssetProduct.autocallable(
        STOCKS, [100.*unit, 80.*unit], obs, OBSERVATIONS[-1], PAYMENTS[-1],
        notional=notional, final_barrier=.70, memory=True, on_autocall="pay",
        on_maturity="forfeit", currency_id=2,
    ).to_graph()
    usd = curve(1, .04)
    markets = [rp.Market.equity(1, 10, 100., usd, curve(3, .015)),
               rp.Market.equity(1, 20, 80., usd, curve(4, .025))]
    option = rp.CrossCurrencyEquityBasket(
        VALUATION, graph, markets, curve(2, .01),
        equity_models=[rp.Model.black_scholes(.20), rp.Model.black_scholes(.28)],
        settlement=settlement, fx_spot=FX, fx_volatility=.12,
        fixed_fx_rate=FX if settlement == "quanto" else None,
        rates=(rp.HullWhiteModel(.25, [0.], [.008]), rp.HullWhiteModel(.15, [0., .5], [.014, .017])),
        correlations=CORRELATIONS, maximum_step=.25,
    )
    if settlement == "floating_fx":
        return option.with_payment_fx_fixings(list(zip(PAYMENTS, OBSERVATIONS)))
    return option


if __name__ == "__main__":
    engine = rp.Engine.randomized_quasi_monte_carlo(
        2048, 8237, scramble_count=12, antithetic=True, brownian_bridge=True)
    for settlement in ["quanto", "floating_fx", "composite"]:
        estimate = note(settlement).price_mc(engine)
        print(f"{settlement:12} {estimate.value:12,.2f} JPY   SE {estimate.standard_error:6.2f} JPY")
