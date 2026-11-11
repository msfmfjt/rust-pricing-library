"""USD coupons, JPY redemptions, USD/EUR shares with stochastic volatility and cash dividends."""
import argparse
from datetime import date
from math import exp, expm1

import rust_pricing as rp
from cross_currency_usd_eur_autocall import (
    USD, JPY, EUR, STOCK_IDS, VALUATION, OBSERVATIONS, PAYMENTS, FX_QUOTES,
    contract_terms, curve, model_inputs,
)

REVALUATION = "2026-04-10"
CASH_DATES = ["2026-03-18", "2026-06-18", "2026-09-18", "2026-12-18", "2027-03-18"]
# Issue-date cash forecasts per share in the stock's own currency.
CASH_MEANS = [[1.20, 1.35, 1.40, 1.50, 1.55], [1.00, 1.10, 1.25, 1.30, 1.40]]
# April 10 model estimates (normalized residual share, cash dividend factor, volatility factor).
APRIL_STATES = [(.97, 1.06, .16), (.94, .91, -.18)]
# Uncentered fast and slow OU estimates, in model order (US, euro-area).
APRIL_TWO_FACTOR = [(.16, -.09), (-.18, .12)]


def years(start, end):
    return (date.fromisoformat(end) - date.fromisoformat(start)).days / 365.


def note(as_of, *, vol_of_vol=(.55, .70), dividend_vols=(.30, .38), maximum_step=1./24,
         usd_rate_vol_shift=0., two_factor=False):
    if as_of not in (VALUATION, REVALUATION):
        raise ValueError("the example supplies two dated market snapshots")
    seasoned = as_of == REVALUATION
    age = years(VALUATION, as_of)
    states = APRIL_STATES if seasoned else [(1., 1., 0.)]*2
    stocks, dividends, equities = [], [], []
    spot = [99., 77.] if seasoned else [100., 80.]
    for i, (ccy, funding, repo, initial_vol, vol_k, vol_rho, div_k, alpha, div_rho) in enumerate([
        (USD, .04, .015, .20, 1.2, -.45, .8, .65, .35),
        (EUR, .025, .030, .28, .9, -.50, 1.1, .55, .30),
    ]):
        f, y, _ = states[i]
        events = []
        for j, (day, mean) in enumerate(zip(CASH_DATES, CASH_MEANS[i])):
            if day <= as_of:
                continue
            t = years(as_of, day)
            # Conditional cash forecasts in this illustrative snapshot. March has already paid.
            forecast = mean*(exp(-div_k*t)*y - expm1(-div_k*t)*(alpha*f + 1.-alpha))
            events.append(rp.DividendEvent.fixed_cash(j+1, t, forecast))
        stocks.append(rp.Market.equity(ccy, STOCK_IDS[i], spot[i], curve(ccy, funding),
                                       curve(4+i, repo), discrete_dividends=events))
        dividends.append(rp.BuehlerDividendModel(mean_reversion=div_k, equity_linkage=alpha,
                                                dividend_volatility=dividend_vols[i], equity_dividend_correlation=div_rho))
        if two_factor:
            factor = rp.VolatilityFactorSpec.bergomi_two_factor(
                [2.*vol_k, .4*vol_k], vol_of_vol[i], .4, [vol_rho, -.25], .25)
        else:
            factor = rp.VolatilityFactorSpec.bergomi(vol_k, vol_of_vol[i], vol_rho)
        equities.append(rp.EquityModelSpec.stochastic_volatility(initial_vol, factor))

    legs = rp.MultiAssetProduct.autocallable_legs(
        **contract_terms("floating_fx", JPY, USD), coupon_currency_id=USD, principal_currency_id=JPY)
    if seasoned:
        history = [(10, OBSERVATIONS[0], 98.), (20, OBSERVATIONS[0], 76.)]
        legs = [leg.remaining_cashflows(as_of, fixings=history) for leg in legs]
    inputs = model_inputs([(USD, 149., .12), (EUR, 171., .14)] if seasoned else FX_QUOTES)
    if usd_rate_vol_shift:
        usd = inputs["rates"][1]
        inputs["rates"][1] = rp.HullWhiteModel(usd.mean_reversion, usd.volatility_times,
                                               [v+usd_rate_vol_shift for v in usd.volatilities])
    inputs.update(equity_models=equities, dividend_models=dividends, maximum_step=maximum_step)
    basket = rp.CrossCurrencyEquityBasket.from_legs(
        as_of, legs, stocks, curve(JPY, .01), settlement_currency_id=JPY, **inputs)
    if seasoned:
        for i, (stock, (f, y, x)) in enumerate(zip(STOCK_IDS, states)):
            if two_factor:
                x = APRIL_TWO_FACTOR[i]
            basket = basket.with_dividend_state(stock, model_age=age, volatility_factor=x,
                                                equity_factor=f, dividend_factor=y)
    schedule = [(p, d) for p, d in zip(PAYMENTS, OBSERVATIONS) if p in legs[0].payment_dates]
    known = [(OBSERVATIONS[0], 147.)] if seasoned else []
    return basket.with_leg_fx(0, schedule=schedule, known_fixings=known)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-factor", action="store_true", help="use two-factor Bergomi for both residual shares")
    two_factor = parser.parse_args().two_factor
    points = 8192 if two_factor else 1024
    engine = rp.Engine.randomized_quasi_monte_carlo(points, 8237, scramble_count=16 if two_factor else 8,
                                                     antithetic=True, brownian_bridge=True)
    for as_of in [VALUATION, REVALUATION]:
        basket = note(as_of, two_factor=two_factor)
        pv = basket.price_mc(engine)
        print(f"{as_of} PV {pv.value:,.2f} JPY   SE {pv.standard_error:.2f}")
    factor_risks = [
        ("bergomi_factor_1", 10, .05, .01, "US fast volatility factor +0.01"),
        ("bergomi_factor_2", 10, .05, .01, "US slow volatility factor +0.01"),
    ] if two_factor else []
    for factor, identifier, bump, scale, label in [
        *factor_risks,
        ("fx_spot", USD, .5, 1., "JPY per USD +1"),
        ("equity_spot", 10, 1., 1., "US share +1 USD"),
        ("equity_spot", 20, 1., 1., "European share +1 EUR"),
        ("equity_volatility", 10, .01, .01, "US residual volatility scale +0.01"),
        ("equity_vol_of_vol", 10, .05, .01, "US volatility of volatility +0.01"),
        ("dividend_volatility", 20, .05, .01, "European dividend volatility +0.01"),
        ("discount_rate", USD, .0001, .0001, "USD curve +1 bp"),
        ("discount_rate", EUR, .0001, .0001, "EUR curve +1 bp"),
        ("discount_rate", JPY, .0001, .0001, "JPY curve +1 bp"),
    ]:
        v = basket.sensitivity(factor, identifier, bump, engine)
        print(f"{label:40} {v.value*scale:9.2f} JPY   SE {v.standard_error*scale:.2f}")
    rate_scenario = note(REVALUATION, usd_rate_vol_shift=.001, two_factor=two_factor)
    pnl = basket.scenario_pnl(rate_scenario, engine)
    print(f"USD short-rate volatility +10 bp          {pnl.value:9.2f} JPY   SE {pnl.standard_error:.2f}")
