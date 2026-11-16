"""Fit each stock's fast/slow Bergomi states and measure the seasoned note's four JPY quote vegas."""
import rust_pricing as rp
from cross_currency_stochastic_dividend_autocall import (
    APRIL_STATES, REVALUATION, STOCK_IDS, VALUATION, note, years,
)

# Illustrative European premiums per share in local currency, rounded to cents
# from a separate local-currency simulation. Forwards are independent quote inputs.
# Expiry, side, strike, market forward, premium.
QUOTES = {
    10: [("2026-07-10", "call", 99., 98.20, 3.34), ("2027-04-10", "put", 96., 95.46, 7.34)],
    20: [("2026-07-10", "call", 76., 75.89, 3.55), ("2027-04-10", "put", 72., 71.91, 7.09)],
}

def seasoned_basket():
    basket = note(REVALUATION, two_factor=True)
    for stock, (f, y, _) in zip(STOCK_IDS, APRIL_STATES):
        basket = basket.with_dividend_state(stock, model_age=years(VALUATION, REVALUATION),
                                            volatility_factor=(0., 0.), equity_factor=f, dividend_factor=y)
    return basket


if __name__ == "__main__":
    engine = rp.Engine.randomized_quasi_monte_carlo(1024, 29737, scramble_count=4,
                                                     antithetic=True, brownian_bridge=True)
    result = seasoned_basket().bergomi_quote_risk(QUOTES, engine, bump=.005,
                                                   replications=8, worker_threads=4)
    pv = result.present_value
    print(f"{REVALUATION} PV {pv.value:,.2f} JPY (SE {pv.standard_error:.2f})")
    for stock, currency in [(10, "USD"), (20, "EUR")]:
        fast, slow = result.states[stock]
        print(f"{currency}  fast {fast.value:+.4f} (SE {fast.standard_error:.4f})"
              f"  slow {slow.value:+.4f} (SE {slow.standard_error:.4f})")
        for j, (expiry, side, strike, forward, premium) in enumerate(QUOTES[stock]):
            iv = result.implied_volatilities[stock][j]
            residual = result.price_residuals[stock][j]
            risk = result.vegas[stock][j]
            print(f"  {expiry} {side:4} K={strike:g} F={forward:.2f} premium={premium:.2f} {currency}"
                  f"  Black IV {iv:.3%}  independent price minus quote {residual.value:+.4f}"
                  f" (SE {residual.standard_error:.4f}) {currency}"
                  f"  JPY/vol point {risk.value:+,.2f} (SE {risk.standard_error:.2f})")
