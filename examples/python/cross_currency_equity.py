"""One-share USD equity calls paid in JPY, using illustrative market inputs."""
from math import exp

import rust_pricing as rp

usd, jpy, stock = 1, 2, 10
valuation, expiry = "2026-01-01", "2027-01-01"
usd_curve = rp.DiscountCurve(1, [0., 1.], [1., exp(-.04)])
jpy_curve = rp.DiscountCurve(2, [0., 1.], [1., exp(-.01)])
dividend = rp.DiscountCurve(3, [0., 1.], [1., exp(-.02)])
market = rp.Market.equity(usd, stock, 100., usd_curve, dividend)
engine = rp.Engine.randomized_quasi_monte_carlo(1024, 817, scramble_count=8, antithetic=True)

# Market FX is 150 JPY per USD. Only the quanto fixes a contractual conversion.
for settlement, strike, fixed_rate in [
    ("quanto", 100., 145.),
    ("floating_fx", 100., None),
    ("composite", 15_000., None),
]:
    product = rp.Product.european_vanilla(stock, jpy, expiry, strike, 1., "call")
    option = rp.CrossCurrencyEquityOption(
        valuation, product, market, jpy_curve, settlement=settlement,
        fx_spot=150., equity_volatility=.20, fx_volatility=.10,
        correlation=.35, fixed_fx_rate=fixed_rate,
    )
    estimate = option.price_mc(engine)
    print(f"{settlement:12}  PV {option.analytical_price():9.3f} JPY  "
          f"RQMC {estimate.value:9.3f} JPY  SE {estimate.standard_error:.3f} JPY")
