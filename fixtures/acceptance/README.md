# European Black–Scholes analytical acceptance data

`european_bs_analytical.csv` is an independently generated binary64 reference
grid. It uses SciPy 1.17.0 `scipy.special.ndtr` for the normal CDF and the
standard forward Black–Scholes formula for Price, spot Delta, spot Gamma, and
unit-volatility Vega.

The fixed grid uses Spot 100 and Notional 1. Strike is defined as
`moneyness × Forward`. Its 18 cases cover both option sides and three levels
each of log-forward-moneyness proxy, maturity, continuously compounded rate,
continuous dividend yield, and volatility. The row construction is a fixed
covering grid rather than a full Cartesian product, keeping CI fast while
retaining regular-domain extremes.

The fixture is reference evidence, not generated during tests. Updates require
an explicit review of both the inputs and output columns.
