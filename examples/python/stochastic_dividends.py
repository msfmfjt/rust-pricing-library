"""Price a call with Buehler stochastic discrete cash dividends (no LSV/AAD)."""
import rust_pricing as rp

request = rp.PricingRequest(
    "2026-09-04",
    rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
    rp.Market.equity(
        2, 1, 100.0,
        rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
        # Legacy API name: carry curve, kept separate from cash amounts.
        rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.5, 6.0),
                            rp.DividendEvent.fixed_cash(2, 1.4, 3.0)],
    ),
    rp.Model.black_scholes(0.2),  # Volatility of normalized residual f, not physical S.
    rp.Engine.randomized_quasi_monte_carlo(1024, 612, scramble_count=8,
        antithetic=True, brownian_bridge=True),
    rp.RiskRequest(),
)
plan = rp.StochasticDividendPlan.compile_bs(
    request, mean_reversion=0.7, equity_linkage=0.6, dividend_volatility=0.4,
    equity_dividend_correlation=-0.2, maximum_step=1/64,
    worker_threads=2, reduction_block_size=64,
)
price = plan.evaluate()
print(f"Price={price.value:.8f}; sampling SE={price.standard_error:.8f}")
print(f"Funded residual={plan.risky_spot:.8f}; scheme={price.scheme}")
# Repeat with smaller maximum_step to assess discretization separately from SE.
