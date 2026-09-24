"""Finite-bump Gamma: all parameters and cash means are held fixed."""
import rust_pricing as rp

request = rp.PricingRequest(
    "2026-09-04", rp.Product.european_vanilla(1, 2, "2027-09-04", 100., 1., "call"),
    rp.Market.equity(2, 1, 100., rp.DiscountCurve(10, [0., 1.], [1., .95]),
        rp.DiscountCurve(11, [0., 1.], [1., .98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, 1., 3.),
                            rp.DividendEvent.fixed_cash(2, 1.4, 12.)]),
    rp.Model.black_scholes(.2),
    rp.Engine.randomized_quasi_monte_carlo(4096, 612, scramble_count=8,
        antithetic=True, brownian_bridge=True), rp.RiskRequest())
plan = rp.StochasticDividendPlan.compile_bergomi_two_factor(
    request, mean_reversions=[.8, 2.1], vol_of_vol=.3, mixing_weight=.35,
    spot_correlations=[-.4, -.2], factor_correlation=.3,
    dividend_mean_reversion=.7, equity_linkage=.6, dividend_volatility=.35,
    equity_dividend_correlation=-.25, dividend_volatility_correlations=[.15, -.1],
    maximum_step=1/32, worker_threads=2, reduction_block_size=64)
risk = plan.evaluate_gamma(gamma_relative_bump=.01)
print(f"Price={risk.price.value:.8f}; Delta={risk.delta:.8f}")
for h, gamma, se in zip(risk.spot_bumps, risk.gamma_estimates, risk.gamma_standard_errors):
    print(f"Spot bump {h:g}: Gamma={gamma:.8g}; paired sampling SE={se:.3g}")
print("Bump gaps (half-base, base-double):", risk.bump_differences)
print("Paired gap SEs:", risk.bump_difference_standard_errors)
print("Risk fingerprint:", risk.risk_fingerprint)
