"""Notebook-style stochastic-dividend Bergomi LSV calibrated from Standard SSVI.

The Standard SSVI surface is the market smile target for the funded residual-equity
coordinate. The library materializes its Dupire local variance on the grid below,
then particle-calibrates the Bergomi leverage function.

The sampling budgets are intentionally small for an interactive example. Increase
both the RQMC points/scrambles and particle_count for convergence work.
"""
import rust_pricing as rp

# %% Contract and market
valuation_date = "2026-09-04"
expiry = "2027-09-04"

product = rp.Product.european_vanilla(
    1, 2, expiry, 100.0, 1.0, "call"
)
market = rp.Market.equity(
    2,
    1,
    100.0,
    rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
    # Legacy API name: carry curve, kept separate from cash amounts.
    rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
    discrete_dividends=[
        rp.DividendEvent.fixed_cash(1, 0.5, 6.0),
        rp.DividendEvent.fixed_cash(2, 1.4, 3.0),
    ],
)

# %% Calibrated Standard SSVI market target
# theta(T) is ATM total variance. These values correspond to a roughly 20% ATM
# volatility term structure, with a negative-equity skew from rho < 0.
calibration_times = [0.0, 0.25, 0.5, 0.75, 1.0]
calibration_log_moneyness = [-0.6, -0.3, 0.0, 0.3, 0.6]
ssvi_target = rp.Model.local_volatility_from_standard_ssvi_power_law(
    theta_times=[0.25, 0.5, 1.0],
    theta_values=[0.01, 0.02, 0.04],
    terminal_theta_slope=0.04,
    rho=-0.4,
    eta=0.8,
    gamma=0.4,
    time_nodes=calibration_times,
    log_forward_moneyness_nodes=calibration_log_moneyness,
    floor=1.0e-8,
    cap=4.0,
)

# %% Lightweight pricing request
# 256 points x 4 independent scrambles = 1,024 RQMC points before antithetic
# expansion. This is a notebook/demo budget, not a production convergence target.
request = rp.PricingRequest(
    valuation_date,
    product,
    market,
    ssvi_target,
    rp.Engine.randomized_quasi_monte_carlo(
        256,
        612,
        scramble_count=4,
        antithetic=True,
        brownian_bridge=True,
    ),
    rp.RiskRequest(),
)

# %% Particle calibration and pricing
plan = rp.StochasticDividendPlan.compile_bergomi_lsv(
    request,
    mean_reversion=0.8,
    vol_of_vol=0.3,
    correlation=-0.4,
    dividend_mean_reversion=0.7,
    equity_linkage=0.6,
    dividend_volatility=0.35,
    equity_dividend_correlation=-0.25,
    dividend_volatility_correlation=0.15,
    particle_count=512,
    calibration_seed=42,
    log_bandwidth=0.35,
    minimum_effective_samples=5.0,
    maximum_step=0.25,
    worker_threads=2,
    reduction_block_size=32,
)

# %% Results and calibration diagnostics
price = plan.evaluate()
print(f"Price={price.value:.8f}; sampling SE={price.standard_error:.8f}")
print(f"Funded residual={plan.risky_spot:.8f}; scheme={price.scheme}")
print(
    "LSV calibration grid:",
    len(plan.lsv_time_nodes),
    "x",
    len(plan.lsv_log_moneyness_nodes),
)
print("Uncertainty scope:", price.uncertainty_scope)
