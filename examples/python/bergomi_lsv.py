"""Particle-calibrated Bergomi LSV: price and recalibrated Dupire variance risk.

Run after installing the wheel. Standard errors are conditional on calibration;
node_adjoints are local-variance sensitivities, not market implied-vol VegaKT.
"""
import numpy as np
import rust_pricing as rp

times = np.linspace(0.0, 1.0, 33)
log_nodes = np.linspace(-0.75, 0.75, 31)
curve = rp.DiscountCurve(10, [0.0, 1.0], [1.0, 1.0])
target = rp.Model.local_volatility_from_grid(
    times, log_nodes, np.full(33 * 31, 0.04), 1.0e-8, 4.0
)
request = rp.PricingRequest(
    valuation_date="2026-09-04",
    product=rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
    market=rp.Market.equity(2, 1, 100.0, curve, curve),
    model=target,
    engine=rp.Engine.randomized_quasi_monte_carlo(
        1024, 91, scramble_count=16, antithetic=True, brownian_bridge=True
    ),
    risk=rp.RiskRequest(),
)
plan = rp.BergomiLsvPlan.compile(
    request,
    mean_reversion=2.0,
    vol_of_vol=0.7,
    correlation=-0.5,
    particle_count=4096,
    calibration_seed=42,
    log_bandwidth=0.12,
    minimum_effective_samples=10.0,
    retain_reverse_trace=True,
    worker_threads=2,
    reduction_block_size=256,
)
result = plan.evaluate_local_variance_risk()
print("LSV price:", result.price.value)
print("BS target:", 7.965567455405804)
print("Pricing standard error:", result.price.standard_error)
print("Uncertainty scope:", result.price.uncertainty_scope)
print("Risk coordinate:", result.coordinate)
print("Risk matrix shape:", (len(result.time_nodes), len(result.log_moneyness_nodes)))
print("Method:", result.method)
print("Plan:", plan.plan_fingerprint)
