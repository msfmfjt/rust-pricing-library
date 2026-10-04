"""Fixed-target model-parameter risk through particle Leverage recalibration.

Absolute parameter units, not market IV Vega. RQMC SEs condition on one
calibration. These small example counts are not an accuracy prescription.
"""
import rust_pricing as rp

p = rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call")
market = rp.Market.equity(
    2, 1, 100.0,
    rp.DiscountCurve(10, [0.0, 1.0, 2.0], [1.0, 0.95, 0.95**2]),
    rp.DiscountCurve(11, [0.0, 1.0, 2.0], [1.0, 0.98, 0.98**2]),
    discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.25, 3.0),
                        rp.DividendEvent.fixed_cash(2, 1.5, 4.0)],
)
times = [0.0, 0.25, 0.5, 0.75, 1.0]
xs = [-0.6, -0.2, 0.13, 0.45, 0.8]
# Zero is deliberately inside a spatial cell, rather than at a slope kink.
target = rp.Model.local_volatility_from_grid(times, xs, [0.04 + 0.002*j for _ in times for j in range(5)],
                                   floor=1e-8, cap=4.0)
request = rp.PricingRequest("2026-09-04", p, market, target,
    rp.Engine.randomized_quasi_monte_carlo(256, 819, scramble_count=4,
                                          antithetic=True, brownian_bridge=True), rp.RiskRequest())
model = rp.RoughVolatilityModel.rough_heston(hurst=0.2, initial_variance=0.04,
    mean_reversion=0.7, long_run_variance=0.055, vol_of_vol=0.15, correlation=-0.6)
plan = rp.RoughFamilyLsvPlan.compile(request, model, particle_count=128, calibration_seed=429,
    log_bandwidth=0.5, minimum_effective_samples=3.0, retain_reverse_trace=True,
    worker_threads=1, reduction_block_size=64)
risk = plan.evaluate_heston_parameter_risk(include_hurst=True)
print("price", risk.price.value)
for name, total, direct, recalibrated, se in zip(
    risk.parameter_names, risk.parameter_adjoints, risk.direct_adjoints,
    risk.calibration_adjoints, risk.standard_errors,
):
    print(name, "total", total, "direct", direct, "recalibration", recalibrated, "conditional_SE", se)
    assert abs(total - direct - recalibrated) < 1e-10
assert risk.price.value == plan.evaluate().value
