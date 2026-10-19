"""Two physical Spot Deltas: fixed leverage versus sticky relative Local Variance.

Both results condition on the same finite calibration. Neither is market-IV
sticky-strike Delta. Small illustrative path counts are not an accuracy budget.
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
    log_bandwidth=0.5, minimum_effective_samples=3.0, retain_reverse_trace=False,
    worker_threads=1, reduction_block_size=64)
for result in [plan.evaluate_frozen_leverage_delta(), plan.evaluate_sticky_moneyness_delta()]:
    print(result.convention, result.price.value, result.delta, result.delta_standard_error)
    assert result.price.value == plan.evaluate().value
    assert result.coordinate == "physical_spot_fixed_curves_and_cash_dividends"
