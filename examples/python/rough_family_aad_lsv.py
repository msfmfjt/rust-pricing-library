"""Rough-Heston Spot Delta and LSV recalibrated local-variance-node AAD.

Small illustrative workloads, not an accuracy certificate. Local variance nodes
are not market IV quotes; no claim of market VegaKT or model-parameter AAD.
"""
import math
import rust_pricing as rp

model = rp.RoughVolatilityModel.rough_heston(
    hurst=0.2, initial_variance=0.04, mean_reversion=0.7,
    long_run_variance=0.055, vol_of_vol=0.15, correlation=-0.6)
curve = rp.DiscountCurve(10, [0.0, 1.0, 2.0], [1.0, 0.97, 0.97**2])
repo = rp.DiscountCurve(11, [0.0, 1.0, 2.0], [1.0, 0.99, 0.99**2])
market = rp.Market.equity(2, 1, 100.0, curve, repo,
    discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.5, 3.0),
                        rp.DividendEvent.fixed_cash(2, 1.5, 4.0)])
product = rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call")
engine = rp.Engine.randomized_quasi_monte_carlo(
    256, 91, scramble_count=8, antithetic=True, brownian_bridge=True)

def request(carrier):
    return rp.PricingRequest("2026-09-04", product, market, carrier, engine, rp.RiskRequest())

pure = rp.RoughVolatilityPlan.compile(request(rp.Model.black_scholes(0.2)), model,
    maximum_step=1/16, worker_threads=2, reduction_block_size=128)
delta = pure.evaluate_delta()
print("Pure SV price, physical Spot Delta, Delta SE:",
      delta.price.value, delta.delta, delta.delta_standard_error)

# Construct the local Dupire target from SSVI using the existing adapter.
# Reported adjoints still refer to local variance nodes, not SSVI parameters/IVs.
times = [i/16 for i in range(17)]
xs = [-0.75 + 0.05*i for i in range(31)]
values = [0.04 for _ in times for _ in xs]
target = rp.Model.local_volatility_from_standard_ssvi_power_law(
    [0.25, 0.5, 1.0], [0.01, 0.02, 0.04], 0.04,
    -0.4, 0.3, 0.5, times, xs, 1e-8, 4.0)
lsv = rp.RoughFamilyLsvPlan.compile(request(target), model,
    particle_count=512, calibration_seed=429, log_bandwidth=0.25,
    minimum_effective_samples=5.0, retain_reverse_trace=True,
    worker_threads=2, reduction_block_size=128)
risk = lsv.evaluate_local_variance_risk()
assert math.isfinite(risk.price.value)
assert len(risk.node_adjoints) == len(values)
print("LSV price, pricing SE:", risk.price.value, risk.price.standard_error)
print("Node coordinate:", risk.coordinate)
print("dPrice/d parallel additive local-variance shift:", sum(risk.node_adjoints))
print("Uncertainty:", risk.price.uncertainty_scope)
print("Grid ends at expiry; later cash remains in escrow:", lsv.time_nodes[-1])
