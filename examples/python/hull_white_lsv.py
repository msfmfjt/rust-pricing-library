"""One-currency BS/LSV + Hull-White, with independent calibration and pricing.

Rate parameters below are illustrative inputs, not a calibration to swaptions.
Pricing standard errors exclude particle calibration noise and numerical bias.
"""
import numpy as np
import rust_pricing as rp

discount = rp.DiscountCurve(10, [0.0, 1.0, 5.0], [1.0, 0.95, 0.95**5])
dividend = rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
rates = rp.HullWhiteModel(0.2, [0.0, 0.45], [0.012, 0.018])
engine = rp.Engine.randomized_quasi_monte_carlo(
    1024, 612, scramble_count=8, antithetic=True, brownian_bridge=True
)


def request(model):
    return rp.PricingRequest(
        "2026-09-04",
        rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
        rp.Market.equity(2, 1, 100.0, discount, dividend),
        model, engine, rp.RiskRequest(),
    )


bs = rp.HullWhiteEquityPlan.compile_bs(
    request(rp.Model.black_scholes(0.2)), rates,
    equity_rate_correlation=0.25, maximum_step=1.0, worker_threads=2,
).evaluate()
target = rp.HullWhiteLsvTarget.flat(
    0.2, np.linspace(0.0, 1.0, 33), np.linspace(-0.75, 0.75, 31)
)
plan = rp.HullWhiteEquityPlan.compile_lsv(
    request(target.model), target, rates,
    vol_mean_reversion=2.0, vol_of_vol=0.4,
    equity_vol_correlation=-0.5, equity_rate_correlation=0.25,
    vol_rate_correlation=-0.1, particle_count=8192, calibration_seed=712,
    log_bandwidth=0.14, minimum_effective_samples=20.0, worker_threads=2,
)
lsv = plan.evaluate()
print("Initial 5Y bond:", rates.bond_price(discount, 0.0, 5.0, 0.0))
print("BS + HW price / SE:", bs.value, bs.standard_error)
print("Recalibrated LSV + HW price / SE:", lsv.value, lsv.standard_error)
print("Uncertainty scope:", lsv.uncertainty_scope)
print("Terminal mean D/P0 (target 1):", plan.calibration_discount_means[-1])
print("Terminal mean D/P0 * normalized equity (target 100):",
      plan.calibration_discounted_equity_means[-1])
print("Terminal fallback nodes:", plan.fallback_nodes[-1])
print("Scheme / calibration:", lsv.scheme, lsv.calibration_method)
print("Plan:", plan.plan_fingerprint)
