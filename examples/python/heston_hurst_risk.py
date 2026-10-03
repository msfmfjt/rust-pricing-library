"""Fixed-scalar-parameter Hurst risk. No lift-factory or calibrated market risk."""
import rust_pricing as rp

model = rp.RoughVolatilityModel.rough_heston(
    hurst=0.1, initial_variance=0.04, mean_reversion=0.7,
    long_run_variance=0.055, vol_of_vol=0.18, correlation=-0.65,
)
plan = rp.HestonFourierPlan.compile(model, 1.0)
hurst_plan = plan.hurst_risk_plan()
for strike in [80.0, 100.0, 120.0]:
    result = hurst_plan.price(100.0, strike, 0.97)
    print(strike, result.price.call, result.hurst_sensitivity)
    assert result.price.call == plan.price(100.0, strike, 0.97).call
# Derivatives are per absolute H unit; multiply by 0.01 for a linearized dH=.01.
# Diagnostics do not bound omitted tails or Riccati/time discretization errors.
