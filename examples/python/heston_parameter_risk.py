"""Fixed-kernel scalar risk, common to call and put; no calibration or bumps."""
import rust_pricing as rp

model = rp.RoughVolatilityModel.rough_heston(
    hurst=0.1, initial_variance=0.04, mean_reversion=0.7,
    long_run_variance=0.055, vol_of_vol=0.18, correlation=-0.65)
plan = rp.HestonFourierPlan.compile(model, 1.0)
risk_plan = plan.parameter_risk_plan()
for strike in [80.0, 100.0, 120.0]:
    result = risk_plan.price(100.0, strike, 0.97)
    d = result.sensitivities
    print(strike, result.price.call, d.initial_variance, d.mean_reversion,
          d.long_run_variance, d.vol_of_vol, d.correlation)
    assert result.price.call == plan.price(100.0, strike, 0.97).call
