"""Forward derivatives at fixed model, maturity, strike and discount.
Not Spot Greeks, calibration risk or automatic numerical error control.
"""
import rust_pricing as rp

rough=rp.RoughVolatilityModel.rough_heston(hurst=0.1,initial_variance=0.04,
    mean_reversion=0.7,long_run_variance=0.055,vol_of_vol=0.18,correlation=-0.65)
lift=rp.RoughVolatilityModel.lifted_heston_from_rough(rough,factors=20,ratio=2.5)
for model in [rough,lift]:
    plan=rp.HestonFourierPlan.compile(model,1.0,time_steps=512,integration_intervals=512,cutoff=128.0)
    for strike in [80.0,100.0,120.0]:
        greeks=plan.price_and_greeks(100.0,strike,0.97)
        print(model.name,strike,greeks.price.call,greeks.call_forward_delta,greeks.forward_gamma)
        print('frequency diagnostics, NOT error bounds:',greeks.delta_quadrature_difference,
              greeks.gamma_quadrature_difference,greeks.delta_tail_indicator,greeks.gamma_tail_indicator)
