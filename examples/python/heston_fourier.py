"""European prices from continuous-time rough/lifted Heston transforms.
These are positive-forward payoffs, not physical-Spot fixed-cash-dividend claims.
Refine time_steps, integration_intervals and cutoff separately before use.
"""
import rust_pricing as rp
model=rp.RoughVolatilityModel.rough_heston(hurst=0.1,initial_variance=0.04,
    mean_reversion=0.7,long_run_variance=0.055,vol_of_vol=0.18,correlation=-0.65)
lift=rp.RoughVolatilityModel.lifted_heston_from_rough(model,factors=20,ratio=2.5)
for m in [model,lift]:
    plan=rp.HestonFourierPlan.compile(m,1.0,time_steps=1024,
                                     integration_intervals=512,cutoff=128.0)
    for strike in [80.0,100.0,120.0]:
        p=plan.price(100.0,strike,0.97)
        print(m.name,strike,p.call,p.put,p.quadrature_difference,p.tail_indicator)
