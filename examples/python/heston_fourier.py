"""Price a strike strip without simulation; numerical grids are explicit."""
import rust_pricing as rp

rough=rp.RoughVolatilityModel.rough_heston(
    hurst=0.1,initial_variance=0.04,mean_reversion=0.7,
    long_run_variance=0.055,vol_of_vol=0.18,correlation=-0.65)
lifted=rp.RoughVolatilityModel.lifted_heston_from_rough(rough,factors=32,ratio=1.7)
for label,model in [('rough_heston',rough),('lifted_heston',lifted)]:
    plan=rp.HestonFourierPlan.compile(model,1.0,time_steps=256,integration_intervals=512,cutoff=128.0)
    for strike in [80.0,100.0,120.0]:
        print(label,strike,plan.price(100.0,strike,discount=0.97))
    diagnostic=plan.refinement(100.0,100.0,discount=0.97)
    print('time, quadrature, cutoff changes:',diagnostic.time_change,
          diagnostic.quadrature_change,diagnostic.cutoff_change)
    # These changes are not standard errors or certified continuous-time bounds.
