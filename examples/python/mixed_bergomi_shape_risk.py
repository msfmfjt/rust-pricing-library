"""Original Forward Variance Curve inputs and sum-preserving mixture transfers."""
import rust_pricing as rp

model = rp.RoughVolatilityModel.mixed_rough_bergomi(
    hurst=.1, correlation=-.6, weights=[.35, .65], vol_of_vols=[.3, .8],
    forward_variance=rp.ForwardVarianceCurve.piecewise_linear(
        [0., .4, 1.4], [.04, .05, .045]),
)
product=rp.Product.european_vanilla(1,2,'2027-09-04',100.,1.,'call')
market=rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,2.],[1.,.97**2]),
    rp.DiscountCurve(11,[0.,2.],[1.,.99**2]),discrete_dividends=[
        rp.DividendEvent.fixed_cash(1,.25,3.),rp.DividendEvent.fixed_cash(2,1.5,4.)])
engine=rp.Engine.randomized_quasi_monte_carlo(256,819,scramble_count=4,antithetic=True,brownian_bridge=True)
request=lambda m:rp.PricingRequest('2026-09-04',product,market,m,engine,rp.RiskRequest())
pure=rp.RoughVolatilityPlan.compile(request(rp.Model.black_scholes(.2)),model,maximum_step=.25,worker_threads=1)
target=rp.Model.local_volatility_from_grid([0.,.25,.5,.75,1.],[-.6,-.2,.13,.45,.8],
    [.04+.001*i+.002*j for i in range(5) for j in range(5)],1e-8,4.)
lsv=rp.RoughFamilyLsvPlan.compile(request(target),model,particle_count=128,calibration_seed=429,
    log_bandwidth=.5,minimum_effective_samples=3.,retain_reverse_trace=True,worker_threads=1)
for label,plan in [('Pure SV',pure),('Recalibrated LSV',lsv)]:
    r=plan.evaluate_mixed_bergomi_shape_risk()
    print(label, 'price', r.price.value, 'coordinate', r.coordinate)
    for name,value in zip(r.parameter_names,r.parameter_adjoints):
        print(name,value)
    print('Marginal sampling errors:',r.standard_errors)
    if label=='Recalibrated LSV':
        print('Direct:',r.direct_adjoints)
        print('Recalibration:',r.calibration_adjoints)
print('weight_transfer[0,1]: increase normalized w0 and decrease w1 by the same amount.')
print('Forward-variance derivatives are NOT market-IV Vega. LSV errors condition on one calibration.')
