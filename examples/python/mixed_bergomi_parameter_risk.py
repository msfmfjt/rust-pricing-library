"""Fixed Hurst/weights/forward-variance eta and rho risk, Pure SV and LSV."""
import rust_pricing as rp

model = rp.RoughVolatilityModel.mixed_rough_bergomi(
    hurst=.2, correlation=-.6, weights=[.35,.65], vol_of_vols=[.3,.8],
    forward_variance=rp.ForwardVarianceCurve.piecewise_linear([0.,.5,1.],[.04,.05,.045]))
product = rp.Product.european_vanilla(1,2,'2027-09-04',100.,1.,'call')
market = rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.,2.],[1.,.97,.97**2]),
    rp.DiscountCurve(11,[0.,1.,2.],[1.,.99,.99**2]),discrete_dividends=[
        rp.DividendEvent.fixed_cash(1,.25,3.),rp.DividendEvent.fixed_cash(2,1.5,4.)])
engine = rp.Engine.randomized_quasi_monte_carlo(256,819,scramble_count=4,antithetic=True,brownian_bridge=True)
times = [0.,.25,.5,.75,1.]; xs = [-.6,-.2,.13,.45,.8]
target = rp.Model.local_volatility_from_grid(times,xs,[.04+.001*i+.002*j for i in range(5) for j in range(5)],1e-8,4.)
request = lambda model: rp.PricingRequest('2026-09-04',product,market,model,engine,rp.RiskRequest())
pure = rp.RoughVolatilityPlan.compile(request(rp.Model.black_scholes(.2)),model,maximum_step=.25,worker_threads=1)
lsv = rp.RoughFamilyLsvPlan.compile(request(target),model,particle_count=128,calibration_seed=429,
    log_bandwidth=.5,minimum_effective_samples=3.,retain_reverse_trace=True,worker_threads=1)
for label,plan in [('Pure SV',pure),('LSV with Leverage recalibration',lsv)]:
    r = plan.evaluate_mixed_bergomi_parameter_risk()
    print(label,'price:',r.price.value)
    for i,name in enumerate(r.parameter_names):
        print(name, r.parameter_adjoints[i], 'sampling SE:', r.standard_errors[i])
        if label.startswith('LSV'):
            print('  direct:',r.direct_adjoints[i],'recalibration:',r.calibration_adjoints[i])
print('Errors exclude calibration-seed variation and time-grid bias. Counts are illustrative.')
