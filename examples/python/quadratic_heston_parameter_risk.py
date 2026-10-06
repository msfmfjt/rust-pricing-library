"""QRH scalar/Hurst risk, both fixed model and fixed-target LSV recalibration.

Small path counts demonstrate API usage, not risk precision. Keep the existing
fixed-driver definition: Leverage changes the stock, not the feedback state Z.
"""
import rust_pricing as rp

model = rp.RoughVolatilityModel.quadratic_rough_heston(
    hurst=.1, initial_state=.1, mean_reversion=.7, vol_of_vol=.4,
    quadratic=.5, shift=.2, variance_floor=.03,
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
    r=plan.evaluate_quadratic_heston_parameter_risk(include_hurst=True)
    print(label, 'price', r.price.value, 'coordinate', r.coordinate)
    for name,value in zip(r.parameter_names,r.parameter_adjoints):
        print(name,value)
    print('Marginal sampling errors:',r.standard_errors)
    if label=='Recalibrated LSV':
        print('Direct:',r.direct_adjoints)
        print('Recalibration:',r.calibration_adjoints)
print('Hurst is the last coordinate, per absolute H unit; H=.5 uses the left derivative.')
print('variance_floor is the model constant c, not a numerical floor or market-IV Vega.')
print('LSV errors condition on one calibration; they omit calibration-seed and grid uncertainty.')
