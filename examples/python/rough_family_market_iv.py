"""Discrete market-IV -> Dupire -> particle LSV quote risk (fixed SV parameters)."""
import rust_pricing as rp

quote_times = [0.2, 0.6, 1.2]
quote_x = [-0.9, -0.45, 0.05, 0.5, 0.9]
vols = [0.2+0.002*i-0.003*x+0.002*x*x for i in range(3) for x in quote_x]
source = rp.MarketIvSurface(quote_times, quote_x, vols)
target = source.local_volatility_model([0.0, 0.13, 0.37, 0.7, 1.0], [-0.55,-0.13,0.18,0.6])
curve = rp.DiscountCurve(10, [0.0,1.0,2.0], [1.0,0.95,0.95**2])
repo = rp.DiscountCurve(11, [0.0,1.0,2.0], [1.0,0.98,0.98**2])
market = rp.Market.equity(2,1,100.0,curve,repo,discrete_dividends=[
    rp.DividendEvent.fixed_cash(1,0.25,3.0), rp.DividendEvent.fixed_cash(2,1.5,4.0)])
product = rp.Product.european_vanilla(1,2,'2027-09-04',100.0,1.0,'call')
engine = rp.Engine.randomized_quasi_monte_carlo(256,819,scramble_count=4,
    antithetic=True,brownian_bridge=True)
request = rp.PricingRequest('2026-09-04',product,market,target,engine,rp.RiskRequest())
model = rp.RoughVolatilityModel.rough_heston(hurst=0.2,initial_variance=0.04,
    mean_reversion=0.7,long_run_variance=0.055,vol_of_vol=0.15,correlation=-0.6)
plan = rp.RoughFamilyLsvPlan.compile(request,model,particle_count=256,
    calibration_seed=429,log_bandwidth=0.5,minimum_effective_samples=3.0,
    retain_reverse_trace=True,worker_threads=1)
risk = plan.market_iv_risk_plan(source).evaluate()
print('Price:',risk.price.value)
print('IV quote adjoints (maturity-major):',risk.quote_adjoints)
print('Parallel Vega per absolute IV unit:',risk.parallel_vega)
print('Parallel Vega per one volatility point:',0.01*risk.parallel_vega)
print('Parallel sampling SE:',risk.parallel_standard_error)
print('Coordinate:',risk.coordinate)
# SE excludes particle-calibration uncertainty and time/space/model bias.
# Small illustrative counts are not a production accuracy prescription.
