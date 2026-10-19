"""Pure-SV Rough Heston MC Hurst reverse; kernel and history derivatives included."""
import rust_pricing as rp

curve=rp.DiscountCurve(10,[0.,1.,2.],[1.,.97,.97**2])
repo=rp.DiscountCurve(11,[0.,1.,2.],[1.,.99,.99**2])
market=rp.Market.equity(2,1,100.,curve,repo,discrete_dividends=[
    rp.DividendEvent.fixed_cash(1,.25,3.),rp.DividendEvent.fixed_cash(2,1.5,4.)])
product=rp.Product.european_vanilla(1,2,"2027-09-04",100.,1.,"call")
engine=rp.Engine.randomized_quasi_monte_carlo(256,819,scramble_count=8,
    antithetic=True,brownian_bridge=True)
request=rp.PricingRequest("2026-09-04",product,market,rp.Model.black_scholes(.2),engine,rp.RiskRequest())
h=rp.RoughVolatilityModel.rough_heston(hurst=.2,initial_variance=.04,
    mean_reversion=.7,long_run_variance=.055,vol_of_vol=.15,correlation=-.6)
plan=rp.RoughVolatilityPlan.compile(request,h,maximum_step=.125,worker_threads=1)
result=plan.evaluate_hurst_risk()
print("price",result.price.value)
print("dPrice/dH",result.hurst_sensitivity,"sampling SE",result.standard_error)
print("first-order price change for H+0.01",.01*result.hurst_sensitivity)
# Fix scalar parameters and the grid. This is not LSV recalibration risk.
