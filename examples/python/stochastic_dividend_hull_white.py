"""Q-mean Buehler cash dividends with correlated Hull--White discounting."""
import rust_pricing as rp

request=rp.PricingRequest('2026-09-04',
    rp.Product.european_vanilla(1,2,'2027-09-04',100.,1.,'call'),
    rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.],[1.,.95]),
        rp.DiscountCurve(11,[0.,1.],[1.,.98]),discrete_dividends=[
            rp.DividendEvent.fixed_cash(1,1.,3.),rp.DividendEvent.fixed_cash(2,1.4,8.)]),
    rp.Model.black_scholes(.2),rp.Engine.randomized_quasi_monte_carlo(256,973,
        scramble_count=8,antithetic=True,brownian_bridge=True),rp.RiskRequest())
plan=rp.StochasticDividendHullWhitePlan.compile_bs(request,
    dividend_mean_reversion=.7,equity_linkage=.6,dividend_volatility=.35,
    equity_dividend_correlation=-.25,rate_mean_reversion=.4,
    rate_volatility_times=[0.,.75],rate_volatilities=[.04,.05],
    equity_rate_correlation=.25,dividend_rate_correlation=-.2,
    maximum_step=1/32,worker_threads=2,reduction_block_size=64)
price=plan.evaluate()
print('Price:',price.value,'Sampling SE:',price.standard_error)
print('Cash dates:',plan.cash_times)
print('Collateral dividend PVs:',plan.initial_dividend_claim_values)
print('Collateral dividend forwards (not input Q means):',plan.initial_dividend_forwards)
print('Initially funded residual equity:',plan.risky_spot)
# This price-only class has no AAD/Gamma/calibration methods. Sampling SE excludes
# time discretization, reserve quadrature, smoothing and model uncertainty.
