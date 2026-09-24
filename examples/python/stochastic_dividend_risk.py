"""Bergomi/Buehler first-order risk; correlations and Bergomi parameters fixed."""
import rust_pricing as rp

request=rp.PricingRequest(
    '2026-09-04',rp.Product.european_vanilla(1,2,'2027-09-04',100.,1.,'call'),
    rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.],[1.,.95]),
        rp.DiscountCurve(11,[0.,1.],[1.,.98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1,1.,3.),rp.DividendEvent.fixed_cash(2,1.4,12.)]),
    rp.Model.black_scholes(.2),
    rp.Engine.randomized_quasi_monte_carlo(1024,612,scramble_count=8,antithetic=True,brownian_bridge=True),
    rp.RiskRequest())
plan=rp.StochasticDividendPlan.compile_bergomi_two_factor(
    request,mean_reversions=[.8,2.1],vol_of_vol=.3,mixing_weight=.35,
    spot_correlations=[-.4,-.2],factor_correlation=.3,
    dividend_mean_reversion=.7,equity_linkage=.6,dividend_volatility=.35,
    equity_dividend_correlation=-.25,dividend_volatility_correlations=[.15,-.1],
    maximum_step=1/32,worker_threads=2,reduction_block_size=64)
risk=plan.evaluate_aad()
print(f'Price={risk.price.value:.8f}, Delta={risk.delta:.8f}')
print('Initial-volatility Vega per +1 vol point:',risk.initial_volatility_vega_per_vol_point)
for label,derivative,se in zip(risk.parameter_labels,risk.derivatives,risk.standard_errors):
    print(f'{label}: {derivative:.8g} (sampling SE {se:.3g})')
print('Discount-node DV01:',risk.discount_node_dv01)
print('Repo-spread-node DV01:',risk.repo_spread_node_dv01)
