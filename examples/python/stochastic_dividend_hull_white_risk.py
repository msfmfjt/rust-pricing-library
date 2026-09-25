"""HW/Buehler first-order risk and paired Gamma at fixed Q cash means."""
import rust_pricing as rp

request = rp.PricingRequest(
    '2026-09-04', rp.Product.european_vanilla(1, 2, '2027-09-04', 100., 1., 'call'),
    rp.Market.equity(2, 1, 100., rp.DiscountCurve(10, [0., 1.], [1., .95]),
        rp.DiscountCurve(11, [0., 1.], [1., .98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, .5, 4.),
                            rp.DividendEvent.fixed_cash(2, 1.4, 8.)]),
    rp.Model.black_scholes(.2),
    rp.Engine.randomized_quasi_monte_carlo(256, 1859, scramble_count=8,
        antithetic=True, brownian_bridge=True), rp.RiskRequest())
plan = rp.StochasticDividendHullWhitePlan.compile_bs(
    request, dividend_mean_reversion=.7, equity_linkage=.6, dividend_volatility=.35,
    equity_dividend_correlation=-.25, rate_mean_reversion=.4,
    rate_volatility_times=[0., .8, 1.15], rate_volatilities=[.04, .06, .09],
    equity_rate_correlation=.25, dividend_rate_correlation=-.2,
    maximum_step=1/32, worker_threads=2, reduction_block_size=64)
risk = plan.evaluate_correlation_aad()
print('Price:', risk.price.value, 'Delta:', risk.delta)
print('Initial residual-volatility Vega per vol point:', risk.initial_volatility_vega_per_vol_point)
print('Q cash-mean adjoints:', risk.cash_mean_adjoints)
print('Discount-curve node DV01:', risk.discount_node_dv01)
print('Repo-spread node DV01:', risk.repo_spread_node_dv01)
for label, derivative, se in zip(risk.parameter_labels, risk.derivatives, risk.standard_errors):
    print(f'{label}: {derivative:.8g}, sampling SE={se:.3g}')
gamma = plan.evaluate_gamma(gamma_relative_bump=.01)
for bump, estimate, se in zip(gamma.spot_bumps, gamma.gamma_estimates, gamma.gamma_standard_errors):
    print(f'Spot bump {bump:g}: Gamma={estimate:.8g}, paired sampling SE={se:.3g}')
print('Adjacent-bump differences:', gamma.bump_differences)
print('Paired gap SE:', gamma.bump_difference_standard_errors)
# These are NOT fixed-dividend-forward or recalibrated market-IV sensitivities.
