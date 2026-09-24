"""Rough/Buehler basic and H/eta risk, plus paired Delta-bump Spot Gamma."""
import rust_pricing as rp

request = rp.PricingRequest(
    '2026-09-04', rp.Product.european_vanilla(1, 2, '2027-09-04', 100., 1., 'call'),
    rp.Market.equity(2, 1, 100., rp.DiscountCurve(10, [0., 1.], [1., .95]),
        rp.DiscountCurve(11, [0., 1.], [1., .98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, 1., 3.),
                            rp.DividendEvent.fixed_cash(2, 1.4, 8.)]),
    rp.Model.black_scholes(.2),
    rp.Engine.randomized_quasi_monte_carlo(256, 884, scramble_count=8,
        antithetic=True, brownian_bridge=True), rp.RiskRequest())
plan = rp.StochasticDividendPlan.compile_rough_bergomi(
    request, hurst=.1, vol_of_vol=.6, correlation=-.4,
    dividend_mean_reversion=.7, equity_linkage=.6, dividend_volatility=.35,
    equity_dividend_correlation=-.25, dividend_volatility_correlation=.15,
    maximum_step=1/32, worker_threads=2, reduction_block_size=64)
risk = plan.evaluate_rough_aad()
print('Price:', risk.price.value, 'Delta:', risk.delta)
for label, value, se in zip(risk.parameter_labels, risk.derivatives, risk.standard_errors):
    print(f'{label}: {value:.8g} (sampling SE {se:.3g})')
gamma = plan.evaluate_gamma(gamma_relative_bump=.01)
for bump, value, se in zip(gamma.spot_bumps, gamma.gamma_estimates, gamma.gamma_standard_errors):
    print(f'Spot bump {bump:g}: Gamma {value:.8g} (sampling SE {se:.3g})')
print('Paired bump gaps:', gamma.bump_differences)
