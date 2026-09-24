"""Deterministic-rate rough Bergomi + stochastic discrete cash dividends.

vol_of_vol is eta in log variance, not the log-volatility nu of 1F/2F Bergomi.
This factory currently exposes price only; existing AAD methods reject it.
"""
import rust_pricing as rp

request = rp.PricingRequest(
    '2026-09-04',
    rp.Product.european_vanilla(1, 2, '2027-09-04', 100., 1., 'call'),
    rp.Market.equity(
        2, 1, 100., rp.DiscountCurve(10, [0., 1.], [1., .95]),
        rp.DiscountCurve(11, [0., 1.], [1., .98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, 1., 3.),
                            rp.DividendEvent.fixed_cash(2, 1.4, 12.)],
    ),
    rp.Model.black_scholes(.2),  # Flat initial residual-equity volatility sigma0.
    rp.Engine.randomized_quasi_monte_carlo(
        1024, 612, scramble_count=8, antithetic=True, brownian_bridge=True,
    ),
    rp.RiskRequest(),
)
plan = rp.StochasticDividendPlan.compile_rough_bergomi(
    request, hurst=.1, vol_of_vol=.6, correlation=-.4,
    dividend_mean_reversion=.7, equity_linkage=.6, dividend_volatility=.35,
    equity_dividend_correlation=-.25, dividend_volatility_correlation=.15,
    maximum_step=1/32, worker_threads=2, reduction_block_size=64,
)
result = plan.evaluate()
print(f'Price={result.value:.8f}; sampling SE={result.standard_error:.8f}')
print('Funded residual equity:', plan.risky_spot)
print('Scheme:', plan.scheme)
