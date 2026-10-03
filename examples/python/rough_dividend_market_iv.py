"""Finite parallel IV risk for a continuous rough-LSV stochastic-dividend Barrier."""
import rust_pricing as rp

expiry = 364 / 365
# Illustrative quotes in the funded residual-forward coordinate. Physical-Spot
# option quotes require a separate model-consistent conversion/calibration.
source = rp.MarketIvSurface([.25, 1.25], [-.75, 0., .75],
                            [.22, .20, .21, .24, .22, .23])
model = source.local_volatility_model([0., 182 / 365, expiry], [-.5, 0., .5])
request = rp.PricingRequest(
    '2026-09-04',
    rp.Product.barrier(1, 2, '2027-09-03', 80., 120., 2., 'call', 'up',
        'knock_out', 'continuous', ['2026-09-04', '2027-03-05', '2027-09-03'],
        '2027-12-04', rebate=7.),
    rp.Market.equity(2, 1, 100., rp.DiscountCurve(10, [0., 1.], [1., .95]),
        rp.DiscountCurve(11, [0., 1.], [1., .98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, 182 / 365, 5.),
                            rp.DividendEvent.fixed_cash(2, expiry, 3.),
                            rp.DividendEvent.fixed_cash(3, 1.4, 12.)]),
    model,
    rp.Engine.randomized_quasi_monte_carlo(1024, 193, scramble_count=8,
        antithetic=True, brownian_bridge=True),
    rp.RiskRequest())
# Requests serialize the explicit target grid. Keep the source separately and
# pass it again when compiling a restored request; the grid alone has no quotes.
plan = rp.StochasticDividendContinuousBarrierPlan.compile_rough_bergomi_lsv(
    request, market_iv_surface=source, hurst=.1, vol_of_vol=.6, correlation=-.4,
    dividend_mean_reversion=.7, equity_linkage=.6, dividend_volatility=.35,
    equity_dividend_correlation=-.25, dividend_volatility_correlation=.15,
    particle_count=256, calibration_seed=42, log_bandwidth=.35,
    minimum_effective_samples=5., maximum_step=expiry / 16,
    worker_threads=2, reduction_block_size=64)
assert plan.supports_market_iv_risk
risk = plan.evaluate_parallel_market_iv_risk(implied_volatility_bump=.01)
print('Price / sampling SE:', risk.price.value, risk.price.standard_error)
print('Parallel quote-IV bumps:', risk.implied_volatility_bumps)
print('Vega ladder / paired SE:', risk.vega_estimates, risk.vega_standard_errors)
print('Value / SE per vol point:', risk.vega_per_vol_point, risk.standard_error_per_vol_point)
print('Bump gaps / paired SE:', risk.bump_differences, risk.bump_difference_standard_errors)
print('Interpolation:', risk.interpolation)
print('Recalibrations / scenario paths:', risk.recalibration_count, risk.scenario_evaluated_paths)
print('Sampling errors condition on the calibration seed and exclude grid/bridge/bump bias.')
