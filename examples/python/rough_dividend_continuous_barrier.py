"""Opt-in continuous rough-LSV Barrier approximation with paired finite-bump Spot risk."""
import rust_pricing as rp

expiry = 364/365
request = rp.PricingRequest(
    '2026-09-04',
    rp.Product.barrier(1, 2, '2027-09-03', 80., 120., 2., 'call', 'up',
        'knock_out', 'continuous', ['2026-09-03', '2026-09-04', '2027-03-05', '2027-09-03'],
        '2027-12-04', rebate=7., historical_hit=False),
    rp.Market.equity(2, 1, 100., rp.DiscountCurve(10, [0., 1.], [1., .95]),
        rp.DiscountCurve(11, [0., 1.], [1., .98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, 182/365, 5.),
                            rp.DividendEvent.fixed_cash(2, expiry, 3.),
                            rp.DividendEvent.fixed_cash(3, 1.4, 12.)]),
    # This target is in funded residual-equity coordinates.
    rp.Model.local_volatility_from_grid([0., 182/365, expiry], [-.5, 0., .5],
        [.045, .04, .035, .05, .045, .04, .055, .05, .045], 1e-8, 4.),
    rp.Engine.randomized_quasi_monte_carlo(1024, 193, scramble_count=8,
        antithetic=True, brownian_bridge=True),
    rp.RiskRequest())  # Price-only request; no smoothing or Greek flags.
plan = rp.StochasticDividendContinuousBarrierPlan.compile_rough_bergomi_lsv(
    request, hurst=.1, vol_of_vol=.6, correlation=-.4,
    dividend_mean_reversion=.7, equity_linkage=.6, dividend_volatility=.35,
    equity_dividend_correlation=-.25, dividend_volatility_correlation=.15,
    particle_count=256, calibration_seed=42, log_bandwidth=.35,
    minimum_effective_samples=5., maximum_step=expiry/16,
    worker_threads=2, reduction_block_size=64)
price = plan.evaluate()
print(f'Approximate price: {price.value:.8g} (sampling SE {price.standard_error:.3g})')
print('Scheme:', price.scheme)
print('Fingerprint:', price.plan_fingerprint)
print('Uses physical log-Spot variance including the stochastic dividend reserve.')
print('Past monitoring is explicitly unhit; diffusion and cash jumps are monitored from valuation.')
print('Sampling SE excludes calibration uncertainty and time-grid bias. Refine maximum_step before use.')

risk = plan.evaluate_spot_bump_risk(spot_absolute_bump=1.)
print('Spot bumps:', risk.spot_bumps)
print('Finite-bump Deltas:', risk.delta_estimates)
print('Paired sampling SEs:', risk.delta_standard_errors)
print('Adjacent bump differences:', risk.bump_differences)
print('Paired gap SEs:', risk.bump_difference_standard_errors)
print('Risk method:', risk.method)
print('Finite bumps include endpoint/jump branch changes; gaps are diagnostics, not error bounds.')
