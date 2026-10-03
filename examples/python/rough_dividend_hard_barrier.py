"""Discrete rough-LSV Barrier price/Spot Delta, with a fixed cash rebate."""
import rust_pricing as rp

expiry = 364/365
request = rp.PricingRequest(
    '2026-09-04',
    rp.Product.barrier(1, 2, '2027-09-03', 100., 95., 2., 'put', 'down',
        'knock_out', 'discrete', ['2027-03-05', '2027-09-03'], '2027-12-04', rebate=7.),
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
    rp.RiskRequest())  # Price-only request; select the hard risk method explicitly.
plan = rp.StochasticDividendPlan.compile_rough_bergomi_lsv(
    request, hurst=.1, vol_of_vol=.6, correlation=-.4,
    dividend_mean_reversion=.7, equity_linkage=.6, dividend_volatility=.35,
    equity_dividend_correlation=-.25, dividend_volatility_correlation=.15,
    particle_count=256, calibration_seed=42, log_bandwidth=.35,
    minimum_effective_samples=5., maximum_step=expiry/16,
    worker_threads=2, reduction_block_size=64, retain_reverse_trace=False)
risk = plan.evaluate_lsv_hard_barrier_spot_risk()
print(f'Price: {risk.price.value:.8g} (sampling SE {risk.price.standard_error:.3g})')
print(f'Spot Delta: {risk.delta:.8g} (sampling SE {risk.delta_standard_error:.3g})')
print('Method:', risk.method)
print('Price fingerprint:', risk.price.plan_fingerprint)
print('Rebate: fixed cash 7 at payment on the inactive branch; not multiplied by notional 2.')
print('Sampling errors exclude calibration uncertainty and time-grid bias.')
