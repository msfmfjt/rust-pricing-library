"""Two-factor Bergomi LSV + shared HW + particle Local Correlation."""
import math
import rust_pricing as rp

TOL = dict(symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12, psd_abs_tol=1e-12,
    psd_rel_tol=1e-12, zero_pivot_abs_tol=1e-12, zero_pivot_rel_tol=1e-12)
def correlation(rho):
    return rp.CorrelationSchedule([1, 2], ['2026-01-01'],
        [[[1., rho], [rho, 1.]]], **TOL)
def target(sigma):
    return rp.HullWhiteLsvTarget.from_market_iv([.5, 1.], [-.5, 0., .5],
        [sigma]*6, [0., .5, 1.], [-.35, 0., .35])

asset, basket = target(.28), target(.235)
curve = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.05)])
markets = [rp.Market.equity(1, i+1, spot, curve,
    rp.DiscountCurve(100+i, [0., 2.], [1., math.exp(-.02*(i+1))]))
    for i, spot in enumerate((100., 90.))]
lsv = rp.MultiAssetLsv2FactorConfig(mean_reversions=[.4, 2.], vol_of_vol=.4,
    mixing_weight=.35, spot_correlations=[-.35, -.2], factor_correlation=.25,
    particle_count=1024, calibration_seed=318, log_bandwidth=.8,
    minimum_effective_samples=2., retain_reverse_trace=True)
local = rp.LocalCorrelationConfig(basket_weights=[.6, .4], target_model=basket.model,
    hull_white_target=basket, second_correlations=correlation(.95),
    particle_count=1024, calibration_seed=8401, log_bandwidth=.7,
    minimum_effective_samples=2., feasibility='project_and_report', retain_reverse_trace=True)
product = rp.MultiAssetProduct.basket([1, 2], [.6, .4], [1., 1.], 'call', 96.,
    '2027-01-01', '2027-02-01', smoothing_half_width=3.)
plan = rp.MultiAssetPlan.compile('2026-01-01', product, markets,
    [asset.model, rp.Model.black_scholes(.3)], correlation(-.3),
    rp.Engine.randomized_quasi_monte_carlo(128, 702, scramble_count=8,
        antithetic=True, brownian_bridge=True), maximum_step=.25,
    lsv_configs=[lsv, None], rate_model=rp.HullWhiteModel(.13, [0., .37], [.012, .0144]),
    rate_correlations=[.1, -.05, .02, .02], lsv_targets=[asset, None], local_correlation=local)
result = plan.evaluate_aad()
print('PV:', result.value, 'SE:', result.standard_error)
print('Basket quote Vega:', result.local_correlation_risk.basket_hull_white.vega_kt_raw)
print('Asset A quote Vega:', result.local_correlation_risk.asset_hull_white[0].vega_kt_raw)
print('Rate corrections:', plan.local_correlation_calibration.rate_corrections)
print('Projected nodes:', sum(plan.local_correlation_calibration.projected_nodes))
