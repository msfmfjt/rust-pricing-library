"""Two-factor Bergomi LSV with one shared HW rate, paid cash and quote-node risk."""
import math
import rust_pricing as rp

r = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.05)])
q = rp.DiscountCurve(2, [0., 2.], [1., math.exp(-.02)])
markets = [rp.Market.equity(1, i+1, spot, r, q,
    discrete_dividends=[rp.DividendEvent.fixed_cash(i+1, .5, cash)])
    for i, (spot, cash) in enumerate([(100., 2.), (90., 1.)])]
# Retained quotes regenerate the paired variance and density at product dates.
# They refer to continuous normalized equity under the affine dividend model.
targets = [rp.HullWhiteLsvTarget.from_market_iv([.5, 1.], [-.5, 0., .5],
    [vol]*6, [0., .5, 1.], [-.5, -.25, 0., .25, .5]) for vol in [.25, .3]]
lsv = [rp.MultiAssetLsv2FactorConfig(mean_reversions=k, vol_of_vol=nu,
    mixing_weight=theta, spot_correlations=rho, factor_correlation=rho12,
    particle_count=2048, calibration_seed=401+i, log_bandwidth=.5,
    minimum_effective_samples=8., retain_reverse_trace=True)
    for i, (k, nu, theta, rho, rho12) in enumerate([
        ([4., .35], .3, .3, [-.65, -.25], .5),
        ([2.2, .12], .24, .65, [-.35, -.1], .25)])]
rates = rp.HullWhiteModel(.13, [0., .37], [.004, .006])
correlation = rp.CorrelationSchedule([1, 2], ["2026-01-01"], [[[1., .4], [.4, 1.]]],
    symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12, psd_abs_tol=1e-12,
    psd_rel_tol=1e-12, zero_pivot_abs_tol=1e-12, zero_pivot_rel_tol=1e-12)
# Price/rate, followed by volatility/rate: A, B, A1, A2, B1, B2.
rate_correlations = [.2, -.1, -.05, .02, .03, -.02]
# Explicit Brownian order: W_A, W_B, V_A1, V_A2, V_B1, V_B2, W_r.
# Cross-asset price/vol correlations can be supplied independently.
drivers = [[1., .4, -.65, -.25, .1, -.05],
           [.4, 1., -.15, -.05, -.35, -.1],
           [-.65, -.15, 1., .5, .2, .04],
           [-.25, -.05, .5, 1., .03, .15],
           [.1, -.35, .2, .03, 1., .25],
           [-.05, -.1, .04, .15, .25, 1.]]
drivers = [row+[rho] for row, rho in zip(drivers, rate_correlations)]
drivers.append(rate_correlations+[1.])
products = {
    "Basket": rp.MultiAssetProduct.basket([1, 2], [.6, .4], [1., 1.], "call", 96.,
        "2027-01-01", "2027-01-15", smoothing_half_width=2.),
    "Worst-of": rp.MultiAssetProduct.worst_of([1, 2], [100., 90.], "put", 1.,
        "2027-01-01", "2027-01-15", notional=100., smoothing_half_width=.05),
    "Autocall": rp.MultiAssetProduct.autocallable([1, 2], [100., 90.],
        [rp.AutocallObservation(t, pay, coupon_amount=5., coupon_level=.9, call_level=1.05)
         for t, pay in [("2026-07-02", "2026-07-16"), ("2027-01-01", "2027-01-15")]],
        "2027-01-01", "2027-01-15", notional=100., final_barrier=.7,
        memory=True, on_autocall="pay", on_maturity="forfeit", smoothing_half_width=.05),
}
engine = rp.Engine.randomized_quasi_monte_carlo(512, 702, scramble_count=8,
    antithetic=True, brownian_bridge=True)
for name, product in products.items():
    plan = rp.MultiAssetPlan.compile("2026-01-01", product, markets,
        [t.model for t in targets], correlation, engine, maximum_step=.125,
        worker_threads=2, reduction_block_size=128, lsv_configs=lsv,
        driver_correlations=[drivers], rate_model=rates,
        rate_correlations=rate_correlations, lsv_targets=targets)
    assert plan.random_factor_count == 8
    result = plan.evaluate_aad(gamma_relative_bump=.001)
    print(f"{name}: {result.value:.8f}; conditional SE {result.standard_error:.8f}")
    print("Delta:", [v.delta.value for v in result.risks])
    print("Cross Gamma:", [[g.value for g in row] for row in result.gamma])
    print("Common discount-node DV01:", [v.value for v in result.hull_white_curve_risk.discount_node_dv01])
    for risk in result.risks:
        print("Asset", risk.underlying_id, "IV VegaKT:", risk.hull_white_lsv.vega_kt_raw)
        print("Parallel IV Vega / conditional SE:", risk.hull_white_lsv.parallel_vega,
              risk.hull_white_lsv.parallel_vega_standard_error)
    print("Calibration fallback cells:", [sum(c.fallback_nodes) for c in plan.hull_white_calibrations])

request = rp.PricingRequest("2026-01-01",
    rp.Product.european_vanilla(1, 1, "2027-01-01", 100., 1., "call"),
    markets[0], targets[0].model, engine, rp.RiskRequest())
single = rp.HullWhiteEquityPlan.compile_lsv_two_factor(request, targets[0], rates,
    mean_reversions=[4., .35], vol_of_vol=.3, mixing_weight=.3,
    spot_correlations=[-.65, -.25], factor_correlation=.5,
    equity_rate_correlation=.2, vol_rate_correlations=[-.05, .02],
    particle_count=2048, calibration_seed=401, log_bandwidth=.5,
    minimum_effective_samples=8., worker_threads=2, reduction_block_size=128,
    retain_reverse_trace=True)
assert single.random_factor_count == 5
print("Single equity IV VegaKT:", single.evaluate_aad().vega_kt_raw)
