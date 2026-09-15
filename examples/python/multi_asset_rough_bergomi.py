"""Two rough-LSV assets, a common HW rate, paid cash and calibrated quote risk."""
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
# vol_of_vol is eta, the log-variance coefficient, for a rough config.
lsv = [rp.MultiAssetRoughLsvConfig(hurst=h, vol_of_vol=eta, correlation=rho,
    particle_count=2048, calibration_seed=401+i, log_bandwidth=.5,
    minimum_effective_samples=8., retain_reverse_trace=True)
    for i, (h, eta, rho) in enumerate([(.1, .5, -.65), (.3, .4, -.35)])]
rates = rp.HullWhiteModel(.13, [0., .37], [.004, .006])
correlation = rp.CorrelationSchedule([1, 2], ["2026-01-01"], [[[1., .4], [.4, 1.]]],
    symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12, psd_abs_tol=1e-12,
    psd_rel_tol=1e-12, zero_pivot_abs_tol=1e-12, zero_pivot_rel_tol=1e-12)
# Brownian input order: W_A, W_B, V_A, V_B, W_r (five drivers).
# Exact step order adds a rate integral and two power integrals (eight blocks):
# dW_A, dW_B, dV_A, dV_B, OU_r, integral_r, J_A, J_B.
rate_correlations = [.2, -.1, -.05, .03]
# Explicit cross-asset price/volatility and volatility/volatility correlations.
drivers = [[1., .4, -.65, .1, .2],
           [.4, 1., -.15, -.35, -.1],
           [-.65, -.15, 1., .3, -.05],
           [.1, -.35, .3, 1., .03],
           [.2, -.1, -.05, .03, 1.]]
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
