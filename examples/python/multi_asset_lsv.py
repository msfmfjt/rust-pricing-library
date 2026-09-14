"""Two correlated Bergomi LSV assets, calibrated target risk and cross Gamma."""
import math
import rust_pricing as rp

r = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.06)])
q = rp.DiscountCurve(2, [0., 2.], [1., math.exp(-.02)])
markets = [rp.Market.equity(1, i+1, s, r, q) for i, s in enumerate([100., 90.])]
models = [rp.Model.local_volatility_from_grid([0., .5, 1.], [-.8, 0., .8], v,
          floor=1e-5, cap=2.) for v in
          [[.048, .04, .035, .06, .048, .039, .065, .05, .041], [.09]*9]]
lsv = [rp.MultiAssetLsvConfig(mean_reversion=k, vol_of_vol=nu, correlation=rho,
        particle_count=2048, calibration_seed=401+i, log_bandwidth=.5,
        minimum_effective_samples=8., retain_reverse_trace=True)
       for i, (k, nu, rho) in enumerate([(.8, .3, -.65), (2.2, .24, -.35)])]
correlation = rp.CorrelationSchedule([1, 2], ["2026-01-01"], [[[1., .4], [.4, 1.]]],
    symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12, psd_abs_tol=1e-12,
    psd_rel_tol=1e-12, zero_pivot_abs_tol=1e-12, zero_pivot_rel_tol=1e-12)
# All spot drivers first, then the two volatility drivers. Omit this keyword
# to use the documented independent-residual coupling instead.
drivers = [[[1., .4, -.65, .1], [.4, 1., -.15, -.35],
            [-.65, -.15, 1., .3], [.1, -.35, .3, 1.]]]
products = {
    "Basket": rp.MultiAssetProduct.basket([1, 2], [.6, .4], [1., 1.], "call", 96.,
              "2027-01-01", "2027-01-01", smoothing_half_width=2.),
    "Worst-of": rp.MultiAssetProduct.worst_of([1, 2], [100., 90.], "put", 1.,
              "2027-01-01", "2027-01-01", notional=100., smoothing_half_width=.05),
    "Autocall": rp.MultiAssetProduct.autocallable([1, 2], [100., 90.],
        [rp.AutocallObservation(t, t, coupon_amount=5., coupon_level=.9, call_level=1.05)
         for t in ["2026-07-01", "2027-01-01"]], "2027-01-01", "2027-01-01",
        notional=100., final_barrier=.7, memory=True, on_autocall="pay",
        on_maturity="forfeit", smoothing_half_width=.05),
}
engine = rp.Engine.randomized_quasi_monte_carlo(2048, 702, scramble_count=8,
            antithetic=True, brownian_bridge=True)
for name, product in products.items():
    plan = rp.MultiAssetPlan.compile("2026-01-01", product, markets, models,
        correlation, engine, maximum_step=.125, worker_threads=2,
        lsv_configs=lsv, driver_correlations=drivers)
    result = plan.evaluate_aad(gamma_relative_bump=.001)
    print(f"{name}: {result.value:.8f}; conditional SE {result.standard_error:.8f}")
    print("Delta:", [v.delta.value for v in result.risks])
    print("Cross Gamma:", [[g.value for g in row] for row in result.gamma])
    for risk in result.risks:
        target = risk.lsv_local_variance
        print("Asset", risk.underlying_id, "target-variance adjoints:", target.node_adjoints)
        print("Conditional target-risk SE:", target.standard_errors)
    print("Calibration fallback cells:", [sum(c.extrapolated) for c in plan.lsv_calibrations])
