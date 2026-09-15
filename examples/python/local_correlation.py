"""Particle Local Correlation on a normalized continuous-equity basket."""
import math
import rust_pricing as rp

TOL = dict(symmetry_abs_tol=1e-12, diagonal_abs_tol=1e-12,
           psd_abs_tol=1e-12, psd_rel_tol=1e-12,
           zero_pivot_abs_tol=1e-12, zero_pivot_rel_tol=1e-12)


def correlation(rho):
    return rp.CorrelationSchedule([1, 2], ["2026-01-01"],
        [[[1., rho], [rho, 1.]]], **TOL)


r = rp.DiscountCurve(1, [0., 2.], [1., math.exp(-.05)])
q = rp.DiscountCurve(2, [0., 2.], [1., math.exp(-.02)])
markets = [rp.Market.equity(1, i+1, s, r, q) for i, s in enumerate([100., 90.])]
times, xs = [0., .5, 1.], [-.6, 0., .6]
models = [rp.Model.local_volatility_from_grid(times, xs,
    [.082, .070, .065, .086, .074, .068, .090, .078, .072], floor=1e-6, cap=1.),
    rp.Model.black_scholes(.3)]
# B=.6*f_A/F_A(t)+.4*f_B/F_B(t), not the physical payoff basket below.
target = rp.Model.local_volatility_from_grid(times, xs,
    [.059, .053, .049, .061, .055, .050, .063, .057, .052], floor=1e-6, cap=1.)
config = rp.LocalCorrelationConfig(basket_weights=[.6, .4], target_model=target,
    second_correlations=correlation(.95), particle_count=2048,
    calibration_seed=8401, log_bandwidth=.4, minimum_effective_samples=8.,
    feasibility="project_and_report", retain_reverse_trace=True)
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
engine = rp.Engine.randomized_quasi_monte_carlo(512, 702, scramble_count=8,
    antithetic=True, brownian_bridge=True)
for name, product in products.items():
    plan = rp.MultiAssetPlan.compile("2026-01-01", product, markets, models,
        correlation(-.3), engine, maximum_step=.125, worker_threads=2,
        reduction_block_size=128, local_correlation=config)
    result = plan.evaluate_aad(gamma_relative_bump=.001)
    cal, risk = plan.local_correlation_calibration, result.local_correlation_risk
    print(f"{name}: {result.value:.8f}; conditional SE {result.standard_error:.8f}")
    print("Delta:", [v.delta.value for v in result.risks])
    print("Cross Gamma:", [[g.value for g in row] for row in result.gamma])
    print("Recalibrated basket variance adjoints:", risk.basket_variance_adjoints)
    print("Recalibrated constituent volatility adjoints:", risk.asset_adjoints)
    print("Conditional basket-risk SE:", risk.basket_standard_errors)
    print("Projected/fallback cells:", sum(cal.projected_nodes), sum(cal.fallback_nodes))
    print("Maximum absolute variance residual:", max(map(abs, cal.variance_residuals)))
    print("Correlation at t=.5, B=1:", cal.correlation_at(.5, 0.))
