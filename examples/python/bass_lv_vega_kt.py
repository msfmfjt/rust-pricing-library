"""Bass-LV market-IV VegaKT, including projection and recalibration.

Run after `maturin develop`. Quote IVs are absolute (0.20 = 20%); log strikes
refer to the driftless martingale coordinate log(K/spot).
"""
import rust_pricing as rp

market = rp.BassMarketIvModel.calibrate(
    100.0,
    maturity_nodes=[0.5, 1.0],
    log_moneyness_nodes=[-2.0, 0.0, 2.0],
    implied_volatilities=[0.20]*6,  # Time-major: all strikes at T=0.5, then T=1.
    projection_nodes=[-2.0+4.0*i/1600 for i in range(1601)],
    config=rp.BassLvConfig(grid_points=801, cdf_tolerance=1e-7),
)
plan = market.compile_vega_kt([0.25, 0.75, 1.0], bump_size=1e-4)
risk = plan.price_asian(100.0, paths=50_000, seed=42)
print(f"Asian: {risk.estimate.price:.6f} +/- {risk.estimate.standard_error:.6f}")
print(f"Method: {risk.method}; absolute-IV bump: {risk.bump_size}")
print("T     log(K/S0)   Vega / vol point    Paired standard error")
for i, t in enumerate(risk.maturity_nodes):
    for j, k in enumerate(risk.log_moneyness_nodes):
        n = i*len(risk.log_moneyness_nodes)+j
        print(f"{t:3.1f}   {k:8.2f}   {risk.vega_per_vol_point[n]: .8f}        "
              f"{risk.standard_errors_per_vol_point[n]:.8f}")
print(f"Parallel / vol point: {0.01*risk.parallel_sensitivity:.8f} "
      f"+/- {0.01*risk.parallel_standard_error:.8f}")
print(f"Bucket sum / vol point: {0.01*risk.bucket_sum:.8f} "
      f"+/- {0.01*risk.bucket_sum_standard_error:.8f}")
print("Scenario max CDF residual:", max(d.cdf_residual for s in plan.diagnostics for d in s.calibration))
print("Scenario max marginal CDF error:", max(d.marginal_cdf_error for s in plan.diagnostics for d in s.calibration))
print("Errors above measure sampling only; check the bump and grid resolution separately.")
