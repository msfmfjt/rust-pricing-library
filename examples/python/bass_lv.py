"""Multi-marginal Bass-LV: calibration, exact Brownian steps and Asian pricing."""
import rust_pricing as rp

marginals = [rp.BassMarginal.lognormal(t, 100.0, 0.20) for t in [0.5, 1.0, 2.0]]
model = rp.BassLvModel.calibrate(100.0, marginals)
for d in model.diagnostics:
    print(f"[{d.start_time:g}, {d.end_time:g}] iterations={d.iterations} "
          f"CDF residual={d.cdf_residual:.3g}, propagated CDF error={d.marginal_cdf_error:.3g}")

# Missing calibration expiries are inserted automatically. Returned paths
# contain only the requested observations; inserted dates are not Asian fixings.
plan = model.compile_simulation([0.25, 0.75, 1.25, 1.75, 2.0])
print("Brownian time grid:", plan.time_nodes)
european = plan.price_european(100.0, paths=100_000, seed=42)
asian = plan.price_asian(100.0, paths=100_000, seed=42)
print(f"European call: {european.price:.6f} +/- {european.standard_error:.6f} (1 SE)")
print(f"Marginal reference: {marginals[-1].call_price(100.0):.6f}")
print(f"Asian call: {asian.price:.6f} +/- {asian.standard_error:.6f} (1 SE)")
print("Local volatility at t=0.75, S=100:", model.local_volatility(0.75, 100.0))
print("First two paths:", plan.sample_paths(2, seed=42))
