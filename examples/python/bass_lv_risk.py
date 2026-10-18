"""Connect an eSSVI smile to Bass-LV and differentiate terminal-map hats."""
import rust_pricing as rp

slices = [rp.EssviSlice(0.5, 0.02, 0.06, -0.025),
          rp.EssviSlice(1.0, 0.04, 0.09, -0.035),
          rp.EssviSlice(2.0, 0.08, 0.13, -0.045)]
log_strikes = [-4.0 + 8.0*i/1600 for i in range(1601)]
projections = [rp.BassMarginal.from_essvi(t, 100.0, slices, log_strikes)
               for t in [0.5, 1.0, 2.0]]
for p in projections:
    d = p.diagnostics
    print(f"T={p.marginal.expiry:g}: mean scale={d.mean_scale:.9f}, "
          f"max call error={d.max_call_price_error:.6g}")

model = rp.BassLvModel.calibrate(100.0, [p.marginal for p in projections])
bumps = [rp.BassMappingBump(i, -1.0, 0.0, 1.0) for i in range(3)]
plan = model.compile_mapping_risk([0.25, 0.75, 1.5, 2.0], bumps)
asian = plan.price_asian(100.0, paths=50_000, seed=42)
print(f"Asian: {asian.estimate.price:.6f} +/- {asian.estimate.standard_error:.6f}")
print("Terminal-map gradients:", asian.sensitivities)
print("Gradient standard errors:", asian.standard_errors)
vanilla = plan.vanilla_call(2, 100.0)
print(f"Semi-analytic 2Y call: {vanilla.price:.6f}")
print("Semi-analytic call gradients:", vanilla.sensitivities)

# These are derivatives per unit of terminal-map amplitude, not IV Vega.
eps = 1e-5
up = plan.bumped_simulation([eps, 0.0, 0.0]).price_asian(100.0, paths=50_000, seed=42)
dn = plan.bumped_simulation([-eps, 0.0, 0.0]).price_asian(100.0, paths=50_000, seed=42)
print("First gradient, AAD / CRN difference:", asian.sensitivities[0], (up.price-dn.price)/(2*eps))
