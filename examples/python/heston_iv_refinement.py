"""Finite-grid validation and bounded recalibration; not a total-error guarantee."""
import rust_pricing as rp

model = rp.RoughVolatilityModel.rough_heston(hurst=.15, initial_variance=.05, mean_reversion=.9, long_run_variance=.05, vol_of_vol=.23, correlation=-.5)
surface = rp.HestonCalibrationSsviSurface.power_law(
    [.25,.75,1.5], [.01,.03,.06], .04, -.5, .35, .5)
quotes = [surface.quote(t,100.,k,.97) for t in [.25,.75,1.5] for k in [85.,100.,115.]]
variables = [rp.HestonCalibrationVariable.create(name, lo, hi, scale)
             for name,lo,hi,scale in [
                 ('initial_variance',.01,.1,.04), ('mean_reversion',.1,2.,.7),
                 ('long_run_variance',.015,.12,.05), ('vol_of_vol',.03,.4,.2),
                 ('correlation',-.95,-.05,.5), ('hurst',.02,.49,.2)]]
problem = rp.HestonIvCalibrationProblem.compile(model, quotes, variables,
    time_steps=64, integration_intervals=256, cutoff=64.)
policy = rp.HestonIvRefinementOptions.create(
    fit_tolerance=5e-4, grid_tolerance=1e-5, max_stages=2)  # 5 IV bp and 0.1 IV bp
result = problem.calibrate_refined(policy, max_iterations=60,
    max_evaluations=100, residual_tolerance=2e-6)  # per-stage optimizer budget
print('Finite-grid status:', result.termination)
for level, stage in enumerate(result.stages):
    check = stage.validation
    print(level, 'optimizer_fit:',stage.calibration.fit_achieved,
          'grid_accepted:',check.accepted,
          'target_error_bp:',check.max_abs_iv_residual/1e-4,
          'grid_change_bp:',check.max_abs_iv_difference/1e-4)
    print(check.grid_names, check.configurations)
# Reuse this model, but do not assume its MC discretization has the same accuracy.
calibrated_model = result.stages[-1].calibration.model
