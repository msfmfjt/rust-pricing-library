"""Fit SSVI training nodes, then validate predeclared unused sites without refitting."""
import rust_pricing as rp

surface = rp.HestonCalibrationSsviSurface.power_law(
    [.25, .75, 1.5], [.01, .03, .06], .04, -.5, .35, .5)
training = [surface.quote(t, 100., k, .97)
            for t in [.25, .75, 1.5] for k in [85., 100., 115.]]
# Declare the holdout BEFORE calibration. Never use its errors to pick parameters.
holdout = [surface.quote(t, 100., k, .97)
           for t in [.5, 1.] for k in [90., 110.]]
policy = rp.HestonIvRefinementOptions.create(
    fit_tolerance=5e-4, grid_tolerance=2.5e-5, max_stages=1)
model = rp.RoughVolatilityModel.rough_heston(
    hurst=.15, initial_variance=.05, mean_reversion=.9,
    long_run_variance=.05, vol_of_vol=.23, correlation=-.5)
variables = [rp.HestonCalibrationVariable.create(name, lo, hi, scale)
             for name, lo, hi, scale in [
                 ('initial_variance', .01, .1, .04), ('mean_reversion', .1, 2., .7),
                 ('long_run_variance', .015, .12, .05), ('vol_of_vol', .03, .4, .2),
                 ('correlation', -.95, -.05, .5), ('hurst', .02, .49, .2)]]
problem = rp.HestonIvCalibrationProblem.compile(model, training, variables,
    time_steps=128, integration_intervals=512, cutoff=128.)
fit = problem.calibrate(max_iterations=60, max_evaluations=100, residual_tolerance=2e-6)
report = problem.validate_holdout(fit.parameters, holdout, policy)
print('Optimizer fit:', fit.fit_achieved)
print('Holdout accepted:', report.accepted)
print('Max holdout error, IV bp:', report.max_abs_iv_residual / 1e-4)
print('Max grid change, IV bp:', report.max_abs_iv_difference / 1e-4)
# This is a finite synthetic holdout check, not total-error or market prediction evidence.
