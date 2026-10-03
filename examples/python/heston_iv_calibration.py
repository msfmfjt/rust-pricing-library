"""Sample an SSVI target and fit a rough-Heston model to absolute Black IV residuals."""
import math
import rust_pricing as rp

surface = rp.HestonCalibrationSsviSurface.power_law(
    [0.25, 0.75, 1.5], [0.01, 0.03, 0.06], 0.04,
    rho=-0.5, eta=0.35, gamma=0.5,
)
quotes = [surface.quote(t, f, f*math.exp(k), d)
          for t, f, d in [(0.25, 100., 0.99), (0.75, 101., 0.97), (1.5, 102., 0.95)]
          for k in [-0.15, 0., 0.15]]
model = rp.RoughVolatilityModel.rough_heston(
    hurst=0.1, initial_variance=0.05, mean_reversion=0.7,
    long_run_variance=0.055, vol_of_vol=0.23, correlation=-0.5,
)
variables = [rp.HestonCalibrationVariable.create(name, low, high, scale)
             for name, low, high, scale in [
                 ('initial_variance', 0.01, 0.1, 0.04),
                 ('vol_of_vol', 0.03, 0.4, 0.2),
                 ('correlation', -0.95, -0.05, 0.5)]]
problem = rp.HestonIvCalibrationProblem.compile(
    model, quotes, variables, time_steps=64, integration_intervals=512, cutoff=128.,
)
result = problem.calibrate(max_iterations=30, max_evaluations=60, residual_tolerance=5e-4)
print('All quotes within 5 IV bps:', result.fit_achieved, '| stop:', result.termination)
print(dict(zip(result.parameter_names, result.parameters)))
for q, iv in zip(quotes, result.evaluation.model_implied_volatilities):
    print(f'T={q.maturity:g} K/F={q.strike/q.forward:.4f}: target={q.target_volatility:.6f}, '
          f'model={iv:.6f}, difference={(iv-q.target_volatility)*10000:.3f} IV bps')
# An SSVI surface need not lie in the chosen Heston family: false is a valid result.
# This finite-grid fit is not a continuous-time or Monte Carlo accuracy certificate.
