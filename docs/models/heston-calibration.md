# Heston Fourier price calibration (experimental)

`HestonCalibrationProblem` fits positive-forward European call/put **prices**
across one or more maturities. Rough Heston permits any selected subset of
`initial_variance`, `mean_reversion`, `long_run_variance`, `vol_of_vol`,
`correlation`, and `hurst`. Lifted Heston permits the first five, preserving all
explicit kernel weights/rates. Unselected parameters remain exactly fixed.
There is no inferred H-to-lift mapping, Spot/dividend adjustment or measure change.

## Inputs and objective

Each `HestonCalibrationQuote` supplies maturity in years, forward, strike,
discount, option side, target price and a strictly positive `price_scale`.
It is validated when the problem is compiled. For selected physical parameters
`p`, the minimized objective is

\[
 r_i(p)=\frac{P_i(p)-C_i}{s_i},\qquad f(p)=\tfrac12\sum_i r_i(p)^2.
\]

Thus a conventional squared-error weight `w_i` corresponds to `s_i=1/sqrt(w_i)`.
Scales are caller-supplied constants; they are not inferred quote uncertainties,
normalized probabilities or automatically Black vegas. All quotes are retained,
including duplicates. Individual intrinsic/upper price bounds are checked;
there is no full surface calendar/convexity arbitrage validation.

`HestonCalibrationVariable` supplies a distinct parameter name, finite box bounds
and a positive solver-coordinate scale. Bounds must remain inside the model
domain. Initial variance has a strictly positive lower bound; selected H lies
in `(0,.5]`, nonnegative scalar parameters remain nonnegative and correlation
lies in `[-1,1]`. Initial values must be inside the box. This API deliberately
requires positive initial variance even when fitting only H. There are at most
6 variables, 4096 quotes and 64 distinct maturities, and at least as many quotes
as variables. Counting quotes does **not** establish full Jacobian rank.

## Algorithm and Jacobian

Each evaluation compiles one price plan per distinct maturity, sharing its
cached transform and derivative plans across that maturity's quotes. The
[scalar tangents](heston-parameter-risk.md) and, when selected, the
[Hurst tangent](heston-hurst-risk.md) supply analytic derivatives of the actual
numerical prices. No production finite differences, recalibration-in-the-risk
calculation or external optimization dependency is used.

`evaluate(parameters)` returns prices, scaled residuals and the residual Jacobian
in input quote order and selected-variable order. Jacobian entries are
`dP_i/dp_j / price_scale_i`, **without** the variable-coordinate scale. Python
returns a list of rows; Rust uses a flat row-major vector. These outputs also
permit explicitly controlled use of an external optimizer.

The model-independent implementation in `pricing-numerics::least_squares` uses
a small dense, scaled, projected damped Gauss--Newton method. Blocking coordinates
at a bound are held fixed; damped normal equations are solved by Cholesky, the
trial point is projected to the box and only a strict objective decrease is
accepted. Damping is divided by three after acceptance and multiplied by ten
after rejection. Invalid trial pricing points are counted and rejected; an
initial failure is an error. Invalid callback shapes/nonfinite arrays in the
numerical solver are errors, not silently dropped residuals. Normal-equation
regularization is not a rank test; ill-conditioning and local minima remain
limitations. This is **not** SciPy's trust-region-reflective algorithm.

## Result semantics

`fit_achieved` means every `abs(scaled_residual)` is no greater than
`residual_tolerance` at the returned model. It does not mean correct parameters,
a unique solution, a global optimum, a continuum-accurate fit or a successful
fit to independent market data.

`termination` separately records `residual_tolerance`,
`projected_gradient_tolerance`, `step_tolerance`, `max_iterations`,
`max_evaluations`, or `no_progress`. A constrained stationary point, a small step,
or an exhausted budget can return **fit_achieved=False**. `active_bounds` marks
coordinates exactly at their declared bounds. `accepted_objectives` includes the
initial objective and accepted decreases only. `invalid_evaluations` is a subset
of `rejected_evaluations`; neither includes failed matrix factorizations.
The projected gradient is measured in scaled solver coordinates, not raw
financial units.

The evaluation budget includes a reserved final complete re-evaluation, which
must reproduce the final residuals and Jacobian exactly. The result contains the
fitted reusable model and unaltered input-order model prices. Rust's nested
`optimizer.evaluations` excludes that final check; the calibration result's
`evaluations` includes it. Python exposes the inclusive count. `max_iterations`
counts trial-step attempts, including a failed factorization.

Fourier frequency-grid differences and tail indicators are retained with each
quote. They do not include Riccati time error, omitted-tail bounds or optimizer,
calibration or model uncertainty. Reprice the fitted model on finer time and
frequency grids before interpreting fit residuals. Small residuals can otherwise
reflect a fit to discretization error.

## Python entry point

See the [executable example](../../examples/python/heston_calibration.py).
Construct quotes with `HestonCalibrationQuote.create(...)`, select variables with
`HestonCalibrationVariable.create(name,lower,upper,scale)`, then call
`HestonCalibrationProblem.compile(model,quotes,variables,...)` and `calibrate()`.
Python uses the numerical solver's default initial damping of `1e-3`; Rust can
set it explicitly through `LeastSquaresOptions`.

For IV or SSVI inputs, callers must explicitly convert quotes to forward Black
prices and select price scales. This addition does not implement an IV objective,
SSVI adapter, automatic initial guesses, regularization/prior fitting, multi-start
search, parameter confidence intervals, generalized forward variance curves,
recalibrated Greeks/AAD/VegaKT, VIX/SSR or LSV/rate/dividend composition.

## References and evidence

The transform is inherited from the separate [Fourier API](rough-heston-fourier.md)
and El Euch--Rosenbaum's [rough Heston transform](https://arxiv.org/abs/1609.02108).
The independent optimizer comparison uses SciPy's documented
[bounded least-squares/TRF method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
on the **same** pricing residuals/Jacobian; it is not an independent model oracle.
See the [validation protocol](../../design/validation/heston-calibration.md).

The returned parameters do not remove time-step or full-truncation bias from
the separate MC path schemes. A Fourier fit is not a verification that those
MC prices reproduce the same quotes.

For exact Black implied-volatility residuals and SSVI targets, see [IV calibration](heston-iv-calibration.md).
