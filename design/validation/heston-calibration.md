# Heston Fourier calibration validation

## Scope and separation of evidence

This addition is a local, box-constrained **price** calibration over the existing
Fourier engine. It does not change MC paths, Fourier prices, scalar/Hurst
sensitivities, model defaults, stable JSON schemas, manifests or Cargo.lock.
The numerical optimizer is independent of finance and lives in pricing-numerics.
The model adapter and analytic Jacobian live in pricing; Python bindings live in
pricing-python. No runtime dependency or additional crate is introduced.

There are three different evidence types; they must not be conflated:

1. Ordinary-Heston and explicit three-factor-lift fits to the existing independent
   P1/P2/adaptive-ODE price fixture `fixtures/rough-volatility/fourier.json`.
2. Six-parameter rough-Heston recovery on a **same-scheme synthetic** price panel.
   This exercises the optimizer and Jacobian, not independent model price accuracy.
3. SciPy TRF versus the new optimizer on a shared finite-grid residual/Jacobian.
   This is independent optimization, not independent pricing.

The independent reference's byte hash and the numerical protocol are frozen in
`fixtures/rough-volatility/calibration.json`. Protocol controls reject target,
threshold, evaluation-budget and archive/CI-gate mutations. CI also reruns the
original independent Fourier reference checker without regenerating the fixture.

## Fixed numerical panel

Truth: v0=.04, kappa=.7, theta=.055, nu=.18, rho=-.65. Rough H=.1; ordinary
Heston fixes H=.5. The explicit lift fixes weights [.2,.4,.5] and rates [.1,1,8].
Forward is100 and discount .97. Ordinary/lift quotes use maturities .25/1 and
strikes80/100/120 (six calls each). Rough quotes use .25/.75/1.5 and85/100/115
(nine quotes; puts at strike85, calls otherwise). The rough quote scale is .1;
the independent Markov quote scale is1.

Two starts per family:

| Start | v0 | kappa | theta | nu | rho | Rough H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .045 | .85 | .06 | .21 | -.55 | .16 |
| 1 | .035 | .55 | .05 | .15 | -.75 | .16 |

All five scalar parameters are free. Rough additionally fits H. Bounds and solver
scales in variable order (v0,kappa,theta,nu,rho,H) are:

| Parameter | Lower | Upper | Scale |
| --- | ---: | ---: | ---: |
| v0 | .01 | .1 | .04 |
| kappa | .1 | 2 | .7 |
| theta | .015 | .12 | .05 |
| nu | .03 | .4 | .2 |
| rho | -.95 | -.05 | .5 |
| H | .02 | .49 | .2 |

Fit grid:256 Riccati steps,512 frequency intervals,cutoff128. Maximum100 attempted
steps and102 evaluations, including final verification. Required maximum scaled
residual is5e-5. Final independent re-evaluation at512 steps,1024 intervals,
cutoff256 must differ from each input target by at most .003 currency units.
The two discretization refinements are combined in this final check, so it does
not separate their individual contributions or certify continuous-time error.

## Results

All six calibrations passed on their **first** numerical run. No model, quote,
initial guess, grid, reference or tolerance was adjusted after seeing outcomes.
The later fixture registration moved the same pre-existing numerical constants
into a retained protocol; it did not change their values. The latest-source
numerical run is separately retained in the delivered evidence.

| Family/start | Evaluations incl. final check | Max scaled residual | Max absolute target-price residual | Max finer-grid target-price difference |
| --- | ---: | ---: | ---: | ---: |
| Heston/0 | 10 | 8.398290e-7 | 8.398290e-7 | 1.248878e-6 |
| Heston/1 | 12 | 4.343956e-5 | 4.343956e-5 | 4.251348e-5 |
| Lift/0 | 11 | 4.138445e-6 | 4.138445e-6 | 3.480793e-6 |
| Lift/1 | 14 | 3.077547e-6 | 3.077547e-6 | 3.389163e-6 |
| Rough/0 | 18 | 3.742736e-7 | 3.742736e-8 | 1.781992e-5 |
| Rough/1 | 10 | 1.049581e-5 | 1.049581e-6 | 1.803779e-5 |

Maxima are finite panel observations. These are price/scaled-price quantities,
not IV basis points. Parameters from each fit and all42 finer-grid quote
comparisons are printed, including cases where the fine-grid discrepancy is
larger than the on-grid fitted residual.

The independent optimizer comparison uses a separate64-step/256-interval/cutoff96
synthetic rough panel. Both optimizers fit all six parameters. The largest final
model-price difference is3.752266120038428e-8; the largest parameter-coordinate
difference is3.590462120395266e-6. The new solver used18 evaluations including its
final check; SciPy reported24 objective evaluations (different accounting).
SciPy uses its own tighter stationarity/step tests, so equal stopping times or
identical parameter solutions are not asserted.

## Fast controls and packaging

Generic numerical tests cover Rosenbrock, scaling by1e-6/1e6, rank-deficient
Jacobians, rejected invalid-domain trials, active-bound stationarity, honest
budget stops and malformed/nonfinite callback data. Rank deficiency is exercised
but not exposed as an identifiability certificate.

Rust financial tests cover both option sides, preservation of quote order,
nonuniform quote scales, analytic Jacobians versus two bump sizes, fixed kernel
arrays, exact Black-variance recovery, selected-Hurst recovery, bitwise
repeatability and bounds/unsupported H/invalid input/resource rejection. A truth
outside the permitted variance box must return an active bound with
`fit_achieved=false`; a budget-limited fit must also remain explicitly unfitted.
Python tests exercise the same public contracts and mutation checks for every
new class/member in the wheel stub. The executable example fits only v0/nu/rho,
keeps other parameters fixed and reprices on a finer grid.

The read-only three-OS workflow executes the ignored calibration acceptance test
explicitly and retains its logs; a separate job compares optimizers. Missing
artifacts fail upload. The source-archive checker enforces13 new members and14
CI/evidence snippets. Defining this workflow is not evidence it ran remotely.

## Development observations

The initial Clippy pass rejected the Python calibrate method's argument count.
Python now uses the numerical solver's fixed default initial damping1e-3; Rust
can still set it. No lint was suppressed and no numerical acceptance threshold
was changed. Two editing-time Python syntax mistakes were corrected before
execution. A premature wheel-test invocation occurred before the wheel build
finished and therefore could not import the extension; later installed-wheel
checks are separate from that failed setup attempt. These are not failed
numerical calibration comparisons.

## Limits

No global optimum, unique parameter recovery, confidence interval, Jacobian-rank
certificate, general stress/short-expiry/wing accuracy or simultaneous statistical
coverage is established. No real market quote set or SSVI surface was supplied.
The helper does not add an IV objective, automatic price scales, target-surface
arbitrage repair, regularization, automatic initial guesses, multi-start search,
generalized initial forward variance, calibrated market Greeks/AAD/VegaKT,
finite-lift coefficient calibration, VIX/SSR, or new LSV/rate/dividend composition.
The other four rough model families are not supported by this affine calibration
route. Pricing and calibration remain experimental.
