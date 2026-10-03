# Heston IV holdout validation

`HestonIvCalibrationProblem::validate_holdout(parameters, quotes, policy)` evaluates
unused IV sites at a **frozen parameter vector** on the five grids from
[IV refinement](heston-iv-refinement.md). Rust accepts a quote slice; Python accepts
a sequence of `HestonIvCalibrationQuote`. The return type is the existing immutable
`HestonIvGridValidation`, preserving quote order, signed residuals, grid differences,
plan counts and all acceptance flags.

This method does not run an optimizer, warm-start another fit, alter training
quotes, or use holdout errors to select a grid or parameter vector. It uses the
original problem's Fourier configuration, not an inferred grid from a prior result.
When validating a staged result, explicitly build a problem with that stage's
model, the original training quotes/variables, and the intended base grid first.
`policy.max_stages` does not affect this single-point check.

## Disjointness

An IV site is `(T, log(K/F))`. Two coordinates a,b are considered equal when
`abs(a-b) <= 64*EPSILON*max(1,abs(a),abs(b))`, separately for time and log-moneyness.
This is a conservative roundoff exclusion rule, not a statistical neighborhood or
an interpolation algorithm. Both times are in years. Every holdout site must differ
from every fitting site and every earlier holdout site. Comparison ignores option
side, discount, absolute price units, target IV and residual scale. Thus flipping a
call to a put or doubling both F and K does not disguise a fitting node as holdout.
Sites just outside the explicit tolerance remain permitted.

The method only checks the fitting sites stored in this problem. It cannot know
whether users already inspected these data, fitted another problem, or selected
models or hyperparameters using the proposed holdout. Freeze the holdout and its
criteria before inspecting results. Repeated selection based on its errors requires
a new untouched validation set. SSVI-derived targets are synthetic surface samples,
not independent observations of future markets.

## Report and failure behavior

For every holdout quote, evaluate `(N,M,U)`, `(2N,M,U)`, `(N,2M,U)`, `(N,2M,2U)` and
`(2N,4M,2U)`. Each grid uses the same model. `accepted` requires all five target-IV
residuals to satisfy `fit_tolerance` and all four probe-minus-base differences to
satisfy `grid_tolerance`. These are absolute annualized IV, independent of `iv_scale`.
One IV bp is `0.0001`. `calibration.fit_achieved` is separate and is never overwritten.
A failed holdout can coexist with a successful fit and numerically stable prices.

At most 4096 quotes and 64 distinct exact maturities are permitted. The existing
per-plan and aggregate work guards count the **holdout** maturities, not the fitting
maturities. One price-only plan is shared per exact maturity/grid; no Jacobians are
constructed. Invalid parameters, overlaps, duplicate sites, invalid targets,
unrepresentable Black IVs, invalid grids and work limits return errors. Numerical
failures never produce accepted partial reports or silently dropped quotes.

## Scope and evidence

The feature supports power-kernel Rough Heston and fixed-kernel Lifted Heston under
the parent's positive-forward, deterministic-discount convention. There is no new
Spot/dividend mapping, pricing formula, IV inversion, optimizer equation, or MC
scheme. It measures finite-site synthetic interpolation/extrapolation. Small grid
differences do not bound continuous-time or Fourier-tail error; holdout acceptance
does not establish generalization to real data, arbitrary wings or future regimes.

See [validation protocol](../../design/validation/heston-iv-holdout.md) and the
[complete Python example](../../examples/python/heston_iv_holdout.py).
