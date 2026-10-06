# Rough-family Gamma validation protocol

Base: PR #137 `a8c36cad4ea9cb3f7f9ea6911b351f2f139dd346`.
This increment does not modify existing path, calibration or adjoint formulas.
See [contract](../../docs/models/rough-family-gamma.md).

## Fixed pre-run checks

The fast tests retain the parent's six nondegenerate models, including stochastic
RFSV initial variance. They exercise pseudo-MC128 antithetic units and RQMC4x64,
seed819, true Brownian blocks only for bridge, h=1/h2=.5, 128 calibration particles,
seed429, log-bandwidth.5, ESS3. Nonuniform Local Variance axes avoid an active
initial kink. Dividends include cash3 at.25 and cash4 at1.5 after expiry.

Pure and sticky-relative Gamma are compared with full physical-Spot request
reconstruction/recalibration at both widths. Pure SABR beta0 and beta.5 are added.
Tolerance is 2e-10*(1+abs(reference)). This is same-pricer finite arithmetic
consistency, not an independent stochastic-model pricing reference.

Frozen price and Gamma/half-Gamma/difference SEs are reconstructed from public
paths/shocks with independent European payoff/escrow and two-pass sampling-unit
statistics. A nonvacuous covariance check rejects independent marginal-SE
combination. Workers1/3, trace absent/present, width errors, risk fingerprints,
old price/Delta/local-variance risk preservation, time-zero/proportional/future
cash are checked. Python independently reconstructs Asian paths/payoff/paired
SE and adds nonflat carry, delayed payment, hard-risk rejection and immutable
return fields.

## Independent Black-limit numerical panel

Six constant-variance limits, v=.04, T1, S=K100, zero carry/cash, seeds91/1973,
4 time intervals; RQMC8x4096 with antithetic and bridge (65,536 base paths per row).
Local Variance target is flat, with spatial nodes [-.4,.4]. 128 calibration particles. Physical widths h1/.5.
Analytic true Gamma: .01984762737385059. Independently calculated Black central
second differences: .019844113046403322 / .019846748725001362 (erfc prices).

Before the first numerical run the fixed gates are:
`abs(estimate - analytic_finite_width_value) <= 5*corresponding_SE + 2e-4`,
`0 < corresponding_SE <= .002`. Both widths, pure and LSV are tested; 12 rows
are retained. Analytic finite-bump bias is reported separately from sampling
error. The more accurate half-width analytic value is not a guarantee of
monotone observed Monte Carlo errors. No gate on a particular speedup or on
observed Gamma positivity is imposed.

The full suite's ignored count does not imply all extended numerical panels
ran. CI explicitly includes this panel in the release target and retains logs
on Linux/macOS/Windows. Local result counts, failed initial builds and actual
configuration are recorded separately in the delivery report and PR.

## Limits

Paired finite-difference SEs exclude all bias and calibration uncertainty.
Vanilla/Asian exercise crossings and leverage interpolation can make small-width
Gamma noisy; h and h/2 are diagnostics, not a convergence certificate. Independent
Black results cover deterministic-variance limits only. New nondegenerate
continuous-time Gamma references and direct second-order estimators remain work.

## Initial fixture-domain failure

The first Black-panel execution used spatial nodes [-1,1] with bandwidth .5.
No node had sufficient supported calibration particles at the first positive
time, so calibration correctly returned `NoSupportedCalibrationNode`. No Gamma
accuracy assertion had failed. The flat grid endpoints were changed to [-.4,.4]
to lie within the finite-particle support. Counts, seeds, model, time grid,
bandwidth, ESS threshold, Black references and all numerical budgets are
unchanged. The original failed log is retained, not counted as a pass.

The next execution reached the Gamma gate but retained nonflat discount/repo
factors from the shared request fixture (.95/.98), while the independently
declared Black reference assumes both factors equal one. The failed comparison
(approximately .0189163 versus .0198441 at h=1) is retained. The test now explicitly
sets both curves to one, matching the predeclared zero-carry protocol; production
arithmetic, reference values, sample counts and tolerance gates are unchanged.
The nonflat-carry reconstruction checks remain in the separate fast/Python tests.
