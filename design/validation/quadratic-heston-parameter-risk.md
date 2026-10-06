# Quadratic rough Heston parameter reverse validation

Base: PR145 head `9f89ed7bd6d711211a78fbd5cfa73e62cfb11859`;
source tree `d35bd745f5a90516b52d46f5b3a84f1575cfce90`.
Existing numerical budgets and pricing arithmetic are unchanged.

## Fixed protocols

All inputs and budgets below are fixed before their first numerical execution.
The scalar order is z0/kappa/nu/a/b/c/H; parameters are
(.1,.7,.4,.5,.2,.03,H). H=1/2 uses second-order backward differences.

- **896 path comparisons**: H=.03/.1/.3/.5,16 paths each, seed91, time nodes
  (0,.07,.21,.63,1), all forward/variance observations seeded, each coordinate at
  widths2e-7/1e-7. Bound3e-5*(1+abs(FD)). Primal paths and optional-H scalar
  prefixes must agree exactly; initial-forward adjoint has a scaling control.
- **42 calibration comparisons**: H=.1/.3/.5,128 particles, seed429, bandwidth.5,
  ESS3, nonuniform5x5 target, all Leverage nodes seeded, widths2e-7/1e-7 and the
  same relative bound. Donor topology must match the bumped calibrations and
  extrapolated nodes must actually be exercised.
- **168 fully rebuilt price comparisons**: Pure SV and LSV, H=.1/.3/.5,
  MC128 or RQMC4x64, antithetic/bridge, seed819, widths1e-6/5e-7, same bound.
  LSV recompiles the model and recalibrates the original target on each bump.
  Nonflat carry and cash before/after expiry remain present. Worker1/3,
  unchanged nested price, explicit names, optional-H prefix and error availability
  are checked. This is discrete derivative consistency, not an independent price.
- **Boundary controls**: kappa/nu/a/c zero with positive preterminal variance use
  inward differences at2e-7/1e-7. The a=0 derivative must be nonzero in the
  selected path. Wrong model, missing trace, malformed/nonfinite seeds and exact
  zero preterminal variance fail without changing old pricing. A terminal-variance
  seed at H=.5 with opposite residual normals independently checks the nonzero
  residual-loading derivative despite identical primal paths.
- **Deterministic LSV cancellation**: nu=0,H=.1/.5, nontrivial c direct risk (>1),
  total z0/kappa/a/b/c/H risks and errors <=1e-9. Nu's inward boundary derivative
  is deliberately not assumed zero. Both terms are computed, not hard-coded.
- **Independent covariance control**: manually reconstruct eight scramble means,
  their direct/calibration decomposition and plain two-pass errors. Require
  total mean/SE agreement within1e-10 and a nontrivial difference from the
  incorrect independent-contribution-SE combination, including Hurst.
- **Variance-only overflow control**: an unused unlevered asset overflows at a
  large spot normal, while valid QRH variance-only reverse must remain available.

## Independent nondegenerate expectation reference

`scripts/check_quadratic_heston_reference.py` imports NumPy and SciPy, not the
pricing extension. Integrate the first asset normal and newest-cell residual
with Gauss-Hermite; integrate the last asset normal analytically by conditional
Black. Time grid0/.5/1, S=K100, unit discount/repo, no cash. This represents the
existing **two-step law**, not its continuous-time limit. H=.1/.3/.5, parameters
as above. Both the first variance's effect on F1 and the later polynomial's
parameter/Hurst effects contribute to the analytic reference.

Orders64/96 must agree within2e-7*(1+abs(reference)). The reference derivatives
also match a price-only stencil at2e-6/1e-6 within2e-6*(1+abs(reference)). Retained
reference regeneration tolerance1e-10*(1+abs(reference)).

Production: two seeds91/1973,8x4096 antithetic/bridge RQMC. Six prices and42
parameter expectations must satisfy abs(gap)<=5SE+.003; every parameter SE must
be strictly positive and <=.2. These bounds are in natural price/parameter units,
not IV basis points. A small pathwise bump discrepancy does not imply a similarly
small expectation error. Numerical failures must be retained, not replaced by
seed/tolerance selection.

## Python and repository checks

Four new Python tests use the existing delayed-payment Asian/nonflat-carry and
complete proportional/future-cash builders. They check all directions in Pure/LSV,
MC/RQMC, H=.5, two bump widths, frozen/copy properties, workers, optional Hurst,
zero-parameter domains, hard Digital rejection and a smoothed risk direction.
The complete new example is registered with the existing wheel smoke suite.

Default/minimal and release Rust targets, covariance/overflow controls, Clippy,
strict Rustdoc, schema/dependency/Markdown/source archive and API/protocol guards
are required checks. Record actual executed scope and distinguish full workspace,
focused tests, ignored numerical panels and clean versus offline-inherited Python
environments. Keep initial compiler/environment failures separate from successful
numerical executions. Delivery must match the tracked source bytes and modes and
replay from the pinned parent tree, without local vendor/toolchain configuration.
