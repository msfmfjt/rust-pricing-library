# Stochastic-dividend correlation risk validation

## Source and contract

Child of PR #88, pinned source `11fe8fc34f448d00147a3b1b5dc51aac12cdad86`,
tree `e9dc8ed28a11bac150fa5b6aa196dfa04ce9df35`.
The [decision](../adr/0017-stochastic-dividend-correlation-risk.md) defines the
opt-in API and admissible raw-entry domain. No existing budgets are modified.
This document specifies tests before native measurement; it is not a pass claim.

## Prespecified checks

- Two coefficient tests: every symmetric entry tangent against independently
  recompiled OU Cholesky factors, including zero correlations, zero/equal/tiny/large
  mean reversions and nonuniform intervals (3e-6 absolute); 2F centering against
  an independent unconditional OU moment formula (2e-13 absolute).
- Four public Rust tests: BS/1F/2F, MC/RQMC, two seeds, European, Asian delayed
  payment, smoothed pre/post-cash barrier, zero correlations, zero vol-of-vol,
  zero mean reversion, mixing endpoints, singular/near-singular domain. Both
  h=1e-5 and h=1e-6 central full-recompile differences must satisfy
  `abs(AAD-FD) <= 3e-5 + 2e-5 * max(abs(AAD),abs(FD))`.
  Sampling SE never widens this derivative budget. Raw correlation bumps modify
  the one symmetric pair, not a Cholesky parameter or a PSD projection.
- Exact preservation of prior risk labels, derivatives, standard errors, price,
  price SE and curve/cash accessors; one/three-worker risk replay.
- A perfect instantaneous V1/V2 correlation with unequal mean reversions is
  rejected by correlation AAD even though price/basic/model-parameter AAD work.
  Near-perfect equity/dividend correlations reject without removing base APIs.
- Three Python tests: all entry FDs, exact prefixes, zero correlations, worker
  replay, immutable result/copy semantics and the explicit domain restrictions.
- Release wheel, executable example, full Rust normal/minimal and Python suites,
  Clippy, rustdoc, schema/reference/link/archive/stub/dependency contracts.

All finite-difference budgets and seeds are fixed before observing results.
The tests validate differentiation of a finite numerical scheme; no absolute
continuous-time correlation-risk accuracy or broad stress/exotic coverage is
claimed. Sampling SE excludes discretization, model and calibration uncertainty.

## Parent evidence checked on 2026-09-24

Native candidate run 35941871329 completed with 556 normal all-feature Rust tests,
545 minimal-feature tests, 109 Python tests and its focused tests passing.
The source archive tree exactly matches PR #88. In normal PR CI 35942041388,
all three platforms passed ordinary tests, statistical and extended-risk gates;
both supported native wheel/replay jobs and the dividend risk/refinement jobs
passed. The Linux extended-price artifact reports 26 passed / 1 failed: the
inherited `two_factor_constituent_refinement_consistency` has 4.0124 bp ensemble
SE against the still-present 4 bp limit. PR #85's separate 5 bp requirement change
is not incorporated here. This is parent evidence, not validation of this child.
