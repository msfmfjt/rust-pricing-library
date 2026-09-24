# Bergomi parameter risk with stochastic cash dividends

## Scope and base

Child of PR #87, source `30ce1f2cae0974b075e5099415735b7a927df797`, tree
`24367b169470d25a4574bf5ff9492d0ca7ddbcf2`. The source artifact from native CI
35940620182 was imported and its full Git tree matched this base exactly.
No parent PR is modified or merged by this change.

See [ADR 0016](../adr/0016-stochastic-dividend-bergomi-risk.md) and the
[model reference](../../docs/models/stochastic-dividends.md#bergomi-parameter-risk).

## Prespecified tests and budgets

- Two internal tests: analytic covariance-factor and decay derivatives versus
  separately compiled OU factors, including zero/equal/tiny/large mean reversions
  and nonuniform steps; max absolute difference 3e-6. Independently integrated
  log-phi derivative has absolute tolerance 2e-12 across its series boundary.
- Four public Rust tests: 1F/2F, MC/RQMC, two seeds, terminal calls, Asian delayed
  payments, smoothed pre/post-dividend barriers, and exact one/three-worker replay.
  All extra derivatives must pass *both* h=1e-5 and h=1e-6 full-recompile central
  price differences under `3e-5 + 2e-5*max(abs(AAD),abs(FD))`. SE does not widen
  this derivative budget. Base price, price SE, labels, derivatives and risk SE
  must equal the old scope exactly, including curve and cash convenience accessors.
- Boundary tests: inward FD h=1e-7 at k=0, nu=0 and theta=0/1, absolute tolerance
  3e-4. At sigma0=0 all appended risks and their SEs must be exactly zero.
- Preflight rejection: singular/near-singular integrated correlation or a BS
  plan rejects the extended method without disabling price or basic risk.
- Three Python tests: full-recompile model parameter checks at two bump sizes,
  exact prefix/worker replay, immutable copy semantics and explicit failures.
  The runnable example exercises both the base scope and appended labels.

All budgets above precede native observation. Existing thresholds, seeds and
frozen replay fixtures are unchanged. The new Rust target runs in normal cargo
test discovery; focused native validation also executes it in release mode.

## Observations

Native validation has not yet been observed for this candidate. The local
container has no Rust toolchain or external network access; no local Rust pass
is claimed. Recorded outcomes will identify the exact source tree and test scope.

## Limitations

Correlations are held fixed. Numerical differentiability requires the explicit
positive-pivot domain in ADR 0016; singular model-parameter derivatives are not
claimed. No absolute continuous-time model-risk accuracy, wide stress/exotic
acceptance, recalibration, performance improvement or full-platform release
acceptance follows from a finite-difference match. The parent price/refinement
and extended legacy acceptance gates remain separate.
