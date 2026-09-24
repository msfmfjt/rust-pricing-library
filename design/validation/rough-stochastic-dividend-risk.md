# Rough stochastic-dividend risk: validation protocol

This protocol is specified before native execution. Parent PR91 supplies the
unchanged rough/Buehler primal, price references and random coordinate contract.
Do not relabel parent evidence as this change's native acceptance.

## Checks

- Two private Rust tests: H-dependent average weights versus independent
  Simpson integration of the derivative of the power kernel (2e-10 absolute);
  transposed causal-history H/eta gradients versus recompiled loading values
  (3e-6 absolute), across H=0.01/0.1/0.49/0.5 and eta=0/0.6, on an irregular
  grid. Last-step normals cannot affect any left-endpoint loading or its risk.
- Eight public Rust tests (including the explicit hinge-stencil diagnostic below): all basic market/dividend/curve risks at H=0.1/0.49/0.5,
  MC/RQMC, multiple seeds and one/three workers; European, Asian delayed payment
  and smoothed pre/post-dividend barriers. Basic and extended results retain
  exact price/price-SE and basic derivative/SE prefixes.
- Full-recompile H/eta price FDs at h=1e-5 and h=1e-6 must satisfy
  `abs(AAD-FD) <= 3e-5 + 2e-5*max(abs(AAD),abs(FD))`. The same requirement applies
  to basic risks. Sampling SE never widens this deterministic comparison.
- Inward boundary differences at h=1e-7: 3e-4 absolute. Cover sigma0=0, dividend
  parameters=0, H=1/2, eta=0. H/eta risks are exactly zero at sigma0=0; H risk is
  exactly zero at eta=0. Unsupported scopes/families explicitly reject.
- Three Gamma ladder estimates versus full-recompiled AAD Delta bumps, for
  rough H=0.1 and H=0.5, MC/RQMC: 2e-12 absolute. Baseline price/Delta are exact,
  as are numerical worker outputs including paired bump-gap standard errors.
- Three Python tests exercise H/eta FDs, exact prefix/accessors, immutable copy
  semantics, worker replay, inward boundaries, zero sigma, singular PSD inputs,
  wrong risk scope, the Gamma ladder and funding errors.
- Update prior rough support tests to require basic/H-eta AAD and Gamma success
  while still rejecting correlation/Bergomi-only scopes. Preserve their price
  references, seeds and numerical tolerances. Run all prior stochastic-dividend
  tests, normal Rust/Python discovery, wheel/stub/source contracts and Clippy.

## Interpretation

No finite-grid gap or finite-difference test is a continuous-time accuracy oracle.
H=1/2 is differentiated from the supported side of the hybrid algorithm, not
by replacing it with a Markovian k=0 model before differentiating H. In
particular the near-cell residual has zero value but nonzero H derivative.
All correlations, dates, grid, payoff smoothing and other model inputs are
fixed. Generic Gamma paired-SE tests remain inherited tests, not an independent
continuous-time rough-Gamma oracle.

Native results, exact source tree, observed failures and any outstanding gates
must be recorded with the pull request. Full legacy acceptance, including the
separate existing 4 bp requirement change, is not waived by focused success.

## Finite-stencil correction recorded after the first native observation

Native run 35985436056 passed both new coefficient tests and 6/7 new public tests.
The all-basic-risk comparison stopped at H=0.1, MC seed 912, last discount pillar,
h=1e-5: AAD -34.18298105655497 versus raw central FD -34.21095437188271.
An independent Python reconstruction located antithetic path 321 at terminal
Spot 99.99970564881927: the negative curve bump moves it above strike 100.
At h=1e-6 no path crosses and raw FD agrees. This is a finite payoff-stencil
hinge contribution, not evidence of a reverse/curve formula error.

Retain the original seeds, markets, both bump sizes, all derivative budgets and
all production arithmetic. For the unsmoothed terminal-call basic-risk panel,
reconstruct the base/up/down discounted intrinsic values with public *primal*
paths and an elementary payoff, independently of AAD/payoff-reverse helpers.
Require each reconstructed bumped call price to agree with the full public
pricing call within 2e-12. Subtract the exactly computed finite hinge remainder
from the raw FD before applying the original derivative budget:

`R_h = mean[(x_plus^+ - I0*x_plus - x_minus^+ + I0*x_minus)/(2h)]`,
where `I0 = 1{x_base > 0}`. The remainder is identically zero when no branch
changes. It is never inferred from the AAD-FD error. Asian/smoothed-barrier and
H/eta tests retain their original direct FD comparisons.

An eighth public test retains the originally failing raw case explicitly:
one crossing at 1e-5, zero at 1e-6, a nonzero coarse hinge contribution, and
corrected FD within 3e-5 of AAD at both widths. Neither the raw coarse FD nor
its mismatch is hidden or labelled a direct-FD pass. This amends the initial
validation interpretation; it is not a relaxation of numerical tolerances.
