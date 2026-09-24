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
- Seven public Rust tests: all basic market/dividend/curve risks at H=0.1/0.49/0.5,
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
