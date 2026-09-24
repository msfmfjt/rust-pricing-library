# Stochastic-dividend Hull–White parameter AAD validation

Scope: constant residual volatility, Buehler cash dividends and stochastic
Hull–White rates. This protocol covers the added rate mean-reversion and
piecewise rate-volatility tangents at fixed correlations, curves, grid and
contract inputs. Results must be tied to this candidate source; parent results
are not child passes.

## Prespecified checks

Three private Rust tests:

- Differentiate the piecewise rate Brownian integral with respect to mean
  reversion and every volatility knot; compare central h=1e-6 parameter bumps
  within 2e-10 absolute.
- Compare all 16 joint transition-covariance entries and all conditional cash
  A/B/C rate tangents with central full-model bumps at h=1e-6. Budgets are 2e-9
  for covariance and 3e-8 for the conditional coefficients.
- Check the mean-reversion derivatives of B and its integrated loading at the
  Ho–Lee boundary against inward h=1e-6 values.

One new public Rust integration test:

- For a delayed-payment Asian payoff, compare the appended derivative for mean
  reversion and each volatility knot against full recompilation under identical
  RQMC coordinates. The rate grid includes a volatility knot after expiry but
  before a cash maturity. Use central h=1e-5 for mean reversion and 1e-6 for
  rate-volatility nodes; budget `2e-5 + 2e-5*max(abs(AAD),abs(FD))`.
- Verify price/price-SE identity with `evaluate()`, label order, and exact
  derivative/SE replay across one and three workers. Basic AAD must still work
  with deterministic zero rates while the new HW parameter method rejects its
  singular simulated covariance before sampling.

One new Python test checks the public method, result identity, method label, rate
label order and finite derivatives on a piecewise-rate plan. The prior four HW
basic-AAD tests remain in place.

## Regression and interpretation

Run the HW price and basic-risk tests, focused Rust tests in release mode, all
Python tests, formatting and Clippy. Check type-stub shape and the complete source
archive include the new API/module. Keep the inherited 4 bp legacy accuracy
threshold untouched; PR #85 is a separate change. Focused passes do not certify
full platform acceptance or continuous-time risk convergence.

Sampling SE covers MC independent units or RQMC scramble means only. It excludes
quadrature, grid, smoothing, calibration and model error. All bump comparisons
reuse random coordinates and recompile the full model; no sampling-SE widening is
applied to derivative budgets.

See the [decision](../adr/0024-stochastic-dividend-hull-white-parameter-risk.md),
[basic-risk protocol](stochastic-dividend-hull-white-risk.md) and
[model reference](../../docs/models/stochastic-dividends-hull-white.md).
