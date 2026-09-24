# Rough stochastic-dividend correlation risk

## Decision

Extend the existing `evaluate_correlation_aad()` to the deterministic-rate,
single-asset rough Bergomi/Buehler plan. Retain the full `evaluate_rough_aad()`
label/derivative/sampling-SE prefix and append, in order:

- `equity_dividend_correlation`;
- `spot_volatility_correlation[0]`;
- `dividend_volatility_correlation[0]`.

These are raw symmetric off-diagonal Brownian-correlation partials: vary both
entries of one pair together, holding the others fixed. They do not represent
physical-stock return correlations, a PSD-projected bump or recalibrated market
risk. The existing method label `buehler-joint-correlation-reverse-v1` is retained;
the price scheme, plan fingerprint and appended labels identify the family and
inputs. No new Python method, constructor, return class or wire field is needed.

## Numerical specification

Require every instantaneous 3x3 Cholesky pivot to exceed `1e-10`, including at
zero sigma0/eta/dividend volatility. Differentiate the actual unpivoted factor
in the open SPD interior; do not regularize/project a singular matrix. This
supersedes ADR 0020's rough-correlation rejection only. Other methods retain
their domains, including fixed singular PSD input for price/basic/H-eta/Gamma.

The finite hybrid driver is linear in the volatility Brownian increments:

`X_i = A_(i-1) dWv_(i-1) + B_(i-1) z_(i-1,3) + sum_(j<i-1) w_ij dWv_j`.

H, eta and the grid are fixed for these correlation partials. Prepare the nine
last-row Cholesky derivatives once. Transpose the volatility-loading adjoints
into past increments (including the newest cell), then through the factor row
into the three correlations. Add the existing direct dividend-driver rotation
term to the equity/dividend partial. The fourth independent residual normal
has no correlation derivative. The volatility-driver marginal remains standard
Brownian, so the finite-grid centering variance has zero correlation derivative.
Do not reuse the 2F Bergomi centering formula, which has a different dependence.

At H=1/2 the newest-cell residual vanishes, but the underlying 3x3 Brownian
matrix can remain SPD. Do not reject solely because the augmented four-variable
newest-cell covariance is singular. Keep all four random coordinates. H/eta
prefix derivatives keep ADR 0020's boundary conventions unchanged.

## Compatibility and cost

No changes to primal arithmetic, scheme/fingerprints, RNG order, request schema,
dependencies, existing methods' results, CI workflows, seed fixtures or prior
numerical budgets. Only previous rejection tests for now-supported interior
rough correlation calls change. The 1F/2F-only method still rejects rough.

Correlation risk adds nine prepared scalar derivatives and O(N) path workspace;
the history transpose costs O(N^2). The combined result also retains the H-risk
Jacobian table introduced by ADR 0020. Existing Arc sharing, dense-history
resource limit (4096 steps) and asymptotic costs are unchanged. No FFT or
Markovian surrogate is introduced.

These are finite-algorithm derivatives. No continuous-time risk convergence,
HW/stochastic rates, LSV/market-IV VegaKT, multiple assets, new calibration,
cross-Gamma or other mixed second derivatives are added. Sampling SE excludes
time-grid/smoothing/model/calibration uncertainty and need not be small near the
SPD boundary. A finite scenario bump must separately remain admissible.

See the [validation protocol](../validation/rough-stochastic-dividend-correlation-risk.md)
and [model reference](../../docs/models/stochastic-dividends.md).
