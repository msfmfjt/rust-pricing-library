# 0019: Rough Bergomi with stochastic discrete cash dividends

## Decision

Extend the deterministic-rate Buehler price plans from
[0013](0013-stochastic-cash-dividends.md) through
[0018](0018-stochastic-dividend-gamma.md) with `compile_rough_bergomi`.
Use the existing `RoughBergomi` domain object and its nonuniform-grid Volterra
hybrid weights. Expose prices only in this slice. All inherited AAD and Gamma
methods explicitly reject this new model before constructing an executor.
Older BS/1F/2F pricing, risk, random coordinates and identities stay unchanged.

The rough input eta is log-variance vol-of-vol:
`v_i=sigma0^2 exp(eta X_i - eta^2 V_i/2)`. It is not the 1F/2F log-volatility nu;
at H=1/2 the boundary is zero-mean-reversion 1F Bergomi with `nu=eta/2`.
Use `v_i` at the left endpoint in the existing positive dividend split.
Full future cash means, including post-expiry dates, remain carry-funded; repo
spread is not reinterpreted as uncertain cash and paid-cash dynamics are not added.

## Joint law and causality

Validate the complete instantaneous correlation of `(W_f,W_D,W_v)` with the
existing PSD-aware unpivoted Cholesky tolerances, even for zero loadings. The
first two independent normals continue to drive f and D as in the existing split.
Use the third Cholesky row for the vol-driver Brownian increment. An independent
fourth normal supplies the residual of the newest singular-kernel cell after
conditioning on the full-cell Brownian increment. Its residual standard deviation
is `dt^H (1/2-H)/(H+1/2)`, avoiding cancellation at H=1/2.

Older cells use the exact average of the power kernel over each cell. Center with
the variance of this finite-grid driver, not continuous `t^(2H)`. Update the rough
history only after evolving the current f/Y step. No same-step future volatility
is used. Maintain four coordinates at H=1/2, eta=0 and singular correlations.
Shared rank-major Brownian bridging occurs before model correlation.

## Resource and compatibility boundaries

Dense history costs O(N^2) time per path and O(N^2) compiled storage; no FFT,
Markovian approximation or caching optimization is introduced. Reject more than
4096 time steps before allocating the triangular history (~64 MiB of weights at
the cap). This is an explicit domain restriction of this new price factory only.
The grid, H, eta, f/vol and D/vol correlations, and the Buehler/base plan inputs are
included in a new scheme/fingerprint domain. There are no request-schema,
dependency, older-method, seed, replay-fixture or tolerance changes.

HW, LSV recalibration, multiple assets, rough-parameter/correlation AAD, basic
rough AAD and Gamma are not in this slice. Existing risk methods must not treat
a rough plan as a constant-volatility plan. Finite-grid covariance and price
checks do not establish continuous-time option-price accuracy or convergence.
See the [validation protocol](../validation/rough-stochastic-dividends.md).
