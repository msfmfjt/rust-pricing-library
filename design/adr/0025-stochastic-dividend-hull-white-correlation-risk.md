# 0025: Hull–White correlation AAD for stochastic cash dividends

Status: proposed. Parent: [0024](0024-stochastic-dividend-hull-white-parameter-risk.md).

## Decision and scope

Add `evaluate_correlation_aad()` to the dedicated BS/Buehler/Hull–White plan in
Rust and Python. Preserve the complete `evaluate_hull_white_aad()` price/risk/SE
prefix and append raw `equity_dividend_correlation`, `equity_rate_correlation`
and `dividend_rate_correlation` partials, in that order. Each direction varies
one symmetric pair, with all other correlations, model parameters, Q cash means,
curves, dates, grid and smoothing width fixed. No market recalibration occurs.

## Derivative

For each fixed-grid transition, use unit-correlation covariance kernels to
differentiate the same unpivoted lower-triangular Cholesky factor used in
pricing. Never divide a covariance entry by its correlation: zero correlations
must work. Propagate exact rate and integrated-rate state tangents. The equity
factor keeps the first independent normal and has zero correlation tangent;
the Buehler dividend split differentiates
`rho_FD*z0 + sqrt((1-rho_FD)*(1+rho_FD))*z1`.

Differentiate the Gaussian-tilt A/B/C cash coefficients analytically under the
existing independently converged quadrature. Their rho_FD partial is zero;
rho_Fr and rho_Dr affect the discounted conditional expectation. Include every
future cash claim, even beyond option expiry, when differentiating the initial
funded risky spot. Propagate the rate-state, dividend-factor and coefficient
tangents into both post-cash and pre-cash physical stock. The pre-cash tangent
additionally contains the realized cash jump derivative. Contractual payoff
adjoints contract these tangents, and the relative payment discount contributes
`-dI_T - B_a(U-T)*dx_T`.

## Domain and compatibility

The instantaneous driver correlation must have Cholesky variance pivots above
`1e-10`. The simulated covariance must also have all Cholesky diagonals divided
by marginal standard deviation above `1e-10`. Both checks are necessary: an
integrated covariance can be full rank at an instantaneous PSD boundary when
the volatility kernel varies within a step. Reject unsupported inputs before
sampling; do not project correlations, silently hold terms fixed, or widen the
domain of earlier parameter derivatives. Pricing and basic AAD retain their
existing singular-correlation and deterministic-rate support.

The additive API uses the existing immutable result type. No constructor,
schema, pricing arithmetic, random coordinate, reduction order, fingerprint,
legacy acceptance budget or seed changes. This method supplies finite-grid
first-order correlation derivatives, not Gamma or a risk-convergence claim.

See the [validation protocol](../validation/stochastic-dividend-hull-white-correlation-risk.md)
and [model reference](../../docs/models/stochastic-dividends-hull-white.md).
