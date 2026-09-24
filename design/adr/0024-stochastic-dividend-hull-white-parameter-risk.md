# 0024: Hull–White parameter AAD for stochastic cash dividends

Status: proposed. Parent: [0023](0023-stochastic-dividend-hull-white-risk.md).

## Decision and scope

Extend the dedicated stochastic-dividend Hull–White plan with
`evaluate_hull_white_aad()` in Rust and Python. Keep `evaluate_aad()` and its
existing result unchanged. Append `rate_mean_reversion`, followed by
`rate_volatility[i]` for each piecewise-constant volatility input knot, to the
existing basic risk labels.

Hold correlations, curves, contract inputs, dates, simulation grid and smoothing
width fixed. Differentiate each rate parameter through the joint covariance
Cholesky and exact rate/integrated-rate path, conditional cash claim coefficients
and bonds, initial funded reserve, pathwise payoff, and delayed payment discount.
Do not recalibrate the dividend means, discount curve or volatility surface.

## Derivative method and domain

Use forward tangents at fixed normal coordinates. Differentiate each covariance
kernel with respect to mean reversion or its active rate-volatility knot, then
solve the lower-triangular Cholesky directional equation. Propagate rate-state
tangents step by step. Differentiate the cash-claim Gaussian tilt using analytic
rate-kernel derivatives and independently converged Simpson integrals. Bond
convexity, duration, every claim in the time-zero reserve, node half-variance,
and payment-date adjustment all remain active in the derivative.

The parameter method requires every simulated covariance Cholesky pivot,
normalized to correlation scale, to exceed `1e-10`. Singular or ill-conditioned
paths return a domain error before sampling. This restriction applies only to
`evaluate_hull_white_aad()`; basic AAD and pricing retain their existing domain,
including fixed singular covariance matrices. Mean reversion zero uses stable
Ho–Lee limits. Rate-volatility knots after expiry remain in the result because
they can change conditional cash claims even when they do not enter simulated
rate increments.

## Compatibility and verification

No price arithmetic, random coordinate, fingerprint, constructor, schema,
correlation treatment or legacy numerical threshold changes. The new risk method
returns the same immutable risk object with appended labels and sampling errors.
Its sampling SE does not include quadrature or time-grid error.

See the [validation protocol](../validation/stochastic-dividend-hull-white-parameter-risk.md)
and [model reference](../../docs/models/stochastic-dividends-hull-white.md).
