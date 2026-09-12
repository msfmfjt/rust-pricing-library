# One-currency Hull–White equity extension

Date: 2026-09-12. Decision: implement an experimental price-only extension.
Authorization: the user requested stochastic rates with Hull–White after the
LSV change and approved the proposed first implementation.

## Context and decision

The accepted pricing baseline and the first Bergomi LSV extension use
deterministic interest rates. Adding a random short rate solely to discounting
would miss equity/rate correlation, integrated-rate drift, payment-lag effects
and the stochastic-rate term in smile calibration.

Add a one-factor Hull–White rate kernel with constant nonnegative mean
reversion and piecewise constant nonnegative rate volatility. Parameters are
external inputs; fitting the initial discount curve is automatic, while fitting
rate volatility to caps or swaptions is outside this change. Combine this kernel
with Black–Scholes and the existing one-factor Bergomi LSV factor. Calibrate LSV
using discounted particle weights and an explicit stochastic-rate correction.

Use a separate experimental Rust/Python compile boundary. Permit continuous
deterministic carry and proportional dividends. Reject fixed-cash dividends,
nonpositive simulation horizons, hybrid Greek requests and reverse-trace requests
until their stochastic-rate contracts are implemented. Preserve existing APIs.

## Impact analysis

| Area | Impact |
| --- | --- |
| Requirements | Adds stochastic rates beyond the deterministic-rate baseline; does not change the baseline's acceptance status |
| Public API | Adds `HullWhite1Factor`, hybrid simulation/calibration types, `HullWhiteEquityPricingPlan` and four immutable Python classes |
| Serialized data | Stable JSON continues to describe the BS model or LV target; HW parameters, paired density and compiled state are explicit arguments and are not serialized by the request |
| Numerical behavior | Exact joint OU/rate-integral innovations; positive equity log-Euler; discounted conditional moments, rate correction and support fallback are versioned separately |
| Reproducibility | Four Gaussian blocks, separate calibration random domain, fixed-block reductions and a fingerprint covering hybrid inputs; existing replay fixtures remain unchanged |
| Tests | Independent covariance quadrature, curve fit, bond parity, BS+HW analytic prices, martingales, zero-factor limits, calibrated vanilla repricing, dividend order, errors and Python wheel API |
| Gates | H0–H3 in the [roadmap](../hull-white-roadmap-v0.1.md); broad refinement, hybrid AAD, stable wire support and benchmark acceptance remain open |

This decision does not claim production acceptance of the hybrid calibration or
extend the existing LSV calibration VJP to stochastic rates. See the
[numerical contracts](../hull-white-numerical-contracts-v0.1.md) for the model,
estimator and finite-sample limitations.
