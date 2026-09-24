# ADR 0016: optional Bergomi parameter reverse with stochastic dividends

Status: implemented candidate; native validation pending.

## Decision and compatibility

Add `StochasticDividendPricingPlan::evaluate_bergomi_aad` and the same method on
Python `StochasticDividendPlan`. Return the existing immutable risk result with
an unchanged basic-risk prefix, then append `bergomi_mean_reversion[0]`, the
second mean reversion for 2F, `bergomi_vol_of_vol`, and `bergomi_mixing_weight`
for 2F. Model inputs are raw, factor ordering is retained, and the new method
has its own risk-method label. Original `evaluate_aad` labels, numerical results
and behavior are unchanged. No request Greek flags, wire schemas, dependency,
crate, price formula, sampler coordinate, plan fingerprint or tolerance changes.
The same curve accessor ranges apply with the longer derivative vector.

The extra scope is opt-in: the basic risk method remains available at a singular
joint covariance. This avoids silently dropping new derivatives or turning a
valid price/basic-risk request into an unsupported model-parameter request.
All correlations, dates, grid topology and payoff smoothing width remain fixed.
No HW, rough, LSV recalibration, multi-asset, Gamma or market-IV VegaKT is added.

## Derivative specification

The precompiled coefficient Jacobians differentiate the normalized finite-step
OU correlations, the same unpivoted Cholesky factor, marginal OU standard
deviations and OU decays. No price bump or finite-difference coefficient is used.
For `phi(x)=int_0^1 exp(-xu)du`, differentiate its log with the analytic expression
`1/expm1(x)-1/x`, evaluated by a Taylor series near zero and exp(-x) at large x.
This gives finite derivatives at k=0 and avoids positive-exponential overflow.
Differentiate Cholesky recursively on its positive-pivot branch; see Murray
(2016), *Differentiation of the Cholesky decomposition*,
[arXiv:1602.07527](https://arxiv.org/abs/1602.07527).

Per-path payoff and Buehler split adjoints provide sensitivities to the equity
volatility loading at each left endpoint. Reverse the recorded OU recurrence
and accumulate derivatives through the prepared Jacobians. Include normalized
2F weight derivatives and derivatives of the deterministic lognormal centering
`nu*Var[Z_t]`. In particular the vol-of-vol derivative of
`nu*(Z_t-nu*Var[Z_t])` is `Z_t-2*nu*Var[Z_t]`; it does not divide by nu.
Both k=0 and nu=0 use inward derivatives, as do theta=0 or theta=1.

Every integrated normalized correlation pivot, at each step and at the positive
left-endpoint horizons used for centering, must be greater than `1e-10`. The
normalized 2F weight variance must also exceed `1e-10`. Reject the extended risk
before path generation otherwise; do not regularize a matrix, change its rank,
clip the derivative, or report zeros as a substitute. This is a conservative
numerical differentiability domain, not a change to the pricing PSD policy.

## Validation and interpretation

The [validation record](../validation/stochastic-dividend-bergomi-risk.md) covers
coefficient derivatives, full-recompile price finite differences at two bump
sizes, boundary derivatives, fixed-cash/curve risk compatibility, exact worker
replay, and singular-domain rejection. These are derivatives of the finite
simulation algorithm and do not certify continuous-time risk convergence or
calibration quality. Sampling SE retains the existing MC/antithetic and RQMC
scramble interpretation; it excludes parameter and discretization uncertainty.
