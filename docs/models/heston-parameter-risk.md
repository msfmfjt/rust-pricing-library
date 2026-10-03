# Heston Fourier: fixed-kernel parameter risk

## Contract

The experimental [Fourier plan](rough-heston-fourier.md) has a separate
`parameter_risk_plan()` method. It caches first derivatives in the order
`initial_variance`, `mean_reversion`, `long_run_variance`, `vol_of_vol`,
`correlation`: v0, kappa, theta, nu, rho. `risk_plan.price(F,K,D)` returns
`HestonFourierParameterRisk` with the unchanged price result and named
`HestonParameterSensitivities` objects for sensitivities and two diagnostics.
Call and put have identical parameter derivatives at fixed F, K, D, T.

Each derivative is per **one absolute unit** of its named parameter, not per
percent, volatility point or basis point. The v0 derivative is a variance
sensitivity, not volatility Vega, implied-volatility Vega, or VegaKT.
No calibration is performed. Hurst is fixed for Rough Heston. Every weight and
rate is fixed for Lifted Heston, including a lift created from a rough-model
factory. There is no derivative through the kernel factory, factor count,
Hurst, or discretization configuration. No derivative of physical Spot,
dividend mappings, stochastic rates, LSV or a recalibrated surface is implied.

```python
plan = rp.HestonFourierPlan.compile(model, 1.0)
risk = plan.parameter_risk_plan()   # differentiate once per maturity/configuration
result = risk.price(100.0, 100.0, 0.97)
print(result.price.call, result.sensitivities.initial_variance)
```

Python plan, result and sensitivity objects are frozen. The additional explicit
`risk.log_transform_derivatives(real, imag)` query performs a new tangent solve
and returns five complex pairs in the order above; it is not a cache lookup.

## Tangent calculation

The underlying affine models are those of
[El Euch and Rosenbaum](https://arxiv.org/abs/1609.02108) and
[Abi Jaber](https://arxiv.org/abs/1810.04868), with the parameter mapping described
in the parent specification. The following differentiation of the implemented
numerical scheme is a library implementation derivation, not a claim to copy
an algorithm from those papers.

At each grid point, the parent solver satisfies

```text
y = past + e R(z,y),
R(z,y) = (z*z-z)/2 + (rho*nu*z-kappa)y + nu^2 y^2/2.
R_y = rho*nu*z-kappa + nu^2 y.
R_p = [0, -y, 0, rho*z*y + nu*y^2, nu*z*y].
y_p = (past_p + e R_p) / (1-e R_y).
dR/dp = R_p + R_y y_p.
```

Power-kernel convolution history or finite exponential factors propagate these
five directions. The derivative of the affine exponent also differentiates
**both prefactors** in `v0 integral R + kappa*theta integral psi`. Treating this
as `v0 psi(T)` would be incorrect for a fractional kernel. No parameter bumps or
Monte Carlo paths are used in production. This is forward tangent propagation,
not a reverse-mode AAD implementation.

The shared primal solver retains its original arithmetic order; const-generic
price-only evaluation omits tangent allocation and arithmetic. The original
price plan is cloned into the risk plan and supplies the returned prices.
Riccati residuals and tangent residuals are checked; a nearly singular implicit
Jacobian fails explicitly rather than returning an unbounded derivative.

For the Fourier correction, differentiate both the model transform and the
v0-dependent Black control. The derivative of the Black price with respect to
v0 is `D F phi(d1) T/(2 sqrt(v0 T))`. The correction uses cached
`(dM_control/dp - M_model dlog(M_model)/dp)/(u^2+1/4)` on the parent's Simpson
nodes. Changing strike after compilation needs no additional Riccati solve.
Work is bounded before evaluating the parameter grid, with a six-direction
primal-plus-tangent work allowance.

## Boundaries and diagnostics

At positive maturity strictly positive initial/control variance is required.
A zero initial variance is rejected even if immigration makes the terminal law
nondegenerate: the current control derivative is singular. Underflow of positive
control variance to zero is also rejected. At **zero maturity all parameter
sensitivities are zero, including ATM**: intrinsic price is independent of these
parameters. This differs from the undefined Forward Delta at an ATM kink.

Zero kappa, theta or nu and correlation endpoints are permitted. The formulas
provide derivatives of the smooth algebraic continuation; at a constrained
parameter boundary their interpretation is the **inward one-sided derivative**.
No two-sided stochastic model outside the admissible parameter domain is claimed.
Nonfinite values and singular tangent Jacobians are errors, never clipped.

`quadrature_differences` compare Simpson N and N/2 at the same cutoff.
`tail_indicators` integrate each derivative correction envelope over the retained
interval [cutoff/2, cutoff]. These are neither omitted-infinite-tail bounds,
Riccati-error estimates, total-error estimates, sampling standard errors nor
calibration uncertainty. Price convergence alone does not certify risk accuracy.

See the [validation protocol](../../design/validation/heston-parameter-risk.md)
and [executable example](../../examples/python/heston_parameter_risk.py).

## Related Hurst risk

[Power-kernel Hurst sensitivity](heston-hurst-risk.md) is available separately
through `hurst_risk_plan()`. It differentiates the Rough Heston kernel, holding
all five scalar parameters fixed; finite lifts are explicitly rejected.
