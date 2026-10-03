# Rough Heston: Fourier Hurst sensitivity

## Scope and public interface

`HestonFourierPlan.hurst_risk_plan()` constructs a reusable, experimental
`HestonFourierHurstRiskPlan`. `price(forward, strike, discount)` returns
`HestonFourierHurstRisk` with the unchanged nested `price`, `hurst_sensitivity`,
`quadrature_difference`, and `tail_indicator`. Rust and frozen Python classes
have the same semantics. An explicit `log_transform_derivative(z)` query performs
a new solve; Python takes `real, imag` and returns the two complex components.

The derivative holds Forward, Strike, Discount, Maturity, v0, kappa, theta and nu,
rho fixed. It is per one absolute H unit. Times remain in years; rescaling the time
unit without reinterpreting the dimensional model parameters is a different risk.
Call and Put sensitivities coincide at fixed parity inputs. This is not implied
volatility Vega, VegaKT, recalibration risk, a parameter Hessian, or reverse-mode AAD.

Only the power-kernel Rough Heston model is accepted. An explicit finite lift does
not encode a unique H-to-weight/rate mapping; even a lift generated from rough
parameters is rejected rather than assigned a misleading Hurst derivative. The
[five fixed-kernel scalar sensitivities](heston-parameter-risk.md) remain unchanged.

## Calculation specifications

Write alpha = H + 1/2 and K(t) = t^(alpha-1)/Gamma(alpha). For t>0,

```text
partial_H K(t) = K(t) [log(t) - digamma(alpha)].
psi = K * R(z,psi)
R(z,psi) = (z^2-z)/2 + (rho*nu*z-kappa)*psi + nu^2*psi^2/2
partial_H psi = partial_H K * R + K * (R_psi * partial_H psi).
```

The implemented tangent differentiates the mathematical product-integration
weights of the existing implicit Riccati discretization, not a finite difference
of two prices. With `psi_n = past_n + e*R_n`,

```text
partial_H psi_n = [partial_H past_n + (partial_H e)*R_n] / [1-e*R_psi,n].
```

Both the initial hat weight and all history weights are differentiated, together
with the endpoint weight. For the latter,

```text
e = dt^alpha / Gamma(alpha+2)
partial_H e = e [log(dt) - digamma(alpha) - 1/alpha - 1/(alpha+1)].
```

For l>=2, binomial tails at 1/l evaluate derivatives without subtracting large,
near-equal powers. A derivative recurrence that never divides by alpha+1-k is
used, including at H=1/2. Sixty-four terms are retained (|1/l|<=1/2). This is a
floating-point implementation of analytic weight derivatives, not exact real
arithmetic. Existing primal weight expressions and operation order are unchanged.

The exponent derivative is

```text
partial_H log M = v0 * integral(R_psi * partial_H psi)
               + kappa*theta * integral(partial_H psi).
```

The Black control variance v0*T is H-independent. Therefore the derivative of the
cached half-moment correction is `-M*partial_H log M / (u^2+1/4)`. The same Simpson
nodes and compensated sums give price sensitivity without a new Riccati solve per
strike. The old price and five-direction parameter solves instantiate no Hurst
weight or tangent work. The new route guards work before allocating transform
nodes and rejects nonfinite or ill-conditioned tangents.

## Boundaries and diagnostics

At T=0 and for an identically zero-variance process the Hurst derivative is zero,
including ATM: the parameter does not move the intrinsic payoff. At H=1/2 the
reported value is the limit from H<1/2; it is not assumed to vanish. The model's
existing 0<H<=1/2 domain still applies. Invalid price inputs and exponents outside
the parent transform's moment strip are rejected.

Unlike v0/Forward sensitivities, this derivative does not divide by the Black
control's square-root variance. Zero initial variance with positive subsequent
variance is supported by this Hurst route, subject to the existing transform/price
checks. This does not extend the other Greek APIs' zero-control support. A positive
maturity whose per-step dt underflows to zero is explicitly rejected.

`quadrature_difference` is the absolute Simpson N/N2 difference on the retained
interval; `tail_indicator` integrates the derivative envelope only over
[cutoff/2, cutoff]. Neither is an omitted-tail bound, time-discretization estimate,
total error bound, or sampling standard error. Small diagnostics do not establish
accuracy, and the price/transform moment checks are only necessary shape checks.

## References and validation

El Euch and Rosenbaum, [The characteristic function of rough Heston models](https://arxiv.org/abs/1609.02108)
provides the affine fractional-Riccati structure. The Hurst tangent above is
derived by differentiating that structure with the library's normalization.
Digamma uses [DLMF 5.5.2](https://dlmf.nist.gov/5.5.E2) and
[DLMF 5.11.2](https://dlmf.nist.gov/5.11.E2). The normalized kernel and fixed time-unit
convention are part of this API's definition.

See the [validation protocol](../../design/validation/heston-hurst-risk.md) and
[executable example](../../examples/python/heston_hurst_risk.py). Independent
nondegenerate references cover selected complex transforms, not a complete rough
option-risk surface. A separate [price calibration API](heston-calibration.md)
uses these tangents. Finite-lift kernel risk, broader stressed/short
maturity/wing cases, total-error control and production admission remain separate.
