# Rough-family physical Spot Gamma: explicit finite bumps

Experimental Rust/Python APIs supplement [Spot Delta](rough-family-lsv-spot-delta.md)
and [market-IV adjoints](rough-family-market-iv.md). These are **finite-bump
estimators, not second-order AAD or a certified derivative of the continuous model**.

## Entry points and conventions

- `RoughVolatilityPlan.evaluate_gamma_bump(spot_bump)` fixes the pure variance
  model, curves, full cash/proportional dividend schedule, and contractual payoff.
- `RoughFamilyLsvPlan.evaluate_frozen_leverage_gamma_bump(spot_bump)` also freezes
  calibrated leverage values, axes and reference funded-forward anchor.
- `RoughFamilyLsvPlan.evaluate_sticky_moneyness_gamma_bump(spot_bump)` holds
  relative Local Variance values/log-moneyness axes fixed and moves the anchor.
  This is **not fixed-absolute-strike market-IV recalibration**. Discrete scale
  equivariance avoids extra particle calibrations, exactly as for parent Delta.

Six pure families are accepted, including beta 0/intermediate/1 Rough SABR.
LSV remains beta=1 only for SABR, with the parent's fixed-driver QRH convention.
All requests remain price-only carriers; no stable JSON Gamma routing is added.

The result is `RoughVolatilityGamma` or `RoughFamilyLsvGamma`, respectively.
In Rust both are aliases of `RoughGammaBump<P>`. Python exposes frozen classes.

## Calculation and units

For the same Gaussian coordinates Z and physical Spot S, calculate payoffs at
S, S+h, S-h, S+h/2 and S-h/2. The caller supplies positive **absolute currency
bump** h; no automatic width selection is performed. For each sampled path,

```text
G_h(Z)   = ((P(S+h,Z)-P(S,Z)) + (P(S-h,Z)-P(S,Z))) / h^2
G_h2(Z)  = ((P(S+h/2,Z)-P(S,Z)) + (P(S-h/2,Z)-P(S,Z))) / (h/2)^2
D(Z)     = G_h2(Z) - G_h(Z).
```

`gamma` estimates E[G_h], `half_bump_gamma` estimates E[G_h2], and
`bump_difference` estimates E[D]. **No Richardson extrapolation, clamping to
positive Gamma, bias correction or automatic acceptance flag is applied.**
This retains payoff-strike crossings that are lost by naively differentiating
an unsmoothed vanilla payoff twice on each path.

The three corresponding standard errors retain covariance by calculating
G_h, G_h2 and D **before reduction**. Antithetic pairs are single independent
MC units. RQMC errors are taken across independent scramble means. The SE of D
is not the square root of the sum of marginal Gamma variances. No uncertainty
is inferred from individual Sobol points. A zero empirical SE or small D is not
proof of accuracy, especially when a small sample never crosses a payoff kink.
SEs exclude finite-bump, time-discretization, truncation/absorption, smoothing
and model errors; LSV errors condition on one calibration, not calibration-seed
uncertainty. A width-ladder difference is **not a remaining-bias bound**.

The nested `price` preserves the original base-price sampling/metadata contract.
Its evaluated path count is the base sample count; `payoff_evaluations` is five
times that count (including antithetic paths). Sticky-relative LSV reuses each
base stochastic path, whereas pure/frozen scenarios re-evolve it. The result's
separate `risk_fingerprint` includes the original plan, convention, method and h;
price fingerprints intentionally do not change when requesting another h.

## Physical cash-dividend mapping

Let A0 be the complete initial escrow reserve, R=S-A0 and delta the Spot shift.
For pure SV the model is evolved from S+delta. Each reference observation scale
is multiplied by `(1+delta/R)/(1+delta/S)`; its cash reserve stays fixed.
This matters for beta<1, where paths do not simply scale with their initial value.

For frozen LSV, use initial funded forward `S+delta*S/R` with the reference
observation map fixed. For sticky-relative LSV, the same physical scenarios are
obtained by scaling the base funded-forward states by `1+delta/R` in the fixed
reference observation map. Complete future cash reserves, time-zero jumps,
pre/post-dividend observations and payment-date discounting are retained.

## Guarded domain

Require finite S,R,h>0, h<min(S,R), h/S>=1e-5 and representable squared-width
arithmetic. The relative minimum is a numerical exclusion, not an appropriate
width for every product. Widths outside this domain fail explicitly. Unsupported
hard discontinuous payoffs require caller-specified smoothing; they are not
silently differentiated. Smooth/payoff conventions remain fixed across scenarios.

The frozen-leverage route retains the parent's exact unequal-slope spatial-knot
rejection, including an active initial knot; it does not treat one branch as a
two-sided derivative. Finite bumps can cross knots away from the base point:
this is part of the central-price experiment, not a branchwise Hessian.
A finite result still does not establish that a true second derivative exists
(e.g. a deterministic ATM kink has width-dependent diverging curvature).
All scenario failures propagate; failed paths are never dropped.

## Example and validation

```python
r = lsv_plan.evaluate_sticky_moneyness_gamma_bump(1.0)
print(r.price.value, r.gamma, r.gamma_standard_error)
print(r.half_bump_gamma, r.half_bump_standard_error)
print(r.bump_difference, r.bump_difference_standard_error)
```

See the [complete example](../../examples/python/rough_family_gamma_bump.py)
and [validation protocol](../../design/validation/rough-family-gamma.md).
The resimulation/direct-derivative distinction is discussed by Broadie and
Glasserman, [Estimating Security Price Derivatives Using Simulation](https://doi.org/10.1287/mnsc.42.2.269)
(1996). This implementation is an explicitly biased finite-difference estimator,
not an implementation of that paper's unbiased direct estimators.

NOT included: second-order AAD, model-parameter/Hurst Gamma, stochastic-parameter
recalibration, new stochastic-rate/dividend/multi-asset combinations, or a
nondegenerate continuous-time pricing/Gamma guarantee. Parent experimental
status and small-H full-truncation/time-grid bias remain unchanged.
