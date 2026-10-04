# Rough-family LSV: physical Spot Delta

**Experimental.** Two explicit conventions extend [rough-family MC/LSV](rough-family-aad-lsv.md).
All six variance families are supported; Rough SABR requires beta=1. Pricing,
calibration, RNG layout, fingerprints and the existing local-variance pullback
are unchanged. See [validation](../../design/validation/rough-family-lsv-spot-delta.md)
and the [complete example](../../examples/python/rough_family_lsv_spot_delta.py).

## Contract

`RoughFamilyLsvPricingPlan` in Rust / `RoughFamilyLsvPlan` in Python provide:

| Method | What stays fixed under a physical Spot move |
| --- | --- |
| `evaluate_frozen_leverage_delta()` | The calibrated leverage values, axes and reference funded-forward anchor; deterministic physical observation map |
| `evaluate_sticky_moneyness_delta()` | Relative Dupire variance values and log-moneyness axes; calibration anchor moves with Spot |

Curves, cash/proportional dividend quotes, contractual strikes/barriers/smoothing
widths, dates, model parameters, calibration and valuation random draws, and
bandwidth are fixed in both methods. The second is NOT sticky-strike market-IV
Delta: it does not keep absolute-strike IV quotes fixed or rebuild Dupire from
those quotes. It is also not an arbitrary recalibration-gradient API.

Both return `RoughFamilyLsvDelta` with unchanged `LsvPrice` as `price`,
`delta`, `delta_standard_error`, `convention`, and `method`. The Python result
is frozen; `coordinate` is `physical_spot_fixed_curves_and_cash_dividends`.
The method/convention distinguish risks sharing the same price-plan fingerprint.
Delta is per one physical Spot currency unit, not per percent Spot move.

## Reference coordinate and fixed-cash chain rule

Let S_ref be the original Spot, A0 the full discounted initial escrow reserve,
and R_ref = S_ref - A0 > 0. A0 includes cash beyond option expiry and time-zero
cash/proportional events under the existing carry convention. The martingale
coordinate f starts at S_ref. Physical observations have affine form

```text
S_j = A_j + B_j f_j,
```

where B includes proportional payouts, the initial R_ref/S_ref scale and
continuous deterministic carry. The existing payoff graph supplies
`q_j = d(discounted payoff)/df_j`, including pre-dividend observations and the
payment-date discount exactly once.

### Frozen leverage

Hold the reference A_j, B_j and leverage L(t,log(f/S_ref)) fixed. A physical
Spot scenario S is represented in that reference coordinate by

```text
f_initial(S) = S_ref * (S - A0) / R_ref.
delta_frozen = (dP/df_initial) * S_ref/R_ref.
```

Reverse the recorded log-Euler path to obtain dP/df_initial, including the
spatial derivative of squared leverage. Simply reporting the low-level
initial-forward adjoint would omit S_ref/R_ref when there is fixed cash.
Neither the surface nor the calibration is bumped.

### Sticky relative Local Variance target

Calibrating at S with the same relative Local Variance grid scales every
calibration particle by S/S_ref. All `log(f/S)` kernel arguments, conditional
moments, donor choices and squared-leverage values are invariant in exact
arithmetic. The stochastic variance history is independent of the equity
level for these six fixed-driver families. The pricing paths scale the same way.
Including the Spot dependence of the fixed-cash escrow map therefore gives

```text
delta_sticky = sum_j q_j * f_j / R_ref.
```

This is a discrete scale-equivariance identity, not an approximation obtained
by freezing a conditional expectation that actually changes. It avoids a
calibration reverse trace and another calibration. Tests compare against full
request recompilation and recalibration at two Spot bumps. Floating-point
roundoff and support ties can affect exact numerical equivalence; no bitwise
cross-Spot scaling guarantee is made. QRH retains the parent fixed-driver
LSV definition, not a feedback model driven by leveraged stock returns.

## Sampling and boundaries

The two methods work with or without retained calibration reverse traces.
Pseudo-MC uses independent path/antithetic-pair units; RQMC uses independent
scramble means, never individual Sobol points as independent samples. Delta
SEs are provided for both engines, conditional on the one calibration. They
exclude calibration-seed variability, time-step/truncation error and model bias.

Hard discontinuous payoffs require the existing explicit smoothing contract;
no silent smoothing is introduced. The fixed-leverage method rejects an exact
spatial node with unequal left/right slopes on an active variance step, including
the common initial k=0 case and flat-tail junctions. A branch-selected derivative
there would not be an ordinary two-sided Delta. Zero-variance identity steps
are exempt because leverage cannot affect their price update. The sticky method
leaves relative lookup positions fixed, so this particular kink is not crossed.
Other payoff and interpolation active-set semantics are inherited.

Nonfinite/invalid derivatives or residual equity fail explicitly. Both methods
return the original plan's price and sampling metadata without changing it.
No MC Gamma, stochastic-parameter/Hurst/covariance AAD, market-IV VegaKT,
stochastic-rate/dividend or multiasset composition is added. Matching finite
bump derivatives does not remove the parent small-H discretization bias.
