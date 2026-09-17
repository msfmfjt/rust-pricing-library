# Market-IV VegaKT for the equity/Hull–White LSV hybrid

Date: 2026-09-13. Status: experimental extension of the
[first-order AAD contract](hull-white-aad.md).
The same quote transpose is available for
[rough-LSV](rough-bergomi.md) with fixed H/eta.

## Input and interpolation contract

Create a Rust `MarketIvSurface` from positive, strictly increasing quote
maturities, strictly increasing log-moneyness nodes and row-major Black IV
quotes, then use `HullWhiteLsvTarget::from_market_iv`. Python exposes the same
pipeline through `HullWhiteLsvTarget.from_market_iv`:

```python
target = rp.HullWhiteLsvTarget.from_market_iv(
    [0.4, 1.0],                         # positive quote maturities
    [-0.75, 0.0, 0.75],                 # quote log(K / forward)
    [0.22, 0.20, 0.21, 0.23, 0.21, 0.22],  # row-major absolute IV
    [0.0, 0.25, 0.5, 0.75, 1.0],       # calibration times, including events
    [-0.75, -0.375, 0.0, 0.375, 0.75], # calibration log nodes
)
# Build the price-only request with target.model, then compile_lsv using
# this target and retain_reverse_trace=True. The complete example is linked below.
```

There are at least two quote maturities and two quote log nodes. Quote and
calibration axes can differ. The calibration grid retains the existing rule
that its final time equals product expiry and includes all required events.
The quote grid may extend beyond expiry; such quotes can affect earlier local
variance through the time slope. No zero-time market-IV bucket is invented.

At each quote maturity, interpolate total variance `w_ij=T_i*sigma_ij^2`
with a natural cubic spline in log moneyness. Its second derivatives are zero
at the two quote-domain edges. Interpolate total variance linearly in time.
Use the right time derivative at an interior quote maturity; at the final
maturity use the constant-IV tail. Before the first and after the last quote
maturity, scale the respective boundary total variance by `T/T_boundary`.
Strike extrapolation is rejected: the entire target log grid must lie inside
the quote domain. Label: `natural-cubic-w-linear-time-v1`.

The constructor validates finite positive IV, valid axes and positive total
variance, calendar slope and Durrleman density at every quote. Target creation
checks all calibration samples and rejects any required variance floor/cap
repair. These are sampled checks, not a guarantee of global absence of arbitrage
between samples. This interpolator does not perform an SSVI/eSSVI constrained
fit, quote weighting, bid/ask selection or an optimizer reverse.

## Quote coordinates and cash dividends

IV quotes refer to the same forward coordinate used by the target. With
proportional dividends and no fixed cash, this is equivalent to physical
equity Black IV at fixed `log(K_S / forward_S(T))`; the corresponding normalized
strike is `K_F=S0*exp(x)`.

For the escrowed cash model, quotes must already refer to the deterministic
escrow coordinate F. Convert physical cash-dividend option quotes using
`K_F=(K_S-A0(T))/c(T)` and `C_F=C_S/c(T)`, then obtain Black IV using forward S0
and the same discount factor. This API returns sensitivity to those **converted
IV inputs**. It does not differentiate the physical-price/IV conversion or
refit raw physical-S quotes. An unconverted physical Black smile must not be
passed as an F smile. The cash example explicitly uses illustrative F quotes.

For VegaKT, Spot, curves, HW/Bergomi parameters, correlations, dividend amounts,
quote axes, calibration axes, random draws and kernel configuration are fixed.
The original Spot and curve adjoints continue to hold target samples fixed;
adding VegaKT does not change them into sticky-physical-smile Greeks.

## Exact discrete transpose

Write `x=log(K_F/S0)`, `w=w(T,x)`, `g` for the Durrleman factor, and
`d2=-x/sqrt(w)-sqrt(w)/2`. Both calibration inputs are generated from one surface:

```text
variance = w_T / g
p_log    = phi(d2) * g / sqrt(w)
```

The hybrid AAD first reverses price paths and the full finite particle
recalibration. Contract both outgoing target adjoints through the two formulas,
then through the natural-cubic linear system and time weights, finally through
`dw_ij/dsigma_ij=2*T_i*sigma_ij`. The result is the exact first derivative of
this discrete interpolation/calibration/pricing algorithm with its recorded
branches fixed. It uses no quote bumps in production and runs no calibration
per quote or per pricing path. The spline coefficient map uses O(n_x^2) storage;
the target transpose is applied once per MC mean or RQMC scramble.

The target's time-zero variance is copied from the first positive calibration
time, so its adjoint reaches that surface evaluation. Time-zero density is
fixed at zero and has no quote derivative. Natural-cubic interpolation is
nonlocal in strike: a quote may affect several strike cells in its time slices.
Removing the density term yields a different risk under stochastic rates.

The retained IV source and its interpolation label are part of the plan
fingerprint. Copying only the numerical variance/density arrays loses quote
provenance. `flat`, `from_essvi`, `from_surface` and `from_grid` retain their
existing price and node-risk behavior and do not implicitly gain VegaKT.
Their VegaKT result is `None`; Python `target.supports_vega_kt` exposes this
distinction before compilation. Sampling an existing fitted smile onto this
quote grid and reconstructing it adopts the new interpolation and can change
the price. It is not a reverse of the original fitted-surface parameters.

## Output, units and uncertainty

Use `plan.evaluate_aad()` after compiling with `retain_reverse_trace=True`.
Existing derivative-vector entries keep their order. A quote-backed LSV plan
appends `market_iv[i]` in row-major quote order, then `parallel_market_iv`.

| Rust method / Python property | Meaning |
| --- | --- |
| `vega_kt_raw` | Currency per unit absolute input IV, row-major |
| `vega_kt_market_scaled` | Raw multiplied by 0.01: price change per +1 vol point |
| `vega_kt_standard_errors` | Raw-unit independent-scramble SE, or `None` for pseudo-MC |
| `vega` | Parallel market-IV derivative, the bucket sum up to reduction roundoff; BS retains its original Vega |
| `parallel_vega_standard_error` | SE of the scramble-wise parallel derivative, including cross-bucket covariance |
| `vega_kt_maturity_nodes`, `vega_kt_log_moneyness_nodes` | Quote-grid axes, distinct from target-grid axes |
| `vega_kt_implied_volatilities` | The original row-major input quotes |
| `vega_kt_method` | The interpolation label, or `None` when unavailable |

Multiply raw bucket SE by 0.01 for market-scaled SE. Adding bucket SEs does not
give the parallel SE. All risk SEs are conditional on the realized calibration;
they exclude calibration noise, time/kernel bias and branch-selection effects.
The calibration's hard digital, ESS/support, donor and interpolation choices
have the same almost-everywhere derivative convention as the original AAD.
Finite differences must be checked for branch stability; these tests do not
establish unbiased continuum Greeks or complete the broad H5 acceptance gate.

See the [runnable Python example](../../examples/python/hull_white_lsv.py).

## References

- [Adrien et al., *Vega KT for the Local Volatility Model: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4107770), for the Local Vega path adjoint, reporting-basis projection and equation (11) recovery used by the quote transpose.
- [Hamdouche and Henry-Labordère, *Vega KT for LSV Models: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4304114), for the LSV variance/density transpose and recalibrated target risk.
- [Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2033323), for the SVI/eSSVI surface and density conventions accepted by the target builders.
- [Hull and White, *Pricing Interest-Rate-Derivative Securities*](https://doi.org/10.1093/rfs/3.4.573), for the stochastic-rate hybrid in which the quote transpose is applied.

Natural-cubic interpolation, row-major quote ordering and the reverse branch
conventions are implementation-specific contracts documented above.
