# Rough-family LSV: discrete market-IV quote risk

**Experimental.** All six additional rough families are supported, with Rough
SABR restricted to beta=1, as in the parent [LSV adapter](rough-family-aad-lsv.md).
Model parameters, time/space axes, Spot, curves, cash dividends, calibration
seed, bandwidth, support selection and payoff smoothing are fixed.

## Contract

`plan.market_iv_risk_plan(surface).evaluate()` differentiates

```text
Black IV quotes -> interpolated total variance -> strict Dupire grid
 -> target interpolation -> particle calibration -> leverage -> payoff price.
```

The retained `MarketIvSurface` uses natural-cubic total variance in relative log
moneyness, linear total variance in time, and constant-IV time tails. Quotes are
maturity-major. For quote `(T_i,x_j)`, `w_ij = T_i * sigma_ij^2`. Quote coordinates
refer to the same normalized residual-equity / funded-forward convention as the
LSV target. In the presence of fixed-cash dividends, physical-stock Black quotes
in a different convention must not be inserted without an explicit conversion.
There is no automatic Spot/dividend conversion.

The returned quote adjoints are price per **one absolute annualized IV unit**.
20% IV is `0.20`; multiply an adjoint by `0.01` for one volatility point, or
`0.0001` for one IV basis point. `parallel_vega` is the derivative for adding the
same absolute IV shift to every supplied quote. No production bumps or extra
calibrations are used. Stochastic-model parameter calibration is NOT included.

This is a discrete quote Jacobian, not the paper-style continuous VegaKT
reporting projection. It does not allocate risk to a separately selected
reporting basis, apply a density cutoff, or supply a projection residual.
Nor does it add Gamma, Hurst/model-parameter AAD or fixed-strike-IV Spot Delta.

## Source binding and failure behavior

A separate immutable `RoughFamilyLsvMarketIvRiskPlan` owns a clone of the pricing
plan and the source. Binding does not change the original plan, its prices,
local-variance adjoints or its price fingerprint. A separate `risk_fingerprint`
includes the price plan, interpolation identifier, quote axes and all IV values.

The source must reproduce the original (pre-time-refinement) Local Variance grid
**bit-for-bit** with the same axes and floor/cap. An arbitrary similar surface is
not an acceptable source. Repaired grid nodes, exact floor/cap kinks, invalid
Dupire densities and strike extrapolation are rejected. A calibration reverse
trace and a pathwise-admissible payoff (or explicit smoothing) are required.
The original time-zero target rule is retained: evaluate the surface at the
first positive target time. Both initial and first-positive rows are reversed.
Binding caps the source at 4096 quotes and target-node-count times quote-space-
node-count at 16,777,216; inherited calibration/resource guards also remain.

## Calculation specifications

Let `w`, `wx`, `wxx`, `wt` be the total variance and derivatives at a target node,
`x` relative log moneyness, `a = wt/g` Dupire relative Local Variance, and

```text
u = 1 - x*wx/(2*w)
g = u^2 - (wx^2/4)*(1/w + 1/4) + wxx/2.
```

For an incoming `a_bar`, reverse the ratio and density:

```text
g_bar   = -a_bar*wt/g^2
w_bar   = g_bar*(u*x*wx/w^2 + wx^2/(4*w^2))
wx_bar  = g_bar*(-u*x/w - wx/2*(1/w + 1/4))
wxx_bar = g_bar/2
wt_bar  = a_bar/g.
```

The existing spline/time transpose maps these four seeds to total-variance
quotes, then multiplies by `2*T_i*sigma_ij`. Financial formulas stay in `pricing`;
Python bindings only convert inputs/results. No new dependency is introduced.
See [market IV](../../crates/pricing/src/market/market_iv.rs) and the
[LSV risk implementation](../../crates/pricing/src/engine/risk/lsv/rough_families/market_iv.rs).

## Sampling uncertainty

For RQMC, each independent scramble's complete local-variance gradient is
transposed to quote coordinates **before** means and sample variances are
computed. The parallel direction is also summed within each scramble. This
retains the cross-node covariance. Adding marginal errors or taking their
root-sum-square does not give the parallel standard error. Returned quote means
and their sum can differ from `parallel_vega` by floating-point rounding only.

For pseudo-MC, quote and parallel standard errors are `None`, following the
existing aggregate-first local-variance risk contract, not zero. All available
SEs are conditional on one realized particle calibration and exclude its seed
uncertainty, time/space/truncation bias and model error. Active interpolation,
support and fallback branches have the parent's local derivative semantics.

## Example and validation

```python
# source is the exact MarketIvSurface used to construct the target request.
risk_plan = lsv_plan.market_iv_risk_plan(source)
risk = risk_plan.evaluate()
print(risk.quote_adjoints)
print(risk.parallel_vega, risk.parallel_standard_error)
```

The [complete example](../../examples/python/rough_family_market_iv.py) includes
surface, market, dividends, request and calibration. The
[validation protocol](../../design/validation/rough-family-market-iv.md)
distinguishes discrete derivative tests, independent Dupire rebuilds, covariance
reconstruction and analytic Black limits. None certifies a nondegenerate LSV
smile to 5 IV bp or removes the parent's small-H full-truncation bias.
