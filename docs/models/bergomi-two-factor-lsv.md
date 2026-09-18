# Two-factor Bergomi LSV

This extends the [multi-asset LSV API](multi-asset-lsv.md) and the
[single-asset LSV API](lsv-calculation-specifications.md) to two volatility
factors per asset. Deterministic rates, the existing affine dividend coordinate,
particle calibration and its discrete target-variance VJP are retained.
One-factor entry points, random coordinates and numerical scheme identifiers
remain unchanged. This extension is experimental and does not change stable JSON.

## Model and parameter convention

For each configured asset the two OU states start at zero:

`dX_j = -k_j X_j dt + dV_j`, for `j=1,2`.

The stochastic volatility multiplier is

`a = exp(nu * alpha * ((1-theta)*X_1 + theta*X_2))`,

where `alpha = 1/sqrt((1-theta)^2 + theta^2 + 2*rho_12*theta*(1-theta))`.
This normalizes the instantaneous variance of the weighted OU driver to one;
`nu` is the volatility of this stochastic log-volatility component. Total LSV
volatility also depends on the calibrated leverage function. The model does
not identify factor correlation with observable implied-volatility correlation.
Deterministic exponential normalization cancels between the multiplier and
particle-calibrated leverage, as in the one-factor implementation.

This two-OU-factor structure follows the construction discussed in
[Bergomi's multi-asset SV/LSV presentation](https://www.lorenzobergomi.com/_files/ugd/c4ff5c_4f08ab0ca48b45dc98748a0fa2a31fa9.pdf).
The equations and finite algorithm in this document define the implemented
contract; observable-volatility correlation calibration is not part of this API.

| Input | Meaning |
| --- | --- |
| `mean_reversions=[k1,k2]` | Nonnegative OU mean reversions, in explicit factor order |
| `vol_of_vol=nu` | Nonnegative multiplier parameter |
| `mixing_weight=theta` | Weight of the second factor, in `[0,1]` |
| `spot_correlations=[rho_S1,rho_S2]` | Instantaneous price/OU Brownian correlations |
| `factor_correlation=rho_12` | Instantaneous correlation of the two volatility drivers |

All inputs must be finite. The three-driver matrix ordered `[W,V1,V2]` must be
positive semidefinite. The model uses the exported
`BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES`: zero symmetry/diagonal tolerance,
64 machine epsilons absolute PSD/zero-pivot tolerance, and zero relative
tolerance. The global sampler also validates its matrix with the correlation
schedule's six supplied tolerances. No jitter, pivot permutation or eigenvalue
clipping is introduced. Zero normalization variance (`theta=.5,rho_12=-1`)
is rejected. Factors are never reordered by mean reversion.

## Rust and Python APIs

Rust adds `models::Bergomi2Factor::new([k1,k2],nu,theta,[rho_S1,rho_S2],rho_12)`.
The existing `calibrate_bergomi_lsv`, `calibrate_bergomi_lsv_parallel`,
`CalibratedBergomiLsv`, `BergomiLsvPlan`,
`BergomiLsvPath` and `BergomiLsvPricingPlan` accept this factor through their
sealed `BergomiDynamics` generic parameter. Their default type remains
`Bergomi1Factor`. Two-factor path states and initial factor adjoints are arrays
of length two; `orthogonal_shocks` contains both residual blocks in factor order.

For mixed multi-asset configurations, call `compile_with_bergomi_lsv` with
`Vec<Option<MultiAssetBergomiLsvConfig>>`. Convert either `MultiAssetLsvConfig`
or `MultiAssetLsv2FactorConfig` with `.into()`. The old `compile_with_lsv`
signature still accepts the original one-factor configurations.
`lsv_calibrations()` retains its one-factor return type and returns `None` for
two-factor assets; `lsv_two_factor_calibrations()` exposes the complementary
two-factor objects. `lsv_volatility_factor_counts()` returns 0, 1 or 2 per asset.

Python adds `MultiAssetLsv2FactorConfig` with the five model inputs above and
the same particle settings as `MultiAssetLsvConfig`. Pass either configuration
or `None` in the existing `MultiAssetPlan.compile(...,lsv_configs=...)` list.
`lsv_calibrations` combines both kinds, and each immutable snapshot reports
`volatility_factor_count`. The single-asset entry point is
`Bergomi2FactorLsvPlan.compile(target_request, ...)`, with the same model inputs
and particle/execution settings. Its price scheme is
`bergomi-two-factor-lsv-log-euler-exact-ou-v1`. Compilation and valuation release
the Python GIL. See the [three-product example](../../examples/python/multi_asset_bergomi_two_factor.py).

## Joint correlations and exact transitions

Let `N` be the number of assets and `m_i` their respective volatility factor
counts (0, 1 or 2). The global matrix dimension is `D=N+sum(m_i)`. Driver order
is all price drivers, then each asset's volatility drivers in asset/factor order.
For two two-factor assets: `[W_A,W_B,V_A1,V_A2,V_B1,V_B2]` (6 by 6).
Full matrices align with every date in `CorrelationSchedule`, including dates
beyond maturity. Each price block and each complete own-asset `[W,V1,V2]` block
must match the corresponding configured/calibrated marginal after canonicalization.
Own correlations are fixed; cross-asset correlations can change at scheduled dates.

If full matrices are omitted, the default is

`dV_i = r_i dW_i + H_i dZ_i`, with `H_i H_i^T = C_i-r_i r_i^T`.

Here `r_i` is the vector of own price/vol correlations and `C_i` is the own
volatility correlation matrix. Residual vectors `Z_i` are independent across
assets and independent of all price drivers. Within one asset, the residuals
retain the covariance needed to recover its specified `rho_12`. Across assets,
`corr(W_i,V_jb)=R_ij*r_jb` and `corr(V_ia,V_jb)=R_ij*r_ia*r_jb`.
This is a conditional-independence assumption, not a market-calibrated default.
An admissible full matrix overrides cross-asset entries.

For an interval of length `h`, use the correlations at its left endpoint. For
`J_ia = integral exp(-k_ia*(t+h-s)) dV_ia(s)` and
`I(k,h)=(1-exp(-k*h))/k`, with `I(0,h)=h`:

- `Cov(Delta W_i,Delta W_j)=R_WW(i,j)*h`;
- `Cov(Delta W_i,J_jb)=R_WV(i,jb)*I(k_jb,h)`;
- `Cov(J_ia,J_jb)=R_VV(ia,jb)*I(k_ia+k_jb,h)`.

The entire joint covariance is normalized and factored together. The same exact
three-driver law is used in each asset's independent calibration. Correlation
of finite OU innovations therefore depends on the mean reversions and interval
length. Different kernels can leave multiple independent interval innovations
even when the underlying Brownian matrix is singular.

Independent Gaussian blocks are bridged before the joint factorization.
Antithetics negate every block. Dimensions remain `D*steps`, including factors
with zero mixing weight, zero vol of vol, or singular driver correlations.
`theta=0` recovers the first one-factor model to floating-point roundoff without
changing the first two marginal random-coordinate blocks. `nu=0` recovers the
LV target and its calibrated target-risk chain rule.

## Risk and scope

Multi-asset price, per-asset Delta, full cross Gamma, target-variance VJP and
conditional MC/RQMC uncertainty contracts are the same as for one-factor LSV.
The reverse propagates through both OU states, the leverage lookup, every
calibration particle and original-grid interpolation. Model parameters and all
correlations remain fixed; model/correlation Greeks are not provided.

Product-specific particle, bandwidth and time-step refinement is still required.
The separate
[Bergomi + HW extension](bergomi-hull-white.md) connects two-factor
single-asset and multi-asset plans to stochastic rates, paired target AAD,
market-IV VegaKT and initial-curve risk. The deterministic-rate interfaces
described here retain their effective-variance risk convention.
[Multi-asset rough-LSV](multi-asset-rough-bergomi.md) is available through
the shared HW adapter and can be mixed with two-factor assets.

## References

- [Bergomi, *Smile Dynamics II*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1493302), for the two-factor forward-variance/Bergomi construction and factor correlations.
- [Guyon and Henry-Labordère, *The Smile Calibration Problem Solved*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032), for particle calibration of the local-stochastic-volatility leverage.
- [Hamdouche and Henry-Labordère, *Vega KT for LSV Models: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4304114), for the LSV calibration reverse and target-risk setting.
- [Glasserman, *Monte Carlo Methods in Financial Engineering*](https://doi.org/10.1007/978-0-387-21617-1), for Monte Carlo, antithetic sampling and common-random-number checks.
