# Multi-asset rough Bergomi LSV and common Hull–White

## Scope and entry points

The experimental common-HW multi-asset adapter accepts rough-LSV assets mixed
with BS and one-/two-factor Bergomi LSV. Basket, Worst-of and unseasoned
memory/no-memory Autocallables retain dated observations, payment lags, affine
cash/proportional dividends, MC/RQMC and calibrated risk. All assets use one
currency and one shared Hull–White rate process and initial discount curve.

Rust adds `MultiAssetRoughLsvConfig { factor: RoughBergomi, particles }` and
`MultiAssetBergomiLsvConfig::Rough`, passed to
`MultiAssetPricingPlan::compile_with_hull_white`. Python accepts the immutable
`MultiAssetRoughLsvConfig` in `MultiAssetPlan.compile(..., lsv_configs=...)`:

```python
rough = rp.MultiAssetRoughLsvConfig(
    hurst=0.1, vol_of_vol=0.5, correlation=-0.65,
    particle_count=2048, calibration_seed=401, log_bandwidth=0.5,
    minimum_effective_samples=8.0, retain_reverse_trace=True,
)
```

Supply `rate_model`, `rate_correlations`, and a matching `HullWhiteLsvTarget`
and Local Volatility model for every LSV asset. A BS asset uses `None` for its
configuration and target. For deterministic rates, explicitly supply an HW
model with zero rate volatility; rough configurations without the paired-HW
adapter are rejected. Native Rust/Python APIs are extended; JSON schemas are
unchanged. The [installed example](../../examples/python/multi_asset_rough_bergomi.py)
prices three products with two different H values and a full correlation matrix.

This entry point is calibrated rough-LSV. Uncalibrated pure rough Bergomi with
flat initial forward variance remains available through the existing
[single-equity API](rough-bergomi.md).

## Rough state and finite-grid normalization

For asset i, `0 < H_i <= 0.5`, `eta_i >= 0`, and

`X_i(t) = sqrt(2 H_i) integral_0^t (t-s)^(H_i-1/2) dV_i(s)`.

The field `vol_of_vol` is **eta, the log-variance coefficient**. It is twice
the log-volatility coefficient used by the Markovian Bergomi configurations
at the Brownian boundary. Spot/volatility `correlation` is the Brownian
correlation between this asset's spot driver and V_i.

The nonuniform hybrid scheme retains the exact newest-cell power integral
`J_i,n` and uses average kernels for older cells:

`Xhat_i(t_n) = J_i,n + sum_(k<n-1) w_i,nk * dV_i,k`,

where `w_i,nk` is the integral of the power kernel over cell k divided by
that cell's length. Its actual finite-grid variance is
`Q_i,n = h_(n-1)^(2H_i) + sum_(k<n-1) w_i,nk^2 * h_k`.
Store `Y_i,n = Xhat_i(t_n) - eta_i * Q_i,n/2`; the equity volatility at the
left node is `L_i(t_n, log m_i) * exp(eta_i * Y_i,n/2)`. Thus the variance
multiplier has unit Gaussian mean on the implemented grid. No future cell
enters a stock step's volatility. This reuses the single-equity hybrid
discretization, based on the power-kernel method of
[Bennedsen, Lunde and Pakkanen](https://arxiv.org/abs/1507.03004).

At H=0.5, J_i,n equals dV_i,n exactly; history gives Brownian motion. This
boundary matches zero-mean-reversion Markovian LSV with `nu=eta/2`. At eta=0
the random variance multiplier is one. Neither boundary drops random blocks.

## Joint drivers and cross-asset correlation

Let N be the asset count, F the number of volatility Brownian drivers, and M
the rough asset count. A rough asset contributes one of the F drivers plus
one additional power-integral innovation. Two-factor Markovian LSV contributes
two of the F drivers.

| Interface | Dimension and order |
| --- | --- |
| Full Brownian matrix | N+F+1: all spots, per-asset volatility drivers, common rate |
| `rate_correlations` | N+F: all spot/rate entries, then volatility/rate entries |
| Step innovations | N+F+2+M: all spot increments, per-asset OU or rough Brownian increments, rate OU, rate integral, near-cell power integrals in rough asset order |
| Two rough assets | 5 Brownian drivers; 8 Gaussian blocks |
| Rough plus two-factor LSV | 6 Brownian drivers; 8 Gaussian blocks |
| Rough plus BS | 4 Brownian drivers; 6 Gaussian blocks |

For two rough assets, the step order is
`[dW_A, dW_B, dV_A, dV_B, OU_r, integral_r, J_A, J_B]`.
The extra J blocks are weighted integrals of the existing volatility Brownian
drivers, not additional independent model factors.

Each dated full Brownian matrix specifies price/price, price/volatility,
volatility/volatility and rate correlations independently. All matrices,
including dates beyond the pricing horizon, undergo the existing PSD checks.
Own-asset marginal correlations must match each calibration config; rate
entries must match the explicit constant rate vector. Cross-asset entries may
vary by date. If a full matrix is omitted, the existing independent-residual
construction is used: `Corr(W_A,V_B)=R_AB*rho_B` remains a modeling assumption
of that default, not a restriction of rough Bergomi. See the
[correlation contract](multi-asset-lsv.md).

For a cell of length h and Brownian correlation R, the new exact covariances
are integrated from the two time kernels:

* Rough/rough newest cells:
  `R * 2 sqrt(H_i H_j) * h^(H_i+H_j) / (H_i+H_j)`.
* Rough newest cell / an OU innovation with reversion k:
  `R * sqrt(2H_i) * integral_0^h u^(H_i-1/2) exp(-k u) du`.
  At k=0 this is `R * sqrt(2H_i) * h^(H_i+1/2)/(H_i+1/2)`.
* Rough/rate and rough/integrated-rate innovations use the same power kernel
  against the exact HW kernels, split at rate-volatility knots.

The joint loading includes J/dV cross terms; each asset's later history uses
those same correlated dV increments. Consequently, different H values and
dated cross correlations propagate to cross-time volatility covariances.
A single correlated normal per driver would not reproduce this joint law.

Factorization uses the existing spot prefix and pivoted volatility/rate suffix,
extended to the power-integral rows. H=0.5 rows enforce their exact equality
to dV. No jitter, eigenvalue repair or tolerance relaxation is applied. All
N+F+2+M independent normal blocks are retained. Bridge construction operates
on factor-major independent blocks before loading; antithetic paths negate
every block. The new fingerprint records the rough scheme and asset mapping.
Plans with no rough assets retain their previous fingerprints and paths.

## Calibration, risk and cost

Each rough asset calibrates with its own rough-HW marginal particle law and
independent calibration random domain. The existing discounted second moment,
short-rate correction, affine dividend rules and per-cashflow conditional
discount are reused. Calibration diagnostics are in
`plan.hull_white_calibrations`; `volatility_factor_count == 1` counts the rough
Brownian driver, not a one-factor Markov approximation.

Retained market-IV sources regenerate both target variance and density on the
common event grid. Direct paired targets must include every common time node
exactly; in particular, time-zero Dirac density is not interpolated to positive
times. See the [HW target rules](bergomi-hull-white.md).

With `retain_reverse_trace=True`, AAD includes particle recalibration for every
effective target-variance and forward-density node. Retained IV sources also
produce quote-node VegaKT and parallel-IV Vega. Physical Spot Delta, cross Gamma
by CRN Delta bumps, common initial discount-curve risk and per-asset dividend
curve risk remain available. H, eta, model correlations and HW parameters are
fixed; their Greeks are not added. Payout quotes and normalized target
coordinates remain fixed under Spot bumps. Autocallable AAD requires explicit
smoothing under the existing payoff contract.

RQMC target/quote errors are conditional on the compiled calibration and use a
VJP per independent scramble. MC target/quote errors remain `None`; price and
pathwise-risk sampling estimates remain available. These estimates exclude
particle, time-grid, target construction and smoothing errors.

For K time steps, each rough asset stores O(K^2) kernel weights and uses
O(K^2) work and O(K) scratch per path. Calibration retains O(particles*K)
rough history checkpoints in addition to its existing reverse trace. There
is no FFT, Markov lift or truncated-memory approximation. Large path counts
and long finely spaced grids require a separate performance assessment.

## Verification and boundaries

[Joint-driver Rust tests](../../crates/pricing/tests/cases/multi_asset_rough.rs)
compare all step covariances against independent power/OU/rate quadrature,
including different H, mixed Markovian factors, dated correlations, rate knots,
zero mean reversion and singular Brownian matrices. They cover every paired
target and IV bucket against complete recalibration, physical Spot and cross
Gamma, initial curves, affine payouts, zero-eta and H=0.5 limits, worker replay
and MC/RQMC trace/error contracts.
[History tests](../../crates/pricing/tests/mc_rough_bergomi.rs) verify cross-time
covariance under dated correlations, adaptedness and antithetic behavior.

The [installed Python tests](../../tests/python/test_multi_asset_rough.py) cover
typed configurations, products and mixtures. An independent two-step exchange
price integrates the first joint spot/rate/power increments with five-dimensional
Gauss–Hermite quadrature and integrates the final conditional payoff analytically.
The reference conditions on the compiled leverage surface; it tests valuation,
not the continuum accuracy of the finite particle calibration.

The finite algorithm remains experimental. Intended production smiles and
products require particle/bandwidth/step/seed refinement. Multi-currency/FX,
multiple rate factors, pure multi-asset rough without calibration, nonflat pure
initial forward variance, correlation/H/eta Greeks, seasoned products, American
exercise and continuous barriers remain outside this extension.
