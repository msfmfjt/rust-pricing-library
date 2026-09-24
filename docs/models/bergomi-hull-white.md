# Bergomi and common Hull–White

## Scope and API

This experimental extension connects one- and two-factor Bergomi LSV to a
single shared one-factor Hull–White rate process. It supports a single equity
or a mixture of BS, one-factor LSV and two-factor LSV equities in one currency.
The [rough-LSV extension](multi-asset-rough-bergomi.md) adds non-Markov
assets with per-asset H and eta to this same adapter.
Multi-asset Basket, Worst-of and memory/no-memory Autocallable payoffs use the
existing payoff tape, dated observations, payment lags and affine dividends.

| Boundary | Entry point |
| --- | --- |
| Single equity, Rust | `HullWhiteEquityPricingPlan::compile_lsv_two_factor` |
| Single equity, Python | `HullWhiteEquityPlan.compile_lsv_two_factor` |
| Multiple equities, Rust | `MultiAssetPricingPlan::compile_with_hull_white` and `MultiAssetHullWhiteConfig` |
| Multiple equities, Python | `MultiAssetPlan.compile(..., rate_model=..., rate_correlations=..., lsv_targets=...)` |

The single-asset entry point takes a `Bergomi2Factor` in Rust, or explicit
`mean_reversions`, `vol_of_vol`, `mixing_weight`, `spot_correlations` and
`factor_correlation` keywords in Python. The additional rate parameters are
`equity_rate_correlation` and `vol_rate_correlations` (two entries).

For multiple equities, reuse `MultiAssetLsvConfig` /
`MultiAssetLsv2FactorConfig` in asset order. Every LSV asset requires both a
`HullWhiteLsvTarget` and its matching Local Volatility model; a BS asset uses
`None` for its configuration and target. Bare LV under HW is rejected: use an
LSV configuration with zero vol-of-vol for the calibrated local-vol limit.
All assets must share the same initial discount curve, including curve identity.

The [installed-wheel example](../../examples/python/bergomi_hull_white.py) prices
three products with two two-factor assets and a common rate, then compiles the
single-equity adapter. Native Rust/Python APIs are used; existing JSON schemas
are unchanged.

## Joint drivers and correlation parameters

For asset i, the two volatility states solve `dX_ij = -k_ij X_ij dt + dV_ij`.
The normalized weights and factor normalization are those of the existing
[two-factor Bergomi contract](bergomi-two-factor-lsv.md). The stock's
diffusion is `L_i(t, log m_i) exp(nu_i * weighted_X_i)`. The shared rate state
solves `dx = -a x dt + sigma_r(t) dW_r`; its deterministic shift reproduces
the input discount curve.

There are N spot Brownian drivers, F volatility Brownian drivers, and one rate
Brownian driver. An exact step also requires the integrated rate innovation.
The orders below describe Markovian plans; M rough assets append M newest-cell
power integrals after the rate integral (N+F+2+M total Gaussian blocks):

| Plan | Brownian matrix | Exact Gaussian innovation order |
| --- | --- | --- |
| Single two-factor equity | 4 × 4 | `[dW_S, OU_V1, OU_r, integrated_OU_r, OU_V2]` |
| Multiple equities | (N+F+1) × (N+F+1) | `[dW_spots, OU_vols, OU_r, integrated_OU_r]` |
| Two two-factor equities | 7 × 7 | `[dW_A, dW_B, OU_A1, OU_A2, OU_B1, OU_B2, OU_r, integrated_OU_r]` |

Multi-asset `rate_correlations` has N+F entries: all price/rate correlations,
then each asset's volatility/rate correlations. A supplied `driver_correlations`
matrix appends the common rate Brownian as its final row and column. It contains
every price/price, price/volatility, volatility/volatility and rate correlation.
There is one matrix per date in the price correlation schedule, including dates
beyond the pricing horizon. Each matrix is PSD checked. The own-asset
price/volatility and volatility/volatility blocks must match the calibration
configuration; every rate entry must match the explicit constant rate vector.
Cross-asset blocks may vary with the correlation date.

If full matrices are omitted, the existing conditional-independent-residual
construction supplies only the price/volatility block, as documented in the
[multi-asset LSV contract](multi-asset-lsv.md). The explicitly provided
rate vector is appended and the resulting matrix is checked. In particular,
`Corr(W_A, V_B) = R_AB * rho_B` is a selectable modeling assumption of that
default construction, not a requirement of Bergomi or HW. Supply full matrices
to specify those cross correlations independently. No invalid matrix is repaired.

Each step integrates the OU kernels and every piecewise rate-volatility knot.
For two volatility innovations the covariance is
`R_ij * B(k_i + k_j, dt)`, where `B(k,h) = (1-exp(-k*h))/k` with its continuous
zero-k limit. Rate and integrated-rate covariances use the existing exact HW
kernel integrals. Brownian increments and OU innovations cannot simply share
one normal per Brownian driver when their time kernels differ.

Multi-asset transition factorization normalizes to correlation scale, keeps the
spot prefix in asset order, then pivots the OU/rate suffix by the largest
remaining conditional variance. This prevents nearly identical OU kernels
from amplifying round-off before the integrated-rate coordinate. The caller's
PSD and zero-pivot tolerances remain unchanged; no jitter or eigenvalue
projection is applied. The permutation is undone for reported covariances and
path states, and recorded in the plan fingerprint. All N+F+2 normal blocks are
retained, including zero/singular columns. The Brownian input matrix still uses
the existing unpivoted validation and exposes its diagnostics separately.

Random coordinates remain factor-major. Brownian bridge is applied separately
to independent blocks before the interval loading; antithetic sampling negates
all blocks. Calibration and valuation use independent random domains. The
single-equity five-coordinate order preserves the old first four coordinates,
including exact one-factor path replay when the mixing weight is zero.

## Discounted marginal calibration and target grids

Each asset calibrates an immutable leverage surface with its own HW marginal
law. The existing discounted conditional second moment and short-rate correction
are retained; deterministic-rate LSV calibration is not substituted. Calibration
uses a paired target: effective local variance and forward log density. The
particle state starts at one in the continuous normalized equity coordinate.
Mean relative discount, mean discounted equity, conditional second moments,
rate corrections, fallback cells and effective sample sizes are exposed in
`plan.hull_white_calibrations`.

The common time grid contains observations, in-horizon target knots, dividends,
correlation changes and maximum-step subdivision. There are two target rules:

* `HullWhiteLsvTarget.from_market_iv` retains the source quotes. Both variance
  and density are regenerated from that same source on the common time grid.
  Target-variance and density adjoints are therefore reported on this common
  grid, while VegaKT is reported on the original maturity/log-moneyness quote
  grid. The original target must still cover the pricing horizon.
* Direct paired grids, including targets from `flat`, `from_grid` or `from_essvi`,
  must contain every common time node exactly. Required rows are selected;
  missing rows are rejected. Variance and density adjoints are mapped to the
  original paired grid, with unused rows receiving zero. A time-zero Dirac
  density is never silently interpolated to positive time.

Use retained IV provenance when dated product events or a finer maximum step
introduce new calibration times. Quotes refer to the continuous normalized
equity, not automatically to physical spot in the presence of cash dividends.
See the existing [HW VegaKT contract](hull-white-vegakt.md).

## Cash flows and dividends

Every asset sees the same rate and integrated-rate innovations. Each asset uses
an [escrowed bond reserve](hull-white-cash-dividends.md) and continuous residual
equity. The entire supplied cash schedule is funded, including beyond expiry.
Pre/post observations apply the contractual cash/proportional jump exactly.
IV targets use the
[escrow quote coordinate](hull-white-cash-dividends.md#lsv-target-coordinate-and-calibration).

Each payoff output is valued at its own latest dependent observation time t.
For payment U, the conditional discount is

`P0(U) * exp(-I_t - B(a,U-t)*x_t - Var(I_U)/2 + Var(J_tU)/2)`.

Here `I_t` is the accumulated zero-mean rate integral and `J_tU` is its future
innovation conditional on time t. This supports payment lags and early coupons
without using observations after that cash flow's information date. Only the
rate state is conditioned for discounting after t; equity is simulated through
the latest product observation.

## Risk and uncertainty

`evaluate_aad()` requires `retain_reverse_trace=True` for every calibrated asset.
It reverses the payoff, affine observation mapping, equity evolution and finite
particle calibration. Both target inputs are differentiated, including the rate
correction. Model parameters, all correlations, quote coordinates, particle
settings, payout quotes and calibration random numbers remain fixed.

| Output | Convention |
| --- | --- |
| Per-asset Delta | Physical initial Spot, cash and normalized target fixed |
| Cross Gamma | CRN finite difference of the pathwise Delta using the explicit relative bump |
| BS Vega | Per-asset BS volatility derivative |
| `risk.lsv_local_variance` | Recalibrated effective target-variance adjoints and grid |
| `risk.hull_white_lsv` | Paired forward-log-density adjoints; IV-source VegaKT and parallel IV Vega |
| `result.hull_white_curve_risk` | Common initial discount log-DF adjoints/DV01 and individual dividend-curve log-DF adjoints |

Raw quote VegaKT is per unit IV; market-scaled VegaKT is 0.01 times raw.
Discount-node DV01 is `-time * 1e-4 * dPV/dlog(DF)`, with nonnegative standard
error scaled by the absolute multiplier. It represents a zero-rate node bump,
not a bump of the HW mean reversion or rate volatility.

RQMC performs the calibration VJP once per independent scramble and reports
target/quote sampling uncertainty conditional on the compiled calibration.
Parallel-IV uncertainty includes covariance across quote buckets. Pseudo-MC
performs one VJP of the mean leverage adjoint; target/quote standard errors are
`None`. Price, Delta, BS Vega and curve risk retain their sampling estimates.
These errors exclude particle-calibration uncertainty, discretization, target
construction, payoff smoothing and finite-bump bias. Autocallable AAD retains
the explicit smoothing requirement of the existing multi-asset API.

## Scope and limitations

Existing one-factor HW, rough-HW, deterministic-rate one-/two-factor LSV and
their JSON/request conventions retain their behavior. Multi-currency/FX,
multiple rate factors, correlation/model-parameter
Greeks, sticky-strike/sticky-delta smile dynamics, seasoned products, American
exercise and continuous barriers are outside this extension. Target construction
and particle/bandwidth/time-step refinement remain product-specific acceptance
work; the finite algorithm is defined by the contracts above.

## References

- [Bergomi, *Smile Dynamics II*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1493302), for the one-/two-factor Bergomi volatility factors and joint-driver correlations.
- [Hull and White, *Pricing Interest-Rate-Derivative Securities*](https://doi.org/10.1093/rfs/3.4.573), for the one-factor stochastic-rate model and curve-fitting shift.
- [Fries, *A Short Note on the Exact Stochastic Simulation Scheme of the Hull-White Model and Its Implementation*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2737091), for exact rate and integrated-rate innovations.
- [Cozma, Mariapragassam and Reisinger, *Calibration of a Hybrid Local-Stochastic Volatility Stochastic Rates Model with a Control Variate Particle Method*](https://arxiv.org/abs/1701.06001), for hybrid stochastic-rate/LSV calibration.
- [Guyon and Henry-Labordère, *The Smile Calibration Problem Solved*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032), for particle leverage calibration and the affine-dividend extension.

The public marginal `mean_discounted_normalized_equity` diagnostic is
`E[Dbar*F/S0]` and starts at one. Internal residual/quote states use initial-Spot
units; the diagnostic retains its unit-forward convention.
