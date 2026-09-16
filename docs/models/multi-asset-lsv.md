# Multi-asset Bergomi LSV

This extends the [multi-asset API](multi-asset.md) with particle-calibrated
one- or two-factor Bergomi LSV. The [two-factor extension](bergomi-two-factor-lsv.md)
defines its parameters, mixed-factor APIs and marginal correlation blocks. BS, LV and LSV assets may coexist in one Basket,
Worst-of or unseasoned Autocallable contract. The three-crate structure,
single-asset APIs and stable JSON schemas are preserved. Rates remain
deterministic; all assets use one currency and the same discount curve.

## Inputs and diagnostics

Rust: `MultiAssetPricingPlan::compile_with_lsv` takes the existing eight
`compile` arguments, then `Vec<Option<MultiAssetLsvConfig>>` and optional
`Vec<Vec<Vec<f64>>>` full driver matrices. Each configuration has a validated
`Bergomi1Factor` and `LsvParticleConfig`. The corresponding model must be a
`ModelSpec::LocalVolatility`; its effective variance grid is the target.
`None` retains that asset's BS/LV model. The original `compile` delegates with
no LSV configurations and preserves its existing fingerprints and random draws.

The following constructor description is for the original one-factor configuration.
Python adds optional keywords `lsv_configs` and `driver_correlations` to
`MultiAssetPlan.compile`. Supply one `MultiAssetLsvConfig` or `None` per asset.
The immutable configuration requires `mean_reversion`, `vol_of_vol`,
`correlation`, `particle_count`, `calibration_seed`, `log_bandwidth` and
`minimum_effective_samples`. `retain_reverse_trace=False` is the price-only
default; set it to `True` for `evaluate_aad`.

The plan exposes `random_factor_count`, `lsv_calibrations`,
`lsv_driver_correlations` and `lsv_transition_covariances`.
Python calibration snapshots expose the refined time/log grids, squared
leverage, conditional effective sample sizes, donor indices, fallback flags,
row diagnostics and calibration seed. The particle mean is in the normalized
martingale coordinate, starting at one. Copies do not expose the retained
particle reverse trace. Rust also exposes the full calibrated objects and
the input correlation factors' PSD/rank diagnostics.

## Marginal calibration, carry and dividends

For each LSV asset, define the deterministic no-jump forward
`F_i(t) = S_i(0) Dq_i(t) / Dr(t)` and normalized martingale `m_i=f_i/F_i`.
The simulation and calibration use

`dm_i/m_i = L_i(t,m_i) exp(nu_i X_i) dW_i`,
`dX_i = -k_i X_i dt + dV_i`, `m_i(0)=1`, `X_i(0)=0`.

The existing [particle calibration and discrete VJP](lsv-numerical-contracts.md)
are reused in `x=log(m_i)`. Every original target time knot, contractual
observation, in-horizon dividend and correlation date contributes to the common
grid before maximum-step subdivision. Interpolate each original effective LV
target onto that grid, calibrate once per asset, then reuse the immutable
surface for independent valuation draws. Calibration uses
`RandomDomain::LsvCalibration`; valuation uses `RandomDomain::Valuation`.
Seeds are explicit per asset; reusing a seed is allowed. Both configurations
and realized squared-leverage values enter the fingerprint.

The physical price is `S_i(t)=A_i(t) S_i(0)+B_i(t) F_i(t) m_i(t)`.
This retains the existing fixed-cash carry and pre/post-dividend rules. No
future-cash reserve is introduced, no payout changes the martingale or OU
factor, and no payout consumes an extra random coordinate. Nonfinite or
nonpositive physical prices are errors. There is no new variance clipping,
physical-price floor or path resampling. `lsv_leverage_boundary_counts` reports
mean flat-space leverage lookups, separately from LV boundary counts.

## Full driver correlations and exact joint transitions

For the one-factor configuration with N assets and M LSV assets, Brownian order is
`[W_1, ..., W_N, V_lsv1, ..., V_lsvM]`, with LSV assets in market order.
`driver_correlations` contains one (N+M)-square matrix for **each** existing
spot-correlation date, including dates beyond the pricing horizon. The dates
are supplied by `CorrelationSchedule`; add a dated entry with the same spot
block when only volatility correlations change.

All full matrices use the spot schedule's six explicit tolerances and the
existing unpivoted PSD-aware Cholesky validation. After canonicalization the
leading spot block must equal the spot schedule exactly, and each own
`corr(W_i,V_i)` must equal the calibration factor's rho exactly. Invalid
dimensions, nonfinite values, inconsistent marginals and indefinite matrices
are rejected, including outside the pricing horizon. Model parameters and
these correlations are held fixed in AAD.

If full matrices are omitted, the specified default coupling is
`dV_i=rho_i dW_i+sqrt(1-rho_i^2) dZ_i`, with Z independent of all W and each
other. Thus cross spot/vol correlation is `R_ij*rho_j`, and cross vol/vol
correlation is `rho_i*rho_j*R_ij`. Each own volatility variance is one.
This default can be replaced by an admissible full matrix.

The interval uses the correlation effective at its **left** endpoint. For
`h=t_next-t`, `I(a,h)=(1-exp(-a*h))/a` and `I(0,h)=h`, the covariance of the
spot Brownian increments and OU innovations is

- `Cov(dW_i,dW_j)=R_WW(i,j)*h`;
- `Cov(dW_i,OU_j)=R_WV(i,j)*I(k_j,h)`;
- `Cov(OU_i,OU_j)=R_VV(i,j)*I(k_i+k_j,h)`.

The OU update is `X_next=exp(-k*h)*X+OU`. The stock update uses the factor and
leverage at the interval start. In particular, correlating only the interval
stock normals and then independently supplying each OU residual would lose
the shared intra-interval kernel covariance for positive mean reversion.

Normalize the joint covariance, factor it with the same explicit PSD
tolerances, and scale the OU innovations back. Stable exponential integrals
handle the Brownian/small-k limit. A normalized analytic kernel overlap can
exceed its Cauchy-Schwarz bound of one only by floating-point roundoff: values
within 32 machine epsilons are set to one; larger violations fail. This
correction never changes supplied Brownian correlations. PSD pivots use the
declared tolerances, with no jitter, spectral clipping or pivot reordering.

Random coordinates remain factor-major, with all stock blocks first. Bridge
each independent block before the joint Cholesky map; antithetics negate all
N+M blocks. Retain every column, including singular/unused columns and factors
with zero vol-of-vol. QMC dimensions therefore equal `(N+M)*steps` and use the
existing dimension limit. Perfect Brownian correlation may still leave a
nonzero OU residual when k>0 because its kernel differs from a flat Brownian
increment. Same-platform worker replay retains fixed reduction blocks.

## Risk and uncertainty

`evaluate_aad` returns per-asset Delta and optional full cross Gamma with the
existing smoothing and contractual-reference conventions. Since the target
and calibration use log(f/F), a Spot change leaves normalized particles,
leverage and pricing m unchanged. Holding cash payouts fixed gives
`dS_i/dS_i(0)=B_i F_i m_i/S_i(0)`. The same exact scaling identity implements
common-random-number central differences of Delta for cross Gamma.

For an LSV asset, `risk.lsv_local_variance` contains `time_nodes`,
`log_moneyness_nodes`, row-major `node_adjoints`, optional `standard_errors`
and the method `lsv-discrete-particle-vjp-v1`. Reverse the payoff, transpose
`B_i F_i` into normalized state seeds, run the existing LSV path reverse,
reverse the particle calibration, then transpose time refinement back to the
original **effective target variance** grid. This includes movement of all
calibration particles. Support/donor choices have the existing piecewise
differentiable convention; changes across selection boundaries are nonsmooth.

BS Vega and ordinary LV node adjoints retain their existing fields. For LSV,
`bs_vega` is absent and `local_variance` is empty; the explicit LSV result
distinguishes recalibrated target risk from direct LV risk.

The calibration VJP is linear in its incoming leverage seeds. It is applied
once to the MC mean or once per independent RQMC scramble mean. RQMC returns
target-bucket standard errors across those transformed scramble means. MC
returns target-bucket adjoints with `standard_errors=None`: no covariance
matrix or path-by-path calibration VJP is retained to estimate those errors.
MC still reports the usual pair-level price, Delta and Gamma standard errors.
All valuation errors are **conditional on the realized calibrations**; they
exclude calibration noise, time/particle/kernel bias and support uncertainty.

## Verification and boundaries

Rust regressions cover nonflat LV limits, exact covariance against independent
Simpson integration on a dated grid, all 18 target buckets against full
recalibration, fixed-cash/proportional carry, Spot and every cross Gamma,
mixed BS/LV/LSV, MC trace/error semantics, singular drivers, worker replay and
invalid/future input rejection. Python tests cover immutable snapshots,
recalibrated AAD, mixed models, Worst-of, memory Autocallable smoothing and a
two-step exchange price checked with four-dimensional Gauss-Hermite quadrature
and an analytically integrated conditional final step. This is an independent
pricing check conditional on the finite calibrated leverage surface.

These establish the implemented finite-grid algorithm. Production acceptance
still requires particle/bandwidth/step/seed refinement for the intended smiles
and products. The separate [Bergomi + HW extension](bergomi-hull-white.md)
adds a shared stochastic rate, paired target calibration, market-IV VegaKT from
retained quotes and initial-curve risk. Supplying an LV reporting basis alone
does not enable market-IV risk. The [rough-LSV extension](multi-asset-rough-bergomi.md)
adds non-Markov assets through the same HW adapter. Model-parameter and
correlation Greeks, multiple currencies, seasoned products, American exercise
and continuous barriers remain outside these extensions. Existing single-asset
adapters retain their own capabilities and conventions.

See the [runnable example](../../examples/python/multi_asset_lsv.py),
[Rust regressions](../../crates/pricing/tests/multi_asset.rs) and
[Python verification](../../tests/python/test_multi_asset_lsv.py). PR CI records
native Linux/macOS/Windows validation for the published revision.

Local verification of the initial one-factor revision `11fbd20` on 2026-09-14,
from main `b8b5d62`: Rust 1.98.1 on Linux
x86-64 passed all **461** default workspace tests and all **5** statistical
acceptance tests. The installed release CPython 3.12 wheel passed the strict
metadata/stub/runtime contract, all **69 Python tests**, and the three-product
LSV example (also added to the wheel CI gate). Formatting, all-target/all-feature
Clippy with warnings denied, Rust API docs, all three reference fixtures,
schemas, Markdown links and dependency-direction checks passed locally.

## References

- [Bergomi, *Smile Dynamics II*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1493302), for multi-factor stochastic-volatility factors and their joint correlations.
- [Guyon and Henry-Labordère, *The Smile Calibration Problem Solved*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032), for marginal local-stochastic-volatility calibration with particles.
- [Jourdain and Zhou, *Existence of a calibrated regime switching local volatility model and new fake Brownian motions*](https://arxiv.org/abs/1607.00077), for the conditional-expectation and interacting-particle calibration context.
- [Margrabe, *The Value of an Option to Exchange One Asset for Another*](https://doi.org/10.1111/j.1540-6261.1978.tb03397.x), for the exchange-option reference used by the mixed-asset validation.
- [Glasserman, *Monte Carlo Methods in Financial Engineering*](https://doi.org/10.1007/978-0-387-21617-1), for multi-asset Monte Carlo, antithetic sampling and common-random-number checks.
