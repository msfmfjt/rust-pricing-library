# Multi-asset pricing v0.1

## Scope and entry points

This experimental extension adds `pricing::multi_asset::MultiAssetPricingPlan`
and Python `MultiAssetPlan`. It accepts an ordered vector of existing equity
markets, an equally ordered vector of BS or Local Volatility models, a dated
correlation schedule, a multi-asset product, and the existing MC/RQMC and
execution policies. Different assets may use different BS/LV models, curves
for continuous dividends, and cash/proportional dividend schedules.

The [Bergomi LSV extension](multi-asset-lsv-v0.1.md) adds optional per-asset
particle calibration of those LV targets, full spot/volatility correlations,
joint OU transitions, Delta/cross Gamma and recalibrated target-grid AAD.
The [common Hull–White extension](bergomi-hull-white-v0.1.md) adds stochastic
rates to BS and one-/two-factor Bergomi LSV, with paired target AAD, market-IV
VegaKT and initial-curve risk. The BS/LV contracts below describe the original
deterministic-rate interface when no LSV configuration is supplied.

All assets share one currency and exactly one discount curve (including its
identity). Three crates remain: `pricing-python -> pricing -> pricing-numerics`.
The implementation reuses the existing payoff compiler/reverse tape, normal
transforms, Brownian bridge, Local Volatility path/reverse, statistical
estimators and fixed-block Rayon executor. It does not alter the single-asset
request or its v1/v2/v3 JSON schemas. Multi-asset objects currently use native
Rust/Python APIs; they cannot be serialized through `PricingRequest.to_json()`.

See the runnable [Python example](../examples/python/multi_asset.py),
[public integration tests](../crates/pricing/tests/multi_asset.rs), and
[Python tests](../tests/python/test_multi_asset.py).

## Contracts and cash flows

`MultiAssetProduct.basket` computes `max(side*(sum(w_i*S_i/c_i)-K), 0)`
multiplied by notional. Weights are finite, signed and never normalized.
Scales are explicit positive contractual constants. A raw-Spot component
has scale 1. `worst_of` uses `min(S_i/R_i)` with explicit positive reference
levels. Reference levels and scales never move when Spot is bumped.

Product components, market/model vectors and correlation rows must have the
same explicit underlying order. Duplicate IDs, missing components and order
mismatches are errors. There is no silent reorder or correlation fallback.
The generic Rust `from_graph` constructor accepts the existing source graph
with one payment date per output. Each cash-flow dependency is checked against
its payment date, so a payment cannot depend on a later observation. Pre- and
post-dividend observations are available in the Rust graph API. Every payment
uses its own deterministic discount factor, including payment lags.

Autocallables use Worst-of performance, optional call barriers per observation,
independent coupon barriers, nominal coupon amounts, additional call coupon
amounts and explicit memory termination (`pay` or `forfeit`). On each date the
coupon is evaluated first; a successful coupon releases accumulated nominal
memory. Next the call is evaluated, remaining memory is treated according to
`on_autocall`, and notional plus the additional call coupon is paid on a call.
The active state masks all later cash flows. At maturity an active contract
pays notional if `W >= final_barrier`, otherwise `notional*W`, then applies
`on_maturity` to unpaid memory. Exact barriers are inclusive. An observation
on maturity occurs before maturity redemption. Previously determined coupons
with later payment dates remain payable after a subsequent call.

This release handles unseasoned schedules, including unknown observations on
the valuation date. It requires at least one future observation. Past fixings,
active autocall snapshots, already called trades and fully determined trades
are not supported at this entry point.

## Correlation and random coordinates

`CorrelationTermStructure` (Python `CorrelationSchedule`) takes ordered
`(effective_date, matrix)` entries and six explicit nonnegative finite
`tolerances`: symmetry, diagonal, absolute/relative PSD, and absolute/relative
zero-pivot tolerances. Dates must increase strictly. An entry must cover the
valuation date; the last entry is held flat. A change owns intervals starting
on its date; the interval ending on that date retains the preceding matrix.
Every matrix is validated, including entries outside the simulated horizon.

The numerical crate supplies scalar unpivoted PSD-aware Cholesky. It checks
shape, finite entries, symmetry and unit diagonal, canonicalizes permitted
roundoff by symmetric averaging and exact unit diagonals, and requires exact
correlation bounds [-1,1] after canonicalization. A numerical zero pivot must
have consistent near-zero residuals. No jitter, projection, eigenvalue clipping
or rank compression occurs. Perfect positive and negative correlations are
supported. Raw/canonical matrices, lower factors, pivots, zero pivots, rank,
thresholds and adjustments are inspectable in Rust; the Python schedule exposes
matrices, factors, ranks, pivots and maximum adjustments.

The common event grid contains all observation dates, in-horizon correlation
changes, each asset's in-horizon dividend events and each LV grid's in-horizon
time knots. An explicit `maximum_step` controls subdivision. LV grids must
start at zero and cover the latest observation; their stored grids are not
resampled. Asset models need not have identical LV grids.

Independent normal coordinates are **factor-major**, with one full time block
per underlying. Brownian bridge is applied separately to each independent
factor. The per-interval correlation factor is then applied to those increments.
Singular periods retain all columns and the same random layout. Antithetics
negate all correlated increments together. RQMC keeps the existing Joe-Kuo
21,201-dimension limit and requires at least two independent scrambles.
Pseudo-MC requires at least two independent units.

## Coordinates and dividends

Each asset evolves continuous equity `f`, initialized at its full Spot, with
forward normalizer `F_f(t)=S0*Dq(t)/Dr(t)`. BS uses exact lognormal interval
transitions; Local Volatility reuses Log-Euler in `x=log(f/F_f(t))`. At every
node, physical Spot is reconstructed using the existing paid-cash affine map
`S=A(t)*S0+B(t)*f`. The cash offset carries between events at deterministic
r-q. Cash remains fixed under risk shocks; future payouts are not reserved.

Ex-dates do not add separate pre/post random draws. At an event, both
observations use the same continuous state and their respective affine maps.
An ordinary observation on an ex-date uses the post-dividend Spot. All simulated
pre/post physical nodes must be positive and finite, including Spot-bumped
Gamma paths. Failure returns an error rather than flooring, dropping or
resampling a path. This continuous-equity volatility convention is the same
as the deterministic affine convention in the single-asset library; it is
not a volatility model directly on physical Spot after cash payouts.

## Risk and uncertainty

`evaluate()` returns price and uncertainty. `evaluate_aad()` uses the same
priced graph and returns the following in canonical asset order:

| Output | Definition |
| --- | --- |
| Delta | dPV/dS0, cash/scales/references and log-forward LV grid fixed |
| Scaled Delta | Delta times 0.01*S0 |
| BS Vega | dPV/dsigma_i; volatilities of other assets fixed |
| Scaled BS Vega | BS Vega times 0.01 |
| Local variance nodes | dPV/d(effective stored variance node), with time/x axes |
| Optional Gamma matrix | Gamma[i,j] = central difference of Delta_i under Spot_j bumps |

The payoff reverse pass seeds observations on every asset/date. BS volatility
sensitivities reverse its lognormal factors; LV adjoints use the existing
multi-node path reverse and interpolation transpose. Spot and its forward
normalizer scale together with fixed log-forward LV coordinates, so `f/S0`
is unchanged and the affine physical-Spot tangent is `B*f/S0`. This includes
canceling the Spot dependence of normalized fixed-cash amounts. Gamma reuses
these exact scaling identities and common paths, which is equivalent to
recompiling those Spot bumps under this convention. The relative bump is
explicit, positive and less than 1; the unsymmetrized matrix records the
finite-difference result without imposing symmetry.

Exact Basket and Worst-of risk use the existing almost-everywhere max/min
branch derivatives. At degenerate ties these are branch conventions, not a
claim of differentiability. Optional compact C2 smoothing is applied through
the existing graph opcodes. For Basket its width is in basket units; for
Worst-of and Autocallable it is in performance units, before notional scaling.
Exact discontinuous Autocallable price is supported, but AAD requires explicit
smoothing of the indicators. The smoothed price and risk use one surrogate;
its estimate does not include smoothing bias.

MC errors use independent sampling units (antithetic pair averages when
requested). RQMC errors use independent scramble means, never individual
Sobol paths. Each risk/Gamma entry has its own estimate, standard error,
95% normal interval and effective-unit count. These statistical errors do
not include time discretization, grid/interpolation, smoothing or finite-bump
bias. Local variance outputs are sensitivities to *effective* stored nodes,
not derivatives through floor/cap repair. Mean flat-wing interpolation counts
are returned per asset. Full cross-risk covariance is not returned.

The plan and result fingerprints include explicit model/market inputs,
source and priced tape fingerprints, curves, payouts, correlation matrices
and tolerances, random configuration, grid, reduction policy, and requested
risk/bump convention. Worker count is reported separately and does not change
the fingerprint or numerical results. QMC checksums are retained. Floating
point replay is promised only on the same platform/toolchain with a fixed
reduction-block size.

## Current boundaries and validation

[Rough-LSV with common HW](multi-asset-rough-bergomi-v0.1.md) is also available.
[Particle Local Correlation](local-correlation-v0.1.md) adds state-dependent
PSD correlations and recalibrated volatility risk for deterministic-rate BS/LV.
Its normalized-basket target and two independent endpoint shock blocks have
their own explicit contracts; the fixed-correlation replay described above
continues to apply when this adapter is absent.
Multiple currencies/FX, correlation Greeks, American
exercise and continuous barrier monitoring are not connected to this API.
Initial-curve risk and market-IV VegaKT are available through the separate
[HW mode](bergomi-hull-white-v0.1.md); the deterministic-rate path retains
effective local-variance adjoints. Supplying an LV reporting basis alone does
not enable VegaKT. Sticky-strike/sticky-delta conventions remain future work.
The existing single-asset adapters retain their own behavior.

The new integration suite checks BS analytical limits, perfectly correlated
duplicates, Margrabe's exchange-option formula with dated correlations and
both bridge settings, an independent three-asset product moment, all cross
Gammas against recompiled Spot bumps, BS Vega and all nine nonflat LV nodes
against CRN finite differences, the BS/flat-LV chain rule, fixed references,
cash carry and event collisions, memory and termination cash flows, exact vs
smoothed Autocallable risk, invalid input rejection and worker replay.
Numerical unit tests cover singular/full-rank factor reconstruction and
rejection of inconsistent singular, indefinite or out-of-range correlations.
Python tests exercise the installed extension, immutable snapshots, typed
results, AAD and validation. Existing CI runs these suites on its three Rust
platforms and both supported native wheels. The existing single-asset replay
goldens and schema fixtures are unchanged.

### Local verification on 2026-09-14

Baseline: main `86c6c9992b2970e3abdbef23c5717b48c407f72e` (includes the latest
paid-cash Hull-White/dividend change). Implementation commit:
`db6dd224abb610e2268ef7351c64bf9ea5a9da5f`; subsequent documentation-only changes
record this evidence. Dedicated branch: `feat/multi-asset-pricing`.

- Rust 1.98.1, Linux x86-64: default workspace **455 passed**, including all
  **14** new public multi-asset integration tests, **3** new correlation unit
  tests and **3** existing Python-extension native tests.
- All **5** normally ignored statistical acceptance tests passed explicitly.
- The all-feature workspace excluding the Python extension passed before the
  final two dividend regressions; the complete default run above includes them.
  Final Clippy covers all workspace targets and all features with warnings denied.
- A release CPython 3.12 wheel was built, installed into a clean environment,
  and passed its metadata/stub/runtime contract and all **65 Python tests**.
  The new installed-wheel example ran all three products and their risk outputs.
- Formatting, Rust API documentation, 67 LV / 114 path-dependence / 61
  early-exercise reference checks, schemas, dependency direction, Markdown
  links, source archive contents and whitespace checks passed.

Native Python unit-test linking initially failed because the relocated local
Python runtime reports `/install/lib` and has a broken unversioned libpython
symlink. A temporary local link plus linker/runtime library paths resolved it;
no repository dependency, CI gate or test was changed to work around it.

Remote CI and native macOS/Windows replay results are tracked on the pull
request for this branch; local results above do not assert native-platform
acceptance.
