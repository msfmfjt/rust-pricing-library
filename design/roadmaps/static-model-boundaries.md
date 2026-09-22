# Static model boundaries and extension roadmap

Date: 2026-09-21. Decision: [ADR 0011](../adr/0011-static-model-boundaries.md).
Baseline: [escrowed dividend decision](../adr/0010-remove-paid-cash.md), PR #72.

The goal is to localize a future volatility/rate model addition to its numerical
implementation, capability registration, adapters and tests. Existing models
must retain their prices, risks, random paths and public interfaces. This is an
incremental implementation plan, not a claim that all combinations are generic.

## Current extension points and constraints

| Concern | Current implementation | Constraint on extension |
| --- | --- | --- |
| Markov volatility | [BergomiDynamics](../../crates/pricing/src/models/bergomi_dynamics.rs), [LSV execution](../../crates/pricing/src/engine/processes/lsv.rs) | Typed state/transition and static dispatch share 1F/2F kernels. Internal innovation views now distinguish normals from OU increments; the legacy public array adapters and Bergomi-specific errors remain. |
| Rough volatility | [RoughKernel](../../crates/pricing/src/engine/processes/rough_lsv.rs), [hybrid rough driver](../../crates/pricing/src/engine/processes/hull_white/rough.rs) | Grid-dependent Volterra weights and history preparation are not a Markov scalar-step operation. |
| Rates and joint increments | [HW process](../../crates/pricing/src/engine/processes/hull_white.rs), [2F/HW driver](../../crates/pricing/src/engine/processes/hull_white/two_factor.rs), [multi-asset drivers](../../crates/pricing/src/engine/multi_asset/hull_white/drivers.rs) | Rate state/integral evolution, discounting, bond-state exposure and Gaussian covariance now have separate internal boundaries; the HW moment implementation and hybrid composition remain concrete. |
| Multi-asset selection | [LSV enums](../../crates/pricing/src/engine/multi_asset/lsv_kernels.rs), [compiler](../../crates/pricing/src/engine/multi_asset/compile.rs) | Concrete enum variants are appropriate boundaries, but driver counts and offsets were reconstructed in several consumers. |
| Calibration and reverse | [HW calibration](../../crates/pricing/src/engine/calibration/hull_white.rs), [calibration reverse](../../crates/pricing/src/engine/calibration/hull_white/reverse.rs), [joint reverse](../../crates/pricing/src/engine/multi_asset/local_correlation/joint_reverse.rs) | Leverage moments, target density, dividend reserve and model-specific traces must remain paired with their exact reverse. |
| Configuration | [composition selection](../../crates/pricing/src/engine/multi_asset/composition.rs), [paired HW lowering](../../crates/pricing/src/engine/compile/hybrid.rs), [Python HW facade](../../crates/pricing-python/src/hull_white.rs) | Existing constructors adapt to private configurations. Product/grid checks and model-specific calibration remain staged to preserve the supported combinations and rejection order. |

The multi-asset rough path currently uses the shared HW adapter even with zero
rate volatility. Its two rate-related random coordinates remain allocated.
This roadmap does not silently replace that behavior with the standalone rough
path or change the fixed-random-number coupling.

## Compiled boundaries

| Boundary | Owns | Excludes |
| --- | --- | --- |
| Volatility kernel | Concrete state, transition/driver preparation, multiplier and its reverse | Payoffs, quote interpolation, dividend reserve funding and random-number generation |
| Rate kernel | Rate state/integral, discounting, conditional bonds and required joint innovation moments | Equity payoff interpretation and particle regression policy |
| Driver layout and covariance | Named innovation ranges, dimensions, correlation mapping, compiled covariance/loading and permutations | Generating a new random sequence or dropping zero-loading coordinates |
| Calibration and reverse capabilities | Target coordinate, required conditional moments, retained trace and matching VJP | Implicit fallback to another model or undocumented frozen calibration |
| Composition compiler | Supported combination validation, enum selection and concrete plan construction | Dynamic model lookup inside every simulation step |

Retain separate capabilities for exact Gaussian innovation covariance, bond
reserve exposure, history preparation, calibration reverse and path reverse.
A future non-Gaussian rate process will require an appropriate discretization
capability; it must not be forced into the existing HW Gaussian covariance API.

For new internal interfaces, prefer associated concrete state/transition/trace
types and caller-owned output storage. A compiled model enum can select a
monomorphized path/batch function. Mixed-asset plans may dispatch once per asset
path. Do not duplicate a generic calibration algorithm merely to change factor
count, and do not create one universal trait containing unsupported default
methods. Existing `BergomiDynamics` is the starting point, not an interface to
replace wholesale.

## First extraction: driver metadata

[DriverLayout](../../crates/pricing/src/engine/multi_asset/driver_layout.rs)
records volatility ranges and the base innovation count at compilation.
Consumers share it for:

1. MC/RQMC dimension validation in the multi-asset compiler.
2. `MultiAssetPricingPlan::random_factor_count()`.
3. Volatility-driver slices in deterministic-rate LSV pricing.
4. HW asset volatility offsets passed at compilation.

The layout preserves the order: spot coordinates, volatility coordinates in
asset order, rate state/integral, then rough newest-cell auxiliaries. These are
innovation coordinates; the Brownian correlation matrix can have a different
dimension. In particular, HW contributes one Brownian rate driver but two
joint transition coordinates.

Local correlation doubles the endpoint innovation blocks after its calibration
is attached. Calibration receives the undoubled base plan. The descriptor
therefore stores the base count, and dimension compilation takes an explicit
endpoint-block count. Caching a doubled count during the initial construction
would change the calibration contract.

This step does not generalize model-specific covariance matrices or remove all
fixed arrays. It adds one boxed array of ranges per compiled plan, with no new
path or timestep allocation. The existing QMC bridge-rank ordering, factor-major
MC/unbridged ordering, covariance summation and fingerprint implementation are
unchanged.

## Second extraction: typed volatility inputs

The private [volatility inputs](../../crates/pricing/src/models/volatility_inputs.rs)
distinguish independent orthogonal standard normals, already correlated OU
increments, and full Brownian/newest-cell histories. Step views borrow their
factor-major buffers with a stride; they do not copy or allocate driver arrays.
S1's `DriverLayout` selects the asset's volatility range before this adapter.

The existing sealed supertrait now owns the internal Markov step and pullback
capability. The 1F and 2F kernels return concrete one- and two-coordinate arrays.
The shared calibration and path loops no longer synthesize two-element padding
or branch on `FACTOR_COUNT == 2`. Calibration draws the same factor-major Philox
coordinates through the concrete kernel's coordinate constructor. The public
`BergomiDynamics` signatures remain compatibility adapters, including ignored
1F padding and the historical two-element adjoint result. This remains a sealed
Bergomi capability; it is not an externally extensible, arbitrary-model API.

Path traces record whether inputs were independent normals or supplied OU
increments, so reverse applies the appropriate loading. Joint OU increments
have identity volatility loading and zero spot-to-volatility loading. Neither
covariance construction nor arithmetic ordering changes in this extraction.

Standalone rough calibration/pricing and HW rough pricing use the separate
`HistoryInnovations` preparation boundary. Existing nonuniform Volterra weights,
newest-cell conventions, compensated sums and memory ownership remain intact.
The HW and deterministic preparations remain distinct: their centring and the
H=1/2 input convention must not be silently identified. No Markov state-step
interface is imposed on rough history.

Focused tests compare typed paths against independently assembled legacy array
inputs, including NaN in unused 1F padding; check price-only buffer reuse; and
finite-difference every supplied OU and spot coordinate through path reverse.
Existing 1F/2F particle-reverse, rough H=1/2/finite-difference and multi-asset
sampling suites remain the wider regression gates. The 2F multi-asset flattening
allocation is unchanged and remains a later measured optimization.

## Third extraction: rate capabilities

The private [rate kernels](../../crates/pricing/src/models/rates.rs) separate
state evolution (`RateEvolution`), relative discounting (`RateDiscount`) and
optional exact Gaussian moments (`GaussianRateCovariance`). There is no trait
object or per-step model registry. `DeterministicRates` has unit state and no
innovations; deterministic forward carry and discount curves stay in the market
layer. A zero-volatility HW configuration continues to use its Gaussian state,
conditional bonds and all allocated random coordinates.

`GaussianRateStep` owns the centered rate factor, its accumulated integral and
the step-integral exposure used by equity carry. The single-asset HW kernel and
joint local-correlation path now use the same advance operation. Local
correlation's reverse uses its matching state/innovation pullback, retaining the
original addition order of equity and accumulated-integral seeds. The equity
kernel retains joint loading, volatility evolution and state validation; it no
longer stores the whole HW transition just to read its rate/volatility decays.

The [HW adapter](../../crates/pricing/src/models/rates/hull_white.rs) exposes named
rate-state/integral covariance and unit-correlation OU cross moments to the
multi-asset driver. Its numerical source remains `HullWhite1Factor::transition`,
including every volatility breakpoint. Joint correlation validation, rough
power/rate quadrature, PSD factorization, suffix permutations and dimensions are
unchanged. Other rate models need not implement this Gaussian capability.

[Bond-state exposure](../../crates/pricing/src/models/rates/bonds.rs) owns the
exponential loading and its first/second state derivatives. Escrowed dividend
nodes retain cash/proportional event policy, reserve sums, curve factors and
coefficient/market adjoint accumulation. Their amount/duration fingerprint
bytes are unchanged. Conditional payment discounting uses named bond moments;
curve validation still occurs between transition compilation and total-variance
evaluation, as before.

Preserve both existing arithmetic conventions: standalone HW valuation uses
relative discount times relative bond (two exponentials); multi-asset cashflows
use their compiled combined exponential. Particle weights and their reverse
share `GaussianDiscount` with exactly the previous half-variance arithmetic.
These expressions are not reassociated merely because they are algebraically
equivalent. Curve adjoints remain derivatives of the existing deterministic
curve factors, with HW parameters fixed.

New tests compare a full hybrid path against the frozen pre-extraction
recurrence bit for bit on a nonuniform grid crossing rate-volatility knots.
They also cover rate-state/increment pullbacks, conditional discounts, bond
reserve derivatives, the Ho-Lee limit and the distinction between deterministic
carry and a zero-volatility HW model conditioned on a nonzero rate state.
Existing escrowed, rough/2F HW, multi-asset/local-correlation, curve-AAD and
VegaKT tests remain required. This stage does not register a new rate model or
make the complete HW calibration algorithm generic; capability selection and
composition lowering are the following stages.

## Fourth extraction: explicit reverse capabilities

The private [calibration reverse](../../crates/pricing/src/engine/calibration/capabilities.rs)
and [recorded-path reverse](../../crates/pricing/src/engine/processes/capabilities.rs)
interfaces are optional capabilities. Their associated adjoint types retain the
existing distinction between deterministic-rate target variance, HW paired
variance/density/dividend exposure, and coupled basket/constituent targets.
They delegate to the existing numerical reverses and do not freeze calibration
or substitute zero derivatives for an unavailable reverse.

The deterministic-rate LSV core separates price evolution from path recording.
Its [price-only and recalibrated modes](../../crates/pricing/src/engine/risk/lsv/evaluation.rs)
are selected at the public API boundary using static dispatch. A price-only
model needs neither reverse capability nor a recorded path type. Recalibrated
risk requires both capabilities in its Rust bounds. MC/RQMC coordinate order,
antithetic accumulation, reduction order and per-scramble calibration pullbacks
remain the same. There are no new path allocations or trait objects; additional
monomorphization has not yet been benchmarked.

An implemented reverse still needs its realized primal. Validation uses the
actual retained calibration trace; HW also checks the existing primal
fingerprint before starting pricing paths. Direct public reverse methods retain
their existing seed validation and error order. Multi-asset validation preserves
the local-correlation, marginal-trace, Gamma and payoff-smoothing check order
and existing missing-trace error messages. Local correlation retains its exact
projection-transition rejection; paired target and supported model-combination
checks remain with the existing compilers.

The public two-stage facades compile a price plan before the caller requests
AAD. Consequently, missing traces or unsuitable payoffs continue to allow
price-only compilation and are rejected when the risk operation is selected,
before any pricing paths or risk workspace are created. This is request
preflight, not an unconditional rejection in the price constructor. A future
model without a reverse cannot be registered with the recalibrated core unless
it supplies the required implementation; S5 will lower composition requests
through these existing boundaries.

Focused tests use adapters around an existing numerical model with no reverse
implementations, compare price results under MC/RQMC, preserve missing-trace
and payoff error precedence, and reject a modified HW primal during preflight.
Zero vol of vol still requires the requested calibration trace. Existing
1F/2F/rough recalibration, HW curve-AAD/VegaKT and coupled local-correlation
finite-difference suites remain the numerical regression gates.

## Fifth extraction: composition through compatibility adapters

The private [multi-asset composition](../../crates/pricing/src/engine/multi_asset/composition.rs)
collects the marginal calibration settings, supplied Brownian correlations,
explicit deterministic/shared-HW rate choice and optional local-correlation
calibration. All existing multi-asset constructors lower into this descriptor;
the one-factor constructor retains its conversion to the mixed-factor input.
It selects the driver compiler and one concrete marginal adapter per asset.
The resulting enum is consumed during compilation; existing plan storage,
path dispatch and fingerprint generation are unchanged. There is no serialized
composition format, public registry or runtime model lookup.

Validation remains staged, in the original observable order:

1. Market/model dimensions, configuration count, rough/HW restriction and
   suitability of each marginal's target model.
2. Product/underlying/currency/curve checks, contractual grid and sampling limits.
3. Joint driver dimensions and covariance consistency.
4. Each asset's dividend plan, paired target and numerical calibration.
5. Conditional payment discounting, then optional joint local-correlation
   calibration against the undoubled base plan.

The descriptor does not eagerly reject a paired target before the product or
driver checks that historically preceded it. AAD trace/payoff preflight remains
the separate S4 request boundary. These stages preserve existing error variants
and messages, including deliberately different single-asset round-off-tolerant
and multi-asset exact target-grid comparisons.

The shared [HW lowering](../../crates/pricing/src/engine/compile/hybrid.rs) binds
the target, volatility factor, rates, correlations and particle settings for
calibration. Its result retains the factor with the realized calibration and
selects the matching pricing-volatility enum and leverage surface together.
Both single-asset and multi-asset HW LSV use this boundary. In particular, 2F's
second volatility/rate correlation and rough history convention survive the
conversion. Existing numerical calibration, refined target/quote reverse,
dividend reserves and multi-asset diagnostic normalization stay in their
original adapters. Direct BS/rough-HW and standalone deterministic LSV keep
their existing concrete compilers; this is not a universal financial-model API.

The multi-asset marginal support contract is unchanged:

| Rate configuration | Marginal input | Result |
| --- | --- | --- |
| Deterministic | BS or LV without LSV settings | Existing direct path |
| Deterministic | LV plus 1F/2F Bergomi settings | Variance-target particle calibration |
| Deterministic | Rough-LSV settings | Rejected; shared HW and a paired target are required |
| Shared HW | BS without LSV settings or paired target | Existing BS-HW path |
| Shared HW | LV plus 1F/2F/rough settings and matching paired target | Paired variance/density calibration |
| Shared HW | LV without LSV settings, missing/mismatched paired target, or target without settings | Rejected with the existing specific error |

Zero rate volatility retains the HW adapter and both rate innovation coordinates;
rough retains its additional newest-cell coordinate. Optional local correlation
still applies its existing joint feasibility checks; this table does not expand
that capability. Python signatures/stubs and the Rust public configuration types
remain compatibility adapters, with no wire, fixture or schema changes.

Focused public-API tests check MC/RQMC constructor equivalence for prices, AAD
and fingerprints; overlapping invalid inputs pin the validation order; and
BS/1F/2F/rough with zero-volatility HW retain their coordinate counts and equal
price/AAD values. Existing mixed-factor covariance, paired target, quote reverse,
cash-dividend, local-correlation and frozen Python replay suites remain the
broader regression gates. Timing and memory comparisons belong to S6.

## Ordered implementation stages

| Stage | Deliverable | Exit condition | Status |
| --- | --- | --- | --- |
| S0 | Current-code inventory, boundaries and compatibility contract | Responsibilities and baseline/deferred issues are explicit | Recorded here |
| S1 | Compiled driver metadata shared by dimension/count/offset consumers | Existing sampling tests, mixed-model factor counts, overflow tests and all existing regressions pass | Implemented; normal tests and both native wheel/replay jobs passed on PR #73; extended gates tracked separately |
| S2 | Typed innovation views for Markov kernels; retain a distinct history-preparation interface | 1F/2F use one calibration/reverse implementation; rough keeps its exact history scheme and allocation behavior | Implemented; normal tests and native wheel/replay jobs passed on PR #74; extended gates tracked separately |
| S3 | Rate evolution, discount/bond exposure and joint innovation capabilities | Deterministic/HW adapters reproduce existing paths, conditional discounts, reserves and curve adjoints | Implemented; normal/native wheel/replay and extended risk passed on PR #75; extended price gate tracked separately |
| S4 | Explicit calibration/path-reverse capability selection | Unsupported model capabilities fail typed composition; late AAD requests validate before path generation; existing recalibrated AAD/VegaKT gates pass | Implemented; normal/native wheel/replay and extended risk passed on PR #76; extended price gate tracked separately |
| S5 | Composition configuration lowered through existing public adapters | Rust/Python signatures, wire fixtures and supported-product matrix remain compatible | Implemented; normal/native wheel/replay and extended risk passed on PR #77; extended price gate tracked separately |
| S6 | Extension exercise and performance comparison | A test-only alternative implementation uses registration/adapters without editing shared calibration/payoff algorithms; representative timing/memory results are recorded | Implemented on PR #78; 54 native and 27 DHAT exact comparisons passed; normal/native wheel/replay and extended risk passed; [measured results and separate accuracy failures](../validation/model-boundary-extension.md) |
| S7 | Broad performance/memory measurement and targeted optimization PRs | Measured bottlenecks and numerical/reproducibility gates justify each optimization | First [coupled-HW scaling and allocation study](../validation/local-correlation-scaling.md) measured 44 native/seven DHAT pairs on PR #80; reverse-moment buffer reuse selected for a separate optimization; broader coverage remains |

Each stage is a separate reviewable change above its verified parent. If a
common interface requires a different numerical scheme, defer that combination
instead of mixing a model change into a structural refactor.

## Validation contract

- Compare the same escrowed requests, seed pairs, time grids, factor counts,
  reduction blocks and quote arrays before and after a structural change.
  Require exact same-platform replay where arithmetic is unchanged; do not
  replace it with a Monte Carlo error tolerance.
- Keep the [sampling-contract tests](../../crates/pricing/src/engine/multi_asset/path/sampling_contracts.rs)
  for leading terminal Sobol coordinates and unchanged MC/unbridged layout.
  Descriptor tests cover mixed BS/1F/2F/rough layouts, endpoint duplication,
  reserved coordinates, count overflow and dimensions exceeding `u32`.
- A synthetic three-volatility-factor descriptor checks that layout arithmetic
  has no 1F/2F assumption. It does not establish support for a new pricing model.
- Preserve [escrowed-default tests](../../crates/pricing/tests/hull_white_escrowed_default.rs),
  all existing model/correlation tests and the
  [recalibrated risk panel](../validation/extended-model-risk-accuracy.md).
  AAD traces, support donors, density terms and calibration derivatives must
  remain part of the comparison.
- Exercise zero rate/vol-of-vol limits without removing their allocated random
  coordinates. Include mixed factor counts, proportional/cash dividends,
  future-only dividends, shared HW and both local-correlation endpoints.
- Public stubs, Python wheel tests, JSON/replay fixtures, source archives and
  dependency direction retain their existing gates.

The inherited one-year 2F constituent timestep noise-gate issue remains deferred
in the earlier PR stack. Record it separately from new failures; do not widen
its 4 bp ensemble-SE threshold or mark the full parent stack accepted here.

## Performance evidence required for later stages

Use the same optimized build, machine, input requests, seeds and worker counts
for parent/candidate comparisons. Record calibration, price-only, AAD and
VegaKT separately for 1F/2F/rough, HW, mixed multi-asset and local-correlation
paths. Separate initial compilation/preparation from repeated valuation.

Measure allocations and peak resident memory as well as elapsed time. Vary
particles, time steps, assets and worker threads; retain compiler/target details,
warm-up policy and repeated measurements. Include executable size and compile
time where additional monomorphizations are introduced. Investigate a persistent
slowdown or memory increase before accepting a kernel abstraction; static
dispatch alone is not evidence of unchanged performance.

S1 retains existing path allocations. In particular, the current 2F innovation
flattening and rough convolution work are not optimized in this extraction.
Scratch reuse, SIMD, observation-slot changes and convolution acceleration need
their own measurements and subsequent PRs.

## Validation status of this change

The local environment has no Rust toolchain. The S6 extension tests, normal
Rust/native wheel/replay gates and extended risk passed in CI run 282. The
[S6 record](../validation/model-boundary-extension.md) pins the measured source,
54 exact native comparisons, 27 DHAT pairs and the timing/memory limitations.
Extended price acceptance still fails the inherited 2F ensemble-SE case and
a BS-HW reference-coordinate mismatch; both also occur in parent run 278.
These need separate follow-up and do not justify relaxed accuracy thresholds.
S7 will use independent workload sweeps and stack attribution before selecting
an optimization; the small S6 panel is not a general speed or memory guarantee.
