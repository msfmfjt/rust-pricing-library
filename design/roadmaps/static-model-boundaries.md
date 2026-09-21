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
| Rates and joint increments | [HW process](../../crates/pricing/src/engine/processes/hull_white.rs), [2F/HW driver](../../crates/pricing/src/engine/processes/hull_white/two_factor.rs), [multi-asset drivers](../../crates/pricing/src/engine/multi_asset/hull_white/drivers.rs) | Exact rate state/integral covariance and bond reserve diffusion are coupled to HW; scalar volatility callbacks are insufficient. |
| Multi-asset selection | [LSV enums](../../crates/pricing/src/engine/multi_asset/lsv_kernels.rs), [compiler](../../crates/pricing/src/engine/multi_asset/compile.rs) | Concrete enum variants are appropriate boundaries, but driver counts and offsets were reconstructed in several consumers. |
| Calibration and reverse | [HW calibration](../../crates/pricing/src/engine/calibration/hull_white.rs), [calibration reverse](../../crates/pricing/src/engine/calibration/hull_white/reverse.rs), [joint reverse](../../crates/pricing/src/engine/multi_asset/local_correlation/joint_reverse.rs) | Leverage moments, target density, dividend reserve and model-specific traces must remain paired with their exact reverse. |
| Configuration | [Python HW facade](../../crates/pricing-python/src/hull_white.rs), [multi-asset configuration](../../crates/pricing/src/engine/multi_asset/lsv.rs) | Constructors encode combinations. Keep adapters until a common configuration can represent every supported capability and rejection. |

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

## Ordered implementation stages

| Stage | Deliverable | Exit condition | Status |
| --- | --- | --- | --- |
| S0 | Current-code inventory, boundaries and compatibility contract | Responsibilities and baseline/deferred issues are explicit | Recorded here |
| S1 | Compiled driver metadata shared by dimension/count/offset consumers | Existing sampling tests, mixed-model factor counts, overflow tests and all existing regressions pass | Implemented; normal tests and both native wheel/replay jobs passed on PR #73; extended gates tracked separately |
| S2 | Typed innovation views for Markov kernels; retain a distinct history-preparation interface | 1F/2F use one calibration/reverse implementation; rough keeps its exact history scheme and allocation behavior | Implemented; native validation pending |
| S3 | Rate evolution, discount/bond exposure and joint innovation capabilities | Deterministic/HW adapters reproduce existing paths, conditional discounts, reserves and curve adjoints | Pending |
| S4 | Explicit calibration/path-reverse capability selection | Unsupported combinations reject at compilation; existing recalibrated AAD/VegaKT gates pass | Pending |
| S5 | Composition configuration lowered through existing public adapters | Rust/Python signatures, wire fixtures and supported-product matrix remain compatible | Pending |
| S6 | Extension exercise and performance comparison | A test-only alternative implementation uses registration/adapters without editing shared calibration/payoff algorithms; representative timing/memory results are recorded | Pending |
| S7 | Broad performance/memory measurement and targeted optimization PRs | Measured bottlenecks and numerical/reproducibility gates justify each optimization | Pending |

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

The local environment currently has no Rust toolchain. S2 Rust tests and
native formatting/build/Clippy checks must run in CI; S1 and historical escrowed
results do not certify the S2 source tree. Local documentation, source packaging and
Python public-stub checks are available. No runtime or memory improvement is
claimed until representative measurements are captured.
