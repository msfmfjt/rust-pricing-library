# ADR-0043: Measure-oriented analysis facade over pricing tasks

Status: Proposed
Date: 2026-11-16
Owners: TBD

Builds on [ADR 0028](0028-pricing-specification.md),
[ADR 0029](0029-product-graph-source.md),
[ADR 0030](0030-pricing-task-capability-report.md),
[ADR 0031](0031-typed-risk-parameters-python-errors.md) and
[ADR 0034](0034-compiled-task-reuse.md).

## Context

The library is meant for model validation and model-risk analysis: price one
contract under several models, compare their sensitivities, and explain the
differences with stated uncertainty. The numerical core supports that. The
public surface does not yet.

Measured on `main` at `2fadb73`:

- The Python module exposes 182 classes, 21 of them `...Plan` entry points.
  Changing the model changes the entry point, the argument list and the result
  type (`PricingResult`, `HullWhitePrice`, `LsvPrice`, `StochasticDividendPrice`,
  `MultiAssetPrice`, `BassEstimate`, ...).
- 44 public classes are model-specific risk results. The same quantity is read
  as `result.delta.raw.value` on one path and `risk.delta` on another.
- `PricingTask` (ADR 0030) is the only model-independent entry point, but its
  operations are three coarse selectors: `evaluate`, `aad` and
  `local_variance_risk`. What `aad` returns depends on the backend.
- In the [operation matrix](../../docs/library/pricing-tasks.md#operations-and-supported-compositions),
  four of eight source rows answer at least one operation with "use dedicated
  APIs" or "unavailable", and three more redirect an operation to another one. `PricingSource::capability` returns
  `DedicatedRiskApi` for those cases. Single-asset Gamma is rejected by the
  common operation and lives in the nested `RiskRequest`.
- The six direct rough-volatility families, parameter risk, correlation risk,
  Hurst risk, market-IV risk and all cross-currency products are reachable only
  through dedicated classes.
- Several dedicated compilers take a `PricingRequest` whose `model` field is a
  placeholder. `examples/python/rough_bergomi.py` passes
  `Model.black_scholes(0.2)` to price rough Bergomi.
- Products have four representations: fixed `Product` variants, `GraphProduct`,
  `Contract` and `MultiAssetProduct`.
- There is no library support for the analysis itself: comparing models,
  sweeping a parameter, or applying a market scenario. `scenario_pnl` exists
  only on the cross-currency basket.

The consequence for a model-risk user is that the question "what is the Vega
of this contract under each model I have" cannot be asked in one form. It has
to be rewritten per model, and the answers come back in different shapes.

Constraints that this decision must keep:

- No silent substitution. An unsupported request is an error, never a fallback
  to another model or an implicit finite difference (ADR 0030).
- Native results, diagnostics, fingerprints and wire documents already
  published stay valid.
- No change to sampling, reduction order, calibration or estimator arithmetic.

## Decision

Make `PricingTask` the documented front door for every model, and change what
a task asks for from a backend-defined operation to an explicit set of
**measures**. Dedicated plans stay as backends behind it.

### 1. Measures replace coarse operations

Add a `Measures` operation carrying an ordered, duplicate-free list of
`MeasureRequest`. A measure names a quantity by its financial meaning, not by
the algorithm that produces it.

```rust
#[non_exhaustive]
pub enum Measure {
    Price,
    SpotDelta,
    SpotGamma,
    CrossGamma,
    ParallelMarketIvVega,   // parallel shift of the retained market-IV surface
    VegaKt,                 // bucketed quote-node Vega
    LocalVariance,          // recalibrated target-grid nodes
    DiscountCurve,          // per log-discount pillar, and its DV01 projection
    DividendCash,           // per cash-dividend mean
    ModelParameter(ModelParameterId),
    Correlation(CorrelationId),
}

pub struct MeasureRequest {
    pub measure: Measure,
    pub method: MeasureMethod,
}

#[non_exhaustive]
pub enum MeasureMethod {
    /// The backend's own estimator: pathwise AAD, likelihood ratio,
    /// semi-analytic. Unsupported when the backend has none.
    Native,
    /// Central bump-and-revalue on common random numbers. Always explicit.
    BumpAndRevalue { size: BumpSize, calibration: BumpCalibration },
}

pub enum BumpCalibration {
    /// The source has no calibration step.
    NotApplicable,
    /// Reuse the calibration of the unbumped source on both sides.
    Fixed,
    /// Recalibrate on each bumped side with the same calibration seeds.
    Recalibrated,
}
```

There is no `ModelVega` measure. A model's volatility input is a constant for
Black-Scholes, a surface for local volatility and an initial forward-variance
curve for Bergomi and the rough families, so a sensitivity to "the model
volatility" is not one quantity across models. The cross-model volatility
measures are `ParallelMarketIvVega` and `VegaKt`, both defined on market
implied volatility:

- Constant-volatility models: the sensitivity to the single volatility, which
  is the flat implied volatility.
- Local volatility, Bass-LV and every LSV source that retains a market-IV or
  reporting-IV basis: the existing parallel implied-volatility risk.
- Sources that retain no market-IV surface (direct Bergomi and rough-family
  SV): `unsupported` with reason `no_market_iv_surface`. Their volatility
  inputs are differentiated as `ModelParameter` instead, and the report does
  not label that as Vega.

`ModelParameterId` and `CorrelationId` are new typed identifiers, one variant
per parameter a backend can differentiate. They extend the `RiskParameterKind`
classification of ADR 0031, so a parameter is identified by an enum, never by
a display string.

Rules:

- `Price` is always present in the result whether or not it is listed.
- `Native` never changes estimator. If a backend has no native estimator for a
  measure, the capability is `unsupported` with reason `no_native_estimator`,
  and the detail names `BumpAndRevalue` as the available alternative when the
  backend can re-evaluate under the bump.
- `BumpAndRevalue` is only ever chosen by the caller, and so is its
  calibration treatment. `BumpCalibration` has no default: a calibrated source
  must state `Fixed` or `Recalibrated`, an uncalibrated source must state
  `NotApplicable`, and any other pairing is a validation error. The report
  records the bump size and maps the choice to `CalibrationTreatment`
  (`FixedCalibration` or `Recalibrated`), with the convention text saying which
  inputs were held.
- `Recalibrated` bumps are part of this facade from its first release, because
  a fixed-leverage LSV sensitivity is not comparable with a local-volatility
  one and that comparison is the purpose of the library. They are limited to
  `SpotDelta`, `SpotGamma` and `ParallelMarketIvVega`, to sources whose
  calibration is deterministic for fixed seeds, and to a preflight that reports
  the number of calibrations and evaluations before any is run.
- The nested `RiskRequest` of a standard single-asset source is frozen: it
  keeps its current flags and meaning and gains no new ones. Under `Evaluate`
  it behaves exactly as today. A `Measures` operation requires that nested
  request to be empty and otherwise fails with `ambiguous_risk_selection`, so a
  task never selects risk in two places.
- The existing `Evaluate`, `Aad` and `LocalVarianceRisk` operations remain
  valid inputs with unchanged task fingerprints. They are defined as fixed
  measure sets so that old and new requests go through one code path.

### 2. Capability is reported per measure

`PricingSource::capabilities(&[MeasureRequest])` returns one
`PricingCapability` per requested measure, with the same three statuses as
today. `CompiledPricingTask` gains the matching method against the realized
plan.

Add `PricingSource::available_measures()`: the measures for which the source
is not statically `unsupported`, each with the methods that apply. This is the
discovery call a user makes first.

`DedicatedRiskApi` stops being returned for any measure that has an adapter
(section 3). It remains only for outputs that are intentionally outside the
common contract: dual bounds, VegaKT Gamma-chain audits and the width ladder.

The operation matrix in the task guide is generated from this code by a script
and checked in CI, so the published table cannot drift from the compiler.

### 3. Backends implement one internal adapter trait

Introduce a crate-private trait that every native plan implements:

```rust
pub(crate) trait MeasureBackend {
    fn preflight(&self, request: &MeasureRequest) -> PricingCapability;
    fn evaluate(&self, requests: &[MeasureRequest])
        -> Result<MeasureOutput, PricingTaskError>;
}
```

`MeasureOutput` holds the native typed result plus the rows that go into the
`ValuationReport`. `TaskBackend` dispatches to the trait instead of matching on
operation and backend pairs.

Adapters call the existing dedicated functions. They add no estimator. The
following are wrapped in this order:

| Step | Backend | Measures gained through the task |
| --- | --- | --- |
| A | Standard single-asset (BS, Black-76, LV, Bass-LV) | `SpotDelta`, `SpotGamma`, `ParallelMarketIvVega`, `VegaKt` as measures, bitwise equal to the frozen `RiskRequest` flags |
| B | Deterministic Bergomi and rough-family LSV | `SpotDelta`, `SpotGamma`, `ParallelMarketIvVega`, `ModelParameter` |
| C | Direct rough-volatility families | `SpotDelta`, `SpotGamma`, `ModelParameter` (including Hurst) |
| D | Stochastic dividends | `SpotGamma`, `Correlation`, `ModelParameter`, `DividendCash` |
| E | Multi-asset and local correlation | `Correlation`, `DiscountCurve`; `CrossGamma` already present |
| F | Cross-currency | new `PricingSource::CrossCurrency` variant, then `Price`, `SpotDelta` and quote risk |

`BumpAndRevalue` is implemented once, above the trait, by rebuilding the
source with the bumped input and evaluating both sides on the same random
coordinates. It therefore works for every backend that can price, which gives
all models a uniform, honestly labelled fallback for first- and second-order
spot and volatility risk.

### 4. One result shape

`ValuationReport` is the common result for every measure. Changes:

- `ReportedRisk` gains a typed `measure` field next to its display `name`, and
  `ValuationReport::risk(&Measure)` looks a row up by it.
- `ReportedRisk` gains `method_detail`: bump size and recalibration flag for
  `BumpAndRevalue`, estimator label for `Native`.
- `ValuationReport::to_records()` (Python: `report.to_records()`) returns flat
  rows of `measure, coordinate, value, standard_error, unit, method,
  calibration, uncertainty_scope`. The rows are plain dictionaries so that a
  caller can build a data frame without the library depending on one.

`native_result` stays on `PricingTaskResult` unchanged. Model-specific
diagnostics are read from it; cross-model analysis reads the report.

### 5. Sources carry the real model

Every model reachable through a dedicated compiler becomes constructible as a
typed `PricingModelSpec` source, so that no task needs a placeholder model.
The dedicated compilers keep accepting their current arguments. Their
documentation is changed to point to the typed source, and the placeholder is
described as ignored input where it is ignored.

### 6. Products converge on the graph

`ProductSpec` already has a `Graph` variant. Add an exact lowering
`ProductSpec::to_graph()` for European, Digital, discrete Barrier, Asian and
Lookback, and state in the compatibility page which backends consume which
form. American exercise and continuous monitoring keep their native variants
because their numerical contract is not a payoff graph (ADR 0042).
`MultiAssetProduct` is documented as a builder that produces a `GraphProduct`.

This step does not remove any constructor. Its purpose is that a contract
written once can be handed to every backend that can price it.

### 7. Analysis layer in Python

Add a pure-Python package `rust_pricing.analysis`, built only on tasks,
reports and fingerprints. It contains no numerics.

```python
from rust_pricing import analysis as an

measures = [an.price(), an.spot_delta(), an.parallel_market_iv_vega(),
            an.spot_gamma(method=an.bump(relative=0.01))]

table = an.compare(
    {"LV": lv_source, "Bergomi LSV": lsv_source, "rough Heston": rh_source},
    measures,
)                          # one row per (model, measure, coordinate)
table.unsupported          # the capability reasons, never silently dropped

sweep = an.sweep(lsv_source, an.parameter("vol_of_vol"), [0.5, 0.7, 0.9], measures)
shock = an.scenario(lsv_source, an.spot_shift(-0.10), measures)
```

- `compare` queries capabilities first, runs what is supported, and returns the
  unsupported cells with their reasons alongside the results.
- `sweep` and `scenario` rebuild the source through its typed constructors and
  keep seeds fixed, so that a difference between two rows is a common-random-
  number difference. The returned table reports the paired difference only
  where both sides share a plan layout; otherwise it reports the two estimates
  and marks the difference as unpaired.
- Every row carries the task fingerprint that produced it, so any number in a
  table can be replayed.

Scenarios are limited at first to inputs the sources already expose: spot,
market-IV parallel shift, discount-curve parallel shift, cash-dividend scale
and named model parameters.

### 8. Naming and argument order

New builders in `rust_pricing.analysis` and all new typed constructors take
identifiers as keyword-only arguments. Existing positional signatures are not
changed. A small helper maps caller-chosen labels to the integer identifiers
on the Python side and prints labels in tables; the wire format keeps integers,
so fingerprints are unaffected.

### 9. As implemented in phases 0 and 1

Phase 1 implements sections 1 to 4 for the standard single-asset backend. It
settles the following points, some of them differently from the sketch above.
Where the two differ, this section is the decision.

- **Vocabulary.** Phase 1 declares the measures that have an adapter: `Price`,
  `SpotDelta`, `SpotGamma`, `ParallelMarketIvVega` and `VegaKt`. The remaining
  measures of section 1, and the `ModelParameterId` and `CorrelationId`
  identifiers, are added with the adapter step that first produces them, so
  that no wire name is published before its meaning is exercised.
- **Standard Gamma is a bump of native Deltas.** The standard backend has no
  bump-free Gamma. It is requested with a second native method,
  `MeasureMethod::NativeSpotBump { bump }`, which is the backend's own
  estimator with the caller's bump. `SpotGamma` with `Native` is `unsupported`
  with `no_native_estimator`. `BumpAndRevalue` (phase 2) remains the
  re-evaluation above the trait and is a different estimator.
- **VegaKT carries its grid.** `Measure::VegaKt(VegaKtConfig)` reuses the
  existing typed grid. A measure's identity for duplicate detection, discovery
  and report lookup is the grid-free `MeasureKind`.
- **The list is a `MeasureSet`.** `PricingOperation::Measures(MeasureSet)`
  shares the list behind an `Arc`, and the constructor rejects a repeated
  measure. `PricingOperation` therefore no longer implements `Copy`; callers
  that used an operation twice now clone it. This is a source-level break for
  Rust callers and is accepted before the first release.
- **Capability types.** A per-measure answer is a `MeasureCapability`, which
  holds the request instead of an operation. The calls are
  `PricingSource::measure_capabilities`, `PricingSource::available_measures`
  and `CompiledPricingTask::measure_capabilities`. `PricingSource::capability`
  on a `Measures` operation returns the first refusal.
- **Compile-time selection.** A standard plan is compiled for one risk
  selection. A `Measures` task compiles the plan that a request with the same
  flags would compile, which is what makes the results bit-identical. As a
  consequence, `CompiledPricingTask::with_operation` accepts another selection
  only when it is the compiled one; price-only measures and `Evaluate` share a
  plan. ADR 0034's rule that reselection never compiles is kept by refusing.
- **Adapter trait.** `MeasureBackend` has `preflight`, `validate_measures` and
  `evaluate_measures`, and returns the native `PricingTaskOutput`. Report rows
  are still built in one place from that output, so a separate `MeasureOutput`
  was not needed. Backends without an adapter implement the trait for `Price`
  only, which is why every source prices through `Measures`.
- **Report.** `ReportedRisk` gains `measure: Option<MeasureKind>` and
  `method_detail: Option<MeasureMethod>`, set together and only by a
  `Measures` task. `ValuationReport::risk(MeasureKind)` returns the raw-unit
  row. `to_records()` flattens the price and the risk rows; bucketed VegaKT
  keeps its typed result and is not flattened.
- **Versions.** Writers emit the oldest version that can express a document:
  task v3 only for a `Measures` operation, report v3 only when a row is typed.
  Every document that could be written before is written byte for byte as
  before. Task v3 and report v3 stay open for additive measure and method
  variants until this ADR is accepted; they are frozen at that point.

### 10. As implemented in phase 2

Phase 2 was found to need data the engines do not return: the paired standard
error needs per-sampling-unit prices on both sides, and `Fixed` needs a way to
evaluate a bumped input on an existing calibration. Neither exists, and adding
them touches every engine. Phase 2 therefore lands in two steps.

Implemented:

- `MeasureMethod::BumpAndRevalue { size: BumpSize, calibration:
  BumpCalibration }` for `SpotDelta`, `SpotGamma` and `ParallelMarketIvVega`.
  Spot is bumped on every family, per asset for baskets. Parallel market-IV
  Vega is bumped only where the source has a single volatility
  (Black-Scholes and Black-76 equities).
- Sides are complete `Evaluate` tasks on sources rebuilt through the typed
  constructors, so they share the task's engine, seeds and random coordinates.
  Delta and Gamma of one size share sides. Results equal the hand-formed
  differences bit for bit.
- `BumpCalibration` has no default. `Recalibrated` on a calibrated source and
  `NotApplicable` on any other are accepted. Other pairings are
  `bump_calibration_mismatch`; `Fixed` on a calibrated source is
  `fixed_calibration_unavailable`.
- `PricingSource::bump_cost` states calibrations and evaluations before any
  compilation; `CompiledPricingTask::bump_cost` reports what ran. A test
  checks they agree for LSV and both basket fixtures.
- Report rows carry `method_detail` with the size and calibration.
- Paired standard error for every engine. The shared executor gained an
  opt-in recorder (`capture_reductions`): while a caller records, the scalar,
  vector and array reductions store each sampling unit's value (component 0,
  the price, for vector and array reductions) and the reduced sum. Engines
  that form their units themselves, Bass-LV and early exercise, record them
  with `record_units` at the point where they already have them. Recording
  changes no value, block or order, and costs one thread-local check per
  reduction when off. Units are read from the recorded shape: one reduction
  over all pseudo-random units, or one per RQMC scramble; any other shape
  falls back to the price difference without an error. A test checks, for
  every fixture family under both samplers and for the standard engine's
  Local Volatility, Bass-LV, American and continuous-barrier branches, that
  the recorded units' mean and error equal the engine's price and error. The
  estimate is the mean of the paired per-unit differences and its error that
  mean's; a test checks that the error matches the spread over 40 seeds
  under both samplers. A Gamma request records the base plan once more, and
  the cost preflight counts it.

- `Fixed` calibration for deterministic-rate single-asset LSV. LSV pricing
  used to start every path at the Spot stored in the calibration, so reusing a
  calibration on a bumped source would have produced zero Delta. Paths now
  start at the request's Spot, which is bitwise the calibration anchor for
  every plan compiled from its own request, so no existing price changes. A
  fixed side reuses the realized calibration object and its leverage surface
  unchanged, which makes the convention sticky-strike leverage; a recalibrated
  side is sticky moneyness. A fixed side has its own fingerprint domain and
  runs no calibration, and the cost preflight counts none. A test checks the
  two conventions agree exactly when the leverage is level-independent (zero
  vol-of-vol, flat target) and differ otherwise.

- `Fixed` calibration for Hull-White LSV and stochastic-dividend LSV. Their
  paths read the leverage at `ln(f / initial_f)` and also start from the
  surface anchor, so a fixed side re-anchors the reused surface at the bumped
  state: every log node moves by `ln(old / new)` and the values are
  unchanged. The leverage is then the same function of the absolute level of
  the calibrated state (escrowed forward, funded residual equity) and the
  paths start at the bumped Spot. Each has its own fingerprint domain.
- `Fixed` is refused for the remaining calibrated engines, by decision:
  - Bass-LV paths are generated by the calibrated marginal maps from a
    Brownian motion; Spot enters only through calibration. A reused
    calibration ignores the bump and would report zero Delta.
  - Baskets. Deterministic-rate LSV marginals are calibrated in
    forward-normalized coordinates (`m = f/F` starts at one), so reusing them
    already is sticky moneyness, the recalibrated convention. Sticky strike
    would need a time-dependent shift whenever cash dividends make
    `F_new(t) / F_old(t)` vary with time, and the same choice for the Local
    Correlation function and the Hull-White marginals, which a basket plan
    builds together. Callers state `Recalibrated`; this can be revisited
    per composition.

- Acceptance against native estimators beyond Black-Scholes, with the same
  small bump: Local Volatility on the standard engine against its native
  Delta (gap 4e-6 and 1.3e-4 against combined errors of 2.4e-3 and 7.8e-4,
  pseudo-random and RQMC), and Bergomi stochastic volatility against its AAD
  Spot derivative under RQMC (gap 2e-11, errors equal).
- Acceptance against native LSV risk. The native Spot Deltas of rough-family
  LSV share the paths and seeds of the bumped sides, so a test compares them
  directly on a skewed target under both samplers, with a bump of 1e-4 on a
  Spot of 100 so that no payoff crosses the strike between sides:
  - `Recalibrated` against the native sticky-moneyness Delta (relative
    local-variance target held in log-moneyness, recalibrated): gap about
    1e-11 against a paired error of 0.02 to 0.04.
  - `Fixed` against the native frozen-leverage Delta (leverage held in the
    absolute forward; no cash dividends, so residual equity is Spot): gap
    below 2e-6 against a paired error of 0.6 to 0.8.
  - In every case the paired error equals the native estimator's error
    within 0.1%.
- The parallel-IV half of the acceptance has no LSV instance, by
  construction: an LSV source states a local-variance target, not market
  implied volatilities, so `ParallelMarketIvVega` by bump is refused for it,
  and the native recalibrated local-variance node risk has no facade measure.
  Adding a local-variance measure would be a new measure, which is out of
  scope; it stays on the native API.

Phase 2 is complete.

### 11. As implemented in phase 3

Step B is split in three, each landing on its own evidence: B1 the Spot
measures of rough-family LSV, B2 parallel market-IV Vega where an LSV source
retains an IV basis, B3 the `ModelParameter` vocabulary and the parameter
risks.

Implemented (B1):

- **Calibrated native methods.** A calibrated source's native Spot estimator
  depends on what is held under the Spot move, so the caller states it, as for
  bumps. Two methods are added, leaving `Native` and `NativeSpotBump` and their
  documents unchanged (task and report v3 are open for additive variants):
  `MeasureMethod::NativeCalibrated { calibration }` and
  `MeasureMethod::NativeSpotBumpCalibrated { bump, calibration }`. `Fixed`
  selects the frozen-leverage estimators, `Recalibrated` the sticky-moneyness
  ones; `NotApplicable` is `bump_calibration_mismatch`. A discontinuous payoff
  without smoothing is `discontinuous_payoff`.
- **A family of its own.** Bergomi LSV has no native Spot estimator and
  rough-family LSV does, and the method table is per family, so rough-family
  LSV is classified as `SourceFamily::RoughFamilyLsv` (`rough_family_lsv`).
  It exposes the same operations as Bergomi LSV. Bergomi LSV keeps
  `bump_and_revalue` only for Spot.
- **Bitwise to the dedicated APIs.** The adapter calls
  `evaluate_frozen_leverage_delta` / `evaluate_sticky_moneyness_delta` and the
  matching `*_gamma_bump` with the absolute bump the request resolves to
  (relative bumps scale the physical Spot). The native result is
  `PricingTaskOutput::RoughFamilyLsvMeasureRisk`, holding each dedicated result
  unchanged; the report copies the estimate and error. The price is the
  Delta's price, which is the plan's price, or the plan's price when only
  Gamma is asked. A test checks every row, error and price bit for bit under
  both conventions and both samplers.
- The Gamma row is the full-bump second difference; the half-bump value and
  the paired difference stay in the native result.
- The rough-family plan fixes no risk selection at compilation, so
  `with_operation` moves between price-only and Spot selections on one
  compiled task.

Implemented (B2):

- **Where an LSV source has market IVs.** Section 10 found no LSV instance of
  parallel market-IV Vega because an LSV target is a local-variance grid. A
  target may also retain a reporting IV basis. When the basis, read as a
  `MarketIvSurface`, reproduces the grid bit for bit through the strict Dupire
  construction on the grid's own axes, floor and cap (the binding check
  `market_iv_risk_plan` already makes), a move of the basis IVs is a move of
  the target. Only then is `ParallelMarketIvVega` offered. A missing or
  non-reproducing basis is `model_combination`; nothing is inferred from an
  arbitrary grid.
- **Methods.** Rough-family LSV: `native_calibrated` and `bump_and_revalue`.
  Bergomi LSV: `bump_and_revalue` only (it has no market-IV estimator). Both
  require `recalibrated`; `fixed` is `bump_calibration_mismatch`, because a
  held calibration does not see the target and would report zero. A relative
  bump is refused: a surface has no single level to scale.
- **Bitwise to the dedicated API.** The native row is
  `market_iv_risk_plan(surface).evaluate()`'s parallel Vega and error. The
  adapter calls the same validation and evaluation without cloning the plan,
  and the node risks stay in the native result. The B1 result type is renamed
  `RoughFamilyLsvMeasureRisk` (it was not yet released) and gains
  `market_iv`. The estimator differentiates through the calibration, so it
  needs `retain_reverse_trace` (`reverse_trace_required` otherwise).
- **Acceptance.** The bump moves every basis IV by 1e-5, rebuilds the target
  and recalibrates both sides with the same seeds. Under RQMC on a skewed
  surface it agrees with the native Vega to 1.2e-9 against an error of 1.13,
  and the paired error equals the native error within 0.1%. This is the
  parallel-IV half of the LSV acceptance that section 10 left open.

Implemented (B3):

- **One request, many parameters.** Section 1 sketched
  `Measure::ModelParameter(ModelParameterId)`, one request per parameter. A
  measure list holds one request per measure kind, and reports and
  `MeasureSet::get` look rows up by kind. So, as `VegaKt` carries its grid,
  the measure carries its parameters:
  `Measure::ModelParameters(ModelParameterSelection)`, a non-empty,
  duplicate-free, ordered list (wire name `model_parameters`). The report has
  one row per parameter, in that order, named
  `model_parameters.<label>.<method>`. Each row carries the typed identifier
  in a new optional `ReportedRisk::model_parameter`, present exactly on these
  rows (report v3, additive); `ValuationReport::model_parameter_risk` finds
  one.
- **`ModelParameterId`.** One variant per role, shared across models:
  `InitialVariance`, `InitialState`, `MeanReversion`, `LongRunVariance`,
  `VolOfVol`, `ComponentVolOfVol { component }`, `Correlation`, `Hurst`,
  `Quadratic`, `Shift`, `VarianceFloor`. On the wire it is
  `{"name": ...}`, with `component` for a component. Python uses labels
  (`"component_vol_of_vol.1"`). A model refuses an identifier it lacks with
  `model_combination`, listing its own. `CorrelationId` waits for its first
  adapter (step D).
- **Scope.** Rough-family LSV only. It is the step-B backend with a native
  parameter estimator; deterministic Bergomi LSV has none, so it does not get
  the measure. The four models with an estimator map to their dedicated APIs:
  rough and lifted Heston, quadratic rough Heston, and mixed rough Bergomi.
  Rough SABR and RFSV are `no_native_estimator`. Mixed Bergomi shape risk
  (weight transfers and forward-variance coordinates) is not a parameter
  vocabulary and stays on its native API.
- **Method.** The estimators differentiate the fixed-target price through the
  particle recalibration, so the method is `NativeCalibrated` with
  `Recalibrated`. `Fixed` is `bump_calibration_mismatch`: the fixed-leverage
  part exists as `direct_adjoints`, but without an error, so it is not
  offered as a row. No bump method is added, since recalibrated bumps stay
  limited to Spot and market-IV measures (section 1).
- **Bitwise to the dedicated APIs.** The adapter runs the model's API with
  Hurst exactly when `Hurst` is listed. It checks the native coordinate names
  against the model's list and copies estimate and error. A test covers four
  models, with and without Hurst, under both samplers, with rows requested in
  reverse order.

Phase 3 step B is complete.

Implemented (C):

- **Fixed-model methods.** A direct rough-volatility source has no
  calibration step, so its native estimators state no treatment and their
  rows are `not_applicable`: `SpotDelta` by `Native`, `SpotGamma` by
  `NativeSpotBump`, `ModelParameters` by `Native`. Spot measures keep
  `bump_and_revalue`. A calibrated method is `no_native_estimator`, as for
  any method outside the family's row. Python's
  `MeasureRequest.model_parameters` now takes an optional calibration;
  omitted, it is `native`.
- **Bitwise to the dedicated APIs.** The adapter calls `evaluate_delta`,
  `evaluate_gamma_bump` with the absolute bump the request resolves to, and
  the model's parameter API. Rough and lifted Heston share
  `evaluate_heston_parameter_risk`; rough Heston's Hurst is the separate
  `evaluate_hurst_risk`, and each runs only when a listed parameter needs it.
  Quadratic rough Heston and mixed rough Bergomi take Hurst in the same run,
  exactly when it is listed. The adapter checks the coordinate names against
  the model's list before copying. The native result is
  `PricingTaskOutput::RoughVolatilityMeasureRisk`. Its price is the Delta's,
  else the parameter run's, else the plan's. A test checks every row, error
  and price bit for bit for six models under both samplers, with parameters
  requested in reverse order.
- **Same vocabulary as B3.** The four models with a parameter estimator use
  the B3 identifiers and native names, so one `model_parameters` request
  reads the same on a direct and an LSV source; only the method and the
  calibration treatment differ (fixed model versus differentiated
  recalibration). Rough SABR and RFSV have Spot measures but no parameter
  risk (`no_native_estimator`).
- **Acceptance.** Under RQMC on rough Heston, the native Delta agrees with
  `bump_and_revalue` (relative 1e-4, same draws) to 3.3e-13 against a
  sampling error of 9.0e-3.

Phase 3 steps B and C are complete. Steps D and E remain.

### Out of scope

- New models, estimators, calibration methods or acceptance claims.
- Removing or renaming any existing plan, result class or wire document.
- A generic calibration interface. Heston, particle LSV and Bass calibration
  keep their own entry points; only their outputs become task sources.
- Distribution (PyPI, Linux wheels). That is a release decision, tracked in
  [release readiness](../release-readiness-v0.1.md).

## Consequences

API:

- One documented path for all models: build a source, ask for measures, read a
  report. The 21 plan classes remain but move to a "native APIs" section.
- `PricingOperation` and `CapabilityReason` gain variants. Both are already
  `#[non_exhaustive]`. `PricingOperation` stops being `Copy` (section 9).
- The Python stub grows by the measure types and the analysis package, and
  shrinks in what a new user has to read.

Wire and reproducibility:

- New `pricing_task` v3 and `valuation_report` v3 documents. v1 and v2 schemas,
  fixtures and fingerprints are frozen. A v1 or v2 task read by the new code
  keeps its fingerprint because the legacy operations are retained as-is.
- A measure list is part of the task fingerprint, in request order. Two tasks
  that differ only in measure order have different fingerprints and identical
  numbers; the guide states this.
- `BumpAndRevalue` adds a bump size and its calibration treatment to the
  fingerprint domain.

Numerical:

- Adapters return bit-identical values to the dedicated API they wrap.
- `BumpAndRevalue` results are new numbers. They are labelled as such, and
  their standard error is the paired-difference error on common random
  coordinates. With `Recalibrated` the error excludes calibration noise, as
  native recalibrated risk already does, and the report's uncertainty scope is
  `ConditionalOnCalibration`.

Performance:

- A measure set is evaluated in one pass where the backend already computes
  the quantities together, as AAD backends do. The adapter must not turn one
  native evaluation into several.
- `BumpAndRevalue` costs two extra evaluations per scalar coordinate and, with
  `Recalibrated`, two extra calibrations. The capability detail states the
  evaluation count before the caller runs it.

Maintenance:

- A new model has one obligation at the public boundary: implement
  `MeasureBackend`. It no longer needs its own public plan, price and risk
  classes to be usable.
- The generated operation matrix removes a hand-maintained table.

Risk:

- The measure vocabulary can be wrong for a future model. The enum is
  non-exhaustive and coordinates are typed, which limits the cost of adding
  one, but renaming a published measure would be a wire change.
- Lowering fixed products to graphs must be exact. Any product for which
  bitwise equality with the native path cannot be shown keeps its native
  variant.

## Alternatives considered

1. **A Python-only facade over the existing plans.** Fast to write and no wire
   change. Rejected as the primary design because Rust callers and saved tasks
   would not benefit, capability logic would be duplicated in Python and could
   drift from the compiler, and results could not be replayed from a single
   document. Section 7 keeps the part of this idea that is sound: analysis
   workflows belong in Python.
2. **Collapse all plans into one generic engine.** This would also reduce the
   internal surface. Rejected because it changes path layouts and reduction
   order, invalidating plan fingerprints, replay fixtures and the accepted
   numerical evidence, for a problem that is at the public boundary.
3. **Keep three operations and add more of them** (`gamma`, `parameter_risk`,
   `correlation_risk`, ...). Rejected because every addition multiplies the
   source-by-operation matrix, and one request still could not ask for several
   quantities in one evaluation.
4. **Allow implicit finite differences when no native estimator exists.**
   Rejected. It contradicts ADR 0030 and hides a change of estimator and of
   calibration treatment from the person comparing models.
5. **Documentation only.** A decision guide would help, and section 2 produces
   one. It does not remove the per-model rewriting that the analysis requires.

## Validation

Each step lands separately and is accepted on its own evidence.

Phase 0 introduces no measure vocabulary. `available_measures()` arrives with
the `Measures` operation in Phase 1, when report rows first carry a typed
measure; until then discovery is expressed in the three existing operations.

| Phase | Content | Acceptance |
| --- | --- | --- |
| 0 (implemented) | `SourceFamily` as the single classification read by capability queries; generated operation matrix; `available_operations()` discovery over the current three operations | The explicit eight-by-four exposure table is reviewed in a test; every saved source fixture classifies into its family and is denied exactly the unexposed operations; the guide's embedded table equals the generated one and the test fails on drift |
| 1 (implemented) | `Measures` operation, task and report v3, adapter trait, step A | Bitwise equality with `PricingPlan.evaluate()` Greeks under MC and RQMC; v1 and v2 fixtures and fingerprints unchanged; a `Measures` task with a non-empty nested `RiskRequest` is rejected |
| 2 (implemented; see section 10) | `BumpAndRevalue` above the trait, with `Fixed` and `Recalibrated` calibration | Agreement with native AAD within the paired standard error on BS, LV and Bergomi SV; exact agreement with the existing CRN bump validation where both exist; `Recalibrated` LSV spot and parallel-IV bumps agree with native recalibrated risk within the paired standard error; a missing or mismatched `BumpCalibration` is rejected; the preflight's calibration and evaluation counts equal the counts actually run |
| 3 (in progress; see section 11) | Adapter steps B to E | Bitwise equality with each wrapped dedicated API; `DedicatedRiskApi` no longer returned for the wrapped measures |
| 4 | `rust_pricing.analysis` | Each table row replays from its task fingerprint to the same value; unsupported cells carry the compiler's reason |
| 5 | Product lowering; cross-currency source (step F) | Bitwise equality between fixed and lowered products per backend; cross-currency task equals the dedicated plan |
| 6 | Documentation reorganised around the task path | Every example under `examples/python` that prices through a dedicated plan has a task-based counterpart, or a stated reason why not |

Standing checks for every phase:

- A capability snapshot test covering every source family against every
  measure and method, reviewed like a wire fixture.
- Python runtime and stub contract tests for the new types.
- Schema checks for task v3 and report v3, and a frozen-fixture check that v1
  and v2 documents still parse to the same fingerprints.
- Existing statistical, refinement and extended-model acceptance gates are
  unchanged and must stay green.

Points that were open in the first draft and are now part of the decision:

1. **No cross-model `ModelVega`.** Volatility is compared across models only
   through `ParallelMarketIvVega` and `VegaKt`. Model volatility inputs are
   `ModelParameter` coordinates.
2. **Recalibrated bumps are in scope from the first release**, restricted to
   spot and parallel market-IV measures and gated by a cost preflight. The
   calibration treatment of a bump is always stated by the caller.
3. **`RiskRequest` is frozen, not removed.** It remains a valid way to ask for
   the standard single-asset Greeks under `Evaluate`; new risk is added only as
   measures, and the two cannot be combined in one task.
