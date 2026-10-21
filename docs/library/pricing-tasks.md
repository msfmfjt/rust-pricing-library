# Complete tasks, capability queries and valuation reports

`PricingTask` saves a validated source together with the operation to run.
`PricingSource` has single-asset, multi-asset and stochastic-dividend variants.
The immutable task compiles through existing native plans and returns both a
common `ValuationReport` and the original typed `native_result`.

The [Python example](../../examples/python/pricing_task.py) runs the
[complete task fixtures](../../fixtures/task-v1). The design and compatibility
impact are recorded in [ADR 0030](../../design/adr/0030-pricing-task-capability-report.md).

## Query before executing

Source `capability(operation)` performs pure inspection: it does not calibrate,
generate paths or allocate an AAD workspace. Its status and reason are typed
Rust enums and stable Python strings:

| Status | Meaning |
| --- | --- |
| `unsupported` | The requested operation is unavailable or a known source prerequisite fails; `reason` and `detail` explain why. |
| `conditional` | Static checks passed; native compilation and risk preflight remain necessary. |
| `supported` | The compiled task passed its native operation preflight. Numerical evaluation can still fail. |

Task construction, capability queries and compilation share the operation
selection rules. Native evaluation and compiled preflight share the same guards
for differentiability, reverse traces and Gamma bumps. Covariance, resource,
realized calibration and numerical acceptance remain the existing compiler's
responsibility. A source query never claims these have passed.

Known rejection reasons include `dedicated_risk_api`, `discontinuous_payoff`,
`reverse_trace_required` and `invalid_gamma_bump`. A source selected for price
can inspect another operation without compiling it. A compiled task uses
`capability()` for its saved operation and `capability_for(operation)` for another
operation against the same realized plan.

## Reuse compilation and calibration

Use `CompiledPricingTask.with_operation(...)` to select another operation while
sharing the compiled native plan, calibrated surfaces and retained reverse traces:

```python
task = rp.PricingTask.from_source(source, operation="evaluate")
price_plan = task.compile()
price = price_plan.evaluate()
if price_plan.capability_for("local_variance_risk").status == "supported":
    risk_plan = price_plan.with_operation("local_variance_risk")
    risk = risk_plan.evaluate()
    assert risk_plan.plan_fingerprint == price_plan.plan_fingerprint
    saved_risk_task = risk_plan.task.to_json()
```

In Rust the same methods accept `PricingOperation`; `task()` returns the selected
replayable task. `PricingTask::with_operation` only selects a source operation,
and its subsequent `compile()` builds a fresh plan. Use the method on an already
compiled task when reuse is intended.

Selection returns a new immutable handle. It shares source and plan ownership,
so dropping the original handle is safe. Different operations and Gamma bumps
have their own task fingerprints and reports while retaining the same native
plan identity. The selected `task` can be serialized and compiled independently
for replay; no compiled state is added to the wire format.

Source prerequisites and native preflight run on every selection. Missing reverse
traces, discontinuous payoffs or inadmissible Gamma bumps remain errors; selection
does not enable traces or change inputs. In Python, failures of these source/native
checks use the existing `compile` error stage and selected operation; invalid
operation labels or argument combinations retain validation errors. A failed selection
leaves the original plan usable. Change source inputs, seeds, grids or execution
policy by creating and compiling a new source task.

This is explicit ownership within a process, with no global cache, eviction policy
or cross-source fingerprint lookup. Cloning a Rust compiled task shares its native
plan. Evaluation still owns separate path/risk workspaces and runs the selected
native algorithm, including any required calibration pullbacks or bump work.
Selection itself performs no compilation/calibration; it still validates the
operation and computes its task fingerprint. Concurrent operations on shared
handles preserve deterministic reduction and existing sampling-error semantics.
The sharing boundary is recorded in [ADR 0034](../../design/adr/0034-compiled-task-reuse.md).

## Operations and supported compositions

| Source | `evaluate` | `aad` | `local_variance_risk` |
| --- | --- | --- | --- |
| Standard single-asset BS / Black-76 / LV / Bass-LV | Existing configured price/risk/LSM request | Use configured standard risk with `evaluate` | Use configured native risk request |
| 1F/2F/rough Bergomi SV | Price | Native AAD | Unavailable |
| Deterministic Bergomi or rough-family LSV | Price | Use dedicated APIs | Recalibrated target-grid risk |
| Single-asset Hull–White BS / Bergomi SV / LSV | Price | Native hybrid AAD, including retained target risk | Use hybrid AAD |
| Direct rough-volatility families | Price | Use dedicated APIs | Unavailable |
| Multi-asset BS/LV, mixed Bergomi LSV, shared HW, joint Local Correlation | Price | Native full AAD; optional full cross Gamma | Included in native AAD |
| Stochastic dividends with BS or 1F/2F/rough Bergomi SV, or BS/HW | Price | Native first-order AAD | Unavailable |
| Deterministic-rate stochastic-dividend Bergomi LSV | Price | Use target-grid operation | Recalibrated residual-equity target-grid risk |

The matrix describes the common task operations. Specialized parameter,
correlation, Hurst, market-IV, Gamma and dual-bound APIs retain their own
contracts when not exposed here. Continuous-barrier monitoring wrappers are
also separate. No new model/product combination or numerical acceptance rule
is implied by a source format. Hard live indicators reject pathwise risk;
explicit graph smoothing or the native supported smoothing contract is needed.

`aad.gamma_relative_bump` is optional and belongs only to multi-asset tasks.
It must be finite and in `(0,1)` and pass the native representability checks.
Without it, basket AAD returns first-order risk. Standard single-asset Greeks
remain part of the source's existing `RiskRequest` and `evaluate` operation.

## Complete saved inputs

A single-asset source embeds the existing [pricing specification](pricing-specification.md).
`MultiAssetSpecification` retains the generic payoff graph, ordered markets,
model per asset, raw dated correlation matrices and all six tolerances, marginal
LSV factors and particle settings, optional full driver correlations, shared
Hull–White inputs and paired targets, joint Local Correlation endpoints,
calibration settings, retained market-IV provenance, engine and execution policy.
Raw matrices are preserved even when native canonicalization tolerates a small
asymmetry; this also preserves native diagnostics and plan identities.

`StochasticDividendSpecification` combines the full equity specification with
the Buehler reserve model, dividend/volatility correlations, optional rate
correlation and dividend split-grid maximum step. Fixed cash dividend events
remain in the nested market. Direct SV/HW step settings must match the dividend
step. LSV retains its target grid separately. Unused or conflicting inputs are
rejected. No compiled leverage grids, particle states or reverse workspaces are
serialized.

In Rust, construct `PricingTask::new(source, operation)`, using
`PricingSource::from(specification)` or the corresponding multi-asset/dividend
source. `MultiAssetModelSpec::new(models, correlation)` supplies default absent
extensions; set its typed optional configurations before creating the immutable
`MultiAssetSpecification`. Serializers and parsers are
`pricing_task_to_json`, `pricing_task_to_pretty_json`,
`parse_pricing_task_json` and `fingerprint_pricing_task`.

## Python

```python
import rust_pricing as rp

# Construct from a single-asset source, or load any complete task from JSON.
source = rp.PricingSpecification.from_json(specification_json)
task = rp.PricingTask.from_specification(source, operation="evaluate")
status = task.capability("aad")
print(status.status, status.reason, status.detail)

saved_task = task.to_json()
plan = rp.PricingTask.from_json(saved_task).compile()
assert plan.capability().status == "supported"
result = plan.evaluate()
report = result.report
print(report.value, report.standard_error, report.uncertainty_scope)
for risk in report.risks:
    print(risk.name, risk.unit, risk.method, risk.calibration, risk.standard_errors)
saved_report = report.to_json()
restored_report = rp.ValuationReport.from_json(saved_report)
```

Use `task.with_operation("aad", gamma_relative_bump=0.01)` for a basket,
or `task.with_operation("local_variance_risk")` for a supported LSV source.
Use `PricingTask.from_source(source)` with typed single-asset, multi-asset or
stochastic-dividend inputs; see [typed Python construction](typed-python-sources.md).
Python objects are immutable, collection getters return owned snapshots, and
compilation/evaluation release the GIL. The [stub](../../rust_pricing.pyi) freezes
public signatures; JSON errors identify the task or report document kind.

## Risk and uncertainty meaning

`ValuationReport` includes task and plan fingerprints, library version, platform,
currency, price and standard error, estimator, independent sampling units,
evaluated paths, calibration seeds, uncertainty scope, typed risk blocks and
[retained calibration diagnostics](calibration-reports.md).
Each block has a name, ordered underlying IDs, Cartesian coordinate axes,
row-major values, optional standard errors, a typed unit, native method,
calibration treatment and fixed-input/coordinate convention.

Scalar blocks have no axes and one value. Vector/grid shapes and standard-error
lengths are checked on import. Cross Gamma is the native unsymmetrized
`dDelta_i/dSpot_j` matrix, represented by ordered underlying pairs. Density
adjoints without exposed native grids use an explicit `native_parameter_index`
axis. The saved task and native result provide the remaining model context.

Raw volatility derivatives are per unit volatility; market-scaled Vega is per
one volatility point. Local-variance derivatives are per unit **variance**.
Discount-curve adjoints are per log discount factor; DV01 is the signed change
for a +1bp continuously compounded zero-rate pillar. Spot-scaled Delta/Gamma
retain native 1% conventions. These quantities are never silently interchanged.

Calibration treatment describes differentiation, independently of uncertainty:
`recalibrated` includes the native finite-particle pullback with fixed random
inputs; standard LSM risk records `fixed_exercise_policy`. Price/risk sampling
errors for LSV or joint calibration remain **conditional on the realized
calibration**. They exclude calibration noise and discretization/kernel bias.
LSM sampling errors are conditional on its learned exercise policy.

An absent standard-error array means **unavailable**, not zero. For example,
several native MC calibration adjoints have no retained covariance; RQMC may
provide errors across independent scramble estimates. The report copies native
estimates and available errors without recomputing sampling reductions.
The existing typed `vega_kt` payload retains its covariance and diagnostics.

Native Hull–White and stochastic-dividend AAD results also carry ordered
`RiskParameterKind` values in Rust (`parameter_kinds`) and expose
`parameter_kinds` / `parameter_units` snapshots in Python. These align with
`parameter_labels` and `derivatives`. Units come from the native differentiation
coordinates, never from the spelling of the display labels. The report rejects
inconsistent metadata lengths. Explicit model-parameter sensitivities retain
their native unit convention; vol-of-vol is not market-IV Vega.

Failures expose [structured Python error metadata](python-errors.md), including
the selected task operation and a capability reason for source-preflight rejection.

The report JSON is a normalized reporting contract. It is not a serialization
of every native diagnostic or a compiled plan. Selected retained calibration
diagnostics are copied from the compiled plan into the common report. Additional
diagnostics remain on native plans/results and are recovered by executing the saved task.
For a single-asset task, that native value is the existing `SpecificationResult`
(and its own `native_result` in Python); other task variants return their native
price/risk object directly.

## JSON versions and replay identity

The current independent document kinds are `pricing_task` v1 and `valuation_report` v2.
Their [task schema](../../schemas/task-v1/pricing_task.schema.json) and
[report schema](../../schemas/report-v2/valuation_report.schema.json) are exported
by `pricing_task_json_schema()` / `valuation_report_json_schema()` in Python
and `current_pricing_task_schema()` / `current_valuation_report_schema()` in Rust.
Schemas describe the current writer. The report reader also accepts v1 and
normalizes it to v2, retaining unavailable calibration diagnostics as an absent
field. Existing v1-only report consumers must upgrade for newly written reports.
Nested single-asset source readers also
accept specification v1 and normalize it to v2.

Optional object fields are omitted; ordered optional configurations use
`{"type":"absent"}` or `{"type":"present","value":...}`. JSON `null`, duplicate
keys, unknown fields and nonfinite numbers are rejected by the existing bounded
parser. Rust additionally validates shapes, values and model consistency.

The task fingerprint uses domain `pricing/task/v1\0` and covers the complete
source **and operation**, including Gamma bump, calibration seeds, worker count
and reduction-block size. JSON whitespace/key order do not affect identity.
The report links task and native plan identities. Common plan fingerprints use
`blake3-256:<hex>`; basket native results keep their existing bare-hex spelling,
with identical hash bytes. A price task and risk task can share a native plan
fingerprint while having different task fingerprints.

Legacy request/result v1–v3, specification v1/v2, graph v1, their fingerprint
domains and numerical outputs are unchanged. A legacy reader rejects the new
document kind. Exact replay still follows existing version/platform/toolchain
and execution-policy guarantees; the format does not claim cross-platform bit
identity.

## Extension and validation

Add a typed source variant or operation, strict versioned wire conversion,
explicit compiler dispatch, capability rule and report adapter together.
Keep model dispatch outside path loops. Reuse native preflight for both direct
and common evaluation; do not duplicate a permissive support table. Add native
conformance/replay, fingerprint sensitivity and rejection tests for each
composition. A changed persisted numerical meaning requires an explicit schema
and migration decision.

Verification covers [Rust conformance](../../crates/pricing/tests/pricing_task.rs),
[Python contracts](../../tests/python/test_pricing_task.py),
[offline schemas](../../scripts/check_task_json_schema.py), existing numerical
regression tests, wheel/stub validation and source-archive inclusion. Small
fixtures exercise contracts and deterministic replay; they are not accuracy
acceptance panels. Existing statistical and extended-model accuracy gates remain
in CI without relaxed tolerances or exclusions.
