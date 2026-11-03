# Calibration diagnostics in valuation reports

`PricingTask.compile().evaluate().report.calibration_diagnostics` exposes owned
snapshots of the calibration diagnostics retained by the native plan. Rust uses
`ValuationReport::calibration_diagnostics()`. Reading a report does not compile
or recalibrate anything. Selecting another operation on a compiled task keeps
the same calibration diagnostic values.

Each `ReportedCalibration` identifies a typed `kind`, ordered `underlying_ids`,
an optional calibration `seed`, Cartesian `axes`, and `CalibrationMetric` columns.
Metric names are stable within a kind. Each metric includes a typed `unit`, a
`convention` describing its native meaning, and row-major `values` on the block's
axes. A column contains exactly the product of the axis lengths. Python objects
are immutable; lists and nested axes are owned copies.

```python
report = task.compile().evaluate().report
for calibration in report.calibration_diagnostics or []:
    print(calibration.kind, calibration.underlying_ids, calibration.seed)
    for metric in calibration.metrics:
        print(metric.name, metric.unit, metric.values, metric.convention)
saved = report.to_json()
```

## Retained coverage

| Kind | Axes | Native columns |
| --- | --- | --- |
| `particle_lsv` | `time` | Minimum supported-node ESS, extrapolated-node count, particle mean equity coordinate |
| `hull_white_lsv` | `time` | Minimum supported-node ESS, fallback count, mean relative discount, mean discounted normalized equity, maximum absolute rate correction |
| `local_correlation` | `time`, `log_basket` | Both endpoint variances, target/attained variance, rate correction, raw mixing, original-node ESS, donor index, fallback/projected/unidentifiable flags |
| `bass_fixed_point` | `end_time` | Start time, iteration count, CDF residual, propagated marginal CDF error, boundary tail probability, Brownian bounds/spacing |
| `bass_surface_projection` | `marginal_index` | Tail/retained probabilities, unscaled mean, mean scale, maximum source-call error, retained-node count |

Particle LSV includes one/two-factor and rough Bergomi, rough-family LSV, and
stochastic-dividend residual-equity LSV. Multi-asset reports list each applicable
marginal in native asset order, followed by the Local Correlation basket block.
The joint simulator uses both marginal surfaces and the basket calibration;
their separate blocks retain distinct asset IDs and seeds. A basket's diagnostic
does not describe every marginal node. Deterministic Bass blocks omit `seed`.

All rows retained by the native producer are copied, including initial and
terminal rows. A terminal row is not necessarily used to evolve a subsequent
pricing step. `source_node` indexes the block's `log_basket` axis within its row;
it is not a global flattened index. Bass projection indices preserve native
marginal order. Task inputs provide the model, particle settings and target grids.

LSV minima are calculated by the native producer over supported donor nodes;
Local Correlation ESS is recorded at the original node before fallback. They
must not be compared as if they measured the same support population. Raw mixing
can lie outside `[0,1]`. Projection and fallback flags are native observations,
not a new pass/fail rule. No automatic feasibility threshold, diagnostic score,
cross-row average or total-error estimate is introduced.

`count` values are exactly represented nonnegative integers up to `2^53`;
`indicator` values are numeric zero or one. ESS is dimensionless and may be
fractional. `time` is in years; `variance` retains the native relative-variance
coordinate; `equity_coordinate` retains the native calibrated equity/quote scale
(which can be normalized or residual equity). The metric's convention and saved
task resolve the coordinate. Diagnostic values are not sensitivity estimates.

## Availability and versioning

Fresh reports have a diagnostics list: an empty list means no applicable retained
calibration diagnostics. An absent field means diagnostics are unavailable, as
with imported report v1. Rust returns `None` and Python returns `None` for that
state. An unavailable metric is omitted, never replaced by zero. The field does
not claim to archive every native workspace, exercise-policy fit, bumped-scenario
calibration, conditional moment or calibrated surface. Those remain under their
existing native APIs and replay contracts.

The current writer emits `valuation_report` **v2**, described by the
[v2 schema](../../schemas/report-v2/valuation_report.schema.json). Readers accept
v1 and v2; importing v1 and saving again emits v2 without fabricating diagnostics.
The [frozen v1 schema](../../schemas/report-v1/valuation_report.schema.json) and
its golden report remain unchanged. A v1 document containing the new field is
rejected, even if the list is empty. Consumers restricted to v1 must upgrade
before reading newly written reports.

Readers validate finite values, unique block identities and metric names,
strictly increasing finite axes, aligned column shapes, and count/indicator
domains. Unknown fields, kinds and units are rejected. Additional named metrics
within a known kind and unit can be preserved without teaching the reader a new
model-specific class; consumers should select metrics by name and tolerate new
columns. New kinds/units require an explicit wire-schema decision.

Source/task documents and fingerprint domains are unchanged. Prices, risks,
sampling errors, uncertainty scopes, seeds, native outputs and numerical
acceptance gates are unchanged. Diagnostics do not add calibration sampling
uncertainty to price errors. See [ADR 0035](../../design/adr/0035-calibration-report-diagnostics.md).
