# ADR 0035: Calibration diagnostics in common reports

Status: Accepted

## Context

The common task report records price/risk uncertainty and calibration seeds, but
users must branch on native plan types to inspect retained ESS, fallback,
projection or convergence diagnostics. Some diagnostics live on compiled plans
and are not carried by native price results. A saved common report therefore
cannot explain the realized calibration without replay.

## Decision

Add owned `ReportedCalibration` blocks to `ValuationReport`. Domain types live
under the API boundary without execution state. Native adapters copy existing
diagnostic rows above the numerical executor, using typed calibration kinds and
units, stable metric names, explicit native conventions, asset IDs, calibration
seeds and Cartesian axes. No numerical acceptance policy is added or changed.

Support particle LSV, Hull-White LSV, Local Correlation and both Bass calibration
and surface projection. Include marginal and basket blocks separately in joint
plans: their seeds, coordinates and ESS populations differ. Retain every native
row, including initial/terminal rows, without aggregating flags into a score.
Do not archive plans, particle histories, bump scenarios or exercise policies.

Use `Option<Vec<ReportedCalibration>>` to distinguish unavailable historical
diagnostics from a known empty list. Python exposes frozen objects and owned
collection snapshots. Operation reuse retains identical calibration diagnostics;
each evaluated report owns its copy independently of the compiled plan.

## Compatibility and impact

- Emit independent `valuation_report` v2; keep a strict v1 reader and the frozen
  v1 schema/fixture. A v1 read followed by a write normalizes to v2 and omits
  unavailable diagnostics. New reports populate the field, including `[]` for
  no applicable calibration. Legacy v1-only consumers must upgrade.
- Preserve every pre-existing report field and native output. Task/source JSON,
  canonical fingerprints, replay identities and calibration ownership remain
  unchanged. There is no recalibration, new RNG stream, covariance reduction,
  model policy or change to numerical/refinement Gates.
- Additive Rust/Python reporting APIs expose types, values and conventions.
  Validate shapes, finite data, unique identities/names, monotone axes and exact
  count/indicator domains. Missing data never means zero uncertainty or success.
- Known kinds can gain named columns with existing typed units without new
  model-specific consumer branches. Consumers preserve unfamiliar metrics and
  select known ones by name. New kind/unit variants require a schema decision;
  Rust enums are non-exhaustive.
- Reports copy retained diagnostics, so memory/JSON size grows with calibration
  grids. Existing bounded JSON parsing remains in effect. There is no persistent
  compiled-state format, global cache or extra numerical work.

This extends ADR 0030's common report scope and supersedes only its report writer
version. Requirements for finite values, reproducibility, explicit conventions,
conditional sampling errors and compatibility remain in force.

## Validation

Compare diagnostic columns against independently compiled native plans across
one/two-factor and rough LSV, rough families, Hull-White, MC/RQMC multi-asset and
joint calibration, and Bass. Keep the frozen v1 numerical fields exactly equal.
Exercise v1 migration, current golden/schema round trips, malformed/duplicate
diagnostics, absent versus empty states, operation reuse, Python immutability,
typed stub contracts and installed-wheel checks. Existing source/plan fingerprint
and full native-result tests remain required.
