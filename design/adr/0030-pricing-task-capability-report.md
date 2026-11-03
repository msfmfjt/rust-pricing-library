# ADR 0030: Complete pricing tasks, shared preflight and common reports

Date: 2026-10-21. Status: implemented; additive facade and wire contracts.

## Context

After ADRs 0028/0029, callers can save single-asset model sources and generic
payoffs, but basket/dividend reconstruction still requires external inputs.
Risk availability is spread across native entry points, and risk units,
calibration semantics and uncertainty availability differ between outputs.
The high-priority architecture review requires a shared discovery boundary,
a common result contract and complete source replay for these adapters.

## Decision

Introduce an immutable `PricingTask`: typed `PricingSource` plus
`PricingOperation`. Preserve the existing single-asset specification and add
immutable multi-asset and stochastic-dividend sources. Save every source input
consumed by the existing compilers, including raw correlation endpoints and
tolerances, retained market-IV targets, stochastic reserve inputs, calibration
random settings, numerical grids and execution/reduction policy.

Expose pure source capability inspection with `supported`, `conditional` and
`unsupported` statuses and machine-readable reasons. A source query can reject
known incompatibilities but reports conditional until native compilation.
Extract existing native AAD/local-variance guard code into shared preflight
functions; direct evaluation and the compiled task call those same functions.
This avoids a support table that can drift from realized calibration checks.
Supported after preflight is not a promise that numerical evaluation succeeds.

Common operations are native evaluation, supported native first-order AAD
(with optional basket cross Gamma), and supported recalibrated target-variance
risk. Preserve configured standard Greeks in the existing evaluation operation.
Do not silently substitute finite differences or reinterpret specialized
parameter/correlation/dual-bound/continuous-barrier APIs as a common operation.

Return `PricingTaskResult` with both a common `ValuationReport` and the native
typed result. The report copies estimates and available errors, recording units,
coordinates, method, calibration treatment and uncertainty scope. Missing risk
uncertainty stays absent. Keep typed VegaKT covariance/diagnostics intact.
The common report is not an exhaustive native diagnostic archive; source replay
recreates native diagnostics. Dispatch/report construction remain above native
loops and private execution workspaces stay out of source/domain modules.

## Compatibility and numerical impact

- Add separate `pricing_task` v1 and `valuation_report` v1 documents, strict
  readers, self-contained schemas and canonical task domain `pricing/task/v1\0`.
  The task hash covers both source and operation. Optional fields are omitted;
  ordered optional slots are tagged, preserving the existing no-null rule.
- Preserve request/result v1–v3, specification v1/v2 and graph v1 contracts and
  fingerprints. Normalize nested specification v1 to the existing v2 writer.
  Do not serialize compiled or calibration workspace state.
- Present common plan fingerprints as `blake3-256:<hex>`. Basket native output
  keeps its existing bare hex; bytes and numerical identity do not change.
- No new dependency, crate edge, sampling algorithm, path traversal, covariance
  canonicalization, calibration math or reduction order. Preflight extractions
  preserve direct evaluation guard order and native errors.
- Calibration differentiation and sampling uncertainty are independent metadata.
  Recalibrated finite-particle risk still excludes calibration noise from its
  reported sampling error. LSM risk remains conditional on a fixed exercise
  policy; MC adjoints without covariance do not acquire invented errors.
- Existing Rust constructors and Python native APIs remain intact. New source
  variants and operation enums are non-exhaustive. Python bindings are frozen,
  expose owned snapshots and release the GIL during compilation/evaluation.

## Validation and rollout

Compare native results exactly against the task adapter under MC/RQMC for
standard Greeks, mixed basket factors, shared HW, joint Local Correlation and
stochastic-dividend BS/HW. Round-trip all 1F/2F/rough dividend SV/LSV families;
retain real cash events, full driver endpoints and market-IV provenance.
Check task/plan fingerprints, raw tolerated correlation asymmetry, Gamma,
missing covariance, malformed sources/reports and source-vs-compiled preflight.

Keep offline schema checks, runnable examples, Python runtime/stub contracts and
source-archive completeness in the existing CI. Preserve existing independent
statistical, refinement and extended-model acceptance gates. The
[usage/support guide](../../docs/library/pricing-tasks.md) documents the exact
operation matrix and extension rules. Broader specialized risk operations and
typed Python configuration builders can extend this boundary separately.
