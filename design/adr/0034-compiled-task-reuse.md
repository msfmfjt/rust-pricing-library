# ADR 0034: Reuse compiled pricing tasks across operations

Date: 2026-10-21. Status: implemented.

## Context and decision

The common task facade saved an operation alongside its source. A caller could
switch operations on `PricingTask`, but compiling the new task rebuilt the native
plan and repeated particle calibration. Cloning a compiled task could also copy
large calibrated surfaces and reverse traces. Price, AAD and target-variance risk
often need the same realized calibration and numerical inputs.

Keep immutable source ownership in `PricingTask` behind a private `Arc`, and
compiled backend ownership in `CompiledPricingTask` behind another private `Arc`.
Add Rust `PricingTask::with_operation` for source sharing. Add
`CompiledPricingTask::with_operation` in Rust and Python for explicit reuse of
the existing native plan and calibration. Preserve the original task and return
a new immutable selection. Do not add a global cache, alternate compiler,
workspace pool or persistent compiled-state format.

Every selection uses the existing source capability rules and realized native
preflight. `capability_for` queries another operation against those same checks;
the existing `capability()` still queries the selected operation. A missing
reverse trace cannot be upgraded, and native Gamma/down-Spot constraints remain
enforced. A failed selection leaves existing handles intact. The selected `task`
is exposed as a replayable source/operation value, independently of compiled state.

Backend construction remains determined by the source alone. Future operations
that need different compilation settings must make those settings explicit in
the source and reject an incompatible existing plan; selection must not silently
recompile or alter the shared calibration.

## Compatibility and numerical impact

- Public methods are additive. Existing compile, evaluation, capability, task and
  report contracts remain. Source and backend fields were already private.
- Task fingerprints continue to cover the full source and selected operation,
  including Gamma bump; the native plan fingerprint is unchanged by selection.
  Task/report schemas and canonical writers are unchanged. An independently
  compiled saved task must reproduce the shared-plan result under the existing
  platform/toolchain/execution-policy guarantees.
- No sampling stream, seed, calibration input, reverse trace, covariance,
  sensitivity unit, uncertainty meaning or deterministic reduction is changed.
  Shared objects are immutable; each evaluation owns its usual workspaces.
- Sharing skips compilation/calibration during operation selection. It does not
  skip native pricing, reverse passes or risk-specific bump work at evaluation.
  Task validation and fingerprint serialization still cost time and allocations.
- Source changes require a new task and compilation. Reuse is explicit within
  one process; there is no hash-based cache hit or cross-source substitution.
- Python selection and alternate preflight release the GIL. Source/native
  selection failures use existing compile-stage metadata with the requested
  operation; invalid labels/argument combinations retain validation errors.

## Validation

Compare full native results and reports from reused operations against independent
compilation, including all task-fixture source families, MC/RQMC, joint Local
Correlation/shared Hull-White, single-asset LSV/hybrid and stochastic dividends.
Check operation and Gamma fingerprint changes, source/task JSON replay, original
owner release, concurrent evaluations and unchanged earlier selections.

Assert shared allocation identity for cloned/reselected calibrated plans. This
detects accidental deep copying or recompilation that numerical replay alone
cannot reveal, without environment-dependent timing thresholds. Reject missing
reverse traces and invalid bumps, including a source-conditional Gamma whose
down-Spot violates the native dividend reserve constraint. Preserve structured
Python error metadata and freeze the new callable declarations independently.

Existing native conformance, schema, wheel, archive, cross-platform and numerical
acceptance gates remain required. The small replay tests validate compatibility
and ownership; they do not establish new financial accuracy claims.
