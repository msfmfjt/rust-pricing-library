# ADR 0038: Bind provider configuration for its entire lifetime

Status: accepted

## Context

`RustPricingProvider` calculated its identity at construction, but exposed
writable `numerical_scope`, `worker_threads` and `reduction_block_size` fields.
Its result-cache key binds snapshot, contract and request. Reassigning a setting
could therefore change acceptance or execution behavior without changing the
provider identity or invalidating previously retained risks and task evidence.

A native reproduction first evaluates a deliberately inconsistent reporting
surface in diagnostic mode, then assigns the P0 scope. The validation property
reports accepted P0 status and evaluation returns the original unvalidated
cached risk. A fresh P0 provider rejects the same input. Clearing the mutated
provider's cache also exposes the rejection. This violates the provider identity
and numerical acceptance contracts.

## Decision

Bind every constructor setting, the native-module reference and the computed
identity in one frozen internal configuration record. Expose settings through
read-only properties. Keep the existing constructor keywords and argument
validation; retain readable properties for the existing settings and expose
`diagnostic_only` and `cache_max_entries` consistently.

Setting or deleting a configuration property raises `AttributeError`.
Reinitializing an existing provider is also rejected before touching its
configuration or retained results. Changing settings requires constructing a
new provider, which has its own identity and empty caches and must pass its own
ordinary acceptance checks. The native binding is read-only as well; tests that
inject a native failure select the test module before provider construction.

Keep mutable execution state separate from this configuration. Snapshot changes
and `clear_cache()` replace only the bounded caches, using the original capacity.
In-flight work continues to hold the same configuration and captured snapshot
state. Retain the lifetime/concurrency rules from
[ADR 0036](0036-backtest-provider-cache.md) and the evidence contracts from
[ADR 0037](0037-backtest-task-report-bridge.md).

## Impact

- Construction and configuration reads remain compatible. Post-construction
  assignment/deletion and reinitialization now fail explicitly.
- No change to native APIs, pricing algorithms, numerical tolerances, sample
  pairing, accepted scope combinations or risk/report schemas.
- Preserve the provider identity format and its settings. Cache capacity remains
  excluded from numerical identity. Existing implementation hashes invalidate
  checkpoints made by older source revisions, including this fix's predecessor.
- A newly constructed provider with different numerical/execution settings
  cannot resume a checkpoint under the original identity. A compatible fresh
  provider, including one with a different capacity, can still resume exactly.
- This fixes the existing backtest provenance/acceptance contract. Native-wheel
  API compatibility checks and policy-definition consolidation are separate work.

## Validation

Retain the native diagnostic-to-P0 reproduction as a regression: setting
assignment must fail, the original provider must remain diagnostic before and
after cache clear, and a fresh P0 provider must reject the same input.

Test read-only configuration before evaluation, on cache hits, after cache clear
and during synchronized in-flight work. Reinitialization must preserve cached
risks/evidence and identity. Verify constructor settings across accepted modes,
new-provider execution identity, task evidence after attempted reassignment,
checkpoint rejection for changed execution and exact compatible resume.

Run existing companion, native ordinary/P0, and multi-step cache/audit checks.
Do not regenerate historical numerical acceptance evidence for this fix.
