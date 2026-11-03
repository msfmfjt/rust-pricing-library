# ADR 0039: Check the provider's native API at construction

Status: accepted

## Context

The companion's native extra pins `rust-pricing==0.1.0`. Native builds sharing
that version can have different APIs. After the common task/report bridge in
[ADR 0037](0037-backtest-task-report-bridge.md), an older wheel can construct a
provider successfully but fail later with `AttributeError` for `PricingTask`.
The previous runtime check covered only `DiscountCurve.discount`. A package
version therefore does not establish that a selected provider path can run.

## Decision

Add a small companion-owned native contract at the provider boundary. After
validating constructor policy and execution settings, lazily import the native
module and check the API before computing identity or allocating caches.
Check required native classes, the exception type, result members and callable
signatures. Bind the exact positional/keyword call shapes without executing
native methods. Allow compatible optional arguments and additional APIs.

Select requirements from the actual execution path. Ordinary diagnostic/P0
evaluation uses common tasks/reports, discount curves and VegaKT result fields.
Multi-step evaluation uses ordinary common tasks/reports plus the dedicated
Gamma audit. Extension/custom-basis scopes also support joint covariance, so
require ordinary plan evaluation and the `retain_samples` argument. Quote-control
uses its dedicated plan/sample path and does not require unused common tasks.
Check task/report serialization used by the public replayable evidence contract.

Reject missing or incompatible APIs with `unsupported_native_version`, including
the detected native version, failing API paths and a matched-build remedy.
Distinguish missing modules and extension-loader failures as
`native_import_failure`, preserving the cause. Propagate unrelated programming
exceptions. Failure to read the native file for binary provenance also raises
`unsupported_native_version`. Keep the native dependency optional at package
import time. Do not cache compatibility results across provider constructions.

## Boundaries and impact

- Fix the existing native boundary/diagnostics contract without changing public
  constructor arguments. Incompatible environments now fail before a run starts.
- Structural checks establish the required API surface and accepted call shapes.
  They do not verify return-value semantics, numerical accuracy, all native APIs
  or platform ABI correctness. Preserve the full installed-wheel contracts from
  [ADR 0033](0033-wheel-api-contracts.md), along with runtime report/audit checks.
- Do not price a synthetic trade or perform calibration at startup. Keep every
  calibration, shape, covariance, recovery and precision gate unchanged.
- No native source/API change, algorithm change, numerical tolerance change,
  dependency/version bump, or ledger/checkpoint schema change.
- Continue hashing the actual native binary and companion implementation for
  provenance. Include the new contract module in the multi-step implementation
  identity; existing backtest source hashes also include it. Older source
  checkpoints remain incompatible under the existing rules. Same-build runs,
  cache release and compatible resume preserve numerical behavior.
- Keep policy consolidation and expanded common report diagnostics separate.

## Validation

Use an independent minimal dedicated-path fixture to demonstrate that startup
performs no pricing, requires no unused common API and checks paired-sample call
support. Cover missing classes/callables/members, old signatures, import/loader
errors, optional native installation, binary provenance and failure before any
configuration/cache is published. Keep cache-only synthetic tests isolated from
native conformance.

Against the installed wheel, construct every provider path and reproduce an
older same-version build by removing APIs. Reject old specification signatures,
discount curves and report/risk members at construction. Verify that ordinary
evaluation does not need a Gamma plan and that joint scopes require retained
samples. Retain existing native pricing/error, task replay, cache, calibration,
joint covariance, persistence and exact-resume regressions. Validate both source
and installed companion wheels; preserve historical numerical evidence.
