# ADR 0032: Typed Python source construction

Date: 2026-10-21. Status: implemented.

## Context and decision

ADRs 0028 and 0030 provide complete Rust source types and replayable common tasks.
Python can construct native model-specific plans, but complete multi-asset and
stochastic-dividend tasks previously required JSON. Single-asset composition also
lacked typed configuration objects, limiting static checking and IDE completion.

Expose frozen Python adapters for the existing Rust equity/rate/factor composition,
particle settings, numerics, basket extensions and dividend source types. Add a
full `PricingSpecification` constructor and `PricingTask.from_source` accepting
all three source classes. Reuse existing markets, products, model constructors,
correlation schedules, native LSV configurations and local-correlation settings.

Configuration records own cloned Rust input values. Scalar types use their native
constructors; complete source validation and numerical compilation retain their
existing boundaries. No Python dictionary schema, intermediate JSON parser,
parallel financial validation rules, compiled-state cache or execution workspace
is introduced. Full parameter annotations and independent frozen contract checks
make the new API reviewable and protect type information from drift.

## Compatibility and numerical impact

- All public changes are additive. Existing Python signatures remain unchanged.
  No new public Rust type, crate, dependency or engine visibility is needed.
- Legacy requests/results, source/task/report/graph schemas, canonical writers,
  fingerprint domains, operation capability rules and native dispatch are unchanged.
- Seeds, worker policy, reduction order, calibration targets/provenance and raw
  tolerated correlation inputs are copied without normalization by the binding.
- No pricing formulas, adjoints, calibration algorithms or numerical thresholds change.
- New native validation failures use the existing structured Python error adapter.
  Scalar Rust CoreError fields are preserved for factor construction; unknown
  errors retain the documented explicit fallback.

## Validation and consequences

Python conformance compares the typed path to pre-existing golden source/task
fixtures, exact fingerprints, prices and complete risk reports. Additional cases
cover two-factor/rough SV and LSV, hybrid density inputs, rough-family LSV,
invalid compositions, conditional native covariance rejection and mutation of
original Python lists. Tests cover default absence semantics and immutable inputs.
Wheel checks independently freeze input types, keyword boundaries, defaults,
returns and static methods; negative mutations verify that the gate rejects drift.
The example, documentation and tests are required source-archive contents.

The additional immutable wrappers mirror existing Rust concepts. They do not
expand the model support matrix. Compile/calibration reuse across operations
remains a separate design item. This implements the API extensibility follow-up
to ADR 0030; existing statistical and cross-platform acceptance gates remain in
force and are not replaced by small replay fixtures.
