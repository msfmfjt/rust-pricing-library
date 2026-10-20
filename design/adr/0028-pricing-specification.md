# ADR 0028: Save complete pricing source compositions

Date: 2026-10-20. Status: implemented; additive API.

## Context

Specialized pricing plans accept model parameters, rate models, correlations,
particle settings and execution policy separately from `PricingRequest`.
Saving the request alone cannot reconstruct those plans and may describe only
the legacy carrier model. The architecture review's first implementation step
is a shared, extensible source specification.

## Decision

Introduce `PricingSpecification` with validated equity/rate composition and
explicit numerical policy. Keep immutable source data separate from compiled
plans and results. Reuse existing domain constructors, pricing backends and
static numerical kernels. A private legacy carrier bridges existing contractual
validation and compilation; it is never presented as the extension's actual
model in the public source or its serialized document.

Put source types in `api`, DTO conversion and hashing in `wire`, and backend
selection in the private engine. Rust and Python use that same compiler.
Use non-exhaustive typed composition/result enums and constructors for public
configuration structs. Dispatch once at compilation/evaluation boundaries;
introduce no per-step dynamic dispatch or numerical-policy change.

Create the independent `pricing_specification` v1 document and fingerprint
domain. Retain the actual SV/rough/rate model, LSV calibration sources, optional
market-IV provenance, random seeds, grid controls and execution policy.
Rebuild compiled state on load. Preserve native result types and diagnostics;
defer a common persisted result contract until the separate risk conventions
can be represented explicitly.

## Scope and consequences

- Support current standard `ModelSpec`, 1F/2F/rough Bergomi SV/LSV, supported
  Hull–White compositions, and the six deterministic-rate rough families.
- Preserve existing standard risk and LSM semantics. Extensions in this common
  entry point are price-only. Reject unavailable risk/LSM and unused settings;
  retain specialized APIs for broader existing capabilities.
- Retain existing product, matrix-PSD, rough-SABR LSV beta and resource checks
  in the selected compiler. The facade does not claim broader model support.
- Keep request/result v1/v2/v3 and native plan fingerprints unchanged. Legacy
  request migration metadata remains in memory; saving a new source normalizes
  it rather than inventing provenance for the new document.
- No runtime dependency is added. The independent offline schema validation
  gate uses a test-only `jsonschema` package.
- Follow ADR 0011's static boundaries. Multi-asset, stochastic dividends,
  specialized risk and universal result serialization remain later work.

## Verification

Compare common compilation/evaluation with native APIs, including complete
native results and plan fingerprints. Round-trip each factor and rough family,
retained IV targets, LSM, MC/RQMC, and escrowed dividends. Test source identity
against model/calibration/numerical changes, unknown or duplicated fields,
unsupported capabilities, parser limits and inconsistent target caches.
Keep fixtures, schema generator, wheel/stub contracts, docs and source-archive
inventory in CI. See the [usage and extension guide](../../docs/library/pricing-specification.md).
