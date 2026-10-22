# ADR 0040: Preserve typed multi-asset numerical causes

Date: 2026-10-22. Status: proposed, implemented on the review branch.

## Context and decision

The experimental basket adapter erased native numerical failures using
`MultiAssetError::Numerical(error.to_string())`. Public Rust callers could not
recover the executor, model or payoff failure. The common Python classifier
could only report a generic basket numerical error, losing distinctions already
available for single-asset tasks. The legacy Python basket compiler also
discarded causes when building its validation issue.

Add `NumericalSource(Box<dyn Error + Send + Sync>)` and make `MultiAssetError`
non-exhaustive. Its private numerical conversion requires an actual `Error`,
retains it and exposes it through `Error::source()`. Keep `Numerical(String)`
for existing text-only diagnostics, including the two local-correlation
feasibility failures that currently have no typed error. A future conversion
cannot silently pass a formatted string through the typed helper.

Python classifies retained causes using the shared variant-based classifier.
Unknown causes keep the existing `multi_asset_numerical_error` fallback; there
is no string parsing. The legacy basket compiler preserves `ValidationError`
and its issue payload while supplying compile-stage metadata from the native
error. PricingTask retains its existing `PricingError` boundary.

The numerical engine and models own the original errors. Product and task
adapters preserve those errors; the Python boundary owns user-facing codes.
No model-specific error-code logic is added to the executor or task facade.
No new dependency or crate is introduced.

## Alternatives

- Retaining only strings cannot support reliable classification or downcasting.
- Parsing diagnostic text would violate ADR 0031 and couple callers to wording.
- Adding a separate variant for every model or executor error would repeatedly
  grow the public basket enum and complicate recursive evaluation wrappers.
- Changing the payload of `Numerical(String)` would also break existing variant
  construction. Keeping it provides a migration path for text-only diagnostics.

## Compatibility and reproducibility

- This is a source compatibility change to an experimental Rust API. External
  exhaustive matches require a wildcard. Numerical-error handlers must accept
  the new variant or inspect the source chain. Existing variant constructors
  and function signatures remain available. See the
  [migration example](../../docs/models/multi-asset.md#rust-error-handling).
- The error remains `Send + Sync`. An arbitrary boxed `Error` does not guarantee
  `UnwindSafe` or `RefUnwindSafe`; callers requiring those auto traits must review
  their unwind boundary rather than assuming the old enum's derived properties.
- Python exception classes, messages, `args` and historical issue fields remain
  intact. Exception codes/fields become more specific and causes become richer.
  Native compiler failures in `MultiAssetPlan.compile()` now identify `compile`
  instead of `validation`; argument preflight still uses `validation`.
- No Python signatures, schemas, serialization, fingerprint domains, random
  streams, reductions, path evolution, AAD, calibration formulas or tolerances
  change. Boxing occurs only when a failure is returned.

## Validation

Public Rust regressions downcast execution, RQMC and graph causes through a
compiled common task. Graph errors retain sampling unit 0 with one and three
workers for price and AAD. Binding tests exercise nested calibration errors and
unknown error messages that look like resource/calibration failures. Installed
Python tests cover the legacy and common-task compile/evaluate boundaries,
exception inheritance, issue fields and detailed allocation-error codes.

Compare complete successful reports and existing failure messages against the
parent wheel with exact equality. Record local and hosted validation separately
in the [review evidence](../validation/multi-asset-error-review-2026-10-22.md).
This implements the existing typed-error requirement and ADR 0031; the numerical
acceptance gates and release requirements remain unchanged.
