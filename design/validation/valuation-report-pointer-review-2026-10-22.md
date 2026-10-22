# Valuation-report error-location review — 2026-10-22

## Scope and finding

This continues the common-report/API review on PR #175, commit
`f4d9a47b5ed3779e63eea1413c4dc5aeef129c36`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The separate mutation-testing
branch in #172 is not included.

P2: `ValuationReport::validate` used the constant JSON pointer `/report` for
price/sampling, replay metadata and risk-block domain failures. Both report
schemas store these fields at the document root. There is no `report` member
in either frozen golden document. A caller resolving an error pointer cannot
locate the rejected field or risk block.

Concrete case: change `standard_error` to `-1.0` in either saved report
fixture. The validator's nonnegative-error check selects `DomainAt`, but the
old pointer names a nonexistent field. Python forwards the same pointer into
`ValidationError.field` and `ValidationIssue.pointer`.

## Change and responsibility

Return the actual scalar or replay field for those checks. Risk identity,
shape, axis and uncertainty checks return `/risks/{index}`, a real containing
block appropriate to validations involving multiple fields. Keep existing
calibration-diagnostic and wire-decoding locations.

The common report validates its own domain invariants. The wire reader and
Python boundary propagate the resulting location; they do not duplicate the
validation rules or infer locations from messages. This extends the diagnostic
location work in the [task-validation review](typed-api-task-validation-review-2026-10-22.md)
to reports.

No validation predicate, accepted value domain, schema, numerical kernel, risk
calculation, calibration, tolerance, random stream or fingerprint changes.
Messages and exception classes stay the same. The corrected pointer/field is
an intentional observable metadata change.

## Regression coverage and validation status

Added [Rust regressions](../../crates/pricing/tests/valuation_report_validation.rs)
and [Python regressions](../../tests/python/test_valuation_report_validation.py)
use both frozen report versions. They cover negative standard error, zero
sampling counts, invalid replay metadata, invalid risk values/errors/conventions
and duplicate risks. Every expected pointer must resolve in the submitted JSON;
Python also checks code, parse stage, exception class and issue-pointer agreement.
The duplicate case checks the appended block's index rather than the first
valid occurrence.

**The new Rust and Python tests have not been executed.** The selected execution
environment is unavailable; the command tool returned
`No usable execution environment is available`. Source and fixture inspection,
patch construction and publication use the GitHub connector. No local Cargo,
Clippy, formatter, wheel, source-archive or numerical replay success is claimed
for this commit. Earlier #175 test results do not validate this change.

Source inspection confirms that both fixtures lack `/report`, that the old
validator emits that pointer, and that the new regression locations exist.
This is static evidence, not a substitute for executing the regressions.

Required follow-up before merge: execute the added tests and the repository's
normal Rust, installed-wheel, schema/documentation and numerical CI gates,
including valid-report replay. Keep the PR in draft while those results are
pending. No merge is performed by this review.

## Remaining review items

The model/capability and Product Graph paths inspected in this batch did not
produce another confirmed finding. That is not an exhaustive approval of those
subsystems. The report boundary still needs a separate assessment of diagnostic
schema-version metadata when reporting errors for legacy v1 inputs.

At this batch's initial CI check, #170 had 85 successful checks, #171 had 69,
and #173 had 37; #174 was starting and #175 remained queued. No new failures
were present in those heads. Older Windows extended-model failures in #164/#165
still return `BlobNotFound` when their logs are fetched and remain unresolved.
