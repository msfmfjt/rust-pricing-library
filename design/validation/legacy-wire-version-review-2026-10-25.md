# Legacy wire input-version review — 2026-10-25

## Finding and scope

This continues the API review on PR #178, commit
`006ac9cbc573bd627efcdbcff268243a22df2ce7`, on
`fix/legacy-wire-error-schema`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`; no merge is authorized. The
separate mutation-testing work in #172 is not included.

P2: the legacy Python readers had the same input-version loss previously
corrected for ValuationReport in #177. Invalid Request/Result v1 or v2 documents
reported `ValidationIssue.schema_version == 3`; invalid PricingSpecification
v1 documents reported 2. For example, a negative Spot in a v1 Request identified
`/market/spot` correctly but associated that failure with the wrong input format.
Unsupported legacy Request features also substituted the current version.

The final new Python regressions produce **17 failing cases** against the
installed parent release wheel, all caused by these version substitutions.
Malformed or duplicated envelopes already use the intended fallback and pass.

## Boundary ownership and compatibility

Rust owns envelope context. Four additive readers expose it through the
existing `WireParseError` type:

- `parse_request_json_with_context`
- `parse_result_json_with_context`
- `parse_monte_carlo_result_json_with_context`
- `parse_specification_json_with_context`

They share the bounded JSON/envelope helper introduced for reports. Syntax,
duplicate-member and resource checks precede acquisition of any version.
Successful envelope decoding retains its declared version, including unsupported
versions and wrong-document-kind failures. Subsequent domain/type failures
retain that same context. Report parsing uses this shared helper without
changing its accepted inputs, diagnostics or v1 migration.

Original Rust reader signatures still return the same `WireError`, with the
same variants, source behavior, messages and pointers. No variant is added to
that enum. The new readers wrap its original typed error and expose an optional
version. There is no extra JSON parse in Python, no version extraction from
messages, and no additional JSON decode compared with the old reader path.
The legacy Result reader still tries rich Monte Carlo metadata first and falls
back to the plain result only for the same missing-metadata error.

Python corrects only the issue's input version on `from_json` failures. When a
valid envelope cannot be obtained, historical fallbacks remain 3 for Request/
Result and 2 for Specification/Report. Constructors and compilation are not
given artificial input provenance. In particular, a Specification correlation
error raised by `compile()` still identifies current format 2 after the parsed
source has been normalized. Task diagnostics retain the outer task version.

No accepted/rejected input domain, wire schema, successful serialization,
migration provenance, fingerprint, model/numerical algorithm, sampling/seed,
AAD or calibration policy, worker/block policy or numerical tolerance changes.

## Validation

- Rust workspace with all features, excluding the separately tested binding:
  **1,086 passed, zero failed, 91 existing ignored**, across 107 summaries.
- Binding Rust tests: **6 passed**. Explicit ignored statistical,
  path-dependence and early-exercise panels: **5 passed**.
- Four new Rust tests cover all four context readers, supported/unsupported
  versions, wrong kind, malformed bodies, missing/invalid envelopes, duplicate
  members, UTF-8/BOM, depth/byte/hard-cap limits, original error/source/pointer
  preservation, migration metadata and fingerprints. The five existing Report
  context/pointer tests also pass after helper reuse.
- The installed release wheel passes **165 Python tests**, including five new
  legacy-context methods and the expanded source/error/replay/model suites.
  Existing Specification tests distinguish parse input versions from the
  unchanged normalized compilation version. Package/ABI/SBOM/runtime checks
  validate **341 callables**; the Python API signature surface is unchanged.
- Eight parent/new captures of valid legacy Request, Result and Specification
  documents match exactly, including canonical JSON, migration/replay data,
  native prices/risks and source/plan fingerprints where applicable.
- **26 complete task/report captures match the parent byte for byte**: six
  task fixtures, 18 worker/block variants and both saved report migrations.
  The capture SHA-256 remains
  `1bace4c40954d228f6e36a726fdeced550bf029b1d833810552a191d4570a3e7`.
- **59 rejected-input comparisons** preserve exception class, message, issue
  dictionaries, code, stage, field, operation, reason and causes, except for
  exactly **17 intended input-version corrections**. The comparison includes
  domain/type/kind/version/feature errors and malformed/duplicated envelopes.
- Eighteen numerical-fixture/schema/documentation scripts and **45 checker
  tests** pass. Formatting, all-target/all-feature Clippy with warnings denied,
  API documentation and dependency-direction checks pass.
- Installed-wheel discovery finds **110 modules / 608 tests**, and the provider
  partition still covers **20 shards / 179 cases** with identical multisets.
  Discovery totals are not claims that every heavy Python case ran locally.

Full heavy Python validation, extended release model/risk panels and benchmarks
on all three platforms remain CI gates for this head.

## Provider/CI follow-up

The long-running local checks recorded as pending in the
[preceding checkpoint](provider-wheel-sharding-review-2026-10-25.md) completed:

- Four-interval coupled-basis Put fixed-lookback ledger: **passed in
  1,607.734 seconds**, including four strategies, exact checkpoint replay and
  three invalid-fixing repair paths. Evidence contains four replays, eight
  snapshot audits, three failures/repairs and one retained sample; its JSON
  SHA-256 is
  `57cd506cace2e62faab5b49bbea0a846ca50a75be032ac6740cb5e9f693beb48`.
- Complete `extension-all_rows-6` shard: **passed in 1,675.376 seconds**, with
  two test methods covering four native panels and both complete dated ledgers.

These are Linux results. The native macOS SIGABRT remains unresolved; no Rust
native-crash fix is claimed. #178's PR description records the completed local
results. At the latest pre-publication snapshot its CI had 14 successful,
three running, 70 queued and four skipped checks (91 created). All 63 planned
wheel selections must complete; queued/skipped work is not counted as passed.
Older failed Windows runs remain tracked in the prior review records.
