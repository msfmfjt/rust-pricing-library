# Report input-version review — 2026-10-25

## Scope and reproduced finding

This continues the common report/API review on PR #176, commit
`81f70fb786a1bc5b567d7ae66f408a3c7a8dec1e`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The branch is
`fix/report-validation-schema-version`; the separate mutation-testing work in
#172 is not included or modified.

P2: Python `ValuationReport.from_json` reported schema version 2 for ordinary
failures in report-v1 input. The Rust reader knew the input envelope's version,
but discarded that context on failure; the Python adapter substituted the
current writer version. Changing a saved v1 report's `standard_error` to
`-1.0` therefore produced a correct field pointer after #176 but an incorrect
`ValidationIssue.schema_version`.

The expanded Python regressions were run against the release wheel built from
#176: **13 failing cases** reproduced the version mismatch. Rejected syntax and
limit cases already retained the intended fallback and passed.

## Responsibility and compatibility

The Rust wire boundary retains the version it already reads. The additive
`parse_valuation_report_json_with_context` entry point returns a
`WireParseError` containing the original `WireError` and optional input
version. Bounded JSON validation, duplicate-member rejection and complete
envelope decoding precede adoption of that version. Deserialization/domain
failures after that boundary retain the input version, including unsupported
versions. There is no Python JSON reparse and no inference from error messages.

Existing `parse_valuation_report_json` delegates to the same implementation and
unwraps the original typed error. Its signature, variants, pointers and messages
stay unchanged. `WireParseError` exposes borrowed/owned access to that error and
a standard error source chain. It does not add a variant to `WireError`.

Python uses this context for report issues. Its historical integer field still
falls back to the current report version 2 when no valid envelope was obtained;
the fallback does not identify the input version. Exception class, other issue
fields, message, code and stage are unchanged. Successful v1 reads retain the
existing migration to the v2 writer.

No wire schema, accepted input domain, numerical kernel, price/Greeks/AAD,
calibration algorithm, RNG coordinate, task/plan fingerprint or tolerance is changed.
Other legacy readers have not been converted to the new contextual entry point
in this batch.

## Local validation

- Workspace Rust tests with all features, excluding the separately tested Python
  crate: **1,082 passed, zero failed, 91 existing ignored**, across 106 summaries.
- Python-binding Rust tests: **6 passed**. Explicit ignored statistical,
  path-dependence and early-exercise acceptance panels: **5 passed**.
- Five Rust report-validation tests pass, covering both versions, exact
  pointers, typed source preservation, wrong kind, unsupported version,
  malformed/duplicate JSON, invalid envelopes, UTF-8/BOM, byte/depth limits,
  hard-cap overrides and unchanged valid v1 migration.
- The installed release wheel passes **159 related Python tests**, including
  all four report-validation methods and the cases that failed against #176.
  Package/ABI/SBOM and installed runtime contracts verify **341 callables**.
- **26 complete reports match #176 byte for byte**: all six task fixtures,
  18 execution-policy variants (workers 1/3 and block sizes 1/7/u64::MAX for
  standard, basket and stochastic-dividend BS), plus both saved report versions.
  The comparison includes every price, risk value/error, calibration diagnostic
  and replay field. The sorted JSON capture SHA-256 is
  `1bace4c40954d228f6e36a726fdeced550bf029b1d833810552a191d4570a3e7`.
- Twenty rejected-input comparisons preserve exception class/message and all
  issue/error metadata except nine intentional v1 version corrections. Inputs
  cover invalid price/sampling/replay/fingerprint fields, schema types, document
  kinds and unsupported versions in both frozen reports.
- Eighteen static/schema/documentation scripts and 34 checker tests pass.
  Formatting and diff whitespace checks pass.

All-target workspace check, all-target/all-feature Clippy with warnings denied,
API documentation and dependency-direction validation also pass. The full
installed-wheel discovery, heavy extended-model/risk panels, benchmark suites
and cross-platform validation remain CI gates for this head.

The initial binding-test link attempt could not find `libpython3.12`: this
runtime's development symlink points to a removed temporary directory and its
Python configuration declares `/install/lib`. A temporary linker search directory
pointing to the available `libpython3.12.so.1.0` restored linking. The successful
binding run uses that real library; no repository source or test expectation
was changed to address the environment issue.

The original [#176 record](valuation-report-pointer-review-2026-10-22.md)
retains its historical unavailable-runtime status. On this resumed review,
#176's two new Rust tests passed locally, and both new Python tests were
confirmed successful in the Linux/macOS/Windows `remaining` wheel-job logs.
Its PR description was updated to distinguish those successes from the
incomplete whole-CI gate.

## Inherited CI state and unresolved priorities

Complete paginated check-run snapshots were read for each exact head below.
Counts include all check runs returned by GitHub, including similarly named
jobs in separate workflows. No running or queued check was returned at this
snapshot. Skipped/cancelled checks are not counted as passed.

| PR | Head | Success | Failure | Cancelled | Skipped |
| --- | --- | ---: | ---: | ---: | ---: |
| #155 | `158155c` | 101 | 0 | 1 | 4 |
| #158 | `9fc6b46` | 102 | 0 | 0 | 4 |
| #160 | `f62aa72` | 101 | 0 | 1 | 4 |
| #161 | `cbd0f80` | 100 | 0 | 2 | 4 |
| #162 | `b31ee0d` | 100 | 0 | 2 | 4 |
| #163 | `0eb025f` | 100 | 1 | 1 | 4 |
| #164 | `76156e8` | 99 | 2 | 1 | 4 |
| #165 | `9b403e3` | 98 | 3 | 1 | 4 |
| #166 | `e39e4fa` | 86 | 1 | 8 | 4 |
| #167 | `c920048` | 88 | 1 | 6 | 4 |
| #168 | `29536cb` | 86 | 3 | 6 | 4 |
| #169 | `aab90e5` | 88 | 2 | 5 | 4 |
| #170 | `376c926` | 96 | 1 | 5 | 4 |
| #171 | `b84a76a` | 96 | 0 | 6 | 4 |
| #173 | `168f16c` | 96 | 0 | 6 | 4 |
| #174 | `5767556` | 92 | 2 | 8 | 4 |
| #175 | `f4d9a47` | 93 | 2 | 7 | 4 |
| #176 | `81f70fb` | 96 | 0 | 6 | 4 |

The dependency chain remains #155 → #158 → #160–#171 → #173–#176.
#172 is a separate branch from #170. All are unmerged.

**P1, unresolved:** the #176
[macOS providers log](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39857902552/job/120389224885)
records `SIGABRT` during the coupled-basis dated-ledger test after 9,295.452
seconds. Its Python stack reaches `multi_step_extension.py:164`, the call to
`evaluate_vega_kt_multi_step_gamma_chain_audit`. The previous completed case was
the four-interval Put Asian ledger; the next loop case is the Put Lookback ledger.
The log does not establish the abort's native cause. A subsequent extension
module completed, then the whole job was cancelled near six hours. This is a
real unresolved process failure, even though the check's final state is cancelled.

**P1, unresolved:** all six heavy `custom-basis`/`providers` wheel groups
on #176 are cancelled. The
[Linux custom-basis log](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39857902552/job/120389224928)
shows native panels passing and dated ledgers still running near six hours.
The
[Windows providers log](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39857902552/job/120389224757)
shows the first module taking 20,002.125 seconds and cancellation in the second.
Those durations are consistent with the hosted job limit; cancellation alone
does not prove every interrupted case would pass. Diagnose the abort and
partition long panels with an explicit complete-case inventory before treating
the release gate as satisfied. No tests, precision gates or sampling sizes
were reduced here.

Older failed Windows jobs also remain unresolved. For #175, the
[stochastic-dividend job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39836571500/job/120026360701)
log shows earlier panels passing and cancellation during rough-LSV Asian/Barrier
refinement. The
[extended-model job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39836571500/job/120026360421)
has a successful risk step but no completed accuracy-step result in its
metadata; its log still returns `BlobNotFound`. #176's corresponding numerical
jobs pass on all three OSes, which does not establish the causes of older failures.

Keep this follow-up in Draft pending the required gates. Leave the stack
unmerged until the user explicitly approves merging.

