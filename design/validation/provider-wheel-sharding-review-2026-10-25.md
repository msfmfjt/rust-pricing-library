# Provider wheel validation review — 2026-10-25

## Scope and observed failures

This continues the review on PR #177, commit
`b778164778b76c72d039081df1e22d7e6a533880`, on branch
`fix/wheel-provider-validation-shards`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The separate mutation-testing
work in #172 is not changed. No PR in this review stack has been merged.

P1 validation gap: all six `custom-basis`/`providers` wheel jobs on #176
were cancelled around the hosted runner's six-hour limit. The
[Windows providers job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39857902552/job/120389224757)
completed the coupled-basis module successfully in **20,002.125 seconds**,
then ran out of time during the next module. The
[Linux custom-basis job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39857902552/job/120389224928)
also exhausted the limit within a single module. Module isolation alone
therefore cannot complete these gates.

The [macOS providers job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39857902552/job/120389224885)
has a separate native failure: `SIGABRT` during the coupled-basis dated-ledger
method, after **9,295.452 seconds**. Its last completed case was four-interval
Put arithmetic Asian; the next case in the original loop is four-interval Put
fixed lookback. That identifies a reproduction candidate, not a proven native
cause. The Python stack ends at the native Gamma-chain audit call. No allocator,
Rust panic or stack-overflow diagnosis appears in the retained log. The runner
continued with another expensive module, so the eventual job cancellation
obscured the earlier process failure in the top-level check status.

## Changes and preserved coverage

CI builds each platform wheel once and consumes that same artifact in **21
validation selections per platform**: 16 numerical provider shards, four
contracts shards, and `remaining`. The plan contains 63 jobs across Linux,
macOS and Windows. A numerical shard selects one basis/extension and four or
six intervals; it keeps all products, sides, strategies, checkpoint replays,
settlement checks and repair cases for that selection. Contracts retain their
original full policy configurations. Each selection has a 330-minute limit.

The numerical loops and discovery use the same explicit case lists. Before
every selection, the installed-wheel interpreter verifies the complete module
test-ID multiset and the expanded provider method/case multiset. All **179
provider cases** are covered exactly with their original multiplicities,
including coupled-basis coverage in both the generic and dedicated modules.
Missing, extra, duplicate or empty selections fail. New ordinary test modules
join `remaining`, and new unparameterized provider methods join contracts.
An independent frozen-domain regression checks products, policies, interval
counts, sides and ordering against the pre-change test definitions.

The default full command and legacy group commands still run complete modules.
Legacy numerical acceptance scripts retain their five/six-method interfaces
and full evidence dictionaries. Their frozen-source inventories now include
the new case helper. No historical evidence is rewritten, and source/binary
matching, exact numerical comparisons, and the existing bin-label-only ULP
rule are unchanged.

Each shard has its own process and class setup. State remains shared within a
shard; process/class lifetimes across policy and interval shards differ from a
full-module run. No production cache policy is changed. This partition is not
evidence that a failure depending on the old process lifetime has been fixed.

Test subprocesses save a running record before launch and a completion record
with elapsed time, exit code and signal. The full inventory is saved separately.
A crashed shard fails immediately rather than starting another heavy module.
Every numerical case logs its method and parameters before evaluation. CI
attempts to preserve these records and any available macOS Python `.ips`/`.crash`
reports after failure or cancellation. Missing/running records are incomplete
evidence, never success. Package/example failures still fail validation.

No Rust, binding, backtest production code, wire schema, model, numerical
algorithm, seed, sampling count, worker/block policy or tolerance is changed.
An AST comparison confirms that all 31 existing functions in the three edited
test modules retain their calculation/assertion bodies; only case iteration
and provider lookup move. A second comparison confirms that the four historical
acceptance checkers only add the new helper's source hash.

## Local validation

- The complete installed-wheel inventory is **109 modules / 603 tests**;
  expanded provider coverage is **20 shards / 179 cases**. These are discovery
  counts, not claims that all numerical tests ran locally.
- **45 checker tests** pass: runner 15, shard plan/archive gate 7, wheel contracts 15,
  and audit parser 8. They cover omitted/repeated/unknown cases, aliased test
  IDs, inherited configuration restoration, class setup/teardown, a complete
  three-platform matrix, signal/launch failures, incomplete execution records,
  and unchanged default/group dispatch.
- The installed-wheel frozen provider-domain regression passes.
- All **14 provider-cache regressions** pass, including single-snapshot
  retention, eviction, detached audit/evidence lifetime and concurrency guards.
- All four contracts shards pass: **11 methods**, with real native risk,
  atomic failure/retry checks, immutable/detached audit checks and CLI gates.
  Measured elapsed times are 69.077 seconds (custom), 13.694 seconds (coupled),
  25.452 seconds (extension) and 21.652 seconds (Put extension).
- Eighteen independent numerical-fixture/schema/documentation scripts pass.
  Rust formatting and diff whitespace checks pass. Workflow YAML and its
  embedded Python parse, and the generated matrix is checked against all
  required selections.

At the pre-publication checkpoint, the full `extension-all_rows-6` numerical
shard and the original four-interval coupled-basis Put fixed-lookback ledger
reproduction are still running; this record does not count them as passes.
Their final outcomes are tracked in the pull request. They use the installed
#177 release wheel with unchanged native and
production sources. Full heavy suites on all three operating systems and the
newly built wheel remain CI gates. Previous #177 Rust/numerical validation is
recorded in the [preceding review](report-schema-version-review-2026-10-25.md),
not counted as a new run here.

## Outstanding gates

- The native macOS abort is unresolved. A successful Linux case cannot establish
  a macOS fix; inspect the new case markers and any native crash report if it
  recurs.
- All 63 validation selections must complete on the new head. The Windows
  coupled-basis log supports splitting by interval count, but runtime on the
  new matrix must be observed; incomplete jobs cannot be waived.
- Older Windows extended-model/stochastic-dividend failures remain tracked in
  the preceding review. #176's corresponding Rust gates passed on all three
  platforms; this does not retroactively turn the older failed runs into passes.
- At 2026-10-24 23:42 UTC, #177 had 20 successful, 14 running, 59 queued and
  four skipped checks (97 created checks). This is a snapshot, not a completed
  CI result. No merge is authorized by this review.
