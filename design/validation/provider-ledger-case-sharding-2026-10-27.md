# Provider ledger case sharding — 2026-10-27

## Observed validation gap

This follows PR #188, commit `0382aed423d090dc69a86ff8cc5ff36300eac051`.
Main remains `45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`; the review stack is
unmerged. The separate mutation-testing work in #172 is not changed.

P1: a required numerical gate still cannot finish with the previous partition.
On PR #187, the [Windows four-interval coupled-basis job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40438388913/job/122109294601)
was cancelled after **330 minutes 23 seconds**, at the configured 330-minute
job limit. Its ten native panels and first three dated ledgers passed; the
fourth ledger (Put fixed lookback) was interrupted. The completed ledgers took
approximately **88.7, 70.5 and 88.4 minutes**, before the final case had run for
another 69.6 minutes. The log reports cancellation without a more specific
reason; timing is consistent with the configured timeout. Retained case markers
and timestamps are in the [evidence record](provider-ledger-case-sharding-2026-10-27.json).
This incomplete job is not a numerical pass or evidence of a numerical failure.

The earlier Windows custom-basis/coupled-row-hats job on #186 was also cancelled.
The completed six-interval named-basis jobs on #187 took at most about 132 minutes;
the four-interval weighted-time case took about 239 minutes. We split the shared
four-interval named-basis workload, leaving the shorter six-interval and extension
groupings unchanged. Timeout, sample counts, seeds, worker counts, tolerances and
numerical assertions are unchanged.

## Partition and compatibility

Each of the four original four-interval named-basis shards becomes five CI
selections: all ten native panels, then one dated ledger for each Call/Put and
arithmetic-Asian/fixed-lookback combination. Each ledger retains its complete
four strategies, settlement checks, snapshots, checkpoint replays, wrong-provider
rejection and atomic failure/repair checks. Duplicate coverage in the custom and
dedicated coupled-basis suites is retained.

The matrix contains **36 provider shards plus `remaining` per platform**,
**111 jobs** across Linux, macOS and Windows. The same producer wheel is consumed
by every validation selection. The plan remains generated from one Python
definition; every selection is required and native failures are not retried.

`configured` applies method/case filters to the existing case lists and restores
the class descriptor and axes in `finally`, including inherited attributes.
Inventory and execution use the same selected cases. The full module/test-ID
Counter and provider method/parameter Counter remain mandatory before execution.
Missing, repeated, unknown or empty cases fail; independent frozen domains retain
the original products, policies, sides, intervals, order and multiplicity.

The four original combined shard names remain accepted by `--shard`, with their
original ordering and shared provider/class state, but are excluded from CI to
avoid duplicate execution. Full-suite and `--group` commands and the historical
acceptance scripts retain their behavior. No historical numerical report is
rewritten or automatically accepted against a new source hash.

The shorter CI process lifetimes change cache reuse **between** dated cases.
They do not change production cache policy or reuse **within** a case. Keeping
the original combined commands matters for investigating native faults that
might depend on lifetime; success in the new partition is not a native-crash fix.

## Local validation

- Six checker suites: **62 tests passed, four macOS-only integration tests
  skipped on Linux** (66 discovered). Coverage includes exact matrix membership,
  missing/overlapping ledgers, duplicate parameter multiplicity, unknown and empty
  selections, inherited descriptor restoration, class setup/teardown, combined
  alias ordering, failure propagation, and a real child SIGTRAP remaining a failure.
- Both installed-wheel provider inventory regressions pass: **179 provider
  cases in 36 shards**, with the four combined aliases preserving each original
  14-case sequence. Complete discovery is **111 modules / 613 tests**; discovery
  counts do not claim execution of the full numerical suite.
- The unchanged #188 release wheel passes archive/ABI/metadata/SBOM/installation
  gates and all **341 callable contracts** in a fresh environment. Its SHA-256 is
  `f52e51c2e29e4bcb3b458acc047b63365987a98897750625204075b09fa1ed0b`.
- Eighteen numerical-fixture/schema/documentation checkers, Rust formatting and
  the staged source-archive gate pass. All Rust, Python binding and backtest production sources, the four provider
  test bodies, manifests/lockfile and workflow YAML are byte-identical to #188.
- The full four-interval coupled-basis native panel and Put fixed-lookback ledger
  shards are running locally through the installed-wheel runner; neither is
  counted as passed at this pre-publication checkpoint.

## Outstanding gates

All new CI selections must complete on the published head, including the real
Windows timing checks and macOS LLDB runs. Local Linux validation does not replace
the supported-platform frozen replay/benchmark gates or the extended risk and
price panels. The previous #188 numerical evidence remains recorded in its
[division AAD review](graph-division-review-2026-10-27.md), not counted as new runs.

The two #186 macOS invalid frees remain unresolved. Their captured system-workqueue
backtraces stop in `libdispatch` continuation-cache cleanup during pthread TLS
destruction; pricing threads are active in the Gamma-chain calculation. The
detection site does not identify where corruption began. This partition does not
change Rust/PyO3, numerical execution, allocators or native dependencies and does
not establish a repair. Retain the earlier failed evidence and combined commands
even if the shorter jobs pass. No merge is authorized by this review.
