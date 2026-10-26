# Dynamic executor vector allocation review — 2026-10-26

This follows [PR #184](https://github.com/msfmfjt/rust-pricing-library/pull/184),
commit `b8c04850beb399629a12978331335b6482a458ce`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`; the review stack is unmerged.
The separate mutation-testing work in PR #172 is outside this change.

## P2: public dynamic result widths could panic instead of returning an error

`DeterministicExecutor::try_map_reduce_statistics_vector` accepts a caller-owned
`usize` component count and returns a typed `Result`. The
[earlier resource review](executor-resource-review-2026-10-22.md) made the logical
block table fallible, but the dynamic marginal, callback, covariance and output
buffers still used infallible allocation.

At the parent commit, both public regression cases below panic with
`capacity overflow`, without evaluating a numerical callback:

```rust
executor.try_map_reduce_statistics_vector(1, usize::MAX, |_, _| Ok::<_, Infallible>(()));
executor.try_map_reduce_statistics_vector(0, usize::MAX, |_, _| Ok::<_, Infallible>(()));
```

The first panic occurs in a Rayon worker. The second occurs while collecting
the empty calculation's output. `catch_unwind` in the public tests confirms
that neither call returns the promised error path. The two tests failed before
the implementation changed and pass afterward.

## Boundary and behavior

The numerical executor owns the fix. Product Graph, Pricing Task, model
definitions and calibration inputs do not acquire a width limit or allocation
policy. Dynamic statistics and callback buffers use `try_reserve_exact` before
initialization. The result buffers are reserved before numerical callbacks, so
an impossible output width is rejected even for an empty sampling plan. The
optional upper-triangular covariance buffers use the same fallible construction.

`ExecutionError`, already non-exhaustive, gains the additive
`VectorStatisticsAllocationFailed { components }` variant. Existing block-table
errors retain their precedence. The Python classifier preserves `resource_limit`
and reports `statistics_components` for this variant; existing block failures
continue to report `reduction_blocks`. A binding test covers the full
executor/MonteCarloError/PricingTaskError source chain.

Block partitioning, sampling order, callback zeroing, compensated sums, centered
moments, covariance accumulation and ordered reductions are unchanged. The
result buffers live earlier in the calculation; the patch makes no performance
or global memory-bound claim. In particular, OS overcommit termination and
other allocations, including scratch vectors indexed by block count, are not
made universally recoverable by this change.

This is consistent with the existing
[requirements](../requirements-v1.0.md) for deterministic reductions and
optional centered covariance, and the
[private execution boundary](../../CONTRIBUTING.md#internal-boundaries-after-crate-consolidation).
There is no wire-schema, RNG, tolerance, calibration-policy or acceptance-gate
change.

## Validation

The five public tests cover impossible widths with zero/nonzero sampling,
executor reuse after failure, empty vectors, positive-zero callback resets,
partial writes, ordered evaluation errors and equality of every exposed
statistic bit against independent scalar reductions. Successful controls use
one/three workers and block sizes 1, 7 and `u64::MAX`. A private regression
covers the covariance entry point. The existing centered-covariance numerical
test remains unchanged.

The reproducible
[capture script](../../scripts/capture_executor_vector_replay.py) compares
complete installed-wheel outputs, with no numeric comparison tolerance:

- 54 Task/report pairs: nine original source fixtures, each with its original
  policy, three workers, block size seven, disabled antithetic sampling, and
  explicit Brownian bridge off/on. These cover single/multi-asset graphs,
  rates, rough/LSV models, dividends, Greeks and calibration diagnostics.
- 16 continuous-barrier reporting-IV results: the existing covariance test's
  complete pseudo-MC/RQMC and antithetic matrix, one/three workers, and full
  covariance off/on. Every public result property, including nested price,
  error estimates, covariance and fingerprints, is captured.

The parent wheel comes from run
[40302138642](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40302138642),
artifact `12646113731`. Its downloaded ZIP SHA-256 was verified as
`bd56f1dc1088904ecc7cc955b0e195434f89a1fe86769dd7426fe9934cca735c`;
the contained CPython 3.12 Linux wheel has SHA-256
`21d1a344eae2d8c291db97f5c1853b5e85a70690af8966ce285825be036fbfa3`.

All **70 complete outputs are byte-identical** between parent and candidate.
Both captures have SHA-256
`d1c94f2fe16e96f4a13c9385c20c492e701f4a2d47a5fda450dc665e9e8be816`.
To reproduce, run this script in separate environments containing the two
wheels and the repository's validation dependencies, then compare the files:

```sh
python scripts/capture_executor_vector_replay.py /tmp/vector-replay.json
cmp /tmp/parent-vector-replay.json /tmp/vector-replay.json
```

Local validation uses the pinned Rust 1.98.1 toolchain and CPython 3.12.
The release wheel passes archive/metadata/SBOM/ABI, clean-install and all 341
runtime callable signature checks. Its SHA-256 is
`d60393afcc12cd4a188fbc08c70104411620fa1dd6ce3f45c3b4805042b8fb1b`.

- Workspace/all-feature tests excluding the Python binding: **1,102 passed,
  zero failed**, with 91 existing ignored acceptance cases.
- Pricing without default features: **1,083 passed**. The five explicitly
  selected statistical, path-dependence and early-exercise acceptance tests
  also pass without changing their sample counts or tolerances.
- Release-mode extended risk acceptance: **11 passed**, including all 738
  AAD/recalibration finite-difference sweeps with no reported gate violations.
- Native Python binding: **7 passed**. The host's relocated Python library
  needed an isolated PyO3 link configuration; repository build settings were
  not changed.
- Installed wheel: **159 tests in 34 modules**, the existing joint covariance
  test, and all **42 examples** pass. All 77 process records confirm completion
  and exit zero. Full import-only discovery checks the 110-module/610-method
  inventory and the unchanged 20-shard/179-case provider partition; that
  inventory check is not execution of the additional heavy Python tests.
- All-target/all-feature Clippy with `-D warnings`, the all-target workspace
  check, API docs, dependency-direction check, all 18 reference/schema/link
  scripts, and the checker unit suites pass.
  The three real macOS debugger integration tests are skipped on this Linux
  host and are not counted as passing debugger evidence.

The full local benchmark harness is **not a passing gate**: the Rust early
exercise benchmark exited zero, but the harness failed to observe positive
peak RSS for its launched process. A separate probe showed that the host
exposes an instrumentation wrapper at the child's `/proc` entry. No fabricated
memory figure or relaxed benchmark check is substituted. A separately generated
European replay is rejected by the frozen-fixture checker because
`linux-x86_64` is outside its supported macOS/Windows platform set. Neither
limitation is a numerical mismatch; the 70-output parent comparison above is
the local byte-exact numerical evidence. Cross-platform frozen replay and full
benchmark validation remain CI gates.

## Outstanding platform investigation

This width error is independently reproducible and does not explain the
intermittent macOS native invalid-free detection. The
[retained crash evidence and LLDB launch review](lldb-launch-review-2026-10-26.md)
remain the basis for that investigation. At review time, #184's main macOS
Rust/debugger integration job and many wheel shards are still queued. Success
of a different macOS model job is not evidence that the debugger or previously
failing ledger selection has passed.

No PR has been merged. The new branch remains dependent on #184; platform CI
and explicit user approval are required before merging the stack.
