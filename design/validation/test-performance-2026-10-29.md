# Test execution performance — 2026-10-29

## Scope and invariants

The baseline is `bd5aaf216feb08ddd6e7d838cb51cb0132f0fe3e`. This change reduces
redundant evidence collection, conditional-payoff allocation, and the serial
critical paths of CI. It does not remove numerical cases, supported platforms,
independent references, seeds, sample populations, refinement panels, tolerances,
checkpoint replays or failure checks. Historical numerical acceptance reports
remain unchanged and are not reinterpreted as candidate acceptance.

## Scheduling

The previous Windows extended accuracy run executed 27 ignored tests in about
253.1 minutes. Four longest-processing-time partitions have historical estimates
of 63.4, 63.6, 63.3 and 62.8 minutes. The full risk suite remains a separate job.
Each platform builds the two release executables once. An independent reusable
workflow per OS starts its risk and accuracy consumers after that OS's build;
a queued build on another OS does not hold ready consumers. The
[per-platform workflow](../../.github/workflows/extended-acceptance.yml) uses
GitHub's supported [matrix workflow calls](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows#using-a-matrix-strategy-with-a-reusable-workflow).
Consumers verify the same binary hash, actual source fingerprint, platform and
compiled inventory.
Exact filters are checked against libtest discovery before execution. A normal
exit is insufficient without the expected successful executed-test count.
The original required-check names aggregate all selections and reject failed,
cancelled, missing or inconsistent evidence.

The previous Linux wheel `remaining` group spent about 93.7 minutes in its
examples and test modules. Four partitions have a historical critical path of
about 25.9 minutes. Every currently discovered module is scheduled, including
new modules absent from the timing hints. Examples run once. Module processes,
discovery multisets, provider inventories and native-failure propagation remain
in place. The matrix contains 36 provider selections plus four remaining groups
per platform, 120 wheel validation jobs in total. Legacy full/group commands
remain available.

These are scheduling estimates from previous completed jobs, not observed
candidate CI wall times. Runner queues, setup, build and concurrent capacity
affect actual results. The [retained schedule](test-performance-scheduling-2026-10-29.json)
lists every assignment and distinguishes estimates from measurements. Timing
sources and original checkout/PR revisions accompany the
[accuracy weights](../../scripts/extended_acceptance_timings.json) and
[wheel weights](../../scripts/wheel_validation/remaining_timings.json).

## Avoided evidence-only repricing

Provider ledger tests previously executed four native strategy runs, repriced
the same history solely to export audit evidence, then executed independent
partial and resumed backtests. The observer now captures a detached audit while
the real evaluation's same-snapshot cache is hot. Every original strategy call
still reaches the provider; independent resumes and failure repairs remain
outside the observer. Full snapshot/contract fingerprints, risk provenance,
repeat-risk equality and exact evidence-set coverage are checked.

An unchanged 4-interval, 4096-sample Call case took 11.428 seconds to export cold
evidence versus 0.002402 seconds to read the captured copy. The entire risk and
audit matched exactly. This measures one avoided evidence traversal, not the
whole ledger. The [record](test-performance-provider-evidence-2026-10-29.json)
identifies the installed binary separately from candidate native acceptance.
Reproduce with `scripts/benchmark_provider_evidence.py --output <new-json>`.

`PRICING_TEST_ORIGINAL_EVIDENCE=1` restores the previous evidence repricing
sequence. Combined reproduction aliases and native ASan diagnostics set it
automatically. Shorter ordinary test execution is not evidence that the unrelated
intermittent native fault has been repaired.

## Conditional-payoff allocation

The graph compiler normally emits consecutive slot IDs. Conditional-payoff
construction now stores those slots in one bounded vector instead of allocating
tree nodes for intermediate values. A sparse fallback retains arbitrary `u32`
IDs without allocating up to the largest ID. Operand evaluation, overwrites,
floating-point operations, branches, line/knot budgets and error behavior remain
unchanged; no `unsafe` code or public API was introduced.

Three paired release microbenchmarks, each using one million compilations per
case, measured 32.4–34.9% lower elapsed time for 4/6-observation Asian/Lookback
graphs. Full signed-Gamma payoff outputs matched in every pair. The
[measurements and source/binary hashes](test-performance-terminal-gamma-2026-10-29.json)
explicitly limit that result to the payoff compiler. The reproducible entry
point is `cargo run --locked --release -p pricing --example benchmark_terminal_gamma -- 1000000`.

A separate public `PricingPlan` Gamma-chain comparison uses freshly built
baseline and candidate wheels, 1024 samples, four workers and three alternating
run pairs on the same host. Median times improve by 18.0–30.5% across the
4/6-interval Asian/Lookback cases. Every input, direction, ordinary-price result
and complete audit matches exactly in every pair. The
[public-path report](test-performance-native-gamma-2026-10-29.json) binds the
baseline's clean `bd5aaf2` checkout, candidate source hashes and both native
binaries. This bounded benchmark supplements the unchanged 8192-sample
conditional-Gamma acceptance test; it does not replace its numerical policy.

## Validation

- Thirty-two native tests passed, covering terminal and conditional Gamma,
  sparse slots, overwrites, malformed operands, path Gamma, multi-step Gamma,
  and an independent decimal-price/initial-Vega reference.
- All 11 production extended-risk tests passed through the new binary consumer
  in 164.89 seconds. The compiled accuracy inventory contains all 27 cases.
- The provider's frozen domain remains 179 cases in 36 selections, including
  intentionally repeated coupled-basis coverage.
- Harness regressions exercise complete/disjoint partitions, newly discovered
  modules, preserved process isolation, real libtest selection and execution,
  changed source/binary rejection, missing reports, tampered logs, failed child
  exits, and original diagnostic evidence traversal.
- Seventeen existing numerical-reference and schema checkers passed.
- The unchanged 8192-sample Python conditional-AD acceptance passed on the
  candidate wheel in 33.189 seconds, including both six-interval Asian forms.
- Rust formatting, workspace/all-target/all-feature Clippy with warnings denied,
  API documentation, dependency direction, Markdown links and the complete
  staged source archive passed.
- A complete unchanged fixed-lookback ledger on candidate native binary
  `60402645…e5f4` passed all four strategies, four exact partial/resumed replay
  comparisons and eight snapshot/contract audits in 630.837 seconds. The
  4096-sample, seed-7, four-worker fixture is unchanged. The
  [candidate integration record](test-performance-candidate-ledger-2026-10-29.json)
  binds the implementation sources at `41f8388`, binary, settings and evidence
  hashes. This is a correctness run, not a paired performance comparison.
  The earlier 768-second helper run used the prior installed wheel and is not
  candidate native acceptance.
- All 15 macOS native debugger checks passed in the normal host environment,
  including real child exits, `exec`, SIGABRT and SIGTRAP. The first sandboxed
  attempt could not launch debugged children and is not counted as a pass.

The platform CI matrix is a separate verification step; scheduling estimates
and local benchmarks do not replace it.
