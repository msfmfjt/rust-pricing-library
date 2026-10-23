# Rust mutation-testing rollout — 2026-10-22

Base: `376c92624f88083b46d86e1087d2b0f03e202b91` (PR #170). Implements a testing
improvement under the existing contributor validation requirements; it does not
change a normative pricing requirement or acceptance policy.

## Changes and compatibility

- Pin cargo-mutants 27.1.0; add scoped local execution, a PR smoke gate, changed-line
  checks against the actual PR base, and four-shard manual surveys.
- Retain selection, commands, source HEAD, local dirty status, outcomes, logs,
  mutation patches and job summaries. Propagate failures, baseline errors and
  timeouts; reject incomplete/check-only evidence and distinguish an empty diff.
- Run both Rust crates' tests for changed/full surveys so numerical utility
  mutations face their consumers. The smoke scope is deliberately smaller.
- Strengthen finite-number and deterministic reduction unit tests using exact
  boundaries and independent hand-computed statistical references.

Production Rust statements, public Rust/Python APIs, wire schemas, numerical
algorithms, RNG sequences, replay conventions and acceptance tolerances are
unchanged. No dependency is added to the library or wheel. CI omits test debug
symbols using an environment setting; production/test optimization profiles are
unchanged. No mutation is excluded just to improve a score.

## Measured pilots

Linux, Rust 1.98.1, cargo-mutants 27.1.0. These are **focused unit-test scopes**;
survivors before the change are not a claim that downstream integration tests
also missed them. Raw count summaries, commands, outstanding mutation names and
tested-file hashes are retained in the [machine-readable record](mutation-testing-2026-10-22.json).

| Mutated module / tests | State | Selected | Caught | Missed | Timeout | Unviable |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `core::number` / its unit tests | Before | 15 | 4 | 11 | 0 | 0 |
| `core::number` / its unit tests | After | 15 | 15 | 0 | 0 | 0 |
| `pricing-numerics::reduction` / numerical crate unit tests | Before | 181 | 155 | 24 | 1 | 1 |
| `pricing-numerics::reduction` / numerical crate unit tests | After | 181 | 179 | 0 | 1 | 1 |

The first scope passed through the new smoke runner. The reduction survey used
`cargo mutants --package pricing-numerics --file crates/pricing-numerics/src/reduction.rs
--cargo-arg=--lib --jobs 2 --build-timeout 300`, with a test timeout of 60 seconds
before and 20 seconds after. Its configuration supplies `--locked` and all
features. The shorter final limit bounds the already-identified infinite loop;
neither timeout is counted as caught. The completed reduction survey returns
exit **3**, not success.

The added tests detect constant-return getters, non-reflexive equality, missing
Display output, compensation when the larger operand arrives second, incorrect
two-pass centered moments, incorrect weighting of unequal partitions, and the
one-observation variance/covariance boundary. The independent unequal-partition
references are mean 5, second moment 82, sample variance 20.5, cross-moment 44 and
sample covariance 11.

Two outcomes remain explicit:

1. Changing `fixed_tree_reduce`'s `while level.len() > 1` to `>= 1` loops forever
   on the final singleton. It is a real nonterminating mutant, reported as a
   timeout. No exclusion or timeout-as-caught rule was introduced.
2. Returning `Some(Default::default())` from the generic reducer does not compile
   because `T: Copy` does not require `Default`. It is unviable, not caught.

## Baseline-scope regression and real-tool verification

An installed-tool test uses two small temporary Rust crates with no numerical
crate tests; only the downstream `pricing` integration test checks the function.
It validates four real generated mutants and three independent outcomes:

- Correct assertions: four caught; exit 0.
- Weak assertions: four missed; exit 2.
- An already-failing downstream test: baseline fails; exit 4 and no mutants run.

This exposed an upstream 27.1.0 behavior in
[`Lab::run_baseline`](https://github.com/sourcefrog/cargo-mutants/blob/v27.1.0/src/lab.rs):
the baseline selects only packages containing generated mutants, even when
`--test-package` requests other packages for the mutant runs. Initially, the
third fixture falsely reported all four mutations caught. The runner now also
passes the test packages through `--cargo-arg=--package=...` so baseline and
mutant commands have the same package scope, and validates the recorded baseline
command. The real-tool regression passes with that fix.

## Other verification and limits

- Nine runner unit tests cover actual Git merge-base selection on an advancing
  stacked parent, local edits, untracked new source rejection, tool error codes,
  empty selections and incomplete/incorrect evidence.
- All 23 `pricing-numerics` unit tests and its doc-test target passed. The smoke
  run also executes all five `core::number` unit tests before mutation.
- Clippy passed for both Rust crates with all targets/features and warnings denied.
- Full inventory only: **34,424 mutations across 263 production files**. Product
  Graph and Pricing Task implementation files are included; `pricing-python`
  is excluded. This inventory was not executed as a complete survey.
- The current PR changes test bodies and infrastructure, so its incremental
  production-code mutation selection is empty and explicitly reports
  `no_mutants`; the smoke job still exercises actual mutations.

Full-library mutation coverage, ignored numerical acceptance tests, Python
mutation coverage and current-head remote CI success are not inferred from these
pilots. Large surveys should use focused file selections and all required shards.
The [operating guide](../../docs/library/mutation-testing.md) documents the scopes,
evidence interpretation and failure handling.
