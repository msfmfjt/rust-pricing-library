# Mutation testing

Mutation testing deliberately changes production code and checks whether tests
fail. A surviving (`missed`) mutant identifies a test gap or an equivalent change
that needs review. It is different from line coverage and from the existing
tests that corrupt JSON fixtures or CI contracts.

The repository uses [cargo-mutants](https://mutants.rs/) **27.1.0**, with the Rust
toolchain in `rust-toolchain.toml`. Install from the repository root:

```bash
cargo install cargo-mutants --version 27.1.0 --locked
python -m unittest discover -s scripts -p 'test_mutation_integration.py' -v
python scripts/run_mutation_tests.py smoke
```

## Scopes and tests

| Scope | Mutated code | Tests run | Use |
| --- | --- | --- | --- |
| `smoke` | `pricing::core::number` | Its five unit tests | Always exercise a small, deterministic mutation gate |
| `changed --base REF` | Changed production Rust lines since the merge base with `REF` | Both `pricing` and `pricing-numerics`, including their integration and doc tests | PRs, including stacked PRs |
| `full` | All production Rust in the selected package(s), optionally filtered/sharded | Both Rust crates, including downstream consumers of numerical utilities | Manual surveys and focused follow-up |

The global [configuration](../../.cargo/mutants.toml) excludes test sources,
examples, benchmarks and `pricing-python`. Inline `#[cfg(test)]` modules are
skipped by cargo-mutants. There is no blanket arithmetic/error-path exclusion or
percentage threshold. Filters on `full` and `changed` select **mutations**, not
tests; selecting only `pricing-numerics` still runs `pricing` tests.

Normal Rust tests marked `#[ignore]` remain ignored. Heavy statistical and
installed-wheel acceptance jobs continue to run in ordinary CI. Python bindings
and the Python backtesting package are **not mutation-tested** by this workflow:
the binding would require rebuilding and installing a wheel for each mutation.
Mutation results do not certify stochastic accuracy or Python coverage.

## Local use

Inspect the workload before starting a large survey:

```bash
python scripts/run_mutation_tests.py full --package pricing-numerics --list
python scripts/run_mutation_tests.py full --file 'crates/pricing/src/product/*.rs' --list

# Use the actual parent branch for a stacked PR, not automatically main.
python scripts/run_mutation_tests.py changed --base origin/fix/product-graph-task-review

# A focused run still includes the integration tests of both Rust crates.
python scripts/run_mutation_tests.py full --package pricing-numerics --file 'crates/pricing-numerics/src/normal.rs'
```

Run from the checked-out repository, after installing the pinned toolchain.
`changed` includes staged and unstaged tracked edits. Stage new `.rs` files with
`git add` first; the runner rejects untracked production Rust so it cannot silently
omit a new file. The diff has fixed prefixes, no external text conversion, and
renames represented as additions/deletions. A test-only change can have zero
changed-line mutants; its status is explicitly `no_mutants`. The smoke gate
still runs. Use a focused `full` survey when changing tests or removing behavior.

All runs first validate an unmutated baseline. Source mutations occur in temporary
copies; the working checkout is preserved. Default parallelism is two workers
(`--jobs N`). Baseline builds have a 600-second limit; mutant test timeouts use
three times the baseline duration with a 60-second minimum. Smoke uses a fixed
60-second test limit. CI adds an overall job deadline. If necessary, reduce the
file scope or use shards rather than weakening numerical assertions.

In 27.1.0, `--test-package` alone expands the mutant tests but not the baseline.
The runner also passes both packages as Cargo arguments, and checks the actual
baseline command in the result. This prevents already-failing consumer tests
from making every numerical mutant appear caught and ensures timeout calibration
uses the same test scope. Do not remove these seemingly duplicate package flags.

`--shard 0/4` through `--shard 3/4` partition a selection in deterministic
round-robin order. Run all shards against the same source and filters before
claiming that selection is complete. An empty shard is allowed only if the
unsharded selection is nonempty. An unfiltered all-library survey is expensive;
the manual workflow starts with the smaller numerical package by default.

## CI and retained evidence

The [Mutation testing workflow](../../.github/workflows/mutation-testing.yml) runs
the smoke and changed-line jobs on relevant pull requests. It checks out the PR
head SHA and compares against the **actual base SHA**, not GitHub's synthetic
merge tree. Manual `workflow_dispatch` runs accept a package and optional source
glob, partitioned across four jobs. A failed job does not cancel the other shards.
There is no recurring schedule and no automatic merge.

The installed-tool integration test uses a tiny temporary two-crate workspace.
Only its downstream integration test observes the numerical function. It checks
that correct assertions kill the mutations, weak assertions produce exit 2, and
a broken unmutated baseline produces exit 4. This validates the real Cargo
selection and CLI behavior in addition to the runner's unit tests.

Every invocation creates a fresh directory under `target/mutation/` containing:

- `run.json`: scope, source HEAD, dirty-worktree status, actual command, tool
  version, selection count, completion state and process exit code.
- `selected.json`: the complete selected mutation inventory; `changes.diff` and
  `merge-base.txt` for incremental runs.
- `mutants.out/`: upstream outcomes, individual patches and build/test logs.
- `summary.md`: the same scoped summary shown in the GitHub job summary.

CI uploads these directories even on failure, with 14-day retention. Local dirty
worktrees are identified as such; only a clean committed checkout pins the tested
source completely. `--list` records `listed_only`, never a passing test result.

Survivors, timeouts, baseline failures and tool errors fail the job. The runner
also rejects incomplete counts, missing baseline tests and check-only output,
even if a tool were to return zero. Unviable builds are reported separately and
are not counted as killed mutations. A nonempty selection with no testable
outcomes fails. Empty incremental selection is a limited no-op, not 100% coverage.
See the upstream [exit code definitions](https://mutants.rs/exit-codes.html).

Aggregate counts must agree with the individual outcome records. Missing or
duplicate mutation names, unknown outcomes, multiple baselines and an unfinished
report are errors, even if the reported totals or console summary look successful.
The result identities must exactly match `selected.json`; a different set with
the same count is rejected. Recorded build/test commands, required packages and
exit statuses must agree with the reported outcome for every scenario. A build
failure cannot count as a caught mutation, and a test phase using `--no-run`
cannot count as an executed test. Unsupported or contradictory phase records
fail the gate. List-only and unsharded inventories are validated too.

Investigate survivors by adding independent boundary, invariant or analytical
reference tests. Keep the existing precision budgets, RNG settings and replay
requirements. For equivalent mutants, record a specific rationale before adding
a narrow exclusion; do not hide a whole module or replace a numerical assertion
with a weaker tolerance. A timeout caused by a mutated infinite loop remains a
timeout and a failed survey, not a caught-mutant result.

The [initial validation record](../../design/validation/mutation-testing-2026-10-22.md)
reports the measured pilot scope and known limitations.
The [numerical follow-up](../../design/validation/mutation-numerics-2026-10-22.md)
extends the unit-test survey to all seven source files in `pricing-numerics`.
The [survivor triage](../../design/validation/mutation-triage-2026-10-22.md)
adds analytical solver and quadrature boundary tests, reducing missed mutations
from 30 to 11. It records the remaining precision/equivalence arguments and
unresolved guard without excluding them or presenting the survey as passing.
The [evidence-integrity follow-up](../../design/validation/mutation-evidence-integrity-2026-10-22.md)
verifies selection identity and execution-phase checks against real reports,
injected inconsistencies and the installed tool.
These diagnostic unit-only runs are narrower than the runner's `full` scope,
which continues to include the downstream pricing tests.

The normal-tail and fractional-OU unit tests embed independent high-precision
reference values. To regenerate those values in a development Python environment:

```bash
python -m pip install 'mpmath==1.3.0'
python scripts/generate_numerics_mutation_references.py
```

This optional generator uses 80 decimal digits, `erfc` for Gaussian tails and
adaptive tanh-sinh integration over the infinite interval for fractional OU.
It is not required by Rust tests or the mutation CI, and does not assert a passing
test result by itself. The validation record retains the generated values.
