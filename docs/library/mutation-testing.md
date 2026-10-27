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

The pinned tool has its own generation limits: it does not visit methods named
`new` inside an `impl` or a trait. For example, `Date::new` and `Calendar::new`
are absent from the generated inventory even when their files are selected.
Keep explicit constructor/domain tests, and inspect the inventory before
claiming coverage of a whole API. A complete run of all generated mutations
does not mean every production expression was mutated.

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

PR changed-line checks use two round-robin shards, each with its own complete
two-crate baseline and retained evidence. The existing `Mutation testing / changed
Rust lines` check aggregates their results and succeeds only when both jobs
succeed. Artifacts are named `mutation-changed-0` and `mutation-changed-1`.
To reproduce one partition, add `--shard 0/2` or `--shard 1/2` to the normal
`changed --base <actual-base>` command; run both to cover the selection.

The installed-tool integration test uses a tiny temporary two-crate workspace.
Only its downstream integration test observes the numerical function. It checks
that correct assertions kill the mutations, weak assertions produce exit 2, and
a broken unmutated baseline produces exit 4. This validates the real Cargo
selection and CLI behavior in addition to the runner's unit tests.
It also verifies that changed-line shards have a disjoint, complete union,
retain downstream detection, and fail when the observing assertion is removed.

Every invocation creates a fresh directory under `target/mutation/` containing:

- `run.json`: scope, source HEAD, dirty-worktree status, actual command, tool
  version, selection count, completion state and process exit code. Completed,
  validated reports also list every missed/timeout/unviable identity, its last
  execution phase and its log/patch paths relative to `mutants.out/`.
- `selected.json`: the complete selected mutation inventory; `changes.diff` and
  `merge-base.txt` for incremental runs.
- `mutants.out/`: upstream outcomes, individual patches and build/test logs.
- `summary.md`: the same scoped summary shown in the GitHub job summary.

CI uploads these directories even on failure, with 14-day retention. Local dirty
worktrees are identified as such; only a clean committed checkout pins the tested
source completely. `--list` records `listed_only`, never a passing test result.

The runner writes `run.json` before tool discovery, before inventory selection,
before execution and before result validation. Each checkpoint replaces the
previous JSON atomically. This retains the available command/source/selection
information if a CI deadline or host termination prevents normal cleanup.
`preparing`, `selecting`, `running` and `validating` are **last recorded stages**,
not proof that a process is still alive or that the run completed. An unfinished
record has no `finished_at` or `runner_exit_code` and cannot establish a pass.
Normal and handled-interruption exits retain both timestamps and the runner
exit code; a handled keyboard interrupt records `interrupted` and exit 130.

For completed, validated results, `summary.md` and the CI summary show up to
20 outstanding mutations, with missed mutations first and timeout/build-failure
phases distinguished. The limit applies only to display; `run.json` retains the
entire list, and raw logs/patches remain in `mutants.out/`. Unvalidated or partial
outcomes do not receive a validated failure list.

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
The [date/calendar follow-up](../../design/validation/mutation-calendar-2026-10-22.md)
adds boundary and calendar-contract tests, fixes a Modified Following year/month
comparison, and records all 135 generated mutations. It distinguishes the zero
remaining survivors from five nonterminating mutations and eight unviable builds.
The [execution-evidence follow-up](../../design/validation/mutation-reporting-2026-10-22.md)
verifies abrupt-termination checkpoints and complete failure lists, and runs the
normal changed-line gate against both Rust crates and their downstream tests.
The [discount-curve follow-up](../../design/validation/mutation-curve-2026-10-23.md)
adds nonuniform-interval, negative-rate, overflow/underflow, accessor and
diagnostic tests, with before/after evidence for all 71 generated mutations.
The [forward/dividend follow-up](../../design/validation/mutation-carry-2026-10-23.md)
adds exact-rational cash-flow and Spot-bump references, ex-date boundaries,
funding, state-validation and tolerance tests, with evidence for all 243
generated mutations in the two source files.
These diagnostic unit-only surveys are narrower than the runner's `full` scope,
which includes the downstream pricing tests.

The [CI encoding follow-up](../../design/validation/mutation-ci-encoding-2026-10-24.md)
fixes the Windows source checker failure exposed by Unicode date fixtures,
adds portable regression checks and reruns the normal smoke and changed-line gates.

The [PCHIP follow-up](../../design/validation/mutation-pchip-2026-10-25.md)
adds exact-rational cubic references, unequal knot intervals, plateau and
endpoint checks, and knot-neighbor/query-domain tests. It compares all 215
generated identities using the focused PCHIP unit tests.

The [PCHIP finite-result follow-up](../../design/validation/mutation-pchip-finite-2026-10-25.md)
reproduces nonfinite computed slopes and evaluation results from finite inputs.
Construction now rejects nonfinite knot derivatives; evaluation rejects
nonfinite theta or derivatives using `MarketError::NonFiniteSurfaceValue`.
Arithmetic formulas and precision budgets are unchanged. This checks computed
results, rather than promising stable arithmetic for all finite inputs; an
extreme case can still return an error even when its exact mathematical result
is finite. The record tracks all 215 retained mutation operations and the two
new finite-check mutations, with the four regression tests failing before the
fix and passing afterward.

The [SSVI phi follow-up](../../design/validation/mutation-ssvi-phi-2026-10-25.md)
adds independent 100-digit shape-function references, exact parameter errors,
admissibility boundaries and floating-point failure cases. It corrects
small-argument Heston-like derivative cancellation and distinguishes retained,
removed and new mutation operations when comparing the two implementations.
The scoped survey selects `PhiSpec`, its parameter validators and the new
series helper; it does not claim mutation coverage of the complete SSVI surface.
Regenerate the independent references without extra Python packages:

```bash
python scripts/generate_ssvi_phi_references.py
```

The [density-factor follow-up](../../design/validation/mutation-density-2026-10-25.md)
tests `durrleman_density_factor` with exact rational references, derivative
validation, zero-boundary neighbors and overflow errors. Its comparison uses
the same generated mutation inventory before and after adding tests; production
arithmetic and tolerances are unchanged.

The [forward-call follow-up](../../design/validation/mutation-forward-call-2026-10-25.md)
checks the default surface method and its pricing helper with independent
100-digit lognormal price/density references, forward scaling, provider-error
propagation, strike overflow/underflow, zero-valued tails and computed-result
errors. It uses the same 42 generated operations before and after adding tests.
Regenerate the references without additional Python packages:

```bash
python scripts/generate_ssvi_forward_references.py
```

The [surface-tolerance follow-up](../../design/validation/mutation-surface-tolerance-2026-10-25.md)
checks the documented absolute/relative scaling policy, strict and inclusive
boundary neighbors, butterfly-bound residuals and overflow rejection. Exact
dyadic references exercise the validation helpers; public constructor tests
also check parameter preservation and propagation of butterfly failures.
These helper tests do not establish that every distinguishing input is reachable
through every admissible phi family. Production arithmetic and tolerances are
unchanged, and constructors omitted by the pinned generator are outside its
mutation inventory.

The [Standard SSVI variance follow-up](../../design/validation/mutation-ssvi-variance-2026-10-26.md)
anchors variance and its log-moneyness/time derivatives to independently
differenced scalar references, with separate reflection, ATM, interpolation
metadata and typed-error contracts. Both comparison runs select the same 71
generated operations. The reference generator varies precision and difference
step together; agreement is a finite-panel check, not a whole-domain error bound.

```bash
python scripts/generate_ssvi_variance_references.py
```

The [eSSVI parameter follow-up](../../design/validation/mutation-essvi-parameters-2026-10-27.md)
uses exact affine references on unequal time intervals, knot-neighbor checks,
time-unit scaling, flat segments and public variance identities. It checks all
six parameter values/derivatives and the selected interpolation region. Both
runs select the same 85 generated operations in `parameters`, `interpolate`
and `long_parameters`; this is not the complete eSSVI surface inventory.
The optional reference generator uses only Python's standard library:

```bash
python scripts/generate_essvi_parameter_references.py
```

The [fractional-OU guard investigation](../../design/validation/mutation-fou-guard-2026-10-25.md)
retains the unresolved final range-guard survivor. A bounded comparison of the
actual source with its single logical mutation found no distinguishing input;
this is neither an independent accuracy oracle nor a proof of equivalence.
Reproduce it with the pinned Rust toolchain and Python's standard library:

```bash
python scripts/probe_fractional_ou_guard.py --output /tmp/fou-guard-probe
```

Use a new output directory. The probe retains input/output bits for both default
and optimized builds, source copies and hashes. Its samples do not change the
generated-mutation score, and the survivor has no waiver or exclusion.

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
