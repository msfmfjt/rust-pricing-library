# Mutation evidence integrity — 2026-10-22

Base: `3c59bc2207ddd9577002c037bbaa86410bb46eba`, the
[numerical survivor triage](mutation-triage-2026-10-22.md) in draft PR #172.
This follow-up strengthens the runner's acceptance of evidence. It does not
change Rust kernels, numerical tests, mutation selection or precision budgets.

## Problem and correction

The runner checked unique result names and reconciled totals, but only received
the **count** of selected mutants. A different same-sized set of results could
therefore pass the evidence check. It also trusted outcome summaries without
reconciling them with recorded build/test exits, and checked test packages only
in the baseline.

The runner now requires exactly the identities in `selected.json`. Selection
inventories must contain unique, nonempty names, including in list mode and when
validating the nonempty selection behind an empty shard. Result order may differ.
For the pinned cargo-mutants 27.1.0 format, every scenario must also have:

- A build followed by a test, or a build that failed/timed out before testing.
- Cargo test commands with the appropriate `--no-run` mode and required packages
  in both phases. Arguments after `--` cannot masquerade as package selection.
- Exit statuses consistent with the outcome summary. A caught mutation requires
  a successful build and a failed test; a successful baseline requires both
  phases to succeed. Build failures and timeouts remain separate outcomes.

Unsupported or contradictory records fail the gate. A nonzero tool exit remains
a failure; internally consistent missed/timeout results still fail even if the
tool unexpectedly exits zero. An empty shard remains an explicit no-op, never a
passing tested selection.

## Fault injection and real-tool validation

Six corruptions were applied separately to a copy of the complete 952-mutation
triage report. The original and updated checkers were compared on the same data:

| Injected inconsistency | Previous checker | Updated checker |
| --- | --- | --- |
| One result replaced by an unselected identity, same count | Accepted | Rejected |
| Successful mutant test labelled caught | Accepted | Rejected |
| Failed baseline test labelled successful | Accepted | Rejected |
| Required package missing from a mutant's test command | Accepted | Rejected |
| Failed build with no test labelled caught | Accepted | Rejected |
| Test phase command only builds with `--no-run` | Accepted | Rejected |

These are injected inconsistencies, not evidence that a real CI run previously
misreported its result. The unmodified reports from all three numerical surveys
pass the strengthened checker, retaining their original counts and identities.
The [JSON record](mutation-evidence-integrity-2026-10-22.json) records report
hashes, outcomes, fault-injection errors and verification commands.

The runner's **19 unit tests** pass. The installed-tool two-crate fixture verifies
four caught mutants (exit 0), four missed mutants (exit 2), an already-failing
consumer baseline (exit 4) and an empty shard over a valid nonempty inventory.
The repository's real numeric-input smoke run checks all **15 selected mutants**.
See the JSON for its completed outcome and exact commands. Documentation links,
the staged source archive and diff checks are also validated for this change.

Reproduce the runner checks with the pinned tool installed:

```bash
python -m unittest discover -s scripts -p 'test_run_mutation_tests.py' -v
python -m unittest discover -s scripts -p 'test_mutation_integration.py' -v
python scripts/run_mutation_tests.py smoke
```

To revalidate saved diagnostic numerical evidence, pass the parsed mutation
inventory rather than its length to
`check_outcomes(report, inventory, expected_packages=("pricing-numerics",))`.
The normal full/changed runner still requires both Rust crates.

## Remaining numerical scope

A further diagnostic evaluated the fractional-OU formula with only its final
finite out-of-range rejection removed. The 1,701 evaluations span 21 Hurst
values, subnormal through very large lags and the lag-64 split. No finite value
crossed the existing `1 + 1e-11` guard. The largest absolute result was about
`1.000000000000022`. The JSON retains the grid construction and measured result.
This finite search does **not** establish that the guard is unreachable or
equivalent, and it is not a numerical acceptance test.

The latest full numerical unit survey therefore remains **931 caught, 11 missed,
3 timeouts and 7 unviable out of 952**, with exit 3. This follow-up revalidates
that retained evidence; it does not claim a new full survey or a passing score.
The 55 numerical unit tests and their previous Clippy validation are unchanged.
Remote CI is reported separately in the PR; no merge is authorized here.
