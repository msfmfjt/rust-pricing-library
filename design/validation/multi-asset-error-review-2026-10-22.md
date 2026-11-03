# Multi-asset error-boundary review — 2026-10-22

## Scope and priority

This follows the [executor allocation review](executor-resource-review-2026-10-22.md)
on PR #174, commit `57675566b63dbe5de71ef989b502550a83efbb2e`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The branch is stacked on #174;
it does not incorporate the separate mutation-testing work in #172.

P2: the experimental multi-asset layer formatted native numerical errors into
strings. This lost typed executor, RQMC, model and graph causes across Rust and
Python task boundaries. Batch callers could not distinguish resource limits
from payoff/model failures, despite those distinctions existing in the native
errors. The legacy Python compiler additionally replaced every native compile
failure with generic validation metadata.

The implementation and explicit Rust compatibility impact are recorded in
[ADR 0040](../adr/0040-multi-asset-error-sources.md). No numerical algorithm is
changed. Existing text-only local-correlation feasibility diagnostics remain
generic; this is not a claim that every error in the library is typed.

## Reproductions and regression coverage

- A task with `u64::MAX` sampling units and reduction block size 1 returns the
  typed allocation failure introduced by #174. Before this change its source
  vanished inside the basket adapter for both price and AAD.
- A one-year, two-driver RQMC task with maximum step `1 / 11000` exceeds the
  21,201-dimension Sobol limit. The native cause survives compilation, without
  allocating a large Sobol table or running a long simulation.
- A graph multiplying a sampled basket Spot by the finite literal `1e308`
  compiles and then fails at runtime. Price and AAD preserve both the executor's
  sampling-unit context and `GraphError::NonFiniteRuntimeValue`.

Before the fix, all three new Rust integration tests failed on missing causes.
The Python suite produced five failing subcases: two allocation classifications,
one legacy compile classification and two runtime graph classifications.
The unchanged small-workload boundary test passed.

After the core fix, all three Rust integration tests pass. They include one-
and three-worker graph evaluation. Binding tests also verify nested LSV
calibration classification and a generic fallback for unrecognized errors,
independent of their text. Python tests retain the legacy compiler's exception
class and issue fields while checking metadata agreement with the common task.

Tests: [Rust causes](../../crates/pricing/tests/multi_asset_error_sources.rs),
[Python basket errors](../../tests/python/test_multi_asset_errors.py), and
[Python execution resources](../../tests/python/test_execution_resources.py).

## Exact comparisons

Twenty-four complete serialized reports were captured from the parent wheel:
the six task fixtures and eighteen cases varying worker count (1, 3) and block
size (1, 7, `u64::MAX`) across standard, basket and stochastic-dividend risk.
All twenty-four are byte-for-byte identical after the fix, including prices,
Greeks, standard errors, calibration diagnostics and replay fingerprints.
The before/after report collections have SHA-256
`64dff509942dc3fa21bd37d501dddcfdf72681ec2937009e61c4a13b7e1003e2`.
The parent report values also exactly match the #174 validation capture.
No comparison tolerance is used or relaxed.

Six failing legacy/common-task calls (RQMC compilation, graph price and graph
AAD) preserve exception classes, complete message strings, operation fields
and legacy issue dictionaries exactly. The intended differences are richer
causes, specific codes/fields and the legacy native compiler's corrected stage.

## Local validation

- `cargo test --locked --workspace --all-features --exclude pricing-python`:
  1,077 passed, 0 failed, 91 intentionally ignored acceptance tests.
- Native Rust binding tests: 6 passed. Explicit statistical, path-dependence
  and early-exercise acceptance targets: 5 passed in total.
- Formatting, all-target workspace check, all-target/all-feature Clippy with
  `-D warnings`, all-feature documentation and dependency-direction checks pass.
- The 18 required static/reference/schema/link scripts and 34 replay/wheel
  checker tests pass.
- Release CPython 3.12 wheel builds. Package/ABI/metadata/SBOM/clean-install
  checks pass, including the 341-callable runtime-signature contract.
- Installed-wheel Python validation: 155 tests passed, covering basket error
  regressions, executor resources, source/task/graph APIs, report replay and
  reuse, HW/LSV/rough/local-correlation models, calibration reports, backtest
  native contracts and wheel contracts.

The full Python discovery suite and release-only extended model/risk panels
remain delegated to hosted CI. The ordinary Rust command does not run those
ignored panels. Hosted CI status is reported separately; queued checks are
not passes.

## Remaining review items

The full library review remains open beyond these error boundaries. Other
legacy string-only errors, general allocation limits, and model-specific
unsupported-operation coverage need independent scope and tests. This change
does not broaden model capabilities or assert universal allocation safety.

Older Windows extended-model failures in #164/#165 still lack retrievable logs
and remain unresolved. At the start of this batch, #170 had 70 successful
checks, #171 had 23, and #173/#174 had no completed successful checks yet; the
remaining jobs were running, queued or skipped. No PR is merged by this work.
