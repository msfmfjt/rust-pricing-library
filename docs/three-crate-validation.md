# Three-crate validation

This records local implementation and comparison evidence for
[PR #53](https://github.com/msfmfjt/rust-pricing-library/pull/53). Readiness also
requires assessment of the supported-platform CI checks on that PR.

- Remote main baseline: `0488c415c03f93bcbf7a5bfff43c3a948938e2cf`.
- Dedicated branch: `refactor/three-crate-workspace`.
- Published implementation commit: `f5810cc6f92ef15082dec50f63d796a7d1ceb9fd`.
- Validated implementation tree: `95e60a197af7e2ae2a221b1a689dfe0121131289`.
  The GitHub tree SHA was checked against the local committed tree before publication.
- The final published commit is the PR head; the evidence commit changes documents,
  captured results and the source-archive allowlist only. No main push, merge or other PR update was performed.
- Baseline native CI had passed in
  [run 34769075352](https://github.com/msfmfjt/rust-pricing-library/actions/runs/34769075352).

## Environment and scope

Both revisions ran on the same Linux x86_64 host, Rust 1.98.1
(`48a229cea`, 2026-09-01), Cargo from that toolchain, CPython 3.12, and maturin
1.15.0. The repository toolchain and release settings (`codegen-units = 1`,
`lto = "thin"`) were unchanged. Registry package versions, checksums and dependency
entries in Cargo.lock are unchanged; workspace package entries necessarily change.

The existing facade has no feature switches. The old risk dependency always
selected `pricing-mc/aad`; the corresponding executor import, method, test and
capability marker are now unconditional. Python retains its separate
`extension-module` feature. No unsupported optional model work was imported.

## Command matrix

“Pass” below means an observed successful command, including a corrected rerun
where explicitly noted. Original failed attempts are retained in the raw logs.

| Command / gate | Baseline | Migration |
| --- | --- | --- |
| `cargo metadata --no-deps --format-version 1` | Pass: 10 members | Pass: exactly 3 members |
| Dependency direction checker | Existing graph retained as inventory | Pass, including rejection of 3 invalid dependency probes |
| `cargo fmt --all -- --check` | Pass | Pass |
| `cargo check --locked --workspace --all-targets` | Pass | Pass |
| `cargo test --locked --workspace` | 425 passed, 5 ignored after environment correction | 426 passed, 5 ignored |
| `cargo test --locked --workspace --all-features --exclude pricing-python` | Pass | Pass |
| `cargo test --locked -p pricing-python` | Pass after environment correction | Pass |
| `cargo test --locked -p pricing --no-default-features` | Pass | Pass |
| `cargo test --locked -p pricing-numerics` | Pass, 8 tests | Pass, 8 tests and no dependencies |
| `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` | Pass | Pass after fixing moved test-module naming |
| `cargo doc --locked --workspace --no-deps` | Pass | Pass |
| `cargo doc --locked --workspace --all-features --no-deps` | Pass | Pass |
| Three ignored statistical/acceptance integration targets, exact debug CI commands | All 5 tests pass | All 5 tests pass |
| Same three acceptance targets with `--release` | All pass | All pass |
| Three independent reference-fixture Python checkers | Pass | Pass |
| `python scripts/check_schemas.py` | Pass | Pass |
| `python scripts/check_markdown_links.py` | Pass | Pass |
| Source archive creation and `check_source_archive.py` | Pass | Pass with moved files and 3 manifests |
| `python -m maturin build --locked --release --out dist` | Pass | Pass |
| `python scripts/smoke_test_wheel.py` | Pass, 61 tests | Pass, 61 tests, including updated SBOM graph |
| Eight Rust example targets | All build and run, two initial link retries noted below | All build and run |
| Existing Python benchmark | Pass | Pass |
| `python scripts/run_benchmark_suite.py` | Fails in this host's RSS collection | Same host limitation |
| Frozen supported-platform replay checker | Rejects Linux | Rejects Linux; native gate retained in CI |

The acceptance target names are `statistical_acceptance`,
`path_dependence_acceptance`, and `early_exercise_acceptance`. Their original
`cargo test --locked -p pricing --test NAME -- --ignored --nocapture` commands
were run before and after; the release runs are additional evidence.

The literal `cargo test --workspace --all-features` including Python's
extension-module linker configuration was not used as a supported test gate.
Baseline CI intentionally splits it into the all-feature non-Python tests and
Python default-feature unit tests shown above. This separation is preserved;
all-feature Clippy and rustdoc still include the complete workspace.

## API, test and numerical preservation

- [Public API inventory](validation/three-crate/public-api.json): 377 original
  public names across 14 facade paths, compiled by
  [facade_compatibility.rs](../crates/pricing/tests/facade_compatibility.rs).
  Root functions/types and nested model, MC, Hull–White, LSV and analytical paths
  are included. Existing function signatures were also included in the mechanical
  comparison. Direct users of deleted crates must follow the
  [migration guide](three-crate-migration.md); that part is breaking.
- [Test mapping](validation/three-crate/test-moves.json): all 430 old Rust tests
  map to a new source location, with the same five default-ignored markers. Cargo
  listings and executed leaf-name multisets show no missing test. One facade
  inventory test is added. The removed integration crates' target names are
  explicitly mapped in the migration guide; all eight example targets survive.
- [Mechanical audit](validation/three-crate/mechanical-audit.json): 2,716 of 2,717
  baseline function definitions are identical after normalizing moved import
  paths, internal visibility, comments and rustfmt trailing commas. The remaining
  function is `pricing_numerics::foundation_role()`: it now returns the same
  literal `"core"` instead of depending on `pricing-core` for that marker. No
  numerical expression, operand order, loop, RNG mapping or reduction was changed.
- [Contract hashes](validation/three-crate/unchanged-contract-hashes.json): all
  28 fixture files, 6 schemas, binding source/manifest files, Python tests/examples,
  type stub, pyproject and toolchain file are byte-unchanged. The moved Joe–Kuo
  direction binary also retains its digest. There are no schema/ABI changes or
  golden/tolerance updates.

## Replay and build information

Each unchanged replay example ran from the baseline and migration release binary
on this host. All JSON output bytes match, including price/Greek/statistical bits,
request/plan/payoff fingerprints, random/scramble checksums and replay metadata.

| Existing example | Cases | Before/after result |
| --- | ---: | --- |
| `replay_european_bs` | 2 | Byte-identical |
| `replay_local_vol` | 4 | Byte-identical |
| `replay_path_dependence` | 6 | Byte-identical |
| `replay_early_exercise` | 4 | Byte-identical |

[Replay hashes and byte counts](validation/three-crate/replay-comparison.json)
and raw outputs are retained. Cargo.lock and source-tree/build identifiers differ
outside these replay outputs because crates and paths changed; these differences
are not numerical differences. No cross-platform bitwise requirement is added.

The repository's frozen replay goldens cover macOS ARM64 and Windows x86_64 only.
Their checker rejected `linux-x86_64` on both revisions; no golden was added or
rewritten to bypass that gate. The original native checks remain in CI.

## Performance

All release examples were built before timing. Each unchanged benchmark ran three
times per revision, with before/after order alternating, original sampling counts
and worker settings. The values below are medians of the three within-run
medians. Negative change means less elapsed time. The complete 32 measurements,
configuration, toolchain and per-run values are in
[comparison.json](validation/three-crate/comparison.json).

| Measurement | Before, ms | After, ms | Change |
| --- | ---: | ---: | ---: |
| European BS price | 2.103 | 2.091 | -0.6% |
| European BS AAD with CRN validation | 21.749 | 22.297 | +2.5% |
| Local Vol price | 4.405 | 3.921 | -11.0% |
| Local Vol VegaKT | 32.860 | 31.202 | -5.0% |
| Continuous Barrier full risk | 24.454 | 23.551 | -3.7% |
| LSM training-dominant price | 10.398 | 10.140 | -2.5% |
| LSM valuation-dominant price, initial comparison | 5.938 | 6.374 | +7.3% |
| LSM valuation-dominant price, 7-pair follow-up | 6.059 | 6.150 | +1.5% |
| Python full-risk evaluation | 23.218 | 22.362 | -3.7% |

The initial LSM slowdown prompted seven additional alternating pairs using the
same binary/settings. Its observed median difference was then +1.50%, with
substantial overlapping run ranges; the initial +7.34% difference did not repeat.
Both sets are preserved in [lsm-confirmation.json](validation/three-crate/lsm-confirmation.json)
and the raw archive. These are noisy single-host measurements, not evidence of a
performance improvement or a statistical noninferiority guarantee. No optimization
or benchmark workload change was made.

## Failures, limitations and remaining gates

1. **Baseline environment correction:** the relocated Python interpreter reports
   `/install/lib`. Initial default workspace/Python unit-test links could not find
   `libpython3.12`. A local linker-search directory pointing to the installed
   library resolved the failures; no repository or Python binding change was made.
2. **Transient baseline links:** the first release links for
   `benchmark_european_bs` and `replay_european_bs` reported LLVM hidden-symbol
   errors. Unchanged retries and the subsequent build of all examples passed.
3. **Migration issue resolved:** moved `tests.rs` modules initially nested another
   `mod tests`, triggering Clippy `module_inception`. Distinct outer conformance
   module names fix it; final default/all-feature tests and strict Clippy pass.
4. **Aggregate benchmark gate unavailable locally:** the existing runner cannot
   observe peak RSS for a child process in this host and aborts, on both revisions.
   Individual timing/replay programs run successfully. Aggregate metadata/report
   acceptance is not claimed locally; the native CI gates are preserved.
5. **Native supported-platform validation:** macOS ARM64 and Windows x86_64 are
   unavailable in this Linux workspace. Native wheels, frozen replay goldens and
   complete benchmark artifact checks must be assessed in PR CI. Until those
   required checks are satisfied, the PR remains Draft.

The baseline had no open PRs. Future work on old crate paths/the former monolith
will require rebasing according to the migration guide. Generic QR/RNG extraction,
further wire/LSM splitting, feature redesign and performance work remain separate
follow-ups; no issue discovered here was used to expand the numerical scope.

## Evidence files

The original attempts are retained rather than overwritten by successful reruns:
[baseline commands](validation/three-crate/baseline-commands.json),
[corrected baseline commands](validation/three-crate/baseline-corrected-commands.json),
[migration commands](validation/three-crate/after-commands.json),
[final corrected gates](validation/three-crate/final-structure-commands.json),
[exact CI acceptance commands](validation/three-crate/ci-acceptance.json),
[baseline targets/features](validation/three-crate/baseline-targets.json),
[migrated targets/features](validation/three-crate/after-targets.json),
[boundary probes](validation/three-crate/boundary-probes.json), and
[frozen replay attempts](validation/three-crate/frozen-replay-checks.json).

[run-artifacts.tar.gz](validation/three-crate/run-artifacts.tar.gz) contains the
raw command logs, before/after replay JSON and all timing samples. Its benchmark
records identify the local validation commit `949e5b8`; that commit's tree is
identical to published implementation commit `f5810cc` as verified above.
