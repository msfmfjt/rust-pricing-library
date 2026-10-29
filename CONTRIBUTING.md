# Contributing

## Baselines

The [frozen requirements](design/requirements-v1.0.md),
[architecture](design/architecture-v0.1.md), and accepted implementation roadmap
govern changes. A normative requirement change needs a recorded decision and
impact analysis covering API compatibility, serialized data, numerical behavior,
reproducibility, tests, and roadmap Gates.

## Documentation layout

Keep user-facing library guidance in [docs/library](docs/library/README.md),
product definitions in [docs/products](docs/products/README.md), and model
conventions, calculation specifications and diagnostics in
[docs/models](docs/models/README.md). Requirements, architecture, ADRs,
implementation roadmaps and validation records belong in
[design](design/README.md). Update the corresponding index and all relative links
when moving or adding a document.

The Markdown link checker and source-archive checker cover all documentation
trees. Preserve captured validation evidence as historical data; update the
surrounding report's links instead of rewriting recorded paths in raw artifacts.

## Pull requests

Keep changes narrow and leave the workspace buildable. A pull request should include:

- the requirement or roadmap Gate it implements;
- tests for observable behavior and failure cases;
- reproducibility impact;
- public API or wire-format impact;
- numerical evidence where applicable.

Do not combine an optimization with an untested numerical-policy change.

## Required checks

```shell
python3 scripts/check_local_vol_reference_fixture.py
python3 scripts/check_local_vol_gamma_numerics.py
python3 scripts/check_local_vol_gamma_recovery.py
python3 scripts/check_local_vol_vegakt_paper.py
python3 scripts/check_local_vol_step_adjoints.py
python3 scripts/check_local_vol_step_recovery.py
python3 -m unittest discover -s scripts -p test_audit_local_vol_replay.py -v
python3 scripts/check_local_vol_direct_variance.py
python3 scripts/check_local_vol_initial_gamma.py
python3 scripts/check_local_vol_vegakt_chain.py
python3 scripts/check_local_vol_density_mass.py
python3 scripts/check_local_vol_weak_chain.py
python3 scripts/check_path_dependence_reference_fixture.py
python3 scripts/check_early_exercise_reference_fixture.py
python3 scripts/check_schemas.py
python3 -m pip install "jsonschema>=4.18,<5"
python3 scripts/check_specification_json_schema.py
python3 scripts/check_product_graph_json_schema.py
python3 scripts/check_task_json_schema.py
python3 scripts/check_markdown_links.py
python3 -m unittest discover -s scripts -p test_smoke_test_wheel.py -v
python3 -m unittest discover -s scripts -p test_provider_shards.py -v
python3 -m unittest discover -s scripts -p test_native_debugger.py -v
python3 -m unittest discover -s scripts -p test_native_memory_diagnostic.py -v
python3 -m unittest discover -s scripts -p test_ci_failure_propagation.py -v
python3 -m unittest discover -s scripts -p test_wheel_validation.py -v
git archive --format=tar.gz --output /tmp/rust-pricing-source-check.tar.gz HEAD
python3 scripts/check_source_archive.py /tmp/rust-pricing-source-check.tar.gz
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features --exclude pricing-python
cargo test --locked -p pricing-python
cargo test --locked -p pricing --test statistical_acceptance -- --ignored --nocapture
cargo test --locked -p pricing --test path_dependence_acceptance -- --ignored --nocapture
cargo test --locked -p pricing --test early_exercise_acceptance -- --ignored --nocapture
cargo test --locked --release -p pricing --test extended_model_acceptance -- --ignored --nocapture --test-threads=1
cargo test --locked --release -p pricing --test extended_risk_acceptance -- --ignored --nocapture --test-threads=1
cargo doc --locked --workspace --all-features --no-deps
cargo metadata --locked --format-version 1 --no-deps | python3 scripts/check_dependency_direction.py
python -m maturin develop --locked
python -m unittest discover -s tests/python -v
python -m maturin build --locked --release --out dist
python scripts/smoke_test_wheel.py
python scripts/run_benchmark_suite.py
python scripts/check_replay_fixture.py benchmark-results/replay.json
python scripts/check_replay_fixture.py benchmark-results/local-volatility-replay.json
python scripts/check_replay_fixture.py benchmark-results/path-dependence-replay.json
python scripts/check_replay_fixture.py benchmark-results/early-exercise-replay.json
python scripts/check_benchmark_reports.py benchmark-results
```

Run `scripts/smoke_test_wheel.py` with the same CPython ABI as the built wheel
tag, for example CPython 3.12 for a `cp312` wheel. The smoke test rejects ABI
mismatches before installation.

The main Rust and supported-wheel jobs use explicit `shell: bash` on every
platform so a failed native command or pipeline fails its step immediately.
Do not rely on PowerShell's final `$LASTEXITCODE` for a block with multiple
commands. The failure-propagation tests inject nonzero native exits into the
retained CI blocks and require that subsequent commands do not run.

The default smoke command runs all package checks, examples and discovered
Python tests. CI builds each platform wheel once, runs `--phase package`, and
passes the same wheel artifact to 37 `--phase validation` jobs per platform:
36 provider shards and the `remaining` group. Every selection is required on
Linux, macOS and Windows. The generated matrix in
`scripts/wheel_validation/provider_shards.py` is the single CI plan.
New test modules automatically join `remaining`; new methods in a provider
suite join its contracts shard unless explicitly declared as numerical cases.
Examples run with `remaining` (or `all`).

Provider numerical shards select one basis/extension and one interval count.
For four-interval named bases, native panels run together and each dated ledger
(side and product) runs separately: their combined execution exceeded the job
limit on Windows. Six-interval named bases and extension shards retain their
combined panels and ledgers. Every ledger keeps all four strategies, checkpoint
replays and failure repairs. Contracts run separately with all original policy
configurations. Before executing any selection, discovery checks both the test-ID
multiset and the provider method/case multiset against the complete suite,
including the original repeated coupled-basis coverage. Missing or extra cases
fail validation. Cases, sampling, tolerances and numerical assertions are unchanged.

Each example, module or shard runs in a separate Python process with fault diagnostics
enabled. Tests within that process keep their shared class state. A failed test
or process fails its job without retrying it. `--report-directory` saves the
inventory and a running/completed record for each test process, including exit
codes and signals. CI attempts to upload these records after failure or
cancellation, together with any Python crash reports emitted by macOS.
Provider shard processes also dump Python stacks on POSIX `SIGTRAP`, which the
standard fault handler does not cover. They then chain to the previous handler;
the native crash remains a failed process. Windows uses its built-in fault and
exception handlers. An absent native crash report does not identify the cause.

macOS wheel selections also use `--native-debugger lldb`, which requires a
report directory. LLDB launches the original command once through its Python
API, retaining literal arguments and the venv executable path. Shell argument
expansion is disabled. Startup `exec` handoffs do not stop the process; native
signals and exceptions still do. The runner captures thread backtraces,
registers, image identities and stop reasons before a stopped process is terminated. Diagnostic
logs and process observations are saved under `native-debugger/` in the same
artifact. Each thread's JSON observation also retains up to 128 frames with
function names, module paths/UUIDs, load and file addresses, and available source
locations. Missing symbols remain null; invalid frames, truncated stacks and
unwind errors are explicit. These are read-only debugger observations, not a
diagnosis of the corruption's origin. Full text backtraces remain in the log.
A zero debugger exit alone cannot pass: a current process record must
confirm normal child exit zero, and the debugger must also succeed. Missing,
stale or malformed records and debugger launch failures fail the job. Crashes
are not resumed or retried. Ordinary positive child exit codes are preserved;
a native stop without a normal exit has gate status 1 and its actual stop state
in the diagnostic record. The mandatory macOS debugger tests exercise normal
exit, nonzero exit, `exec` handoffs, SIGABRT, SIGTRAP and literal argument
handling. A crash probe must reach its pre-signal marker and report the requested
signal or its Mach exception before it counts as diagnostic evidence.

The separate `Native memory diagnostics` workflow investigates the two retained
macOS failures on `macos-15` with CPython 3.12.10 and the observed NumPy/SciPy
versions. One job runs existing calibration kernels without importing native
pricing. Two other jobs build a local diagnostic wheel with the pinned Rust
compiler's AddressSanitizer and execute the original failed selections once.
The normal wheel matrix remains required. A diagnostic pass is not proof that
an intermittent fault is repaired.

`scripts/native_memory_diagnostic.py` rejects an uninstrumented installed
extension, checks both normal execution and detection of a deliberately faulty
standalone probe, then verifies ASan can import the wheel. It also runs normal
allocation/free and deliberate double-free controls in separate Python processes
after importing pricing. The faulty control must abort with an ASan report that
includes both allocation and free stacks before the numerical workload can run.
Framework Python is launched through its actual interpreter with the original
environment retained;
its startup launcher can lose sanitizer interposition. Source/binary identities,
host/compiler versions, detector results and process completion records accompany
the logs. Leak detection is disabled; invalid access/free detection remains on.
CPython, NumPy, SciPy, the standard library and system libraries are not rebuilt
with instrumentation. These diagnostic builds are not distribution artifacts.
The workflow runs when its helper/tests/workflow change, and can also be invoked
manually after it is available on the default branch.

A missing or running record is incomplete evidence, never a passing result.
CI gives each selection 330 minutes; successful completion on all platforms
remains required. Runner loss can prevent the final artifact upload entirely.

The original `--group custom-basis`, `--group providers`, default full execution,
and combined four-interval shard names remain available, but those combined
suites can exceed a hosted runner's time limit. Combined shard aliases are
excluded from the CI matrix to avoid duplicating numerical cases. They retain
the original process/class lifetime for investigating native faults; success
under shorter process lifetimes does not establish a fix for those faults.
To reproduce a CI shard with an existing wheel:

```bash
python scripts/smoke_test_wheel.py dist/<wheel>.whl --phase validation --shard coupled-basis-coupled_row_hats-4-put-fixed_lookback --report-directory wheel-validation-reports
```

To reproduce the original combined shard, including its panels and all ledgers:

```bash
python scripts/smoke_test_wheel.py dist/<wheel>.whl --phase validation --shard coupled-basis-coupled_row_hats-4 --report-directory wheel-validation-reports
```

The CLI delegates archive/metadata checks, static API contracts, and installed
runtime checks to `scripts/wheel_validation`. Independent frozen API expectations
live in its `contracts` package, grouped by domain with each class's members,
signatures and return constraints together. When extending an API, update its
domain record and the module exports; do not regenerate expectations from the
implementation or stub. Existing partial annotation contracts remain in force;
new or changed declarations should freeze input and return annotations explicitly.

Every declared callable (including constructors and special methods, excluding
property getters) is also checked against the installed wheel using
`inspect.signature`. Argument names/order, positional-only and keyword-only
boundaries, required arguments and typed defaults must agree. Only the implicit
instance receiver is normalized. Missing signatures, opaque defaults and
variadic catch-all signatures fail validation. PyO3 exposes constructors on the
class itself; use an explicit `text_signature` when a Rust constant's default is
otherwise shown as `...`, with a behavior test that confirms the native default.
Runtime annotations are not supplied by PyO3, so independent static contracts and
native conformance tests still check types and semantics. See
[ADR 0033](design/adr/0033-wheel-api-contracts.md) for the coverage boundaries.

Extended-model changes must preserve the independent price, seed-dispersion and
refinement gates in the [extended-model accuracy panel](design/validation/extended-model-accuracy.md).
The heavy panels run explicitly in release mode on all three Rust CI platforms;
CI retains per-quote prices, seeds, IV errors, sampling errors and calibration
ESS/fallback/projection diagnostics for review. The
[refinement and stress record](design/validation/extended-model-refinement-stress.md)
distinguishes supported quote nodes from global calibration feasibility.

The [extended-model risk panel](design/validation/extended-model-risk-accuracy.md)
checks public AAD and recalibrated sensitivities with common calibration/pricing
streams, multiple seeds and bump sizes. It runs before the price acceptance
panel and retains its own evidence, so the two gates can be assessed separately.

## Internal boundaries after crate consolidation

Use `pricing` for all financial Rust APIs. See the
[three-crate migration guide](docs/library/three-crate-migration.md) for moved test targets,
public paths and feature behavior. Keep execution in the private `engine` modules;
do not expose that module to solve visibility problems. Preserve source/domain
objects independently of execution workspaces and keep risk-report construction
above the MC executor. Existing public inherent methods can be implemented in a
private engine module and re-exported through their original path.

In addition to the existing gates above, run:

```bash
cargo check --locked --workspace --all-targets
cargo test --locked --workspace
cargo test --locked -p pricing --no-default-features
cargo test --locked -p pricing-numerics
```

The original requirements and ADRs are historical records. Current crate layout
and feature behavior are in architecture sections 3 and 15 and the migration guide.
