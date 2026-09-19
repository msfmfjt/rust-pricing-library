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
python3 scripts/check_path_dependence_reference_fixture.py
python3 scripts/check_early_exercise_reference_fixture.py
python3 scripts/check_schemas.py
python3 scripts/check_markdown_links.py
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

Extended-model changes must preserve the independent price, seed-dispersion and
refinement gates in the [extended-model accuracy panel](design/validation/extended-model-accuracy.md).
The heavy panels run explicitly in release mode on all three Rust CI platforms;
CI retains per-quote prices, seeds, IV errors and sampling errors for review.

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
