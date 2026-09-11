# Contributing

## Baselines

The frozen requirements, architecture, and accepted implementation roadmap govern changes. A normative requirement change needs a recorded decision and impact analysis covering API compatibility, serialized data, numerical behavior, reproducibility, tests, and roadmap Gates.

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
