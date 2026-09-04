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
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
cargo metadata --locked --format-version 1 --no-deps | python scripts/check_dependency_direction.py
```
