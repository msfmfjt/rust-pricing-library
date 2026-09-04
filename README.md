# Rust Pricing Library

An extensible derivatives-pricing library for model validation and quantitative research. The calculation core is written in Rust and will expose a Python interface for interactive single-trade analysis.

The first vertical slice is a European vanilla option under Black–Scholes, with analytical reference values, Pseudo-Monte Carlo, randomized Sobol QMC, AAD Greeks, common-random-number bump validation, and deterministic replay.

## Design baselines

- [Frozen MVP requirements](docs/requirements-v1.0.md)
- [Architecture](docs/architecture-v0.1.md)
- [European Black–Scholes implementation roadmap](docs/european-bs-roadmap-v0.1.md)

## Status

The project is in Gate G0 of the initial implementation roadmap: workspace and continuous-integration scaffolding.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `pricing-core` | Fundamental IDs, dates, errors, configuration, and result primitives |
| `pricing-numerics` | Deterministic numerical utilities |
| `pricing-aad` | Simulation reverse-mode and adjoint execution infrastructure |
| `pricing-market` | Curves, dividends, implied/local-volatility market objects |
| `pricing-product` | Built-in products and compiled Event/Payoff graphs |
| `pricing-models` | Black–Scholes, Black-76, and Local Volatility kernels |
| `pricing-mc` | MC/QMC simulation, path execution, and LSM |
| `pricing-risk` | AAD orchestration, bump validation, and VegaKT |
| `pricing` | Stable public Rust facade |
| `pricing-python` | Python binding boundary |

Dependency direction is checked in CI. Lower-level crates may not depend on higher-level crates.

## Development

The repository pins Rust 1.98.1. After installing [rustup](https://rustup.rs/), run:

```shell
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
cargo metadata --locked --format-version 1 --no-deps | python scripts/check_dependency_direction.py
```

## Platforms

The MVP support targets are:

- Apple Silicon macOS (`aarch64-apple-darwin`)
- Windows x86-64 using MSVC (`x86_64-pc-windows-msvc`)

Linux CI is retained as a fast development signal but is not an MVP distribution commitment.

## License

No license has been selected yet. Publication of this repository does not itself grant a license to use, modify, or redistribute the code.
