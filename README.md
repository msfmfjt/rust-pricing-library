# Rust Pricing Library

An extensible derivatives-pricing library for model validation and quantitative research. The calculation core is written in Rust and exposes a typed Python interface for interactive single-trade analysis.

The first vertical slice is a European vanilla option under Black-Scholes, with
analytical reference values, Pseudo-Monte Carlo, randomized Sobol QMC, AAD
Greeks, common-random-number bump validation, and deterministic replay. The
same public request, wire, and Python surfaces also expose a Black-76
constant-volatility model on the market forward. Price-only Monte Carlo
requests can also use cash-or-nothing and asset-or-nothing Digital calls and
puts, fixed-strike discrete Barrier calls and puts with optional expiry
rebates, forward-starting, partially fixed, or fully fixed arithmetic
average-price Asian calls and puts, plus fixed-strike discrete-monitoring
Lookback calls and puts with future or fully fixed monitoring
under the constant-volatility engines. Pathwise Delta, bumped-AAD Gamma, and
Vega are available for European, Asian, and Lookback products under constant
volatility. Pathwise risk requests for Digital and Barrier products are
rejected until discontinuous-event AAD support is implemented.

## Design baselines

- [Frozen MVP requirements](docs/requirements-v1.0.md)
- [Architecture](docs/architecture-v0.1.md)
- [European Black–Scholes implementation roadmap](docs/european-bs-roadmap-v0.1.md)
- [European Black–Scholes conformance report](docs/european-bs-conformance-v0.1.md)
- [European Black–Scholes diagnostics catalogue](docs/european-bs-diagnostics-v0.1.md)
- [Local Volatility and VegaKT implementation roadmap](docs/local-vol-vegakt-roadmap-v0.1.md)
- [Local Volatility and VegaKT numerical contracts](docs/local-vol-vegakt-numerical-contracts-v0.1.md)
- [Local Volatility and VegaKT diagnostics catalogue](docs/local-vol-vegakt-diagnostics-v0.1.md)
- [Local Volatility and VegaKT conformance report](docs/local-vol-vegakt-conformance-v0.1.md)
- [Release readiness](docs/release-readiness-v0.1.md)

## Status

The European Black–Scholes vertical slice has completed Gates G0–G8 and is the accepted baseline for the Local Volatility/VegaKT stage. Local Volatility/VegaKT has completed Gates L0–L8 and is accepted with exact SSVI/eSSVI, Dupire Local variance, non-uniform interpolation, Log-Euler Local Volatility simulation, Local Volatility Price/Delta/Gamma/Vega/VegaKT MC/RQMC evaluation, Local Vega/VegaKT operators, affine dividends, public Rust/JSON/Python request surfaces, independently checked equation fixtures, same-platform replay fixtures on Apple Silicon macOS and Windows x86-64, and retained benchmark artifacts. The accepted baseline supports deterministic Pseudo-MC and randomized Sobol QMC Price and Greeks, including independent-scramble uncertainty and a non-uniform Brownian-bridge plan. Stable Rust and typed PyO3 request/plan/result facades are available, including immutable diagnostics and warnings. CI builds, installs, smoke-tests, benchmarks, and replay-checks private wheels on Apple Silicon macOS and Windows x86-64.

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
python3 scripts/check_local_vol_reference_fixture.py
python3 scripts/check_schemas.py
python3 scripts/check_markdown_links.py
git archive --format=tar.gz --output /tmp/rust-pricing-source-check.tar.gz HEAD
python3 scripts/check_source_archive.py /tmp/rust-pricing-source-check.tar.gz
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features --exclude pricing-python
cargo test --locked -p pricing-python
cargo test --locked -p pricing --test statistical_acceptance -- --ignored --nocapture
cargo metadata --locked --format-version 1 --no-deps | python3 scripts/check_dependency_direction.py
```

The Python extension is built with [maturin](https://www.maturin.rs/):

```shell
python -m maturin develop --locked
python -m unittest discover -s tests/python -v
python -m maturin build --locked --release --out dist
python scripts/smoke_test_wheel.py
python scripts/run_benchmark_suite.py
python scripts/check_benchmark_reports.py benchmark-results
```

A cell-oriented end-to-end example is available at
[`examples/python/european_bs.py`](examples/python/european_bs.py). It uses the
native builders, NumPy inputs, AAD Greeks, deterministic plan compilation, and
structured diagnostics. The distributed wheel contains `rust_pricing.pyi` and
the PEP 561 `py.typed` marker generated by maturin.

A Local Volatility/VegaKT valuation example is available at
[`examples/python/local_vol_vegakt.py`](examples/python/local_vol_vegakt.py).
It builds Local variance and reporting-IV grids from calibrated eSSVI slices,
requests VegaKT reporting buckets, round-trips the canonical JSON payload, and
evaluates Price, Greeks, and the VegaKT result report.

The Python facade exposes the same versioned JSON boundary as Rust:
`PricingRequest.to_json()`, `PricingRequest.to_pretty_json()`,
`PricingRequest.from_json()`, `PricingResult.to_json()`,
`PricingResult.to_pretty_json()`, and `PricingResult.from_json()`. The bundled
Draft 2020-12 schemas are available through `request_json_schema()` and
`result_json_schema()` for external validation or fixture review. Pricing
results expose Price and Greek estimates with standard errors, confidence
intervals, estimator labels, effective sample counts, raw/market-scaled risk
units, and replay metadata for the schema version, request fingerprint, library
version, and producing platform. Result diagnostics expose replay-critical
seeds, execution policy fields, curve regions, Payoff fingerprints, QMC
direction/scramble checksums, bump validation policy fields, and CRN bump
validation estimates for requested Greeks.

Pull-request and `main` CI retain native CPython 3.12 wheels for the two MVP
platforms as short-lived workflow artifacts. Each wheel is installed into a
fresh virtual environment before the Python API smoke suite runs. CI also
retains a source archive for the exact commit, including the locked Rust
dependency metadata required to consume the Rust crates privately.

The [benchmark baseline protocol](docs/benchmarking-v0.1.md) records native
Rust and installed-wheel Python timings and the host/build metadata required to
interpret them.

## Platforms

The MVP support targets are:

- Apple Silicon macOS (`aarch64-apple-darwin`)
- Windows x86-64 using MSVC (`x86_64-pc-windows-msvc`)

Linux CI is retained as a fast development signal but is not an MVP distribution commitment.

## License

No license has been selected yet. Publication of this repository does not itself grant a license to use, modify, or redistribute the code.
