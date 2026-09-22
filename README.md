# Rust Pricing Library

An extensible derivatives-pricing library for model validation and quantitative research. The calculation core is written in Rust and exposes a typed Python interface for interactive single-trade analysis.

The first vertical slice is a European vanilla option under Black-Scholes, with
analytical reference values, Pseudo-Monte Carlo, randomized Sobol QMC, AAD
Greeks, common-random-number bump validation, and deterministic replay. The
same public request, wire, and Python surfaces also expose a Black-76
constant-volatility model on the market forward. Price-only Monte Carlo
requests can also use cash-or-nothing and asset-or-nothing Digital calls and
puts, fixed-strike Barrier calls and puts with an explicit discrete or
continuous monitoring contract and optional expiry rebates, forward-starting,
partially fixed, or fully fixed arithmetic
average-price Asian calls and puts, plus fixed-strike discrete-monitoring
Lookback calls and puts with future or fully fixed monitoring
under the constant-volatility engines. Pathwise Delta, bumped-AAD Gamma, and
Vega are available for European, Asian, and Lookback products under constant
volatility. Digital products additionally support explicit compact-C2 payoff
smoothing for Price, pathwise Delta/Vega, bumped-AAD Gamma, and CRN validation
through the Rust, JSON, and Python request surfaces. The same explicit policy
supports endpoint- and affine-dividend-jump-smoothed risk for
discrete-monitoring Barrier products with a monotone hit state and
KnockIn/KnockOut rebate parity. Dividend collisions reuse one normalized path
state for pre- and post-jump Spot observations without adding a random
coordinate. Digital and Barrier risk without an explicit smoothing width
remains rejected. Continuous Barrier requests are evaluated under the
constant-volatility models with a conditional Brownian-bridge survival
estimator, including affine-dividend jump boundaries, pathwise Delta/Vega, and
bumped-AAD Gamma. Explicit compact-C2 smoothing applies matched endpoint,
affine-dividend-jump, and bridge-survival weights so Price and Greeks use one
surrogate payoff. Local Volatility uses the same exact or smoothed bridge over
every Log-Euler sub-step with trapezoidal endpoint Local variance, CRN
Delta/Gamma, and reverse Local-volatility Vega/VegaKT.
Result diagnostics report endpoint and dividend-jump hit fractions separately
from the mean conditional bridge hit weight, together with bridge interval and
stable numerical-branch counts.

## Documentation

| Topic | Start here |
| --- | --- |
| Library usage, APIs, diagnostics and compatibility | [Library guide](docs/library/README.md) |
| Supported financial products | [Product reference](docs/products/README.md) |
| Models, calibration, simulation and risk conventions | [Model reference](docs/models/README.md) |
| Requirements, architecture, ADRs, roadmaps and acceptance evidence | [Design and development records](design/README.md) |

The [documentation index](docs/README.md) links the library, product and model
guides.
Design documents are maintained separately in `design/`.

## Status

The European Black–Scholes vertical slice has completed Gates G0–G8 and is the accepted baseline for the Local Volatility/VegaKT stage. Local Volatility/VegaKT has completed Gates L0–L8 and is accepted with exact SSVI/eSSVI, Dupire Local variance, non-uniform interpolation, Log-Euler Local Volatility simulation, Local Volatility Price/Delta/Gamma/Vega/VegaKT MC/RQMC evaluation, Local Vega/VegaKT operators, affine dividends, public Rust/JSON/Python request surfaces, independently checked equation fixtures, same-platform replay fixtures on Apple Silicon macOS and Windows x86-64, and retained benchmark artifacts. The accepted baseline supports deterministic Pseudo-MC and randomized Sobol QMC Price and Greeks, including independent-scramble uncertainty and a non-uniform Brownian-bridge plan. Stable Rust and typed PyO3 request/plan/result facades are available, including immutable diagnostics and warnings. CI builds, installs, smoke-tests, benchmarks, and replay-checks private wheels on Apple Silicon macOS and Windows x86-64.

Path Dependence has completed Gates P0-P8 and is accepted with deterministic
and statistical acceptance, compact-C2 continuous Barrier smoothing,
same-platform replay on Apple Silicon macOS and Windows x86-64, benchmark
coverage, and Rust/JSON/Python conformance evidence.

Early Exercise has completed Gates E0-E8 and is accepted with deterministic
and statistical American Call/Put acceptance, independent Bermudan-tree and
in/out-of-sample evidence, same-platform replay on Apple Silicon macOS and
Windows x86-64, and training/valuation/fixed-policy-risk benchmark workloads.

## Workspace

Pure one-/two-factor and rough Bergomi are available through Rust
`StochasticVolatilityPricingPlan` and Python `StochasticVolatilityPlan`, with
deterministic market curves and no LV target or particle calibration. Their
Hull–White counterparts use `HullWhiteEquityPlan.compile_bergomi`,
`compile_bergomi_two_factor` and `compile_rough_bergomi`. Initial forward
variance is flat; Markovian OU factors are normalized to preserve its mean.
Price, Spot Delta, initial-volatility and initial-curve AAD share the existing
escrowed-dividend engine. See the [pure SV contracts](docs/models/pure-stochastic-volatility.md)
and [example](examples/python/pure_bergomi.py).

An experimental one- or two-factor Bergomi LSV extension is available through Rust
`pricing::lsv::BergomiLsvPricingPlan` and Python `BergomiLsvPlan` /
`Bergomi2FactorLsvPlan`. It calibrates
an existing Local Volatility target with particles, prices with independent
MC/RQMC paths, and differentiates the finite calibration to effective Dupire
Local-variance nodes. It reuses the existing payoff graphs and affine dividends.
Market-IV VegaKT and sticky-smile Spot Greeks are not enabled at this boundary.
See the [LSV example](examples/python/bergomi_lsv.py) and
[acceptance roadmap](design/roadmaps/lsv-roadmap-v0.1.md).

Experimental one-currency stochastic-rate pricing is available through Rust
`HullWhiteEquityPricingPlan` and Python `HullWhiteEquityPlan`. It combines
one-factor Hull–White with BS or particle-recalibrated one-/two-factor Bergomi LSV, including
equity/rate correlation, stochastic discounting, proportional dividends and
payment lags. All simulations use **escrowed** fixed and mixed cash/proportional
payouts: known future dividends are reserved, including beyond option expiry.
Python `cash_dividend_model=None` and `"escrowed"` select the same model; Rust
`*_with_cash_dividends` methods remain compatibility aliases. IV inputs use the
[escrowed quote coordinate](docs/models/hull-white-cash-dividends.md#lsv-target-coordinate-and-calibration).
Explicit `evaluate_aad()` returns Spot
Delta, BS Vega, initial-curve risk/DV01 and recalibrated paired variance/density
adjoints. Targets built with `HullWhiteLsvTarget.from_market_iv` additionally
return quote-node VegaKT, market scaling and a parallel IV Vega, reversing both
variance and density through explicit cubic/time interpolation. Calibration targets
use escrow coordinates; prepare physical option quotes in that coordinate before
constructing the target.
See the
[VegaKT contracts](docs/models/hull-white-vegakt.md). LSV AAD requires `retain_reverse_trace=True`; model parameters and
payout quotes remain fixed. See the [AAD contracts](docs/models/hull-white-aad.md),
[cash-dividend contracts](docs/models/hull-white-cash-dividends.md),
[example](examples/python/hull_white_lsv.py) and
[calculation specifications](docs/models/hull-white-calculation-specifications.md).

The same engine now supports experimental rough Bergomi and particle-calibrated
rough-LSV through `compile_rough_bergomi` / `compile_rough_lsv` and the Python
`RoughBergomiModel`. A nonuniform Volterra hybrid scheme connects the rough
driver to Hull–White, escrowed dividends, first-order AAD and quote-node
VegaKT. Pure rough uses flat initial forward variance; rough-LSV fits the paired
target. H and eta are fixed for risk. Direct convolution costs O(time_steps^2)
per path. See the [example](examples/python/rough_bergomi.py) and
[calculation and API specifications](docs/models/rough-bergomi.md).

Experimental multi-asset pricing is available through Rust
`multi_asset::MultiAssetPricingPlan` and Python `MultiAssetPlan`. Correlated
BS/Local Volatility/Bergomi LSV assets support Basket, Worst-of and unseasoned memory/no-memory
Autocallables, per-asset affine dividends, dated PSD correlations, MC/RQMC,
asset-labelled Delta/BS Vega/Local variance adjoints and optional cross Gamma.
The API uses the shared payoff tape, with explicit smoothing for discontinuous
Autocallable AAD. It is single-currency and supports deterministic rates or a
common Hull–White rate with BS, one-/two-factor Bergomi LSV and rough-LSV. The HW mode adds
explicit price/rate and volatility/rate correlations, exact integrated-rate
innovations, per-cashflow stochastic discounting, paired variance/density AAD,
market-IV VegaKT from retained quotes and initial-curve risk. See the
[HW contracts](docs/models/bergomi-hull-white.md) and
[three-product example](examples/python/bergomi_hull_white.py).
Rough assets select `MultiAssetRoughLsvConfig` with per-asset H and eta. Their
non-Markov Volterra histories are driven jointly with all spot, OU and rate
innovations, including explicit cross-asset correlations. See the
[rough multi-asset contracts](docs/models/multi-asset-rough-bergomi.md) and
[rough three-product example](examples/python/multi_asset_rough_bergomi.py).
Deterministic-rate LSV assets use their LV model as a marginal particle-calibration
target; HW LSV additionally requires the matching paired target. Optional full
spot/volatility/rate Brownian correlations and joint OU/power-kernel transitions are
available. AAD
includes particle recalibration; target-risk standard errors are available for
RQMC and conditional on the calibration. See the
[contracts](docs/models/multi-asset.md) and
[example](examples/python/multi_asset.py), plus the
[LSV contracts](docs/models/multi-asset-lsv.md) and
[LSV example](examples/python/multi_asset_lsv.py).

Particle-calibrated **Local Correlation** is available for BS/LV and Bergomi LSV,
including two factors, rough-LSV and common Hull–White rates, through `MultiAssetPlan.compile(..., local_correlation=...)`.
It fits one positive basket of normalized continuous equity martingales by
varying a scalar mixture of two supplied PSD correlation schedules. It exposes
support/feasibility diagnostics and joint basket/constituent volatility AAD
through the finite particle recalibration. See the
[calculation and coordinate specifications](docs/models/local-correlation.md) and
[three-product example](examples/python/local_correlation.py).
Joint LSV/HW endpoints preserve each asset's spot/vol/rate marginal block.
HW basket calibration requires a paired variance/density target and includes
discount weighting, rate correction, and recalibrated density/quote VegaKT.
See the [LSV/HW example](examples/python/local_correlation_lsv_hw.py).

The experimental LSV/Hull–White adapters reject American exercise,
continuous Barrier monitoring and smoothing-width ladders; these features
remain available through the general BS/Local Volatility facade. The
[integration decision](design/adr/0006-integrate-completed-baseline.md) records
the shared observation and schema boundaries.

| Crate | Responsibility |
| --- | --- |
| `pricing-numerics` | Financially independent normal functions, PSD correlation factorization and deterministic reduction |
| `pricing` | Financial definitions and valuation, with private execution modules |
| `pricing-python` | Python conversion, exceptions and bindings to `pricing` |

The dependency chain is `pricing-python → pricing → pricing-numerics`. CI checks
both the Cargo graph and selected internal boundaries. `pricing::core`, `market`,
`product`, `models`, `mc`, `risk`, `hull_white`, `lsv`, `analytical`, and the root
entry points retain their existing public paths. The former seven financial
subcrates are removed; direct users must follow the
[three-crate migration guide](docs/library/three-crate-migration.md).

## Development

The repository pins Rust 1.98.1. After installing [rustup](https://rustup.rs/), run:

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
cargo test --locked --release -p pricing --test extended_risk_acceptance -- --ignored --nocapture --test-threads=1
cargo doc --locked --workspace --all-features --no-deps
cargo metadata --locked --format-version 1 --no-deps | python3 scripts/check_dependency_direction.py
```

The Python extension is built with [maturin](https://www.maturin.rs/):

```shell
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

A cell-oriented end-to-end example is available at
[`examples/python/european_bs.py`](examples/python/european_bs.py). It uses the
native builders, NumPy inputs, AAD Greeks, deterministic plan compilation, and
structured diagnostics. The distributed wheel contains `rust_pricing.pyi` and
the PEP 561 `py.typed` marker generated by maturin.

The American LSM example at
[`examples/python/american_lsm.py`](examples/python/american_lsm.py) uses
independent training and valuation paths, round-trips the v3 request, and
evaluates the same Put under Black-Scholes and Local Volatility with
fixed-policy Greeks and immutable exercise diagnostics.

A Local Volatility/VegaKT valuation example is available at
[`examples/python/local_vol_vegakt.py`](examples/python/local_vol_vegakt.py).
It builds Local variance and reporting-IV grids from calibrated eSSVI slices,
requests VegaKT reporting buckets, round-trips the canonical JSON payload, and
evaluates Price, Greeks, and the VegaKT result report.

The path-dependence example at
[`examples/python/path_dependence.py`](examples/python/path_dependence.py)
keeps exact contractual Digital Price separate from an explicitly smoothed
Price/Greek calculation and evaluates a caller-ordered, non-adaptive smoothing
width ladder with adjacent-width differences.

The Python facade exposes the same versioned JSON boundary as Rust:
`PricingRequest.to_json()`, `PricingRequest.to_pretty_json()`,
`PricingRequest.from_json()`, `PricingResult.to_json()`,
`PricingResult.to_pretty_json()`, and `PricingResult.from_json()`. The bundled
Draft 2020-12 schemas are available through `request_json_schema()` and
`result_json_schema()` for external validation or fixture review. Pricing
results expose Price and Greek estimates with standard errors, confidence
intervals, estimator labels, effective sample counts, raw/market-scaled risk
units, and replay metadata for the schema version, request fingerprint, library
version, producing platform, and any ordered schema migration provenance.
Result diagnostics expose replay-critical
seeds, execution policy fields, curve regions, Payoff fingerprints, QMC
direction/scramble checksums, bump validation policy fields, and CRN bump
validation estimates for requested Greeks.

Pull-request and `main` CI retain native CPython 3.12 wheels for the two MVP
platforms as short-lived workflow artifacts. Each wheel is installed into a
fresh virtual environment before the Python API smoke suite runs. CI also
retains a source archive for the exact commit, including the locked Rust
dependency metadata required to consume the Rust crates privately.

The [benchmark baseline protocol](docs/library/benchmarking.md) records native
Rust and installed-wheel Python timings and the host/build metadata required to
interpret them.

## Platforms

The MVP support targets are:

- Apple Silicon macOS (`aarch64-apple-darwin`)
- Windows x86-64 using MSVC (`x86_64-pc-windows-msvc`)

Linux CI is retained as a fast development signal but is not an MVP distribution commitment.

## License

No license has been selected yet. Publication of this repository does not itself grant a license to use, modify, or redistribute the code.
