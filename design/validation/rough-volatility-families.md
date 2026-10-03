# Rough-volatility family extension: validation record

## Provenance and acceptance boundary

Prepared 2026-10-02 from upstream main
`fbd9beafe4c01eaac3fae713bf4914bdea23b9bf`. The extracted baseline tree was
verified as `7192ff77f71ffa546abc9af5fc1b8fdb55fef55c`, identical to upstream.
The implementation is isolated on `feat/rough-volatility-families`; existing
PRs #115/#116 and main are not part of this change.

**Experimental, price-only implementation with executed tests, not a production
acceptance report.** The first delivered candidate was uncompiled. A subsequent
validation pass obtained the pinned compiler and locked source dependencies
through an isolated GitHub Actions artifact, verified its hashes, and executed
the Rust code and a freshly built Python wheel locally. No model equations,
production RNGs or numerical budgets were changed to make a failure pass.

Local platform: Linux x86_64, Rust 1.98.1
`48a229ceaefd4985c50990b14116b6d856af0985`, Cargo 1.98.1, Python 3.13.5.
`Cargo.lock` SHA-256 remained
`376f6dd0a680433efc1ec11e133dbe70e05e018761b7a3f12c8d090e3fcd3096`.
The wheel was built with the pinned release profile, not a reduced-optimization
substitute. The offline Python venv inherits system NumPy/SciPy; this is not a
claim of a clean dependency installation. Normal wheel CI retains that gate.

[Model contracts and exclusions](../../docs/models/rough-volatility-families.md)
are separate from this validation record. Logs accompany the delivery, and the
focused CI workflow retains release logs for each platform. Workflow definitions
are not evidence that all remote jobs have completed successfully; consult the
checks for the exact candidate commit.

## Initial failures and corrections

The pinned formatter was run. Clippy then rejected a nested tuple type used for
physical dividend observations. Named `PhysicalObservation` / `DividendJump`
structures replace it without changing arithmetic; no lint was suppressed.

The initial debug run passed 15 family tests and failed one whole-result equality
assertion across worker counts. Numerical prices and errors matched; the plan
fingerprints differed because execution policy is intentionally fingerprinted.
The test now separately requires bitwise price and standard-error equality,
identical sampling metadata, each result's own plan fingerprint, different
fingerprints across execution policies, and exact repeated execution of each
plan. Production fingerprint semantics were not changed.

The newly built wheel initially failed the strict stub-shape check because its
five new public classes were absent from the wheel checker's allowlist. Explicit
class, member, signature, static-method and property contracts were added.
A regression test rejects missing new classes, a changed factory default and a
removed static-method decorator. No existing packaging guard was disabled.

## Executed boundary, API and estimator tests

The `rough_volatility_families` Rust integration target passed all **16 tests**
in debug, release and no-default-features configurations. It covers constructor
and grid rejection, forward-variance interpolation, arithmetic failure, six
constant-variance limits, Heston/lift Brownian limits, one-cell rough Heston,
quadratic feedback, discrete variance centering, mixed/SABR identity, classical
SABR exponential-curve limits, normal/absorbing paths, stationary and conditioned
OU limits, independent fOU covariance references, predictability and replay.
The price adapter checks affine-dividend comparisons with the existing pure
rough Bergomi engine and constant-limit Black prices.

The numerics crate passed **13 tests**, including the two new Gamma/fOU tests.
Workspace/all-target/all-feature Clippy with warnings denied passed after the
correction. These checks do not establish general numerical accuracy.

The actual CPython 3.13 wheel passed the **seven** new Python tests and the
six-family example. Packaging checks cover members, layout, ABI, metadata,
RECORD digests, locked SBOM dependencies, exact source-stub bytes and runtime
symbols. Python tests include class ownership, copied path vectors, model
factories, carrier-sigma semantics, pre-dividend Barrier observation and delayed
payment discounting. Full-suite counts and broader regression results are
reported with the delivery logs, not inferred from this focused coverage.

`independent_primal_payoffs_reconstruct_mc_and_rqmc_sampling_errors` passed in
release. It has **48 panels**: six models, MC/RQMC, antithetic on/off and Brownian
bridge on/off. A separate payoff reconstruction and two-pass variance calculation
check positive, nonvacuous errors and the correct independent units. MC has 64
units; RQMC has four scrambles of 64 points. Price/SE discrepancies must be at
most `2e-12`. This reference shares the production primal path and RNG/bridge,
so it validates payoff/estimator aggregation, not independent model dynamics.

## Independent nondegenerate finite-grid prices

The new [fixture](../../fixtures/rough-volatility/two-step-prices.json) contains
18 nonzero-volatility-randomness prices, generated by an independent
[Python script](../../scripts/check_rough_volatility_prices.py). No production
paths, RNGs, kernels or payoff helpers are called by the generator.

Spot is 100, maturity is one year, the grid is exactly `[0, 0.5, 1]`, strikes
are 90/100/110, rates are zero, and there are no dividends. For five families,
conditioning on the first volatility innovation leaves a conditional Black
price; the asset shocks are integrated out analytically and the remaining
one-dimensional Gaussian integral is evaluated independently. Heston's
truncation kink is split explicitly. RFSV uses two-dimensional Gaussian
quadrature with an independently integrated stationary-fOU spectral covariance.
Two integration tolerances / quadrature orders must agree before fixture checks.

This is an independent benchmark for the **specified finite two-step scheme**.
It is not an exact continuous-time rough-model price, a grid-refinement test or
a bound on rough-Heston/lift approximation error.

The ignored release test
`nondegenerate_two_step_prices_match_conditional_gaussian_reference` is explicitly
run in the focused workflow. It covers **36 comparisons**: six families, seeds
91/1973 and three strikes. Every row uses 16 independent RQMC scrambles, 4,096
antithetic units per scramble, Brownian bridge enabled, and 131,072 evaluated
paths. The following budgets were fixed before the first run and all passed
without changing seeds, sample counts, parameters or budgets:

| Check | Budget | Largest observed value |
| --- | ---: | ---: |
| Sampling standard error | `0 < SE <= 0.005` | 0.003840031 |
| Absolute price gap | `<= 5*SE + 2e-7` | 0.003564368 |
| `abs(gap) + 4*SE` | `<= 0.03` | 0.018746099 |

The middle row is a row-specific statistical test, not a fixed absolute budget.
These finite-panel statistical checks are not simultaneous probabilistic error
guarantees. Prices and budgets are currency units for unit notional and spot
100, not implied-volatility basis points.

## Independent Gaussian and limiting-law references

The original independent script checks 30 fOU covariance values using spectral
and separate nonoscillatory integrals, positive definiteness on five irregular
grids, four analytic one-cell Heston references and a Black value. The maximum
covariance discrepancy was approximately `3.32e-14`. Executable Rust/Python
tests also compare the production implementation to retained reference values.

During the initial algorithm design, a locally divided-tolerance Simpson
prototype failed at `H=0.001, lag=0.0001`. Global largest-error-panel refinement
replaced it. No covariance jitter or clipping was used to conceal that failure.
Independent prototype comparisons support the algorithm choice but do not add
a rigorous numerical integration bound to the production covariance routine.

## Reproduction and remaining work

```shell
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked -p pricing-numerics
cargo test --locked -p pricing --test rough_volatility_families --test rough_volatility_pricing_reference
cargo test --locked --no-default-features -p pricing --test rough_volatility_families --test rough_volatility_pricing_reference
cargo test --locked --release -p pricing --test rough_volatility_families --test rough_volatility_pricing_reference -- --include-ignored --nocapture
cargo test --locked --workspace
python scripts/check_rough_volatility_reference.py
python scripts/check_rough_volatility_prices.py
python -m maturin build --locked --release
# Install the built wheel in the intended environment before running:
python -m unittest discover -s tests/python -v
python examples/python/rough_volatility_families.py
python scripts/check_markdown_links.py
```

The focused workflow runs independent fixture checks and the two new Rust
targets on Linux/macOS/Windows. Existing full CI retains packaging, stable
schema, Clippy, regression and longer model-risk/accuracy gates.

Production admission still requires nondegenerate multi-step time refinement,
lift-kernel/factor refinement over the intended horizon, parameter-stress and
martingale diagnostics, broader path-dependent payoff validation and independent
review. General truncation/absorption, time, kernel, covariance, calibration and
model errors are not included in reported sampling SE. Fourier/Riccati pricing,
calibration, VIX/SSR, AAD/Greeks and new LSV/rate/dividend/multi-asset combinations
remain outside this increment.
