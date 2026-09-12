# Hull–White extension: implementation and acceptance roadmap

Status: experimental. Date: 2026-09-12.
Base: PR #45 at `fde80879fa6ddbe7e6b842890313e421d27ddbe4`, including merged
LSV PR #47. The earlier stack's acceptance/merge status is unchanged.

The [decision](adr/0001-hull-white-equity-hybrid.md) records the extension to
the deterministic-rate requirements. The
[numerical contracts](hull-white-numerical-contracts-v0.1.md) define the model,
discounted measure, calibration estimator and reproducibility boundary.

## Available API

Rust exposes `pricing::models::HullWhite1Factor` and
`pricing::hull_white::HullWhiteEquityPricingPlan::{compile_bs,compile_lsv}`.
Python provides `HullWhiteModel`, `HullWhiteLsvTarget`, `HullWhiteEquityPlan`
and immutable `HullWhitePrice`. See the
[runnable example](../examples/python/hull_white_lsv.py).

Supply one currency's discount curve and explicit Hull–White parameters:
constant a and piecewise constant sigma_r. `compile_bs` takes a price-only
Black–Scholes request, equity/rate correlation and maximum equity step.
`compile_lsv` takes an LV target request plus its matching T-forward density,
Bergomi factor, three correlations and particle settings. Use `target.model`
in the request to keep variance and density paired. Calibration and pricing
run without the Python GIL. Rate parameters are illustrative inputs in the
example; no rate option calibration has been performed.

The payoff integration supports positive-horizon European, Asian, Lookback,
Digital and discrete Barrier prices with continuous deterministic carry,
proportional dividends and payment lags. Fixed-cash dividends and hybrid
Greeks are explicit errors in this release.

## Gates

| Gate | Scope | State |
| --- | --- | --- |
| H0 | Model, measure, input and reproducibility contracts | Recorded |
| H1 | Exact HW rate/integral kernel, curve fit, bond option | Implemented and focused tests pass |
| H2 | BS+HW, correlations, payment lag, dividends, MC/RQMC | Implemented and focused tests pass |
| H3 | Discounted Bergomi LSV calibration and Python integration | Implemented; broad calibration acceptance pending |
| H4 | Fixed-cash dividends with stochastic-bond coordinates | Pending |
| H5 | Hybrid AAD, recalibrated volatility risk and curve DV01 | Pending |
| H6 | Rate instrument calibration, stable wire, native replay and benchmark acceptance | Pending |

Initial numerical checks cover exact covariance against independent quadrature
across piecewise volatility knots and the a=0 limit; discounted stock and bond
martingales; curve fit including negative rates; bond call/put parity; Gaussian
BS+HW prices with positive/negative correlation and delayed payment; deterministic
limits; proportional-dividend barrier order; replay and independent LSV vanilla
repricing. Python tests check owned immutable inputs/results, paired target
constructors, replay and validation failures. The installed-wheel gate checks
the public stub/runtime API, all Python tests and the example.

These are focused implementation checks, not smile-wide calibration acceptance.
H3 requires retained particle-count, bandwidth and time-step studies with
independent calibration seeds, adverse smiles, long maturities and measured
fallback sensitivity. Pricing SE alone is insufficient for that gate. H5 must
differentiate rate drift/discounting and the discounted calibration, then check
recalibrated bumps; the deterministic-rate LSV VJP must not be reused unchanged.
H6 requires parameter calibration to specified instruments and new native
hybrid replay/performance fixtures. Existing platform CI remains a regression
gate for the pre-existing pricing baseline.

## Initial local verification

Linux development validation used pinned Rust 1.98.1 and CPython 3.12:

| Check | Result |
| --- | --- |
| Rust workspace excluding the Python extension | 307 passed, including 13 new hybrid tests |
| Python extension native tests | 3 passed |
| Statistical acceptance | 1 passed |
| Release wheel metadata, stub/runtime API, examples and Python suite | Passed; four examples and 46 tests |
| Clippy with warnings denied, formatting and Rust API documentation | Passed |
| Reference fixtures, schemas, Markdown links and dependency direction | Passed |

The installed-wheel example produced BS+HW Price 9.3459126271 and recalibrated
LSV+HW Price 9.2882475863 (conditional pricing SE 0.0042440927). Its terminal
calibration means were 0.9999186602 for D/P0 and 99.9032757397 for discounted
normalized equity, with four fallback nodes. This is one illustrative parameter
set and calibration seed, not H3 acceptance. Native Python unit tests needed a
local linker search path for the packaged libpython; no repository or CI linker
configuration was changed. Native macOS/Windows wheel, replay and benchmark
checks run in the pull request's CI.
