# Bass-LV numerical validation

[ADR 0027](../adr/0027-bass-local-volatility.md) and
[model specification](../../docs/models/bass-local-volatility.md).

This record describes the numerical Bass-LV gates and their development
observations. Historical workspace totals include concurrent work and are
not evidence for the isolated merge. The integration check on 2026-10-04
uses a fresh `main` worktree containing only Bass-LV changes; separate Local
Volatility paper-VegaKT and backtesting changes are excluded.

## Gates

Rust tests in `engine/bass_lv/tests.rs` cover:

- Gaussian convolution constants and positive-part payoffs against closed
  forms, including tails; central tolerance `1e-12`.
- Uniform-distribution moments, CDF/quantile inversion and calls.
- BS recovery at 0.5/1/2 years, spot 100, volatility 20%: local volatility
  within `0.0015` at spots 80/100/120; deterministic paths within 0.03 currency
  units; boundary reset continuity within `1e-12`.
- Conditional martingale identity against independent normal-density
  integration, within `1e-9`.
- 60,000-path marginal means and calls at strikes 80/100/120 within five
  standard errors plus 0.02 currency units for discretization.
- Skewed marginals from knots 70/90/110/150 and probabilities 0/0.25/0.75/1,
  followed by a mean-preserving 1.5x dilation; 40,000-path calls within five
  standard errors plus 0.025 currency units.
- Deterministic CDF propagation on grids 401/801/1601: finest error below
  0.001 and at least a twofold reduction. Sparse density jumps give first-order
  quantile-map interpolation convergence at the discontinuities.
- Invalid inputs, means, convex order, identical marginals, iteration
  exhaustion, narrow grids, invalid observations/normals and nonfinite payoffs.

Observed skewed-case errors at nine quantiles on 2026-10-03:
`0.00321155`, `0.00208498`, `0.000888692`. A small fixed-point residual alone
does not guarantee an equally small marginal error. Production diagnostics
also probe supplied CDF knots.

Python tests check immutability, error types, seeded replay, date insertion,
requested-only Asian fixings, discounted-price scaling, single-fixing
Asian/European equality and call/put consistency.

## Reproduce

```shell
cargo test --locked -p pricing --lib engine::bass_lv -- --nocapture
python -m maturin develop --locked
python -m unittest discover -s tests/python -p test_bass_lv.py -v
python examples/python/bass_lv.py
```

Acceptance covers specified synthetic cases. The PDF lacks machine-readable
SX5E quotes, so its market calibration, exotic prices and speed ratios are not
replicated.

## Local validation on 2026-10-03

- Nine Bass Rust tests and four Bass Python tests pass.
- Workspace Rust tests (excluding the Python crate): 657 pass, 50 ignored;
  the additional between-knot convex-order test was then run in the nine-test
  Bass suite. Python-crate Rust tests: three pass.
- Full Python discovery: 186 run, two skipped, no failures.
- Workspace all-target/all-feature Clippy with warnings denied, formatting,
  schema checks, three baseline reference fixtures and Python stub parsing
  pass. Markdown links passed initially; the final run found four concurrent
  backtesting links to the still-missing `vega-kt-backtest-budget-v0.1.md`.
  Bass documentation links resolve. Documentation builds with two pre-existing unresolved
  `cash` links in the stochastic-dividend Hull-White documentation.
- The Python extension builds and installs in the repository virtual
  environment; the runnable example matches its marginal call reference
  within one Monte Carlo standard error.
- The dependency-direction guard finds a pre-existing upward import in
  `engine/processes/local_vol_valuation.rs` from concurrent Local Volatility
  allocation work. A full working-tree source archive is rejected because
  pre-existing untracked backtesting files are outside its source manifest.
  Bass source files are explicitly required by the archive gate. These
  unrelated working-tree changes are preserved.
- Ignored release/stress acceptance panels and the complete benchmark/release
  wheel pipeline were not run for this additive model boundary.

## Surface projection and mapping risk follow-up (2026-10-03)

Six additional Rust cases and three additional Python cases cover the surface
adapter and Sections 5.2-5.4. The complete Bass suite now has 15 Rust tests and
seven Python tests, all passing. The existing Python facade smoke suite has
45 passing tests and the Python crate has three passing Rust tests. Workspace
all-target/all-feature checks, Clippy with warnings denied, formatting and
stub parsing pass. The extension builds and installs locally.

The eSSVI fixture uses `(T,theta,psi,rho_psi)` equal to
`(0.5,0.02,0.06,-0.025)`, `(1,0.04,0.09,-0.035)`,
`(2,0.08,0.13,-0.045)`, spot 100, and log strikes from -4 to 4.
At 1601 nodes, maximum projected call errors are respectively
`0.000636011`, `0.000454423`, and `0.000322512` currency units.
The one-year call errors on 401/801/1601 nodes are
`0.00727274`, `0.00181779`, and `0.000454423`.
Tests independently differentiate source calls in strike to verify the CDF,
then check mean, call repricing, calibration, and final Monte Carlo repricing.
Narrow tails and excessive mean correction must fail explicitly.

Risk checks include:

- Pathwise reverse versus central mapping bumps across all three intervals,
  interior observations, and fixed initial spot; absolute tolerance `2e-7`.
- European/Asian AAD prices and standard errors exactly equal to price-only
  evaluation on identical streams; CRN gradient comparisons at `3e-5`.
- Semi-analytic vanilla gradients against independently re-evaluated bumped
  deterministic prices within `3e-6`, with exactly zero later-expiry influence.
- Semi-analytic versus 80,000-path MC gradients within five MC errors plus
  `0.002` for CDF/map interpolation; linear-payoff gradients statistically zero.
- Invalid basis coordinates, unresolved/out-of-domain hats, nonmonotone bumped
  maps, wrong derivative shapes and nonfinite payoff derivatives.

The [eSSVI risk example](../../examples/python/bass_lv_risk.py) gives an Asian
price `6.954605` with one-standard-error `0.048492` on 50,000 paths, seed 42.
Its first mapping sensitivity is `0.003796844919`. Central bumps of
`1e-4`, `3e-5`, `1e-5`, `3e-6` differ from the reverse result by
`1.54e-5`, `-9.26e-8`, `-1.92e-8`, `-5.78e-9`. The larger bump crosses
piecewise payoff/interpolation branches; a bump ladder is therefore useful.

These risks are terminal-map derivatives, not market-IV VegaKT. Surface
projection, marginal calibration and the hat grid are held fixed. The
vanilla-hedge portfolio solve and market-IV projection remain separate work.
The full release/stress and benchmark pipelines were not repeated in this
follow-up. A later repository-wide Markdown check found concurrent links to
the then-missing `local-vol-gamma-numerics-v0.1.md`; Bass links resolve.

## Recalibrated market-IV VegaKT (2026-10-03)

Four further Rust cases and three further Python cases cover the full IV
quote pipeline. The Bass suites now have 19 Rust tests and 10 Python tests,
all passing. The new API is a separately identified central CRN estimator,
not market-IV AAD. The earlier mapping-risk scope above is unchanged.

Acceptance includes:

- Independent quote bumps rebuild the surface, projection and Bass model.
  Returned node means and paired standard errors match a separate two-pass
  calculation from sampled up/down paths within `1e-10`. Bucket-sum error is
  likewise checked including covariance, not by summing individual errors.
- Base prices and errors exactly match the existing price-only plan; replay,
  discount scaling, absolute-IV versus vol-point units, quote ordering and
  diagnostics survive the Rust/Python boundary.
- Flat 20% IV, spot/strike 100, one year: 60,000 paths with seed 948 give
  parallel Vega `39.71791250 +/- 0.30475333` per unit absolute IV versus Black's
  analytic `39.69525475`. The node sum is `39.71763702 +/- 0.30474710`.
- A skew-smile fixture uses vols `0.24/0.22/0.20/0.18/0.16` at log strikes
  `-2/-1/0/1/2`, at both 0.5 and 1 year. Each node's one-year ATM call risk
  agrees with the independent market-call oracle within five paired errors
  plus `0.06`: only the one-year ATM source quote has analytic nonzero Vega.
- Later-expiry IV nodes give exactly zero earlier-expiry call sensitivity.
  A fixed time-zero payoff has zero quote risk, linear terminal payoff risk
  is statistically zero, and call-minus-put risk matches the linear payoff.
- For an Asian with fixings 0.25/0.75/1 year, all quote sensitivities differ
  by less than `0.02` between steps `1e-4` and `5e-5`; the ladder also includes
  `2e-4`. Brownian refinement 401 to 801 changes each by less than `0.08` on
  the specified 3,000-path seed-42 stream. These are fixture-specific gates.
- Invalid quote shapes, nonfinite/unrepresentable/nonpositive bumps, invalid
  pricing inputs and failed arbitrage/calibration scenarios raise errors.
  A scenario failure identifies the quote and signed shift.

The [VegaKT example](../../examples/python/bass_lv_vega_kt.py) uses 801 Brownian
nodes, 1601 projection nodes, `cdf_tolerance=1e-7`, and 50,000 paths, seed 42.
It gives Asian price `5.653587 +/- 0.040830`. Parallel Vega per vol point is
`0.28229782 +/- 0.00225221`; the node sum is
`0.28229734 +/- 0.00225222`. Maximum scenario CDF residual is `8.92e-8`, while
maximum marginal CDF error is `2.33e-5`. These are different diagnostics.

An exploratory `cdf_tolerance=1e-9` did not converge on the coarse fixture:
the 401-node residual reached about `7.74e-9`. The API exposes the failure;
it does not relax tolerances. This underlines the need to refine grids as
well as choose a tolerable finite-difference step. Sampling errors do not
bound discretization, calibration or bump bias.

Reproduce the new Python cases and example with:

```shell
python -m unittest discover -s tests/python -p 'test_bass_lv*.py' -v
python examples/python/bass_lv_vega_kt.py
```

The Python extension builds and installs locally. Full Python discovery runs
192 tests with two skips and no failures; the Python crate's three Rust tests
also pass. The default workspace Rust regression (excluding the Python crate)
completes with 673 passes, zero failures and 50 ignored tests across 47 panels.
Workspace all-target/all-feature Clippy with warnings denied,
formatting, stub parsing and whitespace checks pass. Repository Markdown links
passed before a concurrent documentation update; the final rerun finds three
links to the then-missing `local-vol-gamma-recovery-v0.1.md` in non-Bass files.
Bass documentation links resolve.
The dependency-direction guard still reports the pre-existing upward import
in `engine/processes/local_vol_valuation.rs`, outside the Bass changes. No
new dependency is introduced. Ignored release/stress and benchmark panels
are not executed by the default test run.

## Common PricingRequest integration (2026-10-03)

Six Rust integration cases in `crates/pricing/tests/bass_request.rs` and five
Python cases in `tests/python/test_bass_request.py` pass. Together with the
standalone suites, Bass now has 25 Rust and 15 Python test cases. This stage
adds the normalized residual-equity model to `ModelSpec`, common payoff/risk
execution, Python `Model` construction and current JSON v3 serialization.

The tests cover:

- Request/result round-trips, request/plan fingerprints, rejection of Bass
  payloads claiming v1/v2, exported v3 model shape and unknown-field rejection.
- Cash 6 at 0.25 years, proportional dividend 10% at one year, and a future
  cash reserve of 4 at 1.5 years, with discount/dividend factors 0.95/0.98 at
  one year. A one-year call and central Delta/Gamma/parallel Vega agree with
  an independent shifted-Black oracle within six sampling errors plus
  absolute allowances `0.005/0.002/0.003/0.05`, respectively. This is a
  finite-grid, finite-step fixture, not an exact equality assertion.
- A separate request-level spot recompile at `S0 +/- 0.2` matches Delta within
  `1e-10` and Gamma within `1e-9`. Independent source-IV recompiles at
  `sigma_j +/- 1e-4` match selected quote risks within `1e-8`.
- RQMC uses 8 independent scrambles, 2048 points each and antithetic pairs
  (32768 evaluated paths), with Brownian bridge ordering. Changing worker
  counts preserves price/risk values exactly. Six quote buckets produce a
  36-entry full covariance matrix; serialized results retain all metadata.
- Weighted known Asian fixings, nonunit notional and delayed payment match a
  transformed Black payoff. A fully fixed Asian has deterministic price and
  exactly zero Delta, Vega and quote risks, including zero-dimensional RQMC.
- A discrete up-and-in barrier at a cash ex-date is compared path by path
  against an independently reconstructed payoff using both pre- and
  post-dividend spot. Ignoring the pre-dividend spot changes the price.
- A seasoned single-future-observation lookback equals the corresponding
  call on the same stream. Digital smoothing widths 4/2/1 match separately
  compiled requests for price, Delta and Gamma. Unsupported continuous
  barriers, early exercise, smile conventions, mismatched quote axes and
  observations beyond the calibrated horizon fail explicitly.

The [common-request example](../../examples/python/bass_lv_request.py) has a
past Asian fixing, cash/proportional dividends, delayed settlement and six
source quotes. It produces `1.718337 +/- 0.000855`, raw Delta `0.188722`, raw
Gamma `0.012967` and parallel Vega per vol point `0.180142`. It uses 8 RQMC
scrambles and 32768 paths. The maximum base marginal CDF error is about
`9.18e-5`. This diagnostic is separate from price sampling uncertainty.

Reproduce these integration cases with:

```shell
cargo test --locked -p pricing --test bass_request
python -m unittest discover -s tests/python -p 'test_bass_request.py' -v
python examples/python/bass_lv_request.py
```

The common estimator remains central CRN differences. Input IVs are quoted
in residual-equity coordinates; physical-spot IV conversion, density-based
risk allocation, full-calibration AAD, continuous barriers and American
exercise are not supplied by this integration. All quote nodes are retained
even when the common VegaKT request carries a relative-density threshold.

The all-feature default Rust workspace regression (excluding the Python
crate) finishes with 695 passed, zero failed and 50 ignored across 50 panels.
The Python crate's three Rust tests pass. Full Python discovery runs 197 tests
with two skips and no failures. The extension builds and installs in the
repository virtual environment. Workspace Clippy with warnings denied,
formatting, schema checks, Markdown links, stub parsing, whitespace checks and
a working-tree source-archive check pass. Local-volatility, path-dependence
and early-exercise reference fixtures pass their existing 67/114/61 checks.

Rust documentation builds; its two remaining broken-link warnings refer to
the pre-existing `[cash]` formula in
`engine/processes/stochastic_dividends/hull_white.rs`. The dependency-direction
guard still rejects the existing risk-report import in
`engine/processes/local_vol_valuation.rs`. Neither location is part of this
Bass integration. Concurrent documentation links that were initially absent
resolve on the final Markdown check. Ignored statistical/stress panels,
release wheels and the benchmark suite were not rerun for this stage.
