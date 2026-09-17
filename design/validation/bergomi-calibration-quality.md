# One-factor Bergomi calibration quality

Scope: regression and numerical acceptance for deterministic-rate, single-asset
one-factor Bergomi LSV. This adds tests, not a change to the particle algorithm.
Source baseline: main `935cdd8a339ed4437b0cd6c7ba77edf1c380f633`.

## What the tests establish

The [test target](../../crates/pricing/tests/bergomi_calibration_quality.rs) exercises
the actual Rust `MarketIvSurface`, Dupire grid builder, particle calibrator and
independent LSV path engine. It converts independently repriced OTM options back
to Black IV, rather than checking only ATM prices or sensitivity consistency.
The test-only IV solver uses the existing analytical Black oracle and bisection;
invalid/no-time-value prices fail instead of being clipped or dropped.

| Input | Fixed acceptance setting |
| --- | --- |
| Coordinates | Initial martingale f = 100; zero rates and dividends |
| Bergomi | Mean reversion 2; volatility-multiplier parameter nu = 0.7; rho = -0.5 |
| Quote maturities and evaluation maturities | 0.25, 0.5, 1 year |
| Quote log strikes | -1, -0.5, -0.25, 0, 0.25, 0.5, 1 |
| Evaluation log strikes | -0.2, -0.1, 0, 0.1, 0.2; includes off-quote strikes |
| Flat target | IV 20% everywhere |
| Skew/term target | Quote IV = sqrt((0.04 + 0.004 T)(1 - 0.25 k + 0.1 k^2)); the production interpolator defines between-quote IV |
| Dupire/leverage grid | 128 equal time intervals; 160 log-strike intervals on [-1,1] |
| Particles | 32,768; calibration seeds 42, 137, 711, 2027 |
| Kernel | Log bandwidth 0.02; minimum ESS 20; no reverse trace |
| Independent pricing per calibration | 8 RQMC scrambles x 8,192 points x 2 antithetic legs |
| Pricing seed | Calibration seed XOR 0xd1b54a32d192ed03; never the particle stream |
| Pricing execution | Brownian bridge on both independent factors; dedicated 2-worker pool, fixed block size 256 |

Each calibration is shared across all 15 quotes. Each complete outer run has
independent particles and independent pricing scrambles. No calibration
particle payoff is used as an independent pricing sample. The target must have
zero Dupire floor/cap repairs.

## Error attribution and uncertainty

The same spot shocks also price a pure LV control on the same effective grid.
We report exact LSV-IV minus target-IV and LV-IV minus target-IV. Their paired
price difference, divided by target Black vega, is reported separately as a
first-order IV-equivalent residual, not mislabeled as an exact IV difference.
LV error includes target interpolation, Dupire/grid/time approximations and
pricing noise. LSV minus LV isolates the additional LSV numerical error but
does not prove that all of it is kernel bias.

Within one calibration, the independent observations for pricing uncertainty
are the 8 scramble means, not points or antithetic legs. Across the 4 complete
runs, the sample variance of mean prices includes both calibration variability
and pricing noise. The ensemble price SE is the larger of:

- the SE calculated from the 4 complete-run means;
- sqrt(sum of conditional pricing SE squared) / 4.

We do not add the two variances and double-count pricing noise. Seed-to-seed SD
is also printed and explicitly includes conditional pricing noise. Price SE is
converted to IV units by target vega (a delta-method approximation). Four seeds
are a regression panel, not a claim of well-estimated confidence-interval
coverage or a measurement of systematic bias.

## Fixed pass/fail budgets

One IV bp means 0.0001 absolute IV (20.00% to 20.01%). Budgets are fixed in the
test, apply to every active quote, and are not loosened by a large reported SE.

| Metric | Limit |
| --- | --- |
| Ensemble maximum absolute LSV IV error | 20 bp |
| Ensemble LSV IV RMSE | 10 bp |
| Maximum absolute LV-control IV error | 10 bp |
| LV-control IV RMSE | 5 bp |
| Ensemble total SE in IV units | 5 bp at every quote |
| Ensemble conditional-pricing-only SE in IV units | 3 bp at every quote |
| Worst individual calibration-seed absolute IV error | 35 bp |
| Absolute paired LSV-LV price residual / target vega | 15 bp |
| Fallbacks at evaluation quote interpolation neighbors | Zero |
| Independent pricing martingale mean error | At most 4 SE + 0.02 in f units |
| Martingale mean SE | At most 0.02 in f units |

Global row fallback counts and particle means are diagnostics, not a demand
that remote tails with virtually no particles have zero fallback. The quote
support assertion checks evaluation maturities; it does not establish support
along every path or at every earlier time. Low-vega quotes are not silently
skipped: the fixed panel must have target vega greater than 1 in price units per
unit absolute IV.

Helper unit tests validate IV inversion and failure cases, fixture repair
absence, ensemble variance accounting, and rejection of injected bias, excessive
noise, fallback and nonfinite metrics. The old loose ATM repricing and AAD tests
remain unchanged.

## Running and retained diagnostics

Fast helpers run as part of normal `cargo test`. Run both numerical gates with:

```bash
cargo test --locked --release -p pricing --test bergomi_calibration_quality iv_round_trip_ -- --ignored --nocapture --test-threads=1
```

CI explicitly runs this command on Linux, macOS and Windows. Output is retained
as `bergomi-calibration-quality-<os>`, including failed gates. `BERGOMI_RUN` and
`BERGOMI_QUALITY` lines contain JSON with configuration, seeds, per-quote errors,
conditional and total uncertainty, support counts and martingale diagnostics.

For a one-at-a-time sensitivity report:

```bash
cargo test --locked --release -p pricing --test bergomi_calibration_quality one_factor_bergomi_refinement_report -- --ignored --nocapture
```

This starts from a reference setting identical to acceptance except bandwidth
0.035. It then uses 4,096 particles, bandwidth 0.12, 32 time intervals, or 40
spatial intervals, changing one setting from that reference at a time. The last
run narrows bandwidth to the acceptance value 0.02. Reference and coarse
configurations are diagnostic only; the final fine configuration must pass the
same gate. There is no invalid assertion that one noisy realization
must improve monotonically when a setting changes. Changing the time grid also
changes random-coordinate mapping; this is an ensemble comparison.

## Limits and execution evidence

Passing these two surfaces does not certify stressed vol-of-vol, long maturities,
low-vega wings, all target interpolators, stochastic rates, dividends, multiple
assets, local correlation or two-factor/rough Bergomi. It is a defined quality
regression gate, not general LSV production acceptance. No fixture is a market
data calibration or a pure Bergomi parameter fit.

### Local Linux results

Rust 1.98.1, release profile, on the baseline above plus this test-only change:

[Retained per-quote results](bergomi-calibration-quality-results.json) include
the exact test-source SHA-256, seeds, model parameters and uncertainty metrics.

| Surface, bandwidth 0.02 | Maximum absolute IV error | IV RMSE | Maximum LV-control error | Maximum total SE |
| --- | --- | --- | --- | --- |
| Flat 20% | 10.357 bp | 3.279 bp | 1.036 bp | 2.254 bp |
| Skew / term structure | 7.793 bp | 2.831 bp | 2.469 bp | 2.263 bp |

Both numerical gates and all four helper tests passed in one run (37.20 seconds
after compilation). Existing `pricing` library unit tests (371), `mc_lsv` (6)
and `lsv` (3) passed. Clippy for all `pricing` targets with warnings denied,
formatting, dependency direction, Markdown links, JSON schemas and the existing
LV/path-dependence/early-exercise reference checks passed. This local evidence
does not stand in for the PR's macOS, Windows or Python-wheel CI results.

The flat-surface sensitivity study keeps all other reference settings fixed:

| Configuration | Maximum absolute IV error | IV RMSE | Outcome under the same budgets |
| --- | --- | --- | --- |
| Reference: N=32,768, bandwidth 0.035, 128 time / 160 space intervals | 17.429 bp | 5.349 bp | Paired LSV-LV residual fails (17.150 bp) |
| Bandwidth 0.02 | 10.357 bp | 3.279 bp | Pass |
| N=4,096, bandwidth 0.035 | 30.406 bp | 9.992 bp | Error, total SE and worst-seed budgets fail |
| Bandwidth 0.12 | 89.550 bp | 33.818 bp | Error and RMSE budgets fail |
| 32 time intervals, bandwidth 0.035 | 19.077 bp | 6.270 bp | Paired residual fails |
| 40 spatial intervals, bandwidth 0.035 | 18.875 bp | 5.975 bp | Paired residual fails |

The first trial used 2,048 pricing points per scramble; its maximum pricing SE
was approximately 4.44 bp, above the already-fixed 3 bp budget. Pricing points
were increased to 8,192, **without increasing any error budget**. Bandwidth
0.035 still failed the paired residual gate at 3 months, k=-0.2 (flat 17.150 bp;
skew 15.970 bp). The acceptance configuration therefore uses the tested narrower
bandwidth 0.02; the failing reference/coarse configurations remain reproducible
through the manual report. No production defaults or calibration code changed.

These controlled comparisons demonstrate sensitivity to the numerical settings
in this fixture. They do not establish the cause of an unprovided market-data
calibration failure or prove convergence over the full parameter space.
