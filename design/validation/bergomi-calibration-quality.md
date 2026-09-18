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
| Dupire/leverage grid | 256 equal time intervals; 160 log-strike intervals on [-1,1] |
| Particles | 65,536; calibration seeds 42, 137, 711, 2027 |
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
| Fallbacks at evaluation quote interpolation neighbors | Zero, per quote and in total |
| Minimum kernel ESS at those neighbors | At least 200 |
| Interpolated target minus the fixture's closed form | 0.5 bp at every quote |
| Independent pricing martingale mean error | At most 4 SE + 0.02 in f units |
| Martingale mean SE | At most 0.02 in f units |

### Why the support budget is 200 effective samples

Requiring only that no node fell back is satisfied by an ESS just above the
calibrator's own minimum of 20, whose conditional second moment still carries
roughly 20% relative standard error, so absent support could reach a quote as
an IV error without any fallback being recorded. For the quartic kernel
w(u) = (1 - u^2)^2 on [-h, h], ESS = (sum w)^2 / (sum w^2) = N f(x) h times
(integral K)^2 / integral K^2, and (16/15)^2 / (256/315) = 1.4 exactly, so
ESS is approximately 1.4 N f(x) h. The check takes the smaller ESS of the two
grid nodes bracketing each evaluation strike, so the thinnest node is the outer
neighbour of T = 0.25, k = +0.2, at x = 0.2125, where the 20% lognormal density
gives approximately 687. The measured minimum over the four seeds is 670, so the
budget sits a factor 3.4 below the measured value at the tightest node and a
factor 10 above the calibrator floor: it is not a tripwire at 65,536 particles.
The bandwidth scan below measures 324 at bandwidth 0.01, which still passes. The
budget rejects 8,192 particles (approximately 86), and at the previous 32,768
particles it rejected bandwidth 0.01 (measured 148). The budget is a support
floor derived from the kernel, not a fitted number.

### Why 65,536 particles and 256 time steps

The previous acceptance setting of 32,768 particles, bandwidth 0.02 and 128
steps left a flat-surface maximum IV error of 10.357 bp, at T = 0.25, k = -0.2.
The paired LSV-LV residual there was 9.881 bp against an LV-control error of
0.583 bp, so almost all of it is LSV-specific. At T = 0.25 the residual has a
smile shape, positive in both wings and negative at the money: the calibrated
LSV marginal has fatter tails than the target. The error decomposition removes
one source at a time, on the flat surface:

| Particles | Bandwidth | Steps | Paired residual at T = 0.25, k = -0.2 | Paired residual at T = 0.25, k = 0 | Seed-to-seed SD at k = -0.2 |
| --- | --- | --- | --- | --- | --- |
| 32,768 | 0.02 | 128 | 9.88 bp | -1.50 bp | 3.23 bp |
| 131,072 | 0.02 | 128 | 7.43 bp | -1.57 bp | 1.61 bp |
| 131,072 | 0.01 | 128 | 4.60 bp | -1.31 bp | 1.60 bp |
| 524,288 | 0.01 | 128 | 4.63 bp | -1.50 bp | 0.83 bp |
| 131,072 | 0.01 | 512 | 2.60 bp | 0.11 bp | 4.00 bp |
| 65,536 (acceptance) | 0.02 | 256 | 5.66 bp | -0.55 bp | 0.72 bp |

Three sources account for most of the error, in roughly equal parts at the
previous setting; about 2.6 bp at k = -0.2 remains at the finest setting run and
is not attributed:

- **Leverage noise.** Squared leverage is target local variance divided by a
  kernel estimate of E[a^2 | x]. Estimation noise in that denominator inflates
  the effective local variance on average and adds randomness to each path's
  integrated variance, which fattens both tails. Quadrupling particles at fixed
  bandwidth removes 2.4 bp; at bandwidth 0.01 a further quadrupling changes
  nothing, so the particle-noise part is then exhausted.
- **Kernel bias.** The particle density falls steeply at |k| = 0.2, so the
  kernel window over-weights points nearer the money, where E[a^2 | x] differs.
  Halving the bandwidth at 131,072 particles removes 2.8 bp.
- **Time discretization.** Leverage and the variance multiplier are frozen over
  each Euler step. Quadrupling the steps removes 2.0 bp at k = -0.2 and moves the
  at-the-money residual from -1.3 bp to +0.1 bp, which no particle or bandwidth
  change affects. The finer-step run is noisier, with seed-to-seed SD rising
  from 1.60 to 4.00 bp at k = -0.2; the cause of that noise is not isolated.

The spatial grid is not a source: 640 log-strike intervals instead of 160 moved
the residual at k = -0.2 by 0.1 bp. Doubling both particles and steps addresses
the two sources that do not require a narrower kernel, keeps bandwidth 0.02 and
costs about 2.4 times the previous runtime. It brings the flat maximum IV error
to 6.208 bp and the paired residual to 5.663 bp. No calibration code, budget or
production default changed.

The calibrator's own `minimum_effective_samples` stays at 20. Raising it would
change which nodes fall back and therefore change the calibrated leverage
surface; this gate observes support, it does not alter the object being
measured.

### Why the target interpolation is checked separately

For the skew fixture the target IV is read back from the same production
interpolator that feeds Dupire, so a change of interpolation scheme moves the
target and the pricing reference together and cancels in every round-trip
metric. Both fixtures are exactly quadratic in log strike, and the natural
cubic spline on total variance reproduces a quadratic away from its
zero-second-derivative end conditions, leaving 0.043 bp at worst over the
evaluation panel. A separate fast test compares the interpolated target with
the fixture's closed form at every evaluation node under a 0.5 bp budget,
which catches a scheme, knot or units regression without absorbing any part of
the LSV budgets above. The closed form now has exactly one definition in the
test source and both the quotes and this check read it.

Global row fallback counts and particle means remain diagnostics, not a demand
that remote tails with virtually no particles have zero fallback. The quote
support assertion checks evaluation maturities; it does not establish support
along every path or at every earlier time — at early times a node at k = -0.2
holds almost no particles and is legitimately extrapolated. Low-vega quotes are
not silently skipped: the fixed panel must have target vega greater than 1 in
price units per unit absolute IV. That vega condition is also why the panel
stops at |k| = 0.2 while quotes extend to |k| = 1; the reported maxima are
maxima over that panel, not over the wings.

Helper unit tests validate IV inversion and failure cases, fixture repair
absence, target interpolation against the closed form, ensemble variance
accounting, and rejection of injected bias, excessive noise, thin kernel
support, per-quote fallback, drifted target interpolation and nonfinite
metrics. The old loose ATM repricing and AAD tests remain unchanged.

## Running and retained diagnostics

Fast helpers run as part of normal `cargo test`. Run both numerical gates with:

```bash
cargo test --locked --release -p pricing --test bergomi_calibration_quality iv_round_trip_ -- --ignored --nocapture --test-threads=1
```

CI explicitly runs this command on Linux, macOS and Windows. Output is retained
as `bergomi-calibration-quality-<os>`, including failed gates. `BERGOMI_RUN` and
`BERGOMI_QUALITY` lines contain JSON with configuration, seeds, per-quote errors,
conditional and total uncertainty, support counts and martingale diagnostics.

The retained results file below is rebuilt from that log rather than assembled
by hand:

```bash
python3 scripts/summarize_bergomi_quality.py --log bergomi-calibration-quality.log
```

It records the test-source SHA-256, the resolved baseline commit and the local
toolchain, and refuses to write a document it cannot stand behind: a log with no
runs, a surface reported twice, a null metric (which is how serde writes a
nonfinite value), or reports carrying different models, as happens if the
stressed vol-of-vol or refinement output is in the same log. Pass `--platform`
and `--compiler` when summarizing a log produced on another machine, since
recording the local toolchain for a foreign log would be a false provenance
claim, and `--check` to assert the file is already current for a log without
rewriting it.

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

For the error decomposition behind the acceptance particles and steps:

```bash
cargo test --locked --release -p pricing --test bergomi_calibration_quality one_factor_bergomi_error_decomposition -- --ignored --nocapture --test-threads=1
```

This runs the flat fixture at the six settings in the decomposition table above
and emits one `BERGOMI_ERROR_DECOMPOSITION` line per setting, with the summary,
the T = 0.25 paired residuals and their seed-to-seed SD. It takes about three
minutes and asserts nothing.

For the bandwidth evidence behind the acceptance setting:

```bash
cargo test --locked --release -p pricing --test bergomi_calibration_quality one_factor_bergomi_bandwidth_scan -- --ignored --nocapture --test-threads=1
```

This runs both fixtures at bandwidth 0.01, 0.015, 0.02, 0.025, 0.03 and 0.035
with every other acceptance setting held fixed, and emits one
`BERGOMI_BANDWIDTH_SCAN` summary line per configuration with the maximum and
RMS IV error, the LV-control error, the paired residual, both uncertainty
measures, the minimum ESS and the list of budget failures. Narrowing the
bandwidth trades kernel bias for kernel variance, so the scan brackets 0.02 on
both sides: it is what distinguishes an acceptance value near the minimum of
that trade-off from the first setting that happened to clear the budgets. The
scan asserts nothing and is outside the CI name filter.

Measured on macOS arm64 with the acceptance settings otherwise unchanged:

| Bandwidth | Flat paired residual | Skew paired residual | Flat maximum IV error | Skew maximum IV error | Minimum ESS | Budget failures |
| --- | --- | --- | --- | --- | --- | --- |
| 0.010 | 4.589 bp | 4.747 bp | 5.595 bp | 6.488 bp | 322 | None |
| 0.015 | 4.082 bp | 4.095 bp | 5.094 bp | 5.845 bp | 493 | None |
| 0.020 | 5.663 bp | 5.275 bp | 6.208 bp | 4.970 bp | 670 | None |
| 0.025 | 7.754 bp | 7.249 bp | 8.269 bp | 6.663 bp | 842 | None |
| 0.030 | 10.287 bp | 9.596 bp | 10.754 bp | 8.978 bp | 1021 | None |
| 0.035 | 13.186 bp | 12.321 bp | 13.585 bp | 11.654 bp | 1202 | None |

The error is smallest near 0.015 and rises again at 0.01, where kernel variance
starts to dominate. Bandwidth 0.02 is kept because it maximises the smallest
margin across the binding budgets: a factor 2.65 on the paired residual, 3.2 on
the maximum IV error and 3.35 on support. Bandwidth 0.015 would raise the
paired-residual margin to 3.66 but cut the support margin to 2.47.

At the previous 32,768 particles and 128 steps the same scan measured a paired
residual of 7.389, 8.372, 9.881, 11.894, 14.346 and 17.150 bp on the flat
surface for the six bandwidths, with 0.01 failing the support budget and 0.035
failing the paired-residual budget.

For the stressed vol-of-vol case:

```bash
cargo test --locked --release -p pricing --test bergomi_calibration_quality high_vol_of_vol_one_factor_bergomi_report -- --ignored --nocapture --test-threads=1
```

This repeats both fixtures at nu = 1.5 with mean reversion and correlation
unchanged, and emits `BERGOMI_HIGH_VOL_OF_VOL` summary lines. Kernel regression
degrades as vol of vol grows, so this is where the gate would bite; its budgets
have not been measured, so it reports and does not assert. A gate whose limits
were never measured is worse than an honest diagnostic.

## Limits and execution evidence

Passing these two surfaces does not certify stressed vol-of-vol, long maturities,
low-vega wings, all target interpolators, stochastic rates, dividends, multiple
assets, local correlation or two-factor/rough Bergomi. It is a defined quality
regression gate, not general LSV production acceptance. No fixture is a market
data calibration or a pure Bergomi parameter fit. The stressed vol-of-vol and
bandwidth reports above are diagnostics; neither is a certification, and no
model algorithm, bandwidth selection or quote repair happens in this test target.

### Local results

Rust 1.98.1, release profile, on the baseline above plus this test-only change:

[Retained per-quote results](bergomi-calibration-quality-results.json) include
the exact test-source SHA-256, seeds, model parameters, uncertainty metrics,
per-quote minimum ESS and target interpolation error. The file was rebuilt by
the summarizer from an aarch64-apple-darwin run. Every per-quote metric shared
with the earlier x86_64 Linux run agrees to within 3e-12 bp, and the martingale
diagnostics are bitwise identical. That comparison was made at the previous
setting; the file now records the 65,536-particle, 256-step acceptance run.

| Surface, bandwidth 0.02 | Maximum absolute IV error | IV RMSE | Maximum LV-control error | Maximum total SE | Minimum ESS | Maximum target interpolation error |
| --- | --- | --- | --- | --- | --- | --- |
| Flat 20% | 6.208 bp | 2.270 bp | 1.036 bp | 1.680 bp | 671 | 0.000 bp |
| Skew / term structure | 4.970 bp | 2.014 bp | 1.781 bp | 1.670 bp | 670 | 0.043 bp |

On macOS both numerical gates and all five helper tests passed, the gates in
44.12 seconds after compilation, with formatting and workspace Clippy clean.
The Linux run reported below used the previous 32,768 particles and 128 steps
and predates the support and interpolation budgets.

Both numerical gates and all four helper tests passed in one run (37.20 seconds
after compilation). Existing `pricing` library unit tests (371), `mc_lsv` (6)
and `lsv` (3) passed. Clippy for all `pricing` targets with warnings denied,
formatting, dependency direction, Markdown links, JSON schemas and the existing
LV/path-dependence/early-exercise reference checks passed. This local evidence
does not stand in for the PR's macOS, Windows or Python-wheel CI results.

The flat-surface sensitivity study keeps all other reference settings fixed:

| Configuration | Maximum absolute IV error | IV RMSE | Outcome under the same budgets |
| --- | --- | --- | --- |
| Reference: N=65,536, bandwidth 0.035, 256 time / 160 space intervals | 13.585 bp | 4.470 bp | Pass |
| Bandwidth 0.02 (acceptance) | 6.208 bp | 2.270 bp | Pass |
| N=4,096, bandwidth 0.035 | 7.926 bp | 2.979 bp | Support and total SE budgets fail |
| Bandwidth 0.12 | 87.637 bp | 33.341 bp | Error, RMSE, worst-seed and paired-residual budgets fail |
| 32 time intervals, bandwidth 0.035 | 19.086 bp | 6.327 bp | Paired residual fails (18.868 bp) |
| 40 spatial intervals, bandwidth 0.035 | 14.935 bp | 5.173 bp | Pass |

The 4,096-particle mean error is small only because its seed noise is large:
its total SE reaches 8.9 bp and its support falls below 200 at five nodes.

The first trial used 2,048 pricing points per scramble; its maximum pricing SE
was approximately 4.44 bp, above the already-fixed 3 bp budget. Pricing points
were increased to 8,192, **without increasing any error budget**. At 32,768
particles and 128 steps, bandwidth 0.035 then failed the paired residual gate at
3 months, k=-0.2 (flat 17.150 bp; skew 15.970 bp), and bandwidth 0.02 passed
with a flat maximum IV error of 10.357 bp. The error decomposition above then
moved particles and steps to their current values; the bandwidth scan confirms
0.02 as the setting with the largest smallest margin. No production defaults or
calibration code changed.

These controlled comparisons demonstrate sensitivity to the numerical settings
in this fixture. They do not establish the cause of an unprovided market-data
calibration failure or prove convergence over the full parameter space.
