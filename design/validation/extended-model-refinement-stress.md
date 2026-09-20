# Extended-model refinement and stress

This extends the [price accuracy panel](extended-model-accuracy.md) from base
`92155aa525d3b21e31cbaec987239f0406bc0d77`. It adds independent resolution
comparisons for 1F Bergomi, Hull–White hybrids and joint local correlation,
and stronger-smile/high-vol-of-vol/long-maturity/wing-strike price cases.
Production algorithms, APIs, fingerprints, wire formats and RNG layouts are
unchanged. The existing absolute price and sampling-error budgets are preserved.

**Merge blocker:** 73 of 74 measured panels pass. The three-year joint 2F
basket at nu=0.6 passes its price-error limits but fails the unchanged
ensemble and conditional pricing SE limits. Its test remains a failing
acceptance gate; this change is not ready to merge. The constituent panel
passes at the same parameters. Numerical-resolution/calibration-domain
investigation is still required before this basket case can be accepted.

## Executable contract

```sh
cargo test --locked --release -p pricing --test extended_model_acceptance -- --ignored --nocapture --test-threads=1
```

The existing three-OS CI command automatically includes all new ignored tests
and retains the numerical JSON lines in `extended-model-accuracy.log`. Nine
fast tests run normally, including infeasible-skew and inserted-HW-date
contracts. The 27 heavy tests have separate entries by model/study (and basket/constituent),
so a failure does not hide the other model cases. The
[single-asset](../../crates/pricing/tests/cases/extended_single_asset.rs) and
[multi-asset](../../crates/pricing/tests/cases/extended_multi_asset.rs) helpers
use public compiled plans and independent Black price/IV references.

Every panel contains three strikes and three independent calibration/pricing
seed pairs, shared across strikes. Missing, duplicate or reordered strikes,
invalid IV inversion inputs, nonfinite errors and inadequate Black vega fail.
The original near-money basket/constituent cases are now the base cases of
the multi-asset refinement test, avoiding duplicate evaluations while retaining
the same prices, seeds and budgets.

## Refinement cases

All cases have expiry 2027-01-01 and log-forward strikes -0.12, 0 and 0.12.

| Family | Cases | Price panels, including base |
| --- | --- | ---: |
| Deterministic LSV | 1F, 2F, rough; original skew, no discrete dividends | 12 |
| HW/LSV | 1F, 2F, rough; original skew, cash 2 plus 3% proportional dividend at expiry | 12 |
| Local correlation | BS/BS, 2F-LSV/BS, rough-LSV/HW/BS; basket and first constituent | 24 |

Base resolution is 16,384 particles, 128 requested time steps, bandwidth 0.05,
4,096 RQMC points per scramble and eight scrambles. Change one setting at a
time: 32,768 particles, 256 requested steps, or bandwidth 0.035. The three
refinements use seed offsets 10,000, 20,000 and 30,000, respectively. This
retains the original deterministic 2F/rough study and adds the missing families.
The public plan may insert curve, target or event dates; requested steps are
not an assertion that every compiled timeline has that many intervals.

Every resolution must independently pass maximum IV error 15 bp, panel RMSE
8 bp, worst seed error 30 bp, ensemble SE 4 bp and conditional pricing SE
3 bp. In addition, every quote's change from base must be at most
`5 bp + 3*hypot(SE_base, SE_refined)`. Calibration and pricing streams are
disjoint between resolutions. These checks demonstrate stability at specified
resolutions, not monotonic Monte Carlo error or an asymptotic convergence rate.

## Stress cases

Expiry is 2029-01-01: 1096/365 years from valuation on 2026-01-01. Log-forward
strikes are -0.36, 0 and 0.36, compared with the original ±0.12 wings. The
common settings are 192 requested steps, bandwidth 0.035, eight scrambles
and seed offset 40,000. BS local correlation uses 65,536 particles and 8,192
points per scramble; single-asset LSV/HW uses 131,072 particles and 16,384
points; joint 2F uses 131,072 particles and 32,768 points; joint rough-HW
uses 65,536 particles and 16,384 points. The larger
populations address measured calibration dispersion and the additional RQMC
points address measured conditional pricing noise. Pricing seeds
remain calibration seeds XOR `0xd1b54a32d192ed03`. The same 15/8/30/4/3 bp
budgets apply; uncertainty does not enlarge the absolute error budget.

| Family | Vol-of-vol change | Target / dividend |
| --- | --- | --- |
| 1F, deterministic and HW | nu 0.7 → 1.2 | Stronger marginal skew; no discrete dividends |
| 2F, deterministic and HW | nu 0.5 → 0.9 | Same stronger marginal skew; no discrete dividends |
| Rough, deterministic and HW | eta 0.6 → 1.0 | Same stronger marginal skew; no discrete dividends |
| BS/BS local correlation | No stochastic volatility factor | Skewed basket; flat 28% constituent |
| 2F-LSV/BS local correlation | nu 0.4 → 0.6 | Skewed basket; flat 28% constituent |
| Rough-LSV/HW/BS local correlation | eta 0.5 → 0.9 | Skewed basket; flat 28% constituent |

Other model/correlation parameters retain the original panel values. The HW
rate-volatility knot is a fixed fraction of the longer maturity. The marginal
stress IV is `sqrt((0.0625 + 0.002*T)*(1 - 0.6*x + 0.15*x*x))`; the basket
stress IV is `0.235*sqrt(1 - 0.15*x + 0.025*x*x)`. The stronger basket
formula `0.235*sqrt(1 - 0.3*x + 0.05*x*x)` is retained as an infeasible-target
contract and as an initial failed price measurement below. Market surfaces include the
stress maturity explicitly. The joint rough/HW constituent retains flat 28%
market quotes so the public compiler can regenerate paired variance/density
targets at inserted common refinement dates; an explicit flat grid alone does not
cover those additional dates. A fast compilation test exercises this contract.
Calibration log nodes span -0.8 to 0.8 in steps of 0.02; input market
quotes span -1.2 to 1.2. No spatial-domain refinement is claimed: simulated
paths can leave the calibration grid, so the high-nu uncertainty cannot be
attributed to sample count alone. Fast tests bound the natural-spline/oracle
difference at the actual wing quotes by 0.1 IV bp and ensure vega exceeds 5.

Each multi-asset configuration prices both the basket and its first
constituent through the same joint calibration. Equal carry and equal spots
make the physical basket a deterministic multiple of the normalized target.
The original and refinement cases retain their affine cash-price oracle.
Stress cases exclude discrete dividends: the stronger tails can breach the
positive post-dividend spot domain of a fixed cash payment. Those paths are
rejected, never clipped or dropped. No intermediate stochastic-cash
market-IV conversion is assumed.

## Calibration diagnostics and feasibility

`EXTENDED_CALIBRATION` records the case, resolution, seed, strike, grid size
and fallback counts. Deterministic LSV also reports the two terminal
interpolation nodes' ESS, fallback status and donor index, and requires no
fallback and ESS at least 100 at both. HW exposes row-level diagnostics only:
the log records minimum supported-node ESS, terminal fallback count and
maximum absolute rate correction. It does not infer quote-level support from
those row minima. Finite diagnostics and positive finite conditional second
moments are checked.

`EXTENDED_LOCAL_CORRELATION` includes global fallback, projection and
unidentifiability counts, plus terminal quote-node ESS, target/attained
variance and raw mixing. Both terminal nodes must have ESS at least 100 and
no fallback, projection or unidentifiability. This is a local support check;
prices depend on the whole simulated path, so global projections remain
material even when a requested quote passes its price budget.

The separately retained stronger basket skew is infeasible for the BS/BS
inputs in the far left of the grid. At time zero, identical 28%-volatility assets with
weights 1/2 and correlation endpoints -0.3/0.95 have exact variance bounds
`[0.02744, 0.07644]`. Fourteen initial target nodes, x=-0.80 through -0.54,
exceed the upper bound. A fast test independently checks that interval,
the projection residual at every initial node, and rejection under
`LocalCorrelationFeasibility::Reject`. It also checks target nodes above
0.0784, the universal BS/BS basket-variance ceiling: positive asset-value
weights sum to one, so even unequal composition and correlation 1 cannot
exceed the constituent variance. This second bound establishes infeasibility
beyond the compiler's equal-composition time-zero convention. The universal
instantaneous-variance bound is specific to BS/BS, not stochastic-volatility assets.
The price study uses the gentler basket skew and still records
all projections under `ProjectAndReport`. Passing selected quote prices is
not a claim that the entire basket smile is attainable. Increasing particle
count cannot remove this analytic time-zero constraint.

## Measurements

The initial stronger-basket trial used 65,536 particles and 8,192 points per
scramble. Its BS basket/constituent price errors were within budget (maximum
3.8372/1.0990 bp) despite remote projections. The 2F basket failed: maximum
error 17.1568 bp, RMSE 10.4267 bp, ensemble SE 7.8439 bp and conditional SE
4.7830 bp. It projected 8,009–8,361 of 25,110 calibration nodes, while terminal
quote-node ESS remained above 1,800 and those nodes were not projected.
These failures are not accepted by widening the 15/8/30/4/3 bp budgets.
The target feasibility case and the price study are distinct.

At the gentler basket skew, doubling particles to 131,072 and quadrupling
pricing points to 32,768 reduced the nu=0.8 basket's maximum price error to
6.0043 bp (RMSE 3.6069 bp), but did not pass the uncertainty gates: ensemble
SE reached 9.0619 bp and conditional SE 4.3040 bp. The first constituent
also failed its ensemble-SE gate (4.1977 bp) despite maximum price error
2.7270 bp. This parameter point remains unaccepted at the tested numerical
resolutions. The fixed-budget joint 2F stress uses nu=0.6 versus 0.4 in the
original panel, with the same long maturity and wing strikes. At nu=0.6,
the constituent passes (maximum IV error 3.4707 bp, ensemble SE 3.0794 bp),
but the basket still fails: maximum IV error 6.6614 bp and RMSE 4.1285 bp
are within budget, while ensemble SE reaches 8.9042 bp and conditional SE
4.1290 bp. Both ATM and right-wing quotes fail uncertainty gates. The
nu=0.6 basket and nu=0.8 cases therefore remain unaccepted. Lowering nu
did not establish adequate sampling precision, and the parameter is not
reduced further to obtain a passing regression. No spatial-domain or
bandwidth explanation is established by these measurements.

The [two-factor stress observations](extended-model-two-factor-stress.json)
preserve all five exploratory/final 2F panels, including per-seed prices and
conditional errors. Reproduce the current failing case with:

```sh
cargo test --locked --release -p pricing --test extended_model_acceptance multi_asset::two_factor_basket_stress_repricing -- --ignored --exact --nocapture
```

The initial single-asset cash stress also exposed domain rejections: a
nonpositive post-dividend spot for deterministic 1F and nonpositive final
states for HW with expiry cash. The deterministic 2F price errors were small
(maximum 1.5341 bp) but ensemble SE reached 5.0187 bp. The accepted study
isolates the continuous-equity stress and increases resolution. Fixed-cash
stress is not certified. A separate test-fixture correction pins the final
time-grid node exactly to the ACT/365F expiry; multiply/divide rounding had
made it one ULP too large at 1096/365 years with 192 steps.

On Linux, all 74 specified panels were measured: 73 passed and one failed
its sampling-error limits. The 222 quote points use 666 public-plan price
evaluations, with three independent seed pairs per panel shared across
strikes. Results below aggregate targeted runs; splitting test entrypoints
did not change the already-measured refinement inputs. Each metric is a
group maximum; RMSE is the largest three-strike panel RMSE.

| Group | Pass / panels | Max IV error, bp | Max panel RMSE, bp | Worst seed, bp | Max ensemble SE, bp | Max pricing SE, bp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original deterministic | 5 / 5 | 5.0423 | 3.4384 | 7.2464 | 2.4345 | 1.5830 |
| Original HW | 3 / 3 | 3.1695 | 2.1670 | 5.9626 | 2.2734 | 1.5450 |
| Gaussian reference | 6 / 6 | 0.0365 | 0.0232 | 0.0703 | 0.0526 | 0.0526 |
| Deterministic refinement | 12 / 12 | 5.1082 | 3.8515 | 7.5604 | 2.7273 | 2.0733 |
| HW refinement | 12 / 12 | 5.3528 | 4.0108 | 7.6226 | 2.5494 | 2.0752 |
| Local-correlation refinement | 24 / 24 | 6.1861 | 5.5323 | 11.0449 | 3.8495 | 2.7379 |
| Deterministic stress | 3 / 3 | 3.6507 | 2.1669 | 6.4523 | 2.8862 | 1.4801 |
| HW stress | 3 / 3 | 2.7103 | 1.7541 | 6.4461 | 2.4684 | 1.7486 |
| Local-correlation stress | 5 / 6 | 6.6614 | 4.1285 | 17.5291 | 8.9042 | 4.1290 |

All 108 refinement comparisons passed. The largest IV change was 8.3055 bp;
the largest ratio of observed change to its allowed bound was 0.71974.
The original Gaussian reference and carry regressions also passed.

Across the measured deterministic LSV cases, terminal quote-node ESS was
at least 1,246.22; the local-correlation minimum was 1,095.36. None of those
evaluation nodes used fallback, projection or an unidentifiable mixture.
The local-correlation grids nevertheless recorded up to 1,089 projected
nodes and up to 8,161 fallback nodes across the different configurations.
For HW, minimum supported-node row ESS was 20.0005 and terminal rows had
0–21 fallback nodes. Those row-level observations do not certify support
at a particular HW quote. In particular, supported terminal quote nodes
did not prevent the 2F basket sampling-error failure.

Local validation uses pinned Rust 1.98.1 on Linux, with the same temporary
[build workaround](extended-model-accuracy.md#linux-measurement) as the original
panel. The all-features workspace regression run passed 513 tests; the
no-default-features run passed 499. The nine fast acceptance tests and
all-target/all-feature Clippy passed. Formatting, fixture/schema checks and
Markdown-link checks also passed. These Linux results do not assert native
cross-platform replay acceptance; CI runs the normal release configuration
on Ubuntu, macOS and Windows.

## Remaining scope

These specified cases extend S3/H3 evidence without closing their broad
production acceptance. They do not establish convergence rates, arbitrary
stress-domain coverage, market-data fitting, every local-correlation mixture,
HW quote-node ESS, stochastic-cash physical-IV conversion, risk/AAD accuracy
or performance acceptance. Those require separate evidence.
