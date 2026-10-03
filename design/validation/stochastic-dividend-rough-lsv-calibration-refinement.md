# Rough residual-LSV recalibration and particle replication

The [fixed-surface panel](stochastic-dividend-rough-lsv-refinement.md) isolates
valuation-grid sensitivity. This panel calls the production rough-LSV factory
again for every grid, particle count and calibration seed. Its independent
outer replicates measure the variability of those recalibrated comparisons.
No production pricing, calibration randomness or reported API SE is changed.

## Design and predeclared numerical gates

The market, nonflat residual local-variance target and model parameters are the
same as in the fixed-surface panel: H=0.1/0.3, eta=0.6, dividend mean reversion
0.7, linkage 0.6, dividend volatility 0.35 and correlations -0.25/-0.4/0.15.
Cash remains before, at and after expiry; the payoff is the post-cash terminal
call and its scale-invariant physical-Spot Delta.

- Calibration/valuation grids: 64 and 128 uniform steps, both preserving all
  contractual dates and target knots.
- Calibration particle counts: 256, 1,024 and 4,096, with bandwidth 0.35,
  minimum effective samples 5 and no retained reverse trace.
- Thirty-two independent calibration seeds: 42, 193, 617, 877, 1973, 4179,
  6127, 9109, followed by `100003 + 997*i` for i=0,...,23. Every replicate
  compiles all six grid/particle combinations anew. The count increase from
  the initial eight is explained below.
- Each replicate has 16,384 independent antithetic valuation units. Its
  valuation seed is `900001 + 97*replicate_index`, starting at index zero.

The existing common-Brownian coupling pairs the valuation grids. Within a
valuation grid, every particle-count scenario receives the same valuation
normals. Calibration uses its unchanged production random coordinates in the
separate `LsvCalibration` domain. At a fixed grid, increasing particle count
preserves the earlier particles' random inputs. Changing the grid changes the
factor-major coordinate layout: the same calibration seed does **not** imply
common Brownian calibration paths across grids. Outer replication accounts
for that variability without mislabelling it as valuation-only noise.

The panel checks that surface sizes follow the requested grids and that
nonzero-eta leverage values actually change across counts and calibration
seeds. Thus it cannot silently substitute the earlier frozen-surface panel.

## Estimates and acceptance

Each replicate reports five paired contrasts, for both price and Delta:

1. 64 minus 128 steps at each of the three particle counts.
2. 256 minus 1,024 particles on the 128-step grid.
3. 1,024 minus 4,096 particles on the 128-step grid.

Let `g_b` be the mean paired gap in replicate b and `s_b` its conditional
valuation SE. Across B independent replicates the reported mean is the
average of `g_b`, and:

- `total_replication_se = sample_sd(g_b) / sqrt(B)` includes both calibration
  and valuation randomness in the replicate means.
- `conditional_valuation_se = sqrt(sum(s_b^2)) / B` reports only the inner
  valuation contribution to the average, conditional on all calibrations.

These quantities are not added: doing so would count valuation variance
twice. No subtraction/clipping is used to claim a pure calibration variance.
The outer SE is for the average comparison across calibrations, not the
uncertainty of one production result calibrated with one seed.

Predeclared gates apply to the 64/128-step gap at 4,096 particles and the
1,024/4,096-particle gap at 128 steps, separately for each H:

- `abs(mean gap) + 4*total_replication_se < 0.05` price units and `< 0.005`
  Delta units.
- Total replication SE below 0.01 for price and 0.001 for Delta.

Earlier particle counts are diagnostic. The same numerical limits in the
fixed-surface panel remain unchanged; this panel uses the broader outer SE.
All 640 replicate rows and 20 ensemble rows are emitted before checking the
final gates. The conditional and total errors must both be nonzero in these
nontrivial cases. No monotonicity, convergence order or confidence-coverage
claim is inferred from this small finite panel.

## Fast controls and execution

Two fast tests check the reporting and comparison machinery:

- An elementary three-replicate example distinguishes the total SE from the
  conditional valuation SE and from the incorrect sum of both variances.
- With eta=0, dividend mean reversion zero and flat target variance 0.04,
  every recalibrated leverage is constant. Across two grids, two particle
  counts and two seeds, paired price/Delta differences vanish within 3e-11.
  At least one sample must have a nonzero price and Delta.

Run all three checks with:

```sh
cargo test --locked --release -p pricing --lib lsv_calibration_refinement_tests -- --include-ignored --nocapture
```

The ensemble panel is ignored by normal debug discovery and runs explicitly
in the three-OS release CI, retaining
`stochastic-dividend-rough-lsv-calibration-refinement.log`. Results must be
reported against the tested revision; adding the panel is not evidence that
its numerical gates passed.

## Initial calibration-precision observation

The first release panel used eight calibration replicates and failed the
1,024/4,096-particle **price** gates for both H values:

| H | Mean gap | Total replication SE | Conditional valuation SE | abs(gap)+4*total SE |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 0.018692475 | 0.010370788 | 0.000355032 | 0.060175625 |
| 0.3 | 0.022647388 | 0.011934411 | 0.000334663 | 0.070385031 |

Both exceed the fixed 0.05 price budget and 0.01 total-SE cap. This was not
a pass. The much smaller conditional valuation error shows why more inner
valuation paths would not address the dominant uncertainty. Twenty-four
independent calibration replicates were added, retaining the original eight
calibration and valuation seed pairs as a prefix. All grids, particle counts,
inner valuation counts, numerical budgets and SE limits remain unchanged.

## Limits

The highest grid and particle count are finite references. Small differences
do not bound the remaining time bias, finite-particle bias, fixed-bandwidth
kernel bias or model error. Calibration bandwidth/fallback choices are held
fixed. This panel concerns European price/Delta at the stated inputs, not
all rough-LSV risks, path-dependent accuracy or calibration-ensemble uncertainty
in the public API. Common-Brownian coupling of the calibration trajectories
across grids remains a possible variance-reduction extension.
