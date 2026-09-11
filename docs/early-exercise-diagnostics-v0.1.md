# Early Exercise diagnostics catalogue v0.1

Status: candidate catalogue; Gate E8 is not accepted

Requirements: `requirements-v1.0.md`

Roadmap: `early-exercise-roadmap-v0.1.md`

This catalogue records the stable diagnostics for American Vanilla valuation,
LSM policy training, independent out-of-sample valuation, and fixed-policy
sensitivities.

## Policy identity and random domains

Every successful American result attaches `early_exercise` diagnostics:

| Field | Meaning |
|---|---|
| `policy_fingerprint` | Digest of the product, training configuration, dates, basis, fitted models, tolerances, and numerical policy |
| `training_random_domain` | `LsmTrain` for Pseudo-MC or `RqmcScramble` for randomized Sobol training |
| `valuation_random_domain` | `Valuation` for Pseudo-MC or `RqmcScramble` for randomized Sobol valuation |
| `training_direction_checksum`, `valuation_direction_checksum` | Optional Sobol direction-table checksums |
| `training_scramble_checksum`, `valuation_scramble_checksum` | Optional independently seeded RQMC scramble-plan checksums |
| `training_sampling_units`, `valuation_sampling_units` | Independent Pseudo-MC units or RQMC scramble counts used by each phase |
| `training_trajectories`, `valuation_trajectories` | Path counts after antithetic expansion and, for RQMC, point-by-scramble expansion |
| `in_sample_value` | Discounted value produced on the policy-training paths, retained separately from the reported out-of-sample estimate |

Training and valuation seeds, counts, and domains participate in policy or
request identity. Worker count does not alter the deterministic path mapping or
same-platform result bits.

## Exercise outcomes

| Field | Meaning |
|---|---|
| `exercise_dates` | Strictly increasing permissible dates, including expiry |
| `exercise_counts` | Number of valuation trajectories stopping at each date |
| `exercise_probabilities` | Counts divided by the valuation trajectory count |
| `stopping_indices` | Date index selected for every deterministic valuation-path identity |
| `dividend_collisions` | Whether each exercise date shares a normalized event time with a dividend |

Counts sum to `valuation_trajectories`; every stopping index addresses
`exercise_dates`. Exercise/dividend collisions observe post-dividend Spot.
Expiry is intrinsic value and is not regressed.

## Regression diagnostics

The result retains the canonical polynomial basis (`feature_count`, maximum
total degree, and every exponent vector), ITM absolute tolerance, absolute and
relative CPQR rank tolerances, and the checked matrix-element limit.

Each non-terminal exercise date has one diagnostics record:

| Field | Meaning |
|---|---|
| `candidate_rows` | Training trajectories considered at the date |
| `itm_rows` | Rows whose immediate value is strictly above the ITM tolerance |
| `feature_count` | State features supplied to the regression |
| `warnings` | Ordered zero-ITM, inactive-feature, and rank-excluded-column events |

Each date also retains either a `Regression` decision model or an explicit
`ContinueAll` model. Regression models record feature means, population
variances, scales, zero-scale thresholds, active and pre-excluded columns,
pivot order, diagonal magnitudes, rank threshold, numerical rank,
rank-excluded columns, original-basis-order coefficients, and residual sum of
squares. `ContinueAll` currently records `ZeroItmTrainingPaths`.

## Fixed-policy risk labels

Requested American sensitivities attach all of the following:

| Field | Stable value |
|---|---|
| `exercise_strategy` | `FixedExerciseStrategy` |
| `stopping_indices` | `FrozenStoppingIndices` |
| `exercise_policy_fingerprint` | The base result's policy fingerprint |

Delta and Vega use reverse AAD through the cash flow selected by the base
stopping index. Gamma is a central bump of fixed-index AAD Delta. CRN
validation uses the same fitted policy, valuation coordinates, and stopping
index and does not re-run an exercise comparison. These values are not labelled
as derivatives of a reoptimized exercise boundary.

## Errors and warnings

Configuration and numerical failures terminate evaluation without a partial
result. Important typed failures include an absent or unexpected LSM
configuration, overlapping training and valuation path sets, invalid exercise
schedules, non-finite or negative immediate values, malformed feature or cash
flow matrices, invalid tolerances, unsafe matrix dimensions, allocation
failure, non-finite regression intermediates, and inconsistent replay state.

Valid degenerate fits remain successful and produce structured warnings:

- `ZeroItmTrainingPaths` stores a date-local `ContinueAll` model;
- `InactiveFeature` identifies a feature whose population scale is at or below
  its deterministic zero-scale threshold;
- `RankExcludedBasisColumn` identifies a pivoted basis column excluded by the
  declared CPQR rank threshold.

The v3 result JSON preserves the complete diagnostics and rejects unknown,
malformed, non-finite, or cross-field-inconsistent replay state.
