# Early Exercise Numerical Contracts v0.1

Status: Candidate for roadmap Gate E0

Requirements: `requirements-v1.0.md` Section 5.3

Roadmap: `early-exercise-roadmap-v0.1.md`

## 1. Policy Identity

The policy identifier is `early_exercise_v1`. It fixes the exercise, feature,
regression, and policy-identity rules below. A formula, comparison, traversal,
or tie-break change requires a new policy identifier and new reference
fixtures.

All counts use checked integer arithmetic. Every configured tolerance is a
finite non-negative binary64 value. Numerical failures return typed errors;
the implementation never changes a tolerance, repairs a matrix, or selects an
alternate solver.

## 2. Exercise and ITM Decisions

At a non-terminal exercise date, a valuation path exercises exactly when

```text
immediate_value > continuation_value
```

The direct binary64 comparison is the complete policy. Equality, including
equal signed zeros, continues. NaN or infinity in either value is a numerical
error before comparison. At final expiry, non-negative intrinsic value is paid
directly and no continuation comparison is evaluated.

Training-path ITM membership is a separate decision:

```text
is_itm = immediate_value > itm_abs_tol
```

`itm_abs_tol` is explicit in payoff currency units. Equality is not ITM. A
date with no ITM training rows stores `ContinueAll(ZeroItmTrainingPaths)` and
does not perform scaling or regression.

## 3. Polynomial Basis Ordering

For `feature_count = n` and maximum total degree `d`, the basis contains every
exponent vector `e = [e_0, ..., e_(n-1)]` satisfying

```text
e_i >= 0
sum(e_i) <= d
```

Vectors are ordered first by ascending total degree. Within one degree, earlier
declared features have descending exponent priority. Enumeration is equivalent
to nested loops that choose `e_0`, then `e_1`, and so on from the remaining
degree down to zero. For two features and degree two, the order is:

```text
[0,0], [1,0], [0,1], [2,0], [1,1], [0,2]
```

The checked basis count is `binomial(n + d, d)`. Overflow or an effective
resource-limit breach is rejected before allocating exponent or matrix data.

For standardized features `z_i`, a monomial starts at exact `1.0` and, in
declared feature order, multiplies by `z_i` exactly `e_i` times. Production
evaluation does not call a platform `pow` function, reassociate factors, or
cache algebraically equivalent products across distinct basis columns.

## 4. Training Feature Scaling

For each non-terminal date, rows are visited in ascending deterministic
training-path identity. For every raw feature `x`, the arithmetic mean is
computed first, followed by a second pass over squared deviations:

```text
mu = sum(x_p) / n
variance = sum((x_p - mu) * (x_p - mu)) / n
sigma = sqrt(variance)
```

Both sums use the library's versioned Neumaier accumulator in row order. Each
squared deviation is formed by one subtraction and one multiplication. A
negative variance is a numerical error; there is no clamp to zero.

Define

```text
feature_scale = max(1, max_p(abs(x_p)))
zero_scale_threshold = 64 * EPSILON * feature_scale
```

where `EPSILON` is binary64 `2^-52`, and evaluate the products in the written
order. A feature is inactive when `sigma <= zero_scale_threshold`. Every
non-constant monomial with a positive exponent for an inactive feature is
pre-excluded from QR, receives an exact zero coefficient, and is reported in
original basis order. Active features use `(x - mu) / sigma`; the constant
monomial remains exact one and is never centered or scaled.

The fitted means, scales, inactive-feature indices, and threshold bit patterns
are immutable policy data. Valuation paths reuse them without recomputation.

## 5. Column-Pivoted Householder QR

The regression matrix contains ITM rows in ascending training-path identity
and non-pre-excluded basis columns in original basis order. The target is the
discounted realized continuation cash flow from the already fitted later-date
policy.

The solver is scalar column-pivoted Householder QR, policy
`cpqr_householder_v1`:

1. At step `k`, recompute every remaining column's squared norm over rows
   `k..m` from its current matrix entries using a Neumaier accumulator.
2. Select the greatest finite squared norm. Equal binary64 values select the
   smaller original basis-column index.
3. Swap the selected column into `k`, retaining the original-column
   permutation.
4. Compute the Householder norm with the scaled sum-of-squares algorithm in
   ascending row order. Set `alpha = -norm` when the leading value is
   non-negative, otherwise `alpha = norm`.
5. Store the reflector with implicit leading value one and
   `tau = (alpha - leading) / alpha`. Apply it to remaining columns in
   increasing current-column order and to the target last. Dot products use
   the versioned Neumaier accumulator in ascending row order. Row updates use
   separate multiplication and subtraction operations in ascending row order.
6. Recompute norms from the transformed matrix at the next step. No downdated
   norm estimate is reused.

A zero Householder norm stores exact zero diagonal and `tau = 0`. The solver
does not invoke SVD, Ridge, normal equations, or a platform BLAS/LAPACK call.

## 6. Rank, Solve, and Residuals

Let `r_j = abs(R[j,j])` in pivot order. The effective rank threshold is

```text
rank_threshold = max(abs_rank_tol, rel_rank_tol * r_0)
```

with `r_0 = 0` for an empty QR matrix. The numerical rank is the longest prefix
whose diagonal values satisfy `r_j > rank_threshold`. Equality is excluded.
Once one pivot fails, it and every later pivot are excluded even if a later
stored diagonal would pass independently.

Back substitution visits retained pivot positions in descending order. Each
row's upper-triangular dot product uses the versioned Neumaier accumulator over
columns in increasing pivot position.
Division by a retained diagonal must produce a finite value. Coefficients are
then mapped to original basis order; pre-excluded and rank-excluded columns are
exact positive zero.

Predictions are reevaluated from the original design rows and original-order
coefficients. The residual sum of squares is accumulated in training-row order
with the versioned Neumaier accumulator. Diagnostics retain:

- candidate and ITM row counts;
- original basis size and active QR column count;
- feature means, scales, inactive indices, and scale thresholds;
- original-column pivot order;
- absolute diagonal values and effective rank threshold;
- numerical rank and pre/rank-excluded original column indices;
- original-order coefficients and residual sum of squares;
- explicit warnings for zero-scale features, rank exclusion, and zero ITM.

## 7. Policy and Path Identity

An exercise policy stores one date-local `Regression` or `ContinueAll` model
for every non-terminal exercise date in ascending date order. Its canonical
fingerprint begins with domain tag `pricing/exercise-policy`, the
`early_exercise_v1` and `cpqr_householder_v1` identifiers, and then includes:

- product and normalized exercise-schedule fingerprints;
- strict comparison and ITM policies;
- training engine, random domain, seeds, and sample counts;
- basis specification and original exponent vectors;
- every date-local scaling, QR, coefficient, exclusion, warning, and
  continuation model field in schema order.

Every finite scalar uses its exact big-endian binary64 bit pattern. The policy
fingerprint excludes valuation sample count, valuation seed, worker count, and
realized valuation stopping indices because those apply a policy rather than
define it.

Valuation path identity is `(replicate, sampling_unit, antithetic_lane)` under
the existing engine ABI. The base valuation stores one stopping index for each
identity. AAD and validation bumps consume that immutable table and do not
evaluate exercise comparisons again.

The canonical policy byte stream uses unsigned big-endian integers. Variable
byte strings and arrays carry a `u64` element-count prefix; native `usize`
values are converted to `u64` before hashing. A date is encoded as big-endian
`u16` year followed by `u8` month and day. Fixed 32-byte product and training
configuration fingerprints are included raw. Finite `f64` values use their
big-endian `to_bits()` representation, booleans use `0` or `1`, and tagged
variants use the documented zero-based declaration order. The stream includes
the fixed `LsmTrain = 1` random-domain identifier, configured resource limits,
date-local diagnostics, and warnings in deterministic stored order. BLAKE3-256
hashes the stream, and the displayed form is lowercase
`blake3-256:<64 hexadecimal digits>`.

## 8. Reference Artifact

`../fixtures/early-exercise/reference-cases-v0.1.json` freezes decision, basis,
scaling, pivot, rank, coefficient, and residual cases. The independent
`../scripts/check_early_exercise_reference_fixture.py` evaluator uses Decimal
arithmetic and does not import production Rust or Python bindings.

Decimal fixtures validate the mathematical policy at 80-digit precision.
Production binary64 tests additionally freeze operation-order bit patterns and
compare finite differences using separately declared tolerances.
