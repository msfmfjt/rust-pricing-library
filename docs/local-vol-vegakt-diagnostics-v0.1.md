# Local Volatility and VegaKT diagnostics catalogue v0.1

Status: accepted catalogue for the Local Volatility/VegaKT slice

This catalogue covers diagnostic and warning surfaces introduced by the
Local Volatility/VegaKT roadmap. It complements the frozen European
Black-Scholes catalogue and records the stable names that tests and replay
fixtures may rely on.

## Surface and Local-variance diagnostics

Local-variance grids store explicit time nodes, log-forward-moneyness nodes,
row-major values, Floor, Cap, and deterministic repair records. Repairs are
recorded after a valid implied-volatility surface evaluation; surface invariant
failures remain typed errors and are not repaired.

Repair records preserve the time index, log-moneyness index, coordinates,
original binary64 bits, applied value, policy label, and row-major occurrence
index.

| Stable code | Condition | Applied value |
|---|---|---|
| `local_variance_nan` | Raw Local variance is NaN | Floor |
| `local_variance_negative_infinity` | Raw Local variance is `-Infinity` | Floor |
| `local_variance_positive_infinity` | Raw Local variance is `+Infinity` | Cap |
| `local_variance_non_positive` | Raw Local variance is finite and `<= 0` | Floor |
| `local_variance_below_floor` | Raw Local variance is finite and below Floor | Floor |
| `local_variance_above_cap` | Raw Local variance is finite and above Cap | Cap |

Horizontal grid use outside the log-forward-moneyness range is flat at the
nearest boundary. Runtime interpolation statistics record:

| Field | Meaning |
|---|---|
| `left_flat_count` | Number of Local-variance interpolations assigned to the left boundary |
| `right_flat_count` | Number of Local-variance interpolations assigned to the right boundary |
| `max_left_excursion` | Maximum absolute distance beyond the left boundary |
| `max_right_excursion` | Maximum absolute distance beyond the right boundary |

Public Local Volatility Delta and Gamma are reported with risk method
`central_bump`. The bump is evaluated with common random numbers, fixed
reduction order, and Spot-specific recompilation of forward normalizers and
affine dividend transforms. Public Local Vega and VegaKT output remain separate
acceptance work.

## Affine dividend diagnostics

Discrete dividends are represented by deterministic affine transforms between
the contract Spot coordinate and the continuous Local Volatility martingale
coordinate. Event timelines record pre-event and post-event coordinates. The
path engine treats a non-positive reconstructed post-dividend Spot as a typed
market error, not as a warning or clamp.

The matching-condition checker reports expected and actual call-surface values,
absolute error, and both accepted tolerances when the affine dividend surface
condition is violated.

## VegaKT diagnostics

VegaKT reports retain the configured maturity and log-forward-moneyness
coordinates, row-major bucket order, implied-volatility value at each reporting
bucket, raw and market-scaled units, covariance layout, and whether the full
bucket covariance matrix was requested.

| Field | Meaning |
|---|---|
| `policy_label` | Equation (11) approximation policy, currently `equation_11_first_order_v1` |
| `truncation_order` | Documented time-step order, currently `O(delta_t_k)` |
| `active_domain` | Connected density domain containing the forward node |
| `excluded_probability_mass` | Probability mass outside the active domain when available |
| `signed_residual` | Signed sensitivity excluded from reported buckets |
| `pre_projection` | Bucket sum plus residual target before reporting-basis projection |
| `left_edge_count`, `right_edge_count` | Reporting-IV projections assigned to edge buckets |
| `left_edge_sensitivity`, `right_edge_sensitivity` | Signed sensitivity assigned to each reporting edge |

Bucket estimates expose raw means in currency per unit absolute volatility and
market-scaled means in currency per volatility point. Per-bucket sample
variance and covariance with Price are present when sample data is supplied.
The full bucket covariance matrix is populated only when explicitly requested:
the full-matrix layout requires `full_bucket_covariance`, and the compact
price-and-bucket variance layout omits that field.

## Errors

Local Volatility and VegaKT validation failures terminate the calculation and do
not produce a partial pricing result. Important typed failures include invalid
SSVI/eSSVI admissibility, invalid Local-variance grids, unbracketed surface tail
quantiles, non-positive post-dividend Spot, invalid VegaKT density thresholds,
an inactive forward density node, equation (11) denominator failures,
transition-mass defects, reporting-basis shape mismatches, covariance shape
mismatches, and bucket-plus-residual reconciliation failures.

Python maps these failures through the existing `ValidationError` surface for
request construction and the existing runtime error surface for failed
evaluation. Successful warning ordering remains deterministic.
