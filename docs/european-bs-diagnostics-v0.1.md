# European Black–Scholes diagnostics catalogue v0.1

Status: frozen for the first vertical slice

This catalogue covers successful valuation diagnostics and warnings emitted by
the European Black–Scholes Pseudo-MC/RQMC slice. Stable codes and deterministic
ordering are shared by Rust, JSON, and Python unless a field is explicitly an
execution-only diagnostic.

## Estimates and risk output

Every Price and risk estimate reports value, standard error, 95% confidence
interval, estimator kind, and effective independent sampling-unit count. Delta,
Gamma, and Vega each expose Raw and Market-scaled estimates with their units.
The slice identifies Delta and Vega as `aad_reverse`, and Gamma as
`central_bump_of_aad_delta`.

CRN validation diagnostics are returned separately for Delta, Gamma, and Vega.
Each contains the bump-and-revalue estimate and the paired
`bump_minus_primary` estimate. Method metadata records the Gamma Spot bump, the
validation Spot and volatility bumps, smile convention, and bump-policy version.

## Monte Carlo execution diagnostics

| Field | Meaning |
|---|---|
| `master_seed` | Philox seed or independent RQMC scramble seed |
| `estimator` | Pseudo-MC or randomized QMC |
| `scramble_count` | Independent RQMC replicates; absent for Pseudo-MC |
| `direction_checksum` | Joe–Kuo direction-table checksum for RQMC |
| `scramble_checksum` | Compiled LMS plus Digital-shift table checksum |
| `policy_version` | Deterministic execution-policy ABI version |
| `worker_threads` | Dedicated Rayon pool size |
| `reduction_block_size` | Logical fixed reduction block size |
| `aad_tile_policy_version`, `aad_tile_capacity` | Effective SoA AAD tiling |
| `checkpoint_policy_version`, `checkpoint_interval` | Effective reverse checkpoint policy |
| `antithetic` | Whether each independent unit evaluates the exact `Z`/`-Z` pair |
| `discount_region`, `dividend_region` | Pillar, interpolation, or right-extrapolation region |
| `payoff_fingerprint` | Deterministic compiled payoff-tape fingerprint |

Replay metadata additionally carries schema version, normalized Request
fingerprint, library version, and platform. Plan fingerprint and execution policy
are available from the compiled Rust facade and are included in replay fixtures.

## Successful-result warnings

Warnings are emitted in the order shown below and never replace an error.

| Order | Stable code | Condition | Message |
|---:|---|---|---|
| 1 | `discount_curve_extrapolation` | Expiry is beyond the last discount-curve pillar | `expiry uses flat-forward discount-curve extrapolation` |
| 2 | `dividend_curve_extrapolation` | Expiry is beyond the last continuous-dividend curve pillar | `expiry uses flat-forward dividend-curve extrapolation` |

No other warning can currently be emitted by this vertical slice. The Python
`PricingWarning` object exposes immutable `code` and `message` properties; JSON
preserves the same ordered list.

## Errors

Request-domain mismatches, invalid dates, unsupported Black–Scholes VegaKT,
invalid Gamma bumps, insufficient stochastic units, non-finite total variance,
market/graph/RQMC/AAD failures, and wire validation failures terminate the
calculation. They do not return a partial result. Python converts validation
failures to one `ValidationError` containing an immutable ordered `issues` list
with stable code, RFC 6901 pointer, and message.
