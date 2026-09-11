# Path Dependence diagnostics catalogue v0.1

Status: candidate catalogue; Gate P8 is not accepted

Requirements: `requirements-v1.0.md`

Roadmap: `path-dependence-roadmap-v0.1.md`

This catalogue records the stable diagnostic surfaces for Digital, Barrier,
arithmetic average-price Asian, and fixed-strike Lookback calculations. It
complements the accepted European Black-Scholes and Local Volatility/VegaKT
catalogues.

## Valuation and smoothing diagnostics

Every result identifies one payoff valuation kind:

| Value | Meaning |
|---|---|
| `ExactContractual` | Price uses the contractual payoff and no smoothing diagnostics are attached |
| `SmoothedSurrogate` | Price and requested Greeks use the same explicitly configured surrogate payoff |

When smoothing is active, `payoff_smoothing` records:

| Field | Meaning |
|---|---|
| `kernel` | Compact C2 quintic kernel identity |
| `policy_version` | Version of the smoothing formula and branch convention |
| `half_width` | Positive transition half-width supplied by the caller |
| `full_transition_width` | Twice the half-width |
| `width_unit` | Native signed-distance unit, currently Spot |
| `price_and_greeks_share_payoff` | Confirmation that Price and Greeks came from the same compiled surrogate |
| `endpoint_count` | Number of endpoint indicator nodes affected by smoothing |
| `dividend_jump_count` | Number of affine-dividend jump predicates affected by smoothing |

Exact and smoothed Source graphs have distinct fingerprints. Changing the
kernel version or half-width changes the smoothed graph fingerprint. A Width
ladder returns the separately declared primary result, preserves caller order,
and attaches adjacent Price/Delta/Gamma/Vega differences to each non-primary
entry after the first.

## Barrier bridge diagnostics

Continuous monitoring attaches `barrier_bridge` diagnostics:

| Field | Meaning |
|---|---|
| `abi` | Stable bridge implementation identity |
| `policy_version` | Bridge formula, interpolation, and branch-policy version |
| `indicator_mode` | Endpoint and dividend-jump indicator mode |
| `endpoint_hit_fraction` | Fraction of paths with a deterministic endpoint hit |
| `dividend_jump_hit_fraction` | Fraction of paths with a deterministic affine-dividend jump hit |
| `mean_conditional_bridge_hit_weight` | Mean conditional diffusion-bridge hit weight |
| `mean_interval_count` | Mean number of monitored diffusion intervals |
| `mean_finite_correction_count` | Mean number of finite bridge corrections |
| `mean_zero_variance_count` | Mean number of deterministic zero-variance intervals |
| `mean_survival_underflow_count` | Mean number of log-survival products that underflow at consumption |
| `mean_certain_survival_count` | Mean number of intervals with certain survival |

Endpoint contact is inclusive in exact mode. The bridge uses transformed
continuous-martingale barrier coordinates, trapezoidal endpoint Local
variance, linear log-barrier interpolation, and analytic conditional survival;
it consumes no random coordinate. Affine-dividend jumps remain separate
deterministic events and are never folded into a diffusion bridge probability.

The current continuous bridge exposes only `BarrierHitIndicatorMode::Exact`.
Gate P8 therefore remains open until explicitly smoothed endpoint and
dividend-jump predicates are implemented for continuous monitoring, including
matched reverse rules and corresponding diagnostics.

## Contractual path-state diagnostics

Arithmetic Asian results retain:

| Field | Meaning |
|---|---|
| `known_observation_count` | Number of observations carrying contractual fixings |
| `unknown_observation_count` | Number of modelled observations |
| `known_weight_sum` | Sum of weights attached to known observations |
| `unknown_weight_sum` | Sum of weights attached to unknown observations |
| `weighted_known_fixing_sum` | Fixed contribution of known observations to the arithmetic average |

Fixed Lookback results retain:

| Field | Meaning |
|---|---|
| `past_monitoring_count` | Number of declared monitoring dates before valuation |
| `future_monitoring_count` | Number of declared monitoring dates on or after valuation |
| `historical_extremum` | Required fixed running maximum or minimum when past monitoring exists |

Known fixings and historical extrema are contractual state. They remain fixed
under Spot, volatility, curve, and dividend bumps and carry zero market
adjoint. Fully fixed products report exact zero market risks.

## Errors

Validation and numerical failures terminate the calculation without a partial
result. Important typed failures include:

- unsupported risk for an exact discontinuous product;
- smoothing on an unsupported product or a Width ladder without a primary
  smoothing width;
- zero, negative, non-finite, excessively large, duplicate, or non-monotone
  smoothing widths;
- invalid observation ordering, duplicate dates, invalid Asian weights or
  fixing classifications, and invalid Lookback historical state;
- past Barrier monitoring without an explicit supported state;
- non-positive or undefined transformed continuous barriers;
- negative or non-finite bridge variance and non-finite bridge inputs;
- unsupported continuous-barrier payoff smoothing while the open P8 item
  described above remains unresolved.

Python maps construction failures through `ValidationError` and evaluation
failures through the existing runtime error surface. Successful diagnostics,
Width-ladder entry order, and replay serialization remain deterministic.
