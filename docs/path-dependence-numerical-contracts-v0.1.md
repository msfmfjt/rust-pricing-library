# Path Dependence Numerical Contracts v0.1

Status: Frozen for roadmap Gate P0; continuous smoothing amendment candidate

Requirements: `requirements-v1.0.md` Sections 6 and 7.3

Roadmap: `path-dependence-roadmap-v0.1.md`

## 1. Policy Identity

The policy identifier is `path_dependence_v2`. It fixes the scalar formulas,
branches, and boundary conventions below. A formula or branch change requires a
new policy identifier and new reference fixtures.

Every smoothing half-width `h` is finite, strictly positive, and expressed in
the signed condition's native units. The transition band is `[-h, h]`.

## 2. Compact C2 Indicator

For signed condition `x`, define

```text
u = (x + h) / (2h)
q(u) = 6u^5 - 15u^4 + 10u^3
```

The smoothed indicator `I_h` and its first two derivatives are

```text
x <= -h: I_h = 0, I_h' = 0, I_h'' = 0
-h < x < h:
  I_h = q(u)
  I_h' = 30u^2(u - 1)^2 / (2h)
  I_h'' = (120u^3 - 180u^2 + 60u) / (4h^2)
x >= h: I_h = 1, I_h' = 0, I_h'' = 0
```

The closed exterior branches own both band boundaries. At `x = 0`, the
indicator is exactly one half. Positive signed distance always denotes the hit
or true side of a condition; Put and down-barrier behavior is represented by
the sign used to construct `x`, not by a second kernel.

## 3. Smooth Positive Part and Extrema

The compact smooth positive part `P_h` is the integral of `I_h` from `-h`:

```text
x <= -h: P_h(x) = 0
-h < x < h: P_h(x) = 2h(u^6 - 3u^5 + (5/2)u^4)
x >= h: P_h(x) = x
```

Its first derivative is `I_h`; its second derivative is `I_h'`. Smoothed
Maximum and Minimum use one evaluation of this positive part:

```text
maximum_h(a, b) = b + P_h(a - b)
minimum_h(a, b) = a - P_h(a - b)
```

This construction fixes operand order while preserving symmetry,
translation equivariance, exact exterior values, and
`minimum_h(a,b) + maximum_h(a,b) = a + b`.

## 4. Barrier Endpoint and Jump Scores

For an up barrier `H`, the signed endpoint distance is `S - H`. For a down
barrier it is `H - S`. Exact mode treats distance zero as a hit.

In smoothed mode an affine-dividend event computes signed distances on both
sides of the deterministic jump and combines them as

```text
jump_score = maximum_h(distance_before, distance_after)
jump_hit_weight = I_h(jump_score)
```

The jump consumes no random coordinate. This score is used whether the jump
touches, crosses, or remains on one side of the barrier, so its reverse rule is
defined by the same smoothed primitives as ordinary Payoff nodes.

## 5. Continuous Bridge Survival

Let `v` be the strictly positive integrated log variance over one monitored
diffusion interval. With two endpoints on the safe side, the conditional hit
probability is

```text
up:   exp(-2 ln(H / S0) ln(H / S1) / v)
down: exp(-2 ln(S0 / H) ln(S1 / H) / v)
```

and survival is `1 - hit_probability`. A touched or breached endpoint has hit
probability one. If both endpoints are safe and integrated variance is zero,
hit probability is zero. Negative or non-finite integrated variance is a typed
numerical error rather than a probability repair.

The Local Volatility engine uses the architecture's transformed
continuous-martingale barrier and trapezoidal interval Local variance. Products
of interval survival probabilities are accumulated in the log domain. The
analytic probability consumes no pseudo-random or Sobol coordinate.

### 5.1 Smoothed continuous endpoints

Smoothed continuous monitoring uses the same Spot-unit signed hit distance `x`
and compact-C2 kernel as discrete monitoring. For a transformed barrier `H_f`,
continuous state `f`, and positive affine scale `B`,

```text
up:   x = B * (f - H_f)
down: x = B * (H_f - f)
```

The endpoint hit weight is `w = I_h(x)`. A C2 safe-side Spot distance is

```text
y = P_h(-x)
```

and the effective non-negative log distance supplied to the conditional bridge
formula is

```text
C = B * H_f
up:   d_h = -log(1 - y / C)
down: d_h =  log(1 + y / C)
```

An up-barrier smoothing width shall be smaller than every positive transformed
Spot-barrier numerator `C = H_S - A*S0` used by the monitored plan. This makes
the logarithm valid for every positive simulated state. Failure is a typed
compile error rather than a runtime clamp.

Outside the safe edge of the band, `d_h` is the exact bridge log distance.
Outside the hit edge, it is zero. Inside the band it provides a C2 continuation
of the safe-side distance. The conditional bridge survival for positive
integrated variance is

```text
q_h = 1 - exp(-2 * d_h(left) * d_h(right) / v)
```

and is one at zero variance. Each unique deterministic location contributes one
hit predicate to the monotone state. The initial endpoint contributes
`I_h(x_initial)`. A non-dividend interval endpoint contributes `I_h(x_post)`.
At a dividend endpoint, the ordinary endpoint predicate is replaced by the
Section 4 pre/post jump predicate, so it is not counted twice.

For endpoint weights `w_i`, jump weights `j_k`, and interval bridge survivals
`q_l`, total smoothed path survival is

```text
survival_h = product_i(1 - w_i)
           * product_k(1 - j_k)
           * product_l(q_l)
```

Products are accumulated in the log domain. A zero factor has zero reverse
contribution at its closed C2 exterior branch. Otherwise, the reverse rule
differentiates every factor through the Indicator, positive part, transformed
barrier, affine scale, both path endpoints, both Local-variance endpoints, and
the interval length. This surrogate consumes no additional random coordinate.

## 6. Reference Artifact

`../fixtures/path-dependence/reference-cases-v0.1.json` freezes boundary,
interior, extrema, bridge, and affine-jump cases. The independent
`../scripts/check_path_dependence_reference_fixture.py` evaluator uses Decimal
arithmetic and does not import production Rust or Python bindings.

The fixture stores decimal strings to avoid JSON binary64 parsing becoming the
reference. Absolute comparison tolerance is `1e-45` at 70-digit Decimal
precision.
