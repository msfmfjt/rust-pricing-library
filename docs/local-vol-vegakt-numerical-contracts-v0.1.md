# Local Volatility and VegaKT Numerical Contracts v0.1

Status: Accepted for Gate L0
Date: 2026-09-08
Policy identifier: `local_vol_vegakt_v1`
Requirements baseline: `requirements-v1.0.md` (Frozen)

## 1. Purpose and authority

This document freezes the numerical meaning of the first Local Volatility and
VegaKT implementation. It is normative where it is more specific than the
roadmap, but does not expand the frozen product or model scope.

The formula authorities are:

- Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*, for Standard
  SSVI, static-arbitrage sufficient conditions, and the density factor;
- Corbetta, Cohort, Laachir, and Martini, *Robust calibration and arbitrage-free
  interpolation of SSVI slices*, for eSSVI slice consistency and interpolation;
- Adrien et al., *Vega KT for the Local Volatility Model: An AD Approach*, SSRN
  4107770, especially equations (2)-(5), (9)-(11), and Appendix B;
- Guyon and Henry-Labordere, *The Smile Calibration Problem Solved*, SSRN
  1885032, Appendix "Discrete dividends", especially equations (25)-(28).

Paper figures are not numerical fixtures because their complete market surfaces
and grids are not published. The committed cases are independently evaluated
equation fixtures with explicit provenance and reproducible inputs.

## 2. Policy ownership and explicit inputs

Every compiled Local Volatility plan contains the policy identifier and every
effective number below. A helper may propose a value, but compilation stores the
materialized value and never re-runs the helper.

The following are mandatory calculation inputs and have no runtime default:

- Local-variance `floor` and `cap`, with `0 < floor <= cap`;
- maximum simulation step `max_delta_t`;
- explicit Local-variance time and log-moneyness nodes;
- eSSVI terminal ATM forward-variance slope;
- VegaKT reporting maturity and log-moneyness nodes;
- active-density relative threshold;
- covariance layout and any validation bump sizes.

`LocalVolVegaKtHelperPolicy::V1` proposes, but does not silently apply:

| Field | V1 proposal |
| --- | ---: |
| left tail probability | `1e-6` |
| right tail probability | `1e-6` |
| left/right log-moneyness padding | `0.25` / `0.25` |
| left/right interval count | `32` / `32` |
| left/right sinh shape | `2.0` / `2.0` |
| active-density ratio | `1e-10` |
| quantile bracket limit | `abs(k) <= 16` |
| quantile bisection limit | `128` iterations |
| quantile termination | `width <= max(1e-12, 1e-12 * max(1, abs(k_mid)))` |

The maturity helper returns the sorted unique union of positive surface knots,
state-changing event times, requested observation times, and product expiry, all
not later than expiry. Equality is exact normalized-year-fraction equality; no
fuzzy deduplication is performed.

When a materialized Local-variance grid includes the required initial
simulation node `T=0`, the Dupire helper fills that row from the first strictly
positive Local-variance time node. Reporting-IV maturity nodes remain strictly
positive, so calibrated-surface Python helpers exclude `T=0` from the retained
reporting basis even when it is present in the Local-variance grid.

## 3. Validation arithmetic

### 3.1 General rules

- All source parameters, nodes, tolerances, and derived quantities must be
  finite unless an outer integration boundary is explicitly infinite.
- Positive quantities and strict ordering are checked exactly. Tolerances do not
  turn zero, a negative variance, or reversed nodes into valid input.
- Constraint residuals calculated from otherwise valid inputs use
  `abs_tol=1e-14` and `rel_tol=1e-12` under policy V1. The effective tolerance is
  `max(abs_tol, rel_tol * max(1, abs(lhs), abs(rhs)))`.
- A non-strict inequality `lhs <= rhs` passes when `lhs - rhs <= tolerance`.
  A mathematically strict inequality must retain positive slack: it passes only
  when `rhs - lhs > tolerance`.
- Passing within tolerance never mutates a calibrated parameter. Raw and
  compiled fingerprints therefore remain identical for surface parameters.

### 3.2 Standard SSVI

For `theta > 0`, `abs(rho) < 1`, and `phi(theta) > 0`, define

```text
y = phi(theta) * k
a = y + rho
q = sqrt(a*a + 1 - rho*rho)
w = theta/2 * (1 + rho*y + q)
```

The Power-law family is

```text
phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))
phi'(theta) = -phi(theta) * (gamma/theta + (1-gamma)/(1+theta))
```

with `eta > 0`. The public syntactic range remains `0 < gamma < 1`, but a
surface using the V1 zero-maturity extrapolation is statically admissible only
for `gamma <= 0.5` and `eta * (1 + abs(rho)) <= 2` under the constraint rule
above. Thus a larger `gamma` is parsed but rejected as an inadmissible surface.

For Heston-like SSVI, set `z=lambda*theta` and

```text
phi(theta) = (z - 1 + exp(-z)) / z^2
phi'(theta) = lambda * (2 - z - (z + 2)*exp(-z)) / z^3.
```

It requires `lambda > 0` and
`lambda >= (1 + abs(rho))/4`. For `abs(z) <= 1e-4`, V1 evaluates `phi` and
`phi'/lambda` using the fixed Horner series

```text
phi = 1/2 + z*(-1/6 + z*(1/24 + z*(-1/120 + z*(1/720
      + z*(-1/5040 + z/40320)))))
phi'/lambda = -1/6 + z*(1/12 + z*(-1/40 + z*(1/180
                  + z*(-1/1008 + z/6720))))
```

and otherwise uses the displayed exponential formulas. Evaluation does not use
FMA. The branch and Horner order are fingerprinted policy.

Standard SSVI also enforces the Gatheral-Jacquier sufficient conditions over its
entire represented theta range:

```text
d(theta * phi)/dtheta >= 0
phi'(theta) < 0
theta * phi(theta) * (1 + abs(rho)) < 4
theta * phi(theta)^2 * (1 + abs(rho)) <= 4.
```

The family-specific global conditions above prove these inequalities for the
zero-maturity extrapolation; the implementation additionally evaluates their
residuals at every theta knot and extrapolation boundary for diagnostics.

### 3.3 Theta interpolation

Positive maturity knots are strictly increasing and theta values are positive
and non-decreasing. Interior PCHIP slopes use the Fritsch-Butland weighted
harmonic mean when adjacent secants have the same non-zero sign and zero
otherwise. Endpoint slopes use the one-sided three-point estimate, clipped to
zero on a sign change and to three times the adjacent secant when the next
secant changes sign. A two-knot curve is linear.

Before the first knot, `theta(T)=T*theta_1/T_1`. Beyond the last knot,
`theta(T)=theta_n + terminal_slope*(T-T_n)`, where `terminal_slope` is explicit,
finite, and non-negative. The analytic PCHIP derivative is the sole Standard
SSVI source of `dtheta/dT` inside the knot range.

### 3.4 eSSVI

Each slice stores `(theta, psi, rho_psi)`, where `psi=theta*phi > 0` and
`rho=rho_psi/psi` satisfies `abs(rho)<1`. A slice must satisfy

```text
psi * (1 + abs(rho)) < 4
psi^2 * (1 + abs(rho)) <= 4*theta.
```

For consecutive slices, exact monotonicity requires

```text
theta_next >= theta
psi_next >= psi
abs(rho_psi_next - rho_psi) <= psi_next - psi
```

with only the last derived residual classified by the V1 constraint tolerance.
Between maturities, `theta`, `psi`, and `rho_psi` are linearly interpolated in
time, then `rho=rho_psi/psi` is recovered. No component-wise interpolation of
`rho` is permitted.

At the short end, `theta`, `psi`, and `rho_psi` are multiplied by `T/T1`, which
keeps `rho` constant. At the long end, `psi` and `rho_psi` are constant while
theta grows with the explicit terminal slope.

## 4. Surface derivatives, density, and Dupire

With `y`, `a`, and `q` from Section 3.2:

```text
w_k  = theta*phi/2 * (rho + a/q)
w_kk = theta*phi^2/2 * (1-rho^2)/q^3
```

`w_T` is obtained by analytic chain rule through the selected Standard SSVI or
eSSVI time interpolation. Numerical maturity differencing is not a production
path.

Define the Durrleman density factor in this exact operation order:

```text
u = 1 - k*w_k/(2*w)
g = u*u - (w_k*w_k/4)*(1/w + 1/4) + w_kk/2.
```

For forward `F`, strike `K=F*exp(k)`, and
`d2=-k/sqrt(w)-sqrt(w)/2`, the undiscounted call density is

```text
d(T,K) = normal_pdf(d2)/(K*sqrt(w)) * g.
```

In the drift-free forward coordinate, raw Dupire Local variance is
`local_variance_raw=w_T/g`. Evaluation first requires finite positive `w`,
finite derivatives, finite positive `g`, and finite `w_T`. A failed surface
invariant is a `MarketError`, not a clamp.

After a valid surface evaluation, raw Local variance is classified in this
stable priority order:

1. NaN -> floor, `local_variance_nan`;
2. negative infinity -> floor, `local_variance_negative_infinity`;
3. positive infinity -> cap, `local_variance_positive_infinity`;
4. value `<= 0` -> floor, `local_variance_non_positive`;
5. value `< floor` -> floor, `local_variance_below_floor`;
6. value `> cap` -> cap, `local_variance_above_cap`;
7. otherwise unchanged.

Every repair records source coordinates, original `to_bits`, applied value,
reason, policy, and deterministic row-major occurrence index.

## 5. Explicit grid helper

At every requested maturity the helper solves for left and right quantiles of
the analytic SSVI/eSSVI call distribution. It forms the common rectangular
range from the minimum left and maximum right log-forward-moneyness across all
maturities, then applies the two explicit paddings.

For left boundary `k_min < 0`, right boundary `k_max > 0`, interval counts
`n_l,n_r >= 1`, and shapes `a_l,a_r >= 0`, nodes are

```text
k_left(j)  = k_min * sinh(a_l*(1-j/n_l))/sinh(a_l), j=0..n_l
k_right(j) = k_max * sinh(a_r*j/n_r)/sinh(a_r),     j=1..n_r.
```

`sinh(a*t)/sinh(a)` is defined as `t` when `a=0`. The shared left endpoint's
`j=n_l` is exact `+0.0`; the right side does not add another ATM node. Generated
nodes are validated and serialized like caller-supplied nodes.

## 6. Local Vega and hat projection

Following SSRN 4107770 equation (2), `Vloc(t,K)` is a sensitivity density with
respect to strike-coordinate integration. Appendix B's path adjoint is first
deposited on the explicit non-uniform log-moneyness nodes with the linear finite
element hats. For `x in [x_i,x_(i+1)]`:

```text
h_i(x)     = (x_(i+1)-x)/(x_(i+1)-x_i)
h_(i+1)(x) = (x-x_i)/(x_(i+1)-x_i).
```

The support of an interior hat is its two adjacent grid intervals. Values left
or right of the grid are assigned fully to the nearest edge node and counted.
This makes the deposit weights sum to one exactly apart from binary64 rounding.

Accumulated node adjoints `A_i` are converted to a density in `x` by
`Vloc_x_i=A_i/m_i`, using mass-lumped hat areas

```text
m_0     = (x_1-x_0)/2
m_i     = (x_(i+1)-x_(i-1))/2
m_last  = (x_last-x_(last-1))/2.
```

Thus the discrete conservation identity is
`sum_i(m_i*Vloc_x_i)=sum_paths(path_adjoint)`. At strike `K=F*exp(x)`, the
strike-coordinate density required by the paper is
`Vloc_K=Vloc_x/K` because `dK=K dx`.

## 7. Equation (11), active domain, and residual

The V1 first-order recovery is exactly SSRN 4107770 equation (11):

```text
Gamma_local(t_(k+1),K) =
    Vloc_K(t_k,K) / (K^2 * d(t_k,K) * sigma_local(t_k,K) * delta_t_k).
```

It is labelled `equation_11_first_order_v1`; its truncation error is
`O(delta_t_k)`. `sigma_local` is the positive square root of the effective,
possibly clamped Local variance used by the path step.

At each maturity, the node with minimum `abs(x)` is the forward node, with a
lower index breaking a tie. Let `d_max` be the maximum positive analytic call
density. Nodes qualify when `d/d_max >= explicit_threshold`. The active domain
is the maximal contiguous run of qualifying nodes containing the forward node.
Failure of the forward node to qualify is `RiskError::VegaKtForwardInactive`.
Disconnected qualifying nodes remain excluded.

Signed Local Vega outside the active domain is retained in the residual. Results
report active endpoints, threshold, `d_max`, excluded probability mass when
available, excluded signed sensitivity, edge-assigned reporting sensitivity,
and the reconciliation

```text
pre_projection = sum(VegaKT_raw buckets) + signed_residual.
```

## 8. Discrete Gamma operator and cell integration

SSRN 4107770 equations (3)-(5) use the next-time Local Gamma under the
Black-Scholes gamma kernel. For current strike/spot coordinate `S`, variance
`v=sigma_local(S)^2*delta_t`, and integration strike `K`, define

```text
G(S,K;v) = d^2 BS(S,K;v)/dS^2
q(S,K;v) = exp(-v) * G(S,K;v).
```

`q` is a probability density: it is lognormal with
`log(K) ~ Normal(log(S)+3*v/2, v)`. Consequently `integral(q dK)=1`, while
`integral(G dK)=exp(v)`. This normalization is metadata, not an omitted factor.

Local Gamma is piecewise linear inside finite cells. For each cell `[a,b]`, V1
computes both

```text
p_ab = integral_a^b q(S,K;v) dK
m_ab = integral_a^b K*q(S,K;v) dK
```

from lognormal CDF differences. A linear reconstruction `c0+c1*K` contributes
`exp(v)*(c0*p_ab+c1*m_ab)`. The first and last cells are `[0,b]` and
`[a,+infinity)` and use the nearest edge Gamma constantly, so all probability
mass is retained without linear tail extrapolation.

CDF differences use the smaller of the direct lower-tail difference and the
equivalent survival-function difference. `erfc` supplies both tails. Cells are
visited from low to high strike; the final cell probability is set to
`1-NeumaierSum(previous cells)` and must lie in `[-8*EPSILON,1+8*EPSILON]`
before being clipped to `[0,1]`. A larger conservation defect is a typed error.

## 9. Reporting-IV basis and units

The reporting basis samples canonical `f` implied volatility at explicit
`(T,k)` nodes. Inside a rectangle it is reconstructed by bilinear basis weights,
with time weight formed first and strike weight second. Each continuum vanilla
weight from equation (4) is projected through equation (9) using those basis
derivatives. Strike contributions outside the reporting range go to the nearest
edge bucket and emit one aggregated warning per maturity.

Bucket order is row-major `[maturity][log_moneyness]`. Raw VegaKT is currency per
unit absolute volatility (`sigma=1.0`). Market-scaled VegaKT is raw times `0.01`
and is currency per one volatility point. Canonical Scalar Vega is the
deterministic sum of in-domain raw buckets; residual is not silently included.

Each bucket carries its variance and covariance with Price. Full bucket
covariance is absent unless explicitly requested. All sums use the accepted
fixed-block Neumaier and balanced-tree reduction policy.

## 10. Affine dividends

For event dividend `D(S_minus)=alpha*S0+beta*S_minus`, require `alpha>=0` for a
fixed non-negative cash amount compiled at the current `S0`, and
`0<=beta<1`. With `S=A*S0+B*f`, where `f` is continuous across the event:

```text
A_after = (1-beta)*A_before - alpha
B_after = (1-beta)*B_before.
```

For post-event strike `K`, the matching condition equivalent to SSRN 1885032
equation (27) is

```text
C_after(K) = (1-beta) * C_before((K + alpha*S0)/(1-beta)).
```

The matching check uses explicit `abs_price_tol` and `rel_price_tol`; helper V1
proposes `abs_price_tol=1e-12*S0` and `rel_price_tol=1e-11`, both materialized in
the plan. Same-day ordering remains Dividend then Expiry. A Spot bump holds a
fixed cash amount constant and recompiles `alpha=D/S0`.

## 11. Reference and acceptance rules

`fixtures/local-vol/reference-cases-v0.1.json` freezes focused cases for both
Standard SSVI families, eSSVI interpolation, constant-vol Dupire, equation (11),
non-uniform hats, normalized gamma-kernel cells, and affine dividends.
`scripts/check_local_vol_reference_fixture.py` independently recomputes them
with Python `Decimal` where elementary functions permit and checks transition
cells through the standard-library normal CDF.

Production code must pass, in order:

1. parameter and admissibility tests;
2. analytic derivative and density fixtures;
3. hat and transition conservation identities;
4. equation (11) and dividend matching fixtures;
5. refinement, CRN, statistical, replay, and cross-language tests in later
   Gates.

No later performance result may weaken one of these numerical contracts.

## References

- [Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2033323)
- [Corbetta et al., *Robust calibration and arbitrage-free interpolation of SSVI slices*](https://arxiv.org/abs/1804.04924)
- [Adrien et al., *Vega KT for the Local Volatility Model: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4107770)
- [Guyon and Henry-Labordere, *The Smile Calibration Problem Solved*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032)
