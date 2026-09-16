# Particle Local Correlation v0.1

This experimental adapter prices Basket, Worst-of and unseasoned Autocallable
payoffs with deterministic rates and BS/Local Volatility constituents. Rust
uses `MultiAssetPricingPlan::compile_with_local_correlation`; Python uses
`MultiAssetPlan.compile(..., local_correlation=LocalCorrelationConfig(...))`.
The [example](../../examples/python/local_correlation.py) runs all three products.
Existing fixed-correlation plans keep their random dimensions and fingerprints.

## Coordinates and identification

Let `f_i` be the continuous equity coordinate used by the existing affine
dividend engine, `F_i(t)` its deterministic forward normalizer, and
`m_i = f_i / F_i(t)`. Each normalized constituent starts at one and satisfies

\[
dm_i/m_i=\sigma_i(t,\log m_i)\,dW_i,\qquad
B=\sum_i w_i m_i,\quad w_i>0,\quad\sum_i w_i=1.
\]

The supplied target is **effective relative local variance of B**, on
`(time, log B)`. It must be a `LocalVarianceGrid` (Python: an LV `Model`),
start at time zero, cover the complete observation horizon, and span
log basket zero. Weights are explicit, ordered with the assets, and are not
renormalized; their sum is checked to tolerance `1e-12`. At least two assets
are required. Product payoff weights are separate from calibration weights.

Physical observations still reconstruct spot from `f_i` using each asset's
affine dividend coordinates, including cash/proportional payments. Unequal
carries or affine offsets mean that a fixed-weight physical cash index is
generally **not** a constant multiple of B. Market index IVs must first refer
to this normalized continuous basket; this API does not automatically convert
physical-index quotes. With equal carries and no affine offsets, suitably
scaled physical weights can represent B directly.

One basket local-variance surface does not identify every pairwise correlation.
The caller chooses a one-dimensional family using two PSD endpoint schedules:

\[
R(t,m)=(1-\lambda(t,\log B))R_0(t)+\lambda(t,\log B)R_1(t),
\qquad 0\leq\lambda\leq1.
\]

The normal `correlations` argument is R0; `second_correlations` supplies R1.
Asset order, schedule dates (including dates beyond the horizon) and numerical
PSD tolerances must match exactly. Each endpoint uses the existing validated
canonical correlation matrix. R1 need not have larger correlations than R0;
the sign of the conditional variance span is retained. Convex mixing preserves
PSD and unit diagonals, including rank-deficient endpoints. There is no implied
calibration of a free N-by-N matrix or correlation-parameter Greek.

## Finite particle calibration

For each particle, define

\[
a_i=w_i m_i\sigma_i/B,\qquad q_e=a^\top R_e a,\qquad
c_e(t,x)=\mathbb E[q_e\mid\log B=x].
\]

The Markovian projection condition gives the scalar equation

\[
v_B(t,x)=c_0(t,x)+\lambda(t,x)(c_1(t,x)-c_0(t,x)),
\qquad
\lambda_{raw}=\frac{v_B-c_0}{c_1-c_0}.
\]

Particles start at `m_i=1`. At each common grid time the algorithm estimates
both conditional means by Nadaraya–Watson regression, using the compact quartic
kernel `K(u)=(1-u*u)^2` for `abs(u)<1`, zero otherwise. Bandwidth is in log B.
It then solves for lambda and advances the interacting population with that
new row. At time zero the identical initial-state moments are used at every
spatial node; no kernel estimate of a Dirac mass is attempted. Only log B=0 is
visited initially. Off-state time-zero values are an explicit extension.

This finite regression averages each particle's **relative** variance q_e
(including its own B squared denominator). In the zero-bandwidth limit it
agrees with conditioning absolute basket variance and dividing by exp(2x).
Those two finite-bandwidth estimators are different; this implementation uses
the former consistently in calibration and reverse.

The common grid includes constituent/target time knots, dated correlations,
observations, dividends and the `maximum_step` refinement. Lambda is constant
over each time interval and linear in log B, with flat spatial wings. The
terminal row is calibrated and diagnosed, although no evolution uses it.

The effective sample count is `(sum K)^2/sum(K^2)`. An under-supported query
cell takes the conditional moments from the nearest usable spatial node,
with lower-index tie breaking. It keeps its own target variance. If every cell
has insufficient support, compilation fails. `effective_samples` reports the
query cell's support; `source_nodes` identifies the donor and its support can
be looked up separately. Time-zero support equals the particle count.

Feasibility is explicit:

| Policy/condition | Behavior |
| --- | --- |
| `feasibility="reject"` (default) | Fail if any node target is outside the endpoint range |
| `feasibility="project_and_report"` | Clamp raw lambda to [0,1] and report the missed target |
| `abs(c1-c0) <= minimum_variance_span` | Mark unidentifiable and select lambda=0; a target mismatch above this tolerance is rejected or reported according to policy |

`LocalCorrelationCalibration` exposes row-major endpoint, target and attained
variances; residuals are `attained-target`. It also exposes raw/effective lambda,
ESS, donor/fallback/projected/unidentifiable flags, time/log axes and constituent
particle means. `correlation_at(t,x)` returns the effective covariance matrix.
These are owned, read-only Python snapshots. A zero node residual means the
finite conditional equation was solved; it is **not** proof that independently
priced basket options match the target. Kernel bias, finite population, time
and spatial interpolation, fallback, and projection can all affect that fit.

## Simulation and replay

Given independent standard-normal N-vectors z0,z1, each step uses

\[
z=\sqrt{1-\lambda}\,L_0z_0+\sqrt{\lambda}\,L_1z_1,
\qquad L_eL_e^\top=R_e.
\]

Thus the conditional covariance is exactly the PSD mixture. Interpolating
Cholesky factors applied to the same normals would not give this covariance.
The Gaussian dimension is `2*N*time_steps`, with endpoint 0 asset factors
followed by endpoint 1 asset factors. Brownian bridging is applied to the
independent blocks before the state-dependent loading. Antithetic paths negate
those independent blocks and recompute lambda on their own evolving states.
Normalized states follow adapted log Euler, with the existing physical spot
positivity checks. No flooring or resampling repairs a failed path.

Calibration uses a separate Philox calibration domain/seed; valuation uses the
existing MC or randomized Sobol domain. Calibration reductions have fixed
particle order with compensated sums. Valuation uses deterministic block
reduction, so worker-count changes preserve exact replay. The fingerprint
includes the coordinate/algorithm tag, both endpoint schedules, basket target,
weights, all particle options, lambda surface and donor/projection decisions.

For K steps, P particles, N assets and X basket nodes, calibration is
O(K*P*(X+N squared)). Valuation is O(K*N squared) per path. Retaining the
calibration trace for reverse uses O(K*P*N) state memory. There is no compressed
rank sampling or checkpointed reverse in this version.

## Calibrated risk

`evaluate_aad()` requires `retain_reverse_trace=True`. The reverse differentiates
the actual finite algorithm: payoff and physical-coordinate seeds, constituent
LV lookups, log Euler, basket-dependent lambda interpolation, the conditional
moment quotient, kernel motion, both endpoint quadratic forms and all previous
interacting particle states. Both direct valuation and recalibration paths
contribute to each constituent's volatility risk.

| Result | Units and interpretation |
| --- | --- |
| `result.risks[i].delta`, `result.gamma[i][j]` | Physical spot Delta and optional CRN cross Gamma, with normalized calibration weights and targets fixed |
| `result.local_correlation_risk.basket_variance_adjoints` | Per unit effective basket local variance, row-major original target nodes |
| `asset_adjoints[i]` for BS | One derivative per unit sigma, including correlation recalibration |
| `asset_adjoints[i]` for LV | Per unit effective constituent local variance, row-major original grid nodes, including correlation recalibration |
| `basket_standard_errors`, `asset_standard_errors` | RQMC scramble errors conditional on the fixed calibration population; MC returns None for these joint volatility errors |

Old per-asset BS Vega and LV-adjoint fields are empty for Local Correlation
plans; the joint result is the authoritative volatility-risk output. Basket
and LV axes identify the original input grids, not the refined time grid.
BS axis lists are empty. These are **effective variance** risks, even if an LV
grid originated from market IV; source-quote VegaKT is not implemented here.

Reverse holds endpoints, weights, axes, bandwidth, seeds, ESS donor choices,
variance-span and projection branches fixed. Strictly projected and
unidentifiable nodes contribute zero calibration derivative. An exact raw
lambda of zero or one at a non-unidentifiable node in a used time row rejects
AAD: the active-set transition/square-root mixture has no ordinary pathwise
derivative there. Derivatives near such transitions can be noisy. No
derivative through donor selection or active-set changes is reported.

RQMC recomputes the joint calibration VJP for each independent scramble before
estimating errors, preserving covariance between direct and recalibrated
contributions. These errors exclude calibration sampling error and numerical
discretization bias. Vary particles, bandwidth, grid and calibration seed to
assess those separately.

## Scope and validation

This adapter accepts deterministic-rate BS/LV only. It explicitly rejects
LSV/Bergomi/rough configurations, Hull–White and full spot/vol/rate driver
matrices. Extending it to those models requires a joint driver construction
that preserves constituent spot/vol/rate marginals while changing cross-asset
correlation, and a consistent coupled particle calibration/reverse. The
existing fixed-correlation LSV/HW adapters remain available independently.
Multiple basket targets, physical-index quote conversion, correlation Greeks,
and source-IV VegaKT are not included. The adapter has no JSON schema boundary.

The acceptance tests check independently recalibrated finite differences for
all basket and constituent volatility nodes, dividends and cross Gamma,
projection/support/trace contracts, singular endpoints, dated multi-asset
PSD mixing, worker replay, and an independent two-step exchange-option
quadrature. A separate normalized-index option test compares independent
valuation with the target Black–Scholes prices under particle/grid refinement,
and checks constituent marginal prices.

Background: [Langnau, Introduction into Local Correlation Modelling](https://arxiv.org/abs/0909.3441)
discusses state-dependent PSD correlation families and non-uniqueness.
[Jourdain and Zhou](https://arxiv.org/abs/1607.00077) discuss conditional-expectation
calibration and kernel particle methods in the related LSV setting. The
specific convex family and finite regression/reverse conventions implemented
here are defined by the equations above.
