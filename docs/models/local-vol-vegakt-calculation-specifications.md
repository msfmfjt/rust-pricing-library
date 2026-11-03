# Local Volatility and VegaKT Calculation Specifications

Status: Accepted for Gate L0
Date: 2026-09-08
Policy identifier: `local_vol_vegakt_v1`

Implementation note (2026-09-29): the
[independent backtest numerical panel](../../design/validation/vega-kt-backtest-numerics-v0.1.md)
found that the native public VegaKT path does not yet implement the complete
gamma pipeline specified below. Its grid-scaling defect has been corrected,
but end-to-end numerical acceptance still fails. These normative requirements
remain unchanged.

Implementation update (2026-10-04): the
[restricted terminal-measure path](../../design/validation/local-vol-terminal-pathwise-v0.1.md)
is now used by ordinary pricing for eligible flat-variance terminal payoffs,
with actual per-path IV risk and joint sample statistics. The unchanged
numerical gate retains 6 of 154 failures. General recovery and full acceptance
remain open; the implementation does not alter the normative requirements below.

The subsequent [reporting-date extension](../../design/validation/vega-kt-reporting-maturities-v0.1.md)
removes the requirement that every simulation time be a reporting maturity in
that restricted path. Reporting IV is checked against cumulative left-Euler
variance at each quote date, including dates inside an interval. The contract
expiry must remain the final reporting maturity; general smile is still outside
this path's scope.

The [general-smile terminal extension v0.3](../../design/validation/vega-kt-smile-acceptance-v0.3.md)
adds an exact static-replication route for continuous piecewise-affine terminal
payoffs. With `g(S)=a+b*S+sum(q_j*(S-K_j)+)`, IV risk is the transpose
interpolation of `q_j * D(T) * BlackVega(F(T),K_j,IV(T,K_j))`. The order of
Vega multiplication and interpolation follows equation (9). Signed call masses
are preserved. This closed-form terminal case needs no noisy AD Gamma inversion.
General nonterminal recovery below remains a separate, unaccepted scope.

The new policy is `terminal_payoff_replication_iv_v1`; sampling moments are not
applicable (absent/unavailable), residual is zero, and density truncation is not
used. Pricing/Delta/local Vega still use MC, with independent statistical errors.
The exact expiry must be a reporting node; it need not be the last reporting
node. Early exercise, path-dependent or smoothed payoffs and discrete dividends
are excluded. The companion's accepted P0 scope further requires zero dividends
and model/market price consistency. Explicit flat grids retain cumulative-IV
consistency checks at every quote maturity up to expiry.

The [2026-10-05 path Gamma component](../../design/validation/local-vol-path-gamma-recovery-v0.1.md)
integrates the full equation-(3) convolution against source hats, then applies
an explicitly regularized signed-measure inverse to actual compiled-payoff AD.
It supports positive-time rows of flat-space continuous path payoffs and audits
noise amplification and same-draw reconstruction errors. It does not use the
first-order diagonal approximation, claim unique identification, recover the
initial boundary from one scalar, or replace ordinary IV risk reports.

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
`lambda >= (1 + abs(rho))/4`. For `abs(z) <= 0.5`, V1 evaluates the degree-14
series `sum(n=0..14, (-z)^n/(n+2)!)` and its analytic derivative using this
fixed backward Horner order:

```text
value = 1
derivative = 0
for divisor = 16, 15, ..., 3:
    derivative = -(value + z*derivative)/divisor
    value = 1 - z*value/divisor
phi = 0.5*value
phi' = lambda*(0.5*derivative)
```

Above that cutoff it uses the displayed exponential formulas. Evaluation does
not use FMA. The branch and Horner order are fingerprinted policy. The
[SSVI phi validation record](../../design/validation/mutation-ssvi-phi-2026-10-25.md)
documents the cancellation defect and the correction from the former `1e-4`,
degree-6 policy; its historical measurements remain unchanged.

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

At an exact slice time, the parameter values reproduce that slice. Derivatives
use the interval to its right; at the final slice they use the terminal theta
slope and zero slopes for `psi` and `rho_psi`. A knot is reported as `Knot`,
even when its derivatives follow an extrapolation rule. These one-sided
derivatives need not equal a centered difference across the knot. The
[parameter mutation record](../../design/validation/mutation-essvi-parameters-2026-10-27.md)
checks this convention and the immediately adjacent binary64 times.

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

### 6.1. Actual simulation steps versus the original variance grid

The native `LocalVolPath::reverse_state_adjoints` retains a `LocalVolStepAdjoint`
for every actual simulation interval, in forward time order, before aggregation
onto the original grid. Entry `i` belongs to the plan's interval `[t_i,t_(i+1)]`,
including the initial interval. With `a_i` the effective interpolated variance:

```text
A_variance_i = adjoint(S_(i+1))*S_(i+1)*(-dt_i/2 + G_i*sqrt(dt_i)/(2*sqrt(a_i)))
A_volatility_i = adjoint(S_(i+1))*S_(i+1)*(-sqrt(a_i)*dt_i + G_i*sqrt(dt_i)).
```

The second expression is paper equation (10). Its reverse state adjoint includes
all future state dependence and fixing seeds. Both quantities are retained in
`LocalVolReverseAdjoints::step_adjoints()`. Nonfinite input seeds or propagated
adjoints produce typed errors. Existing state, shock and grid-adjoint operations
keep their arithmetic order for finite valid inputs.

`step.accumulate_local_volatility_nodes(row)` deposits `A_volatility_i` onto
the spatial hats of that one interval, without time-interpolation weights and
without hat-area division. Storage is sparse (two spatial weights per step);
the caller only materializes a row when needed. Path averaging must retain the
fixed-block compensated reduction convention. Density, variance samples and
`delta_t` for equation (11) must use the actual step's source time and coordinate,
including event-inserted times absent from the original grid.

For comparison with an original grid node's volatility `sigma_ab=sqrt(a_ab)`,
the chain rule is instead

```text
dP/dsigma_ab = sum_i A_volatility_i * time_hat_a(t_i) * space_hat_b(x_i)
                        * sigma_ab/sqrt(a_i).
```

On nonflat grids, simply aggregating the effective-volatility step sensitivities
would omit this ratio. The retained variance adjoints reproduce the original
grid-variance reverse exactly by its original reverse-time deposition order.
The time-zero sensitivity is retained but cannot be passed to pointwise density
division. See the [step-adjoint validation](../../design/validation/local-vol-step-adjoints-v0.1.md).

These are transition sensitivities. Direct payoff dependence on variance, such
as a continuous barrier bridge's variance seeds, is retained in a separate
actual-node tape (Section 6.3), not in the transition tape. Production
VegaKT buckets still use the diagnostic original-grid allocation; the new tape
must not be described as an accepted paper decomposition.

### 6.2. Step-local Monte Carlo statistics and recovery

`VegaKtStepRecoveryPlan` compiles effective variances and analytic density rows
at the actual positive source times, including inserted event times. Its grid
must cover the simulation horizon. The caller supplies the implied-variance
surface in the same forward coordinate and remains responsible for consistency
with the simulated model. This constructor does not certify a calibration.

`sample_from_path` validates the path's cached transitions against the compiled
simulation/grid and then reverses the supplied payoff state seeds. Initial state
must equal the plan's initial forward normalizer. A sample can only be reduced
by its originating plan or a clone sharing that plan. Samples retain a dense
step-by-spatial-node vector; memory scales with steps times spatial nodes for
each live sample and reduction block, not the total number of paths.

`estimate` uses the existing deterministic executor's fixed blocks, Neumaier
sums and balanced-tree centered moments. Independent pseudo-MC paths and means
of antithetic pairs are explicit sampling units. A pair is averaged **before**
accumulating moments and counts as one observation; the caller must generate
opposite shocks from the same draw. Zero observations are rejected. With one
observation, sample variance and standard error are absent, not zero. RQMC
point-level observations must not be passed as independent paths; replicated
RQMC estimation remains outside this API.

For node contributions `A_ij`, each sampling unit separately forms the row,
retained-domain, excluded-domain and all-step totals. Reducing these additional
channels preserves within-row and across-time covariance in total variances.
Adding individual node variances would be incorrect. The API returns pointwise
node moments and these total moments, not a full covariance matrix or covariance
with price.

Positive-time mean rows feed equation (11) with their own density, variance and
interval length. Gamma pointwise standard errors use the same positive linear
normalization applied to node standard errors. These errors are conditional on
the fixed surface, density and active domain; they exclude calibration error,
time/space discretization bias and the unresolved initial boundary. The initial
row's local Vega and moments are retained, with no Gamma or active-domain split.
The original `sample_from_path` remains transition-only. Explicit variance
seeds use the separate entry point and recovery restrictions in Section 6.3.

The [step-recovery validation](../../design/validation/local-vol-step-recovery-v0.1.md)
checks independent Decimal price-bump moments, antithetic units, deterministic
worker counts and a 4096-pair CRN price-bump comparison. This component is not
yet connected to production IV bucket construction.

### 6.3. Direct payoff variance dependence, including bridge endpoints

`LocalVolLogEulerPlan::reverse_path_with_variance_seeds` takes partial payoff
derivatives with respect to states (holding effective node variances fixed) and
effective variances (holding states fixed). Both arrays have N+1 entries, not N
interval entries. Variance/state feedback is added exactly once before reverse:

```text
state_seed_i += direct_variance_seed_i * partial_x(a(t_i,x_i)) / S_i
direct_volatility_adjoint_i = 2 * sqrt(a(t_i,x_i)) * direct_variance_seed_i.
```

The result contains transition adjoints with this state feedback, a separate
direct-node tape with actual maturities, and the combined original-grid variance
adjoints. Spatial hats deposit each direct contribution on its own node's row.
Transposition onto original grid nodes also requires time hats, and conversion
to nodal Vol requires the nodal/effective volatility ratio as in Section 6.1.
The combined effective-volatility sum is therefore not generally the sum of
original-grid nodal Vol derivatives for a nonflat surface.

The native exact and smoothed continuous-barrier callers now supply endpoint
variance seeds to this reverse. A bridge uses the average of its left and right
effective variance. Its direct risk can occur at the last monitored node,
including maturity. That node has no outgoing simulation interval, so it cannot
be assigned an invented `delta_t` or merged into the previous transition row.

`sample_from_path_with_variance_seeds` feeds this reverse into deterministic
statistics. `direct_variance_rows()` exposes all N+1 direct rows;
`combined_total()` reduces transition plus direct contributions per sampling
unit, preserving their covariance. Existing `total()` remains the transition
contribution (including state feedback). Antithetic pairs retain the same
one-pair/one-observation convention.

Any nonzero input direct seed sets `has_unrecovered_direct_variance()`. In that
case **all Gamma means and standard errors are absent**, because the direct
contributions' equation-(11) interpretation has not been validated. Presence is
tracked independently of signed sensitivities, so cancellation between paths or
within an antithetic pair cannot restore Gamma recovery. All-zero direct seeds
preserve the transition-only behavior. This is a guard for an incomplete
decomposition, not an assertion that direct risk is an excluded-domain tail.

The [direct-variance validation](../../design/validation/local-vol-direct-variance-v0.1.md)
compares 215 path derivatives and 80 statistical price derivatives to independent
Decimal bridge prices, including terminal risk and shortened monitoring.

### 6.4. Conditional repricing of the initial Gamma row

`VegaKtInitialGammaPlan` obtains **additional conditional price information**;
it does not invert or redistribute the scalar first-step local Vega. It fixes
`S1 = K` at the first positive simulation node and replays only later intervals
with common remaining shocks at every strike and bump. The native continuation
returns `[S0, fixed S1, S2, ...]` with original fixing indices and no synthetic
first shock. For a deterministic discounted state-only payoff callback, define

```text
U1(K) = E[payoff(S0, K, S2, ..., SN) | S1 = K]
Gamma1(K) = d2 U1(K) / dK2.
```

The diagnostic uses five-point second price differences at `h = K * relative_bump`
and `h/2`, sharing seven prices per strike/path. Both estimates and their signed
difference are reduced per sampling unit. Antithetic pairs count once. A single
observation has no estimated variance or standard error. Stencil differences
are bump-sensitivity diagnostics, not rigorous bias bounds. With a kinked payoff,
Gamma is validated in expectation, not as a classical second derivative of an
individual nonsmooth Monte Carlo path. Reducing the bump can increase sampling
variance and floating-point cancellation.

The current validated scope requires constant forward, spatially flat variance
(each time row may differ), and at least two simulation intervals. Nonflat
piecewise-linear variance, changing forward, a one-interval terminal Dirac,
invalid strikes, and unrepresentable stencils are rejected. The callback must
not include direct variance/bridge dependence, affine-dividend reconstruction,
or a refitted exercise policy. No production caller silently selects this mode.
Its label is `initial_gamma_conditional_fd_diagnostic_v1`.

Equation (3) at time zero is a useful independent audit without density division:

```text
first_step_vega = integral Gamma1(K) * BS_Vega(S0, K, sigma0, dt0) dK
                = S0^2 * sigma0 * dt0 * frozen_Gamma_transition(Gamma1).
```

The implementation reconstructs this integral separately for every sample,
using the existing piecewise-linear Gamma operator. Its standard error therefore
includes cross-strike covariance. Constant extrapolation beyond the supplied
strikes is explicit: the result reports the two normalized tilted-kernel tail
masses and signed fine-stencil tail Vega contributions. These are not model
state-density tail probabilities or bounds on unknown Gamma-tail error.

The estimated first row connects to `vanilla_hedge_density_from_next`, then to
the existing equation-(9) projection. This composition does not certify the
next equation-(11) row or provide a hedge standard error: joint covariance
between the two estimates is still needed. See the
[initial-boundary validation](../../design/validation/local-vol-initial-gamma-v0.1.md).

### 6.5. Joint sample composition through the reporting-IV matrix

`VegaKtChainPlan` compiles `VegaKtInitialGammaPlan`, `VegaKtStepRecoveryPlan`
and a reporting basis into `flat_variance_paper_chain_diagnostic_v1`. Both
plans must use identical simulation and variance grids. The initial plan's
constant-forward / spatially-flat restrictions remain in force. Every positive
simulation time must be an exact reporting expiry; at least two active strikes
are required in each Gamma row. Densities and reporting IV must match the
**discrete** lognormal distribution obtained by summing the actual left-node
variance times interval lengths. The construction consistency tolerance is
relative `1e-10`, without an absolute floor; it does not alter acceptance policy.
Time-dependent variance rows are allowed when this discrete law is matched.

For one full shock vector, the native reverse evaluates state partials while
conditional initial repricing reuses the suffix shocks. The callback returns
discounted state-only price and state partials, and must be deterministic and
consistent. Explicit variance dependence, bridge risk, affine reconstruction,
and refitted policies are unsupported. They must not be hidden in the callback;
this API cannot establish the mathematical correctness of user-supplied seeds.

Each sample forms Gamma at `t1` from conditional price stencils and Gamma at
`t2..tN` from equation (11). Equation (4) uses next-Gamma propagation minus
current Gamma; the terminal density is minus Gamma. These linear operators are
compiled once. The piecewise-linear hedge density is integrated **in strike**,
splitting at both Gamma knots and reporting-IV knots. Midpoint quadrature with
`2q` subdivisions is used for the reported buckets; the signed difference from
`q` is retained per sample. Vega multiplies the measure before IV-hat allocation.

Path samples or antithetic pair means are composed before moment reduction.
Bucket and hedge standard errors therefore include initial/positive-Gamma
correlation. Scalar errors come from scalar sample totals, not sums of bucket
variances. Bump and quadrature differences have their own paired moments.
The API does not return a full bucket covariance matrix or price covariance.
One independent unit has no estimated standard error, zero units are rejected,
and independent RQMC-point standard errors are not supported.

The estimate retains finite strike bounds, recovery-source times, excluded
source-density probability, signed excluded **local** Vega, initial reconstructed
Vega minus native first-step AD (with joint uncertainty), initial kernel-tail
Vega, propagated constant-Gamma-tail IV risk, and signed left/right reporting
edge IV risk. Constant extrapolation in equation (4) is explicit. Equation (9)
omits the outside of each finite strike interval; its unknown contribution is
not estimated as zero. These quantities are diagnostics, not error bounds or
interchangeable volatility units. Neither residual normalization nor a new
numerical acceptance flag is introduced. See the
[joint-chain validation](../../design/validation/local-vol-vegakt-chain-v0.1.md).

### 6.6 Weak recovery numerical diagnostic

The explicit `PricingPlan.evaluate_vega_kt_chain_diagnostic` adapter uses the
compiled payoff, observation map and RNG. It subdivides initial/recovery grids
and applies integrated next-time `S^2`-weighted log-hat masses. The conditional
source-hat replacement is an approximation, not exact inversion of equation (3)
or the pointwise equation (11). All grids and density times are auditable.
See the [numerical record](../../design/validation/local-vol-weak-gamma-v0.1.md).
The original low-level point-density chain and ordinary pricing reports retain
their distinct labels; neither a zero projection residual nor full acceptance
is inferred from the weak diagnostic's 63 passing scalar comparisons.


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

The Rust component `recover_equation11_gamma_row` composes full-grid hat-area
normalization, the strike Jacobian, active-domain selection, and equation (11).
Its input is **integrated local-volatility adjoints for one time interval**,
not variance adjoints, density values, or an aggregate over several time steps.
Variance adjoints from the reverse sweep must first be multiplied by
`2*sqrt(effective_local_variance)`.

`Equation11GammaRow` retains the positive source maturity, explicit time step,
first-order recovery maturity `source + delta_t`, active strike/Gamma arrays,
and independently summed input/retained/excluded local Vega. Active-domain
indices refer to the original grid; hat areas must not be recomputed after
trimming it. Finite adjoints and positive variances are required even outside
the active domain. An unrepresentable recovery maturity is an error.

`propagate_to_source` applies Section 8's **frozen-variance** kernel over that same
stored time interval, with an explicit effective source variance. Source strike and the
recovered strike nodes must use the same drift-free coordinate. The method
continues the edge Gamma constantly; it does not reconstruct excluded local
Vega. A single active node can be inspected but cannot define this transition.
The singular `T=0` density cannot be represented by `AnalyticCallDensityRow`.

The recovery component alone does not define vanilla weights or IV allocation.
The separate paper operators in Sections 8.1 and 9 now supply those operations;
neither component is wired into public pricing results. The recovery component's
[independent price-bump and refinement checks](../../design/validation/local-vol-gamma-recovery-v0.1.md)
verify the specified positive-time recovery/propagation bridge only.

## 8. Discrete Gamma operator and cell integration

For current strike/spot coordinate `S`, variance `v=sigma_local(S)^2*delta_t`,
and integration strike `K`, first define the **partial derivative with v held
fixed**, used by the existing frozen-variance primitive:

```text
G(S,K;v) = partial_S^2 BS(S,K;v)  (v fixed)
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

Numerical correction (2026-10-03): the former minimum-of-two-differences rule
lost representable tail mass; the final-cell complement also erased small right
tails. The continuous kernel and piecewise-linear Gamma convention are unchanged.
The corrected finite-precision rules are:

- Use lower-tail probabilities for a negative normal interval, and survival
  probabilities for a positive interval. Never take the minimum of a stable
  tail difference and a difference between two rounded numbers near one.
- Evaluate the survival function from the existing central approximation for
  `0 <= z < 3`, and the 128-level backward Laplace continued fraction for `z >= 3`.
  The repository's shared CDF and other pricing paths retain their operation order.
- Compute near-unity log strike ratios with `log1p((K-S)/S)`. Preserve the drift
  shift separately from `log(S)` to avoid cancellation at tiny time steps.
- For normal interval width `h`, when `h*(1+max(abs(z_left),abs(z_right))) < 1e-3`,
  integrate the Gaussian midpoint expansion through order `h^4`. Obtain `h`
  directly from `log(b/a)/sqrt(v)`, not subtraction of two close standardized nodes.
- Compute both unbounded tail cells directly. Visit cells in increasing strike
  order and check `abs(1-NeumaierSum(all probabilities)) <= 8*EPSILON`. Apply only
  that roundoff correction to the largest-probability cell (last index wins a tie).
  A larger defect or nonfinite first moment is a typed error.

The closed-form linear-cell expression above is ill-conditioned for nearly equal
strikes. In the narrow-normal-interval regime above, or when `log(b/a) < 1e-3`,
normal width `< 16` and both standardized endpoints have absolute value `< 40`,
use eight-point Gauss-Legendre integration in standard-normal coordinates.
Panels number `max(1, ceil(h*(1+max(abs(z_left),abs(z_right)))))`. Reconstruct the
linear-strike weight as `expm1(log(b/a)*u)/expm1(log(b/a))`, avoiding differences
of nearly equal strike values. Positive quadrature weights preserve a nonnegative
Gamma integrand; infinite tails still use the nearest constant Gamma.

The independent 120-digit Decimal cases and validation boundaries are recorded in
the [Gamma numerical correction](../../design/validation/local-vol-gamma-numerics-v0.1.md).
Definitions of the error function and its continued fraction are given by
[NIST DLMF §7.2](https://dlmf.nist.gov/7.2) and [§7.9](https://dlmf.nist.gov/7.9).

### 8.1. Paper equation (5): total spot derivative

The user-supplied May 2022 paper, pages 3 and 6 (Appendix A), places the local
variance `v(S)` inside the spot derivative. For a nonflat smooth local variance,
the frozen kernel above is not the full equation-(5) operator:

```text
H(S,K) = d_S^2 BS(S,K;v(S))
       = BS_SS + 2*BS_Sv*v_S + BS_vv*v_S^2 + BS_v*v_SS.
```

`integrate_local_gamma_total_variance_transition` accepts the explicit jet
`[v, v_S, v_SS]` at a fixed interval. With `a=S*v_S`, `b=S^2*v_SS`, and
`z=(log(K/S)-3*v/2)/sqrt(v)`, it integrates Gamma against

```text
H = G * (c0 + c1*z + c2*z^2)
c0 = 1 + 2*a + b/2 + a^2/2 - a^2/(4*v)
c1 = (a + 3*a^2/4)/sqrt(v)
c2 = a^2/(4*v).
```

The total kernel can be signed. Only `q`, not `exp(-v)*H`, is a probability
density. Constant-Gamma integration gives the independent quadratic-payoff
identity `exp(v)*(1+2*S*v_S+S^2*(v_S^2+v_SS)/2)`.

For a normal interval `[l,h]`, use moments `J0=P(l<Z<h)`,
`J1=phi(l)-phi(h)`, and `J2=J0+l*phi(l)-h*phi(h)`, taking infinite boundary
fluxes as zero. Strike-weighted moments use the shifted interval
`[l-sqrt(v),h-sqrt(v)]` and factor `S*exp(2*v)`. Finite cells with endpoint
absolute values less than 40 and either normal width less than 1, or log-strike
width less than 0.01 with normal width less than 16, use the same eight-point quadrature
and panel rule above with the polynomial included. The wider quadrature regime
avoids cancellation between polynomial moments. The endpoint bound prevents
unbounded panel counts for remote short-step cells. Exactly zero derivatives
delegate to the frozen operator, preserving its binary64 result.

Equation (4) defines **offsetting vanilla hedge density**, not position Vega:

```text
Delta(t_k,S) = integral Gamma(t_(k+1),K)*H_k(S,K) dK - Gamma(t_k,S)
Delta(t_n,S) = -Gamma(t_n,S).
```

`Equation11GammaRow::vanilla_hedge_density_from_next` checks that the next
row's source maturity equals the current row's recovered maturity. Its jets
cover the next row's interval. It returns density per unit strike at the current
active nodes; quadrature weights must still include `dK`. The terminal method
returns negative Gamma. Active-tail extension remains an approximation and
does not recover excluded local Vega. See the
[paper operator validation](../../design/validation/local-vol-vegakt-paper-v0.1.md).

The paper assumes smooth local volatility. Piecewise-linear variance in
log-strike has derivative jumps at knots, so setting curvature to zero there
is not a justified discretization. A weak/discrete derivative convention or a
validated smooth reconstruction is required before runtime integration.

## 9. Reporting-IV basis and units

The reporting basis samples canonical `f` implied volatility at explicit
`(T,k)` nodes. Inside a rectangle it is reconstructed by bilinear basis weights,
with time weight formed first and strike weight second. At a reporting expiry
`T_i`, paper equation (9) includes a **ratio of Black-Scholes Vegas**:

```text
hedge_quantity_ij = integral Delta(T_i,K)*hat_ij(K)*Vega(T_i,K) dK / Vega(T_i,K_j)
raw_position_risk_ij = -hedge_quantity_ij*Vega(T_i,K_j)
                    = -integral Delta(T_i,K)*hat_ij(K)*Vega(T_i,K) dK.
```

`project_vanilla_hedge_measure_to_reporting_iv` accepts integrated hedge weights
(`Delta*dK`, or Dirac masses), multiplies by continuum Vega **before** transposed
IV interpolation, and divides by nodal Vega to report hedge quantities. An
owned unit call at an exact node produces positive raw Vega and hedge quantity
`-1`. Multiplying the result of unweighted interpolation by nodal Vega would be
incorrect off-node. The API uses `K=S0*exp(k)` in the paper's zero-carry unit;
affine-market conversions must be explicit at the caller.

This API requires an exact maturity node: paper equation (9) only specifies
strike interpolation at that expiry. Off-node maturity allocation requires an
additional validated convention and is rejected here. Strike contributions
outside the reporting range go to the nearest edge and retain edge counts and
signed risk. A nonzero risk with zero/underflowed nodal Vega is an error. Zero
projection residual certifies allocation of the supplied finite measure only;
it is not evidence of zero initial-time, recovery, or tail-truncation error.

Outside the restricted terminal-measure scope, the production fallback
`local_vol_node_allocation_diagnostic_v1` still directly
allocates local-grid adjoints. It does not implement these paper operators and
remains unvalidated. In particular, aggregated grid-row adjoints cannot replace
equation (10)'s simulation-step adjoints when multiple steps share one grid row.
The reverse sweep now retains those separate step adjoints (Section 6.1), and
the step-recovery component aggregates them with uncertainty (Section 6.2), but
pricing's bucket construction does not yet consume them. At `t=0`, `d(0,S)` is a
Dirac measure, so the scalar first-step local Vega cannot
be divided pointwise to recover a whole Gamma row. The supplied paper does not
provide an operational boundary recovery for that singular case. Section 6.4
adds a separately labelled conditional repricing diagnostic for a restricted
model class; the local-node fallback does not use it. Eligible terminal-measure
pricing instead compiles signed payoff kinks, uses exact semigroup cancellation
at intermediate maturities and projects same-path terminal IV derivatives at
their strikes. Its pre-projection total is independent of both reported buckets
and local parallel AD, and its affine remainder has exact zero IV risk in the
validated zero-carry scope. General
boundary treatment and the remaining decomposition issues must be resolved
before the native acceptance guard can be removed.

Bucket order is row-major `[maturity][log_moneyness]`. Raw VegaKT is currency per
unit absolute volatility (`sigma=1.0`). Market-scaled VegaKT is raw times `0.01`
and is currency per one volatility point. Canonical Scalar Vega is the
deterministic sum of in-domain raw buckets; residual is not silently included.

Each bucket carries its variance and covariance with Price. Full bucket
covariance is absent unless explicitly requested. All sums use the accepted
fixed-block Neumaier and balanced-tree reduction policy.

## 10. Affine dividends

The simulated coordinate is escrowed. With the entire future schedule funded,
`A(0-)=Ainit/S0` and `B(0-)=(S0-Ainit)/S0`; require `S0>Ainit`.
A carries at deterministic r-q between events. Its post-event value is the
remaining cash reserve divided by S0. This convention applies to BS, LV and
LSV. IV inputs describe the escrowed equity coordinate.

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

No later performance result may weaken one of these calculation specifications.

## References

- [Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2033323)
- [Corbetta et al., *Robust calibration and arbitrage-free interpolation of SSVI slices*](https://arxiv.org/abs/1804.04924)
- [Adrien et al., *Vega KT for the Local Volatility Model: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4107770)
- [Guyon and Henry-Labordere, *The Smile Calibration Problem Solved*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032)
- [Dupire, *Pricing with a Smile*](https://www.risk.net/derivatives/equity-derivatives/1500211/pricing-with-a-smile)
- [Black and Scholes, *The Pricing of Options and Corporate Liabilities*](https://doi.org/10.1086/260062), for the Black–Scholes gamma kernel and constant-volatility limit.

The SSVI/eSSVI formulas, Dupire transformation, VegaKT operators and affine
dividend matching are linked to the corresponding implementation sections
above. Grid generation, floors/caps, branch ownership and finite precision
policies remain repository-specific contracts.


## Conditional initial Gamma connection (2026-10-05)

The [initial-measure diagnostic](../../design/validation/local-vol-initial-gamma-measure-v0.1.md)
uses the compiled conditional payoff, with future Gaussian increments fixed, to
obtain a signed Gamma measure at the first positive simulation time. It integrates
the first flat-variance transition exactly. Fixed history and valuation-date
observations are constants; future observations retain their own multipliers.
The initial scalar AD value is used only for an independent same-draw comparison,
not to identify or normalize the measure. Display bins retain both tails and do
not relocate atoms before calculating Vega. Positive-time regularization and
complete path-dependent equation-(4)/(9) acceptance remain separate open work.


## Split-sample regularization selection (2026-10-05)

The [Gamma recovery audit](../../design/validation/local-vol-gamma-selection-v0.1.md)
selects a fixed candidate inverse on disjoint pilot counter indices and validates
it once on the original pricing range. The original signed Tikhonov operator is
unchanged. Selection uses both reconstruction error with an estimated MC norm
margin and relative Gamma MC error, across every positive-time row. The report
retains every candidate and any held-out failure and includes the separate
conditional initial measure. Its positive-time policy status does not establish
full path-dependent market-IV risk or cross-time covariance.

## Signed-atom equation-(4)/(9) connection (2026-10-05)

The [atom-chain diagnostic](../../design/validation/local-vol-atom-chain-v0.1.md)
uses current Gamma minus backward-propagated next Gamma for position IV risk.
For source variance V, next variance v, current maturity t and a unit next atom K,
continuum IV Vega times propagated Gamma integrates to
`S0*phi(d1(V+v,K))*sqrt(t*V/(V+v))`. Its log-strike hat probabilities have mean
`V/(V+v)*log(K/S0)` and variance `V*v/(V+v)`. Thus projection covers the full
positive domain without strike quadrature or moving the initial atoms to bins.
Joint sampling preserves initial/positive-time and cross-maturity covariance.
Reporting IV must match discrete cumulative variance; a held-out recovery failure
withholds the estimate. These operator/statistical checks do not establish
independently recalibrated market-IV bucket sensitivity for path payoffs.

## Equality-constrained Gamma inverse (2026-10-05)

The optional [conservative inverse](../../design/validation/local-vol-conservative-gamma-v0.1.md)
minimizes the original Tikhonov objective subject to `1'Bq=1'b`. With
`c=B'1`, `u=R1`, its fixed recovery matrix is `R + u*(1'-c'R)/(c'u)`.
The constraint is compiled independently of payoff samples; signed atoms,
spatial residuals and updated inverse-noise diagnostics remain observable.
Disjoint pilot selection and held-out quality checks apply to the constrained
operator itself. The default unconstrained inverse and its historical outputs
remain unchanged. Flat maturity quote shifts are tested by rebuilding every
forward interval from differences of market cumulative variances.

## Finite two-step smile quote calibration (2026-10-05)

The companion [discrete calibrator](../../design/validation/local-vol-discrete-smile-calibration-v0.1.md)
fixes the first lognormal marginal and fits terminal quote prices as
`E[Black(S1,K,variance_1(log(S1/S0))*dt_1)]`. The variance grid uses the same
piecewise-linear interpolation and clamped spatial tails as the native process.
Newton increments minimize Euclidean norm in annual local-variance coordinates;
full quote-row rank and residual reduction are required, without floor/cap
clipping. A doubled quadrature order and an analytic omitted-tail price bound
confirm the fit. The model extension between and beyond quote strikes is not
uniquely identified by those quotes.

At a flat baseline the quote-IV Jacobian is
`dIV_T/dvariance_(1,j) = dt_1/(2*IV_T*T) * E[hat_j(X1) | X_T=log(K/S0)]`.
The Gaussian bridge has mean `V1/V_T*log(K/S0)` and variance
`V1*(V_T-V1)/V_T`. Its minimum-norm right inverse U defines a reproducible
calibration direction. For fixed first marginal, local AD gives quote risk
`U' b/(2*sigma_1)`; recovered Gamma gives `U' Bq/(2*sigma_1)`. Their difference
is the actual projected spatial residual `U'(Bq-b)/(2*sigma_1)`. Integral
conservation alone does not force this quote residual to zero. The validation
record keeps quote-Gamma failures separate from successful price calibration.

## Simultaneous local-Vega direction constraints (2026-10-05)

The optional [direction-constrained inverse](../../design/validation/local-vol-gamma-quote-conservation-v0.1.md)
solves the same Tikhonov objective subject to `D'Bq=D'b`, including the total
direction in D when requested. Given the unconstrained `R=H^-1 B'`,
`Q=R D`, `C=B'D` and `G=C'Q`, the fixed inverse is
`R_D=R+Q G^-1 (D'-C'R)`. All directions are fixed from market/model data before
payoff samples are observed. Columns of D are normalized and G is equilibrated
before a Cholesky solve. A pivot <=1e-10 or a normalized matrix constraint
residual >1e-9 rejects the solve; constraints are not dropped or clipped.
Dependent directions, including a duplicate total, are rejected.

Signed masses and every spatial residual are retained. Noise gain and resolution
trace are recalculated from R_D. Direction projections are computed on each path
before antithetic pairing and reduction, so their SE includes the source-node
covariance. Every pilot and the single held-out validation use the same declared
directions. The atom-chain estimator rebuilds exactly the selected constrained
inverse and its storage budget includes the extra direction statistics. Neither
the constraints nor successful calibrated quote tests identify the entire Gamma
measure or establish general nonflat market-IV acceptance.


### 2区間・非平坦な第2区間のGamma回復

`PricingPlan::evaluate_vega_kt_two_step_smile_gamma_audit` は第1分布を固定した
2区間モデル専用の診断で、通常のVegaKT出力を置換しない。
第1区間の年率分散は空間一定とし、`X1 ~ N(-V1/2,V1)` を厳密に用いる。
第2区間の年率局所分散 `v(x)` は実際のgridに従う線形補間・境界一定延長である。
終端分布をGaussianに置き換えない。

入力は実際の経路ADの年率分散節点感応度 `b_j = dP/dv_j`。
非平坦な場合、実効局所ボラ感応度を各節点の `2 sigma_j` で割る変換は使えない。
作用素は `B[j,k] = E[hat_j(X1) S1 phi(d1) dt / (2 sqrt(v(X1)dt))]`、
`d1 = log(S1/K_k)/sqrt(v(X1)dt) + sqrt(v(X1)dt)/2` とする。
sourceの各区間を32点と64点のGauss–Legendreで積分し、最大差と±12標準偏差外の
解析的な各要素上界を別々に確認する。32/64点の差は収束の診断であり、厳密な総誤差上界ではない。
平坦極限では `B_variance = B_volatility/(2 sigma)` に一致する。

較正方向 `U = d(local annual variance nodes)/d(terminal quote IV)` を指定する場合、
`variance_conservation_directions = U.T` とする。追加の `2 sigma` 換算は不要。
逆問題は `D B q = D b` を同時に満たす固定の線形制約付き正則化で、
pilotの最強合格候補をheld-out標本で一度だけ評価する。
同一経路で方向別感応度を合計してからantithetic pairing・標本分散を計算するため、
方向内の節点相関を保持する。全方向間の共分散を返すものではない。

非平坦な将来遷移に対して既存のaffine continuationによる初期Gamma回復は適用できない。
初期年率分散ADを `unrecovered_initial_variance_sensitivity` として残し、
`initial_gamma_recovery='not_implemented'`、全面IV受入は `not_established` と明示する。
任意の第1満期スマイル、多区間、キャリー・配当・バリア・早期行使は対象外。
詳細は[検証記録](../../design/validation/local-vol-nonflat-gamma-v0.1.md)を参照。


### 非平坦な将来を持つ初期Gamma（2区間）

`evaluate_vega_kt_initial_smile_gamma_audit` は、前節の第2区間診断から独立した追加API。
第1区間が一定年率分散の2区間モデルで、第2区間の局所分散を固定する。
第1状態sを固定した条件付きペイオフを、実際のcompiled graphから終端状態の
連続区分線形関数へ変換し、状態依存の第2分散によるBlack積分で継続価格G(s)を求める。
G(s)そのものを物理spotで区分線形補間した傾きの差を、符号付きGamma原子として保持する。
局所分散節点・報告bin境界を格子へ含め、すべての既存区間を半分にして収束を検査する。

第1分布は `log(S1/S0)=-V1/2+sqrt(V1) Z`。
Gamma原子を第1区間のBlack年率分散感応度で積分した結果を、独立なscore積分
`E[G(S1) * t1 * ((Z²-1)/(2V1)-Z/(2 sqrt(V1)))]` と照合する。
標本の価格やADからGammaを再調整することはない。定数・線形項も価格復元へ保持する。

compiled payoffから `|G(s)| <= A+B s` を認証する。±12標準偏差外について、
価格にはGaussian尾確率、年率分散微分には2次モーメントの解析的上界を用いる。
Gamma質量自体は有限領域のみの近似であり、領域外の質量をゼロと認定しない。
32/64点積分差と格子細分化差は収束診断であり、厳密な全近似誤差上界ではない。

決定的な回復判定と、native価格・初期年率分散ADとのMC判定を分離する。
どちらかが失敗すれば `estimate=None`。成功時も返すのは第1年率分散の感応度であり、
第2満期quoteを固定して再較正する第1市場IV bucketではない。
[範囲・精度・検証](../../design/validation/local-vol-initial-smile-gamma-v0.1.md)。

### 第1満期parallel IVと終端quote IVの共同回復

`evaluate_vega_kt_joint_gamma_audit` は上記2つの回復を結合する追加診断。
native入力の各readoutは `[uniform_first_annual_variance, second_variance_nodes...]`。
市場較正はnativeでは認証せず、companion `evaluate_joint_market_gamma` が同じrequestを
検証・compileし、`TwoStepSmileCalibrator.joint_iv_tangent()` を設定する。

終端IVの年率局所分散接線をU、終端callの第1年率分散微分をh、終端Black VegaをV、
第1IVをsigma0とすると、第1IVを動かして終端quoteを固定する接線は
`a=2 sigma0`, `w=-a h/V`, `u0=U w`。
共同接線は `[[a,0],[u0,U]]`。hは初期Gaussianのscore積分で求め、32/64点差・
解析的tail上界・終端quote応答 `[0,I]` を検証する。第1満期を動かしたモデルでは
第1rowを一定 `(sigma0+shift)^2` とし、第2rowを同じminimum-norm Newtonで再較正する。

第2Gammaの独立保存制約はU.Tのみ。u0はUの列空間に属するため追加制約にしない。
従属する方向はreadoutとして許可する。方向jの回復値は
`a_j * initial_Gamma_variance + W_j * B R b_second`、
native対照は `a_j * b_first + W_j * b_second`。
全方向のnative・回復・差を同一経路で作成し、antithetic平均の後に中心化共分散を計算する。
固定順序のブロック還元でworker数に依存しない。共分散は標本単位で、平均の共分散はNで割る。

初期または第2回復が不合格なら共同推定を返さない。共同判定は各方向について
`abs(mean difference) <= 6 SE + abs(a_j)*initial functional tolerance + roundoff allowance`。
roundoff allowanceは `256 epsilon * (abs(native mean)+abs(recovered mean)+native SD+recovered SD)`。
deterministic初期誤差をMC標準誤差に混ぜない。第2区間のpilot選択・3%空間・25%Gamma SE基準は維持。
同時信頼区間や一般スマイルの全面受入は主張しない。
[共同較正と価格差分の検証](../../design/validation/local-vol-joint-smile-gamma-v0.1.md)。

### 3区間の第3分散row回復

`from_two_transition_variance_source` は実際の第1・第2遷移を積分したsource密度で
第3区間のBlack年率分散作用素を構成する。第2分散rowが非平坦ならsourceも非Gaussianとなる。
sourceの確率総量・株価第1モーメント・作用素について32/64点の収束を検査し、再正規化しない。
総積分分散上界AによるChernoff boundと第1積分のGaussian tailを別々に扱う。
2..65 source節点・2..65 Gamma節点に限定し、精度不足の狭い遷移は拒否する。

`evaluate_vega_kt_three_step_terminal_gamma_audit` は実際の3区間gridと一定第1分散を要求する。
先行2区間を固定して較正した終端IV接線U.Tを保存制約に設定できる。
回復対象は第3分散rowのみで、初期・中間Gammaや全満期chainをこの結果から推論しない。
[数値受入・範囲](../../design/validation/local-vol-three-step-gamma-v0.1.md)。

### 中間Gammaと中間・終端quoteの接続

第1状態S1を固定した継続価格G(S1,S2)のS2方向のGammaを回復し、条件付き第2遷移の
Black年率分散核で積分する。native補間hatはS1に評価する。第1状態の密度積分で第2rowの
感応度を得るため、履歴平均Gammaに無条件のBlack核を掛ける近似は使わない。
4水準のPWLに2段階Richardson補正を施し、bin境界の一次誤差と滑らかな汎関数の二次誤差を除く。
範囲は3区間、第1row一定、局所2..33節点、bin境界1..33点。

中間と終端のquote接線をU2,U3、終端価格の中間分散微分をH、終端quote VegaをV3とすると、
結合接線は `[[U2,0],[-U3 diag(V3)^(-1) H U2,U3]]`。
中間quote変更後の終端quote再較正を含み、第1rowは固定する。
終端の独立保存制約はU3だけとし、補償列を依存readoutとして評価する。
[数値受入・独立照合](../../design/validation/local-vol-middle-smile-gamma-v0.1.md)。

### 3区間の初期Gammaと第1満期IVの補償

初期継続価格を `F(s1)=E[payoff(s1,S2,S3)|S1=s1]` と定義する。
第1fixingを含む履歴と、第2遷移の分散 `v2(log(s1/S0))` の双方を動かす。
第3遷移をcompiled PWL payoffのBlack積分で消去し、第2遷移を条件付きGaussianで積分する。
初期状態の±12標準偏差をN,2N,4N,8Nで細分化し、物理spot方向の傾き差を符号付きGamma原子にする。
隣接3水準に `(a-6b+8c)/3` を適用する。binの切替に負の原子があっても削除しない。
原子を第1区間のBlack年率分散kernelで積分し、最初のrowの平行年率分散感応度を得る。
有限windowのbin質量と全領域のGamma質量は区別する。収束差は厳密な総誤差上界ではない。

第1平行IVの変化に対する第1年率分散接線をa、中間の補償をbとする。
終端価格の第1分散微分をH0、第2row微分をH2、終端quote VegaをV3、終端単独接線をU3とすると、
第3rowの補償は `c=U3[-diag(V3)^(-1)(H0 a+H2 b)]`。
全接線の第1列は `[a,b,c]`、残りは既存の中間・終端接線である。
独立制約U3だけで終端Gammaを回復し、最大17方向（1＋中間8＋終端8）の読み出しを許す。
追加方向も候補逆作用素のすべてで保存されることを検査する。
[数値受入・誤差の扱い](../../design/validation/local-vol-initial-three-gamma-v0.1.md)。


### 可変区間数と終端条件付きVega

`from_multi_transition_variance_source` は2..16区間の実際のGaussian遷移密度を順に積分する。
第1遷移は `v0(0)`、以降は状態依存分散を用い、最後の区間のsource密度を構成する。
各積分cellを最小遷移標準偏差の4倍以下に分割し、32/64点の確率・martingale・
作用素の収束と累積tailを検査する。再正規化やモーメント近似は行わない。

新しい終端診断はprefix全履歴を固定し、PWL payoffの各符号付きGamma原子をBlack積分する。
年率分散微分 `sum mass*S*phi(d1)*dt/(2*sqrt(v*dt))` をsource位置のhatに配分してから、
保存制約付き逆作用素を適用する。線形部分のVegaは0で、最終乱数に由来する相殺ノイズを除く。
通常native価格と旧Gamma APIはそのまま保持する。
先行rowを固定した終端IV較正だけを接続し、全満期のGammaとは扱わない。
[式・領域制約・独立照合](../../design/validation/local-vol-multi-step-gamma-v0.1.md)。


### 選択した分散rowの条件付きGamma連鎖

過去履歴と後続乱数を固定した `F(S_(r+1))` を、後続状態・局所分散をともに動かして構成する。
終端payoffは解析Black積分。物理spotのPWL傾き差を符号付きGamma原子とし、
現在の遷移のBlack年率分散核とsource hatで節点感応度へ写す。
N・2N・4Nの入れ子格子から粗・細のRichardson測度を作り、パスごとの絶対細分化差を検査する。
各条件付きwindowのGamma binは履歴・後続乱数で平均し、無条件全領域のGammaとは区別する。

各市場満期直前のrowを較正する延長規則では、選択rowの接線は下三角になる。
実際の遷移密度微分から価格Jacobianを作り、先行quoteの変更が後続quoteを保持するよう補償する。
先行・終端のGamma寄与を同一経路で合成し、方向別SEに相殺と相関を含める。
[式、誤差診断、独立価格差分](../../design/validation/local-vol-path-gamma-chain-v0.1.md)。


### 条件付きGamma連鎖の全方向共分散

選択rowの局所分散感応度を各読み出し方向に同一経路内で合算し、
`X = [native_AD, recovered_Gamma, recovered_Gamma-native_AD]` を作る。
antitheticの場合は `X` のペア平均を1独立標本とし、中心化した標本共分散を
`N-1` で除して求める。平均の共分散はさらにNで除する。
固定blockと固定二分木の還元により、既存の平均・分散を保持する。

D方向に対する行列は `(3D) × (3D)`、行優先。上三角の `3D(3D+1)/2` 成分を
集計し、既存の統計量と合わせてblock数倍が400万成分を超える場合は実行前に拒否する。
これは同一商品のMC誤差であり、回復格子・較正・モデルの誤差や異なる商品の共分散を含まない。
詳細は[受入記録](../../design/validation/local-vol-path-covariance-v0.1.md)。


### 中間区間のまれな支払いと条件付きnative AD

可変区間Gamma診断は、最終遷移を解析積分したnative ADとの同一標本差と、
raw AD−条件付きADの差を別々に検査する。後者は解析的な条件付き分散をSE評価に含める。
raw ADが全標本で0でも、不確実性を0とは扱わない。従来のGamma・raw AD・共分散は保持し、
旧直接照合の結果は `raw_node_validation_passed` に残す。
[数式・検証・適用範囲](../../design/validation/local-vol-conditional-ad-v0.1.md)。
