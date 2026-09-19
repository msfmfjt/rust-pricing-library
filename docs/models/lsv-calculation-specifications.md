# LSV calculation specifications

Status: Implemented experimental numerical boundary, 2026-09-12.
## 1. Coordinates and stochastic factor

Use the continuous martingale f, starting at S0, and the existing deterministic
affine dividend map

`S(t) = A(t) * S0 + B(t) * (F_cont(0,t)/S0) * f(t)`.

`F_cont(0,t)=S0*P_q(0,t)/P_r(0,t)` supplies continuous carry. The shared LV
payoff adapter expects this carried coordinate; LSV scales the martingale and
its payoff adjoints before using that adapter. Dividend A/B coordinates alone
do not include continuous carry. This correction is covered by the
[extended-model price panel](../../design/validation/extended-model-accuracy.md).
The 1F/2F/rough LSV plan fingerprint domains are version 2; calibrated paths and
their RNG layouts are unchanged, but nonzero-carry prices and risks are corrected.

The LSV dynamics are

`df/f = L(t,f) * a(t) * dW`,
`a(t) = exp(nu * X(t))`,
`dX = -k * X dt + dW_vol`, `X(0)=0`, `corr(dW,dW_vol)=rho`.

Require finite `k>=0`, `nu>=0`, and `-1<=rho<=1`. The deterministic normalization
of a(t) cancels in `a / sqrt(E[a^2 | f])`, as in the second paper's (2.7).
No separate forward-variance-curve calibration is inferred from Vanilla smiles.

Our `L` and `sigma_Dup` are relative volatilities in `df/f`. The second paper's
`sigma` and `sigma_loc` in (2.1)-(2.3) are additive volatilities in `df`:

`sigma_abs(t,K) = K * L(t,K)`,
`sigma_Dup_abs(t,K) = K * sigma_Dup(t,K)`.

All grid abscissae are `x=log(f/f0)`, with deterministic f0 held fixed by the
target-grid VJP. Dividend dates update the affine map, not f or X. Contractual
observations at expiry/ex-date collisions use the existing post-event rule;
Barrier jump observations retain both sides. No jump consumes a random number.

## 2. Exact OU joint law and log-Euler step

For h>0, let `d=exp(-k*h)`, `q=(1-exp(-2*k*h))/(2*k)` and
`c=rho*(1-exp(-k*h))/k`. At k=0, take `q=h`, `c=rho*h`.

With independent standard normals z1,z2:

`X_next = d*X + (c/sqrt(h))*z1 + sqrt(q-c*c/h)*z2`.

Then `Var(X_next | X)=q` and `Cov(X_next-d*X, sqrt(h)*z1)=c`.
`expm1` stabilizes small k*h; a negative residual variance from rounding is
floored at zero. Invalid or unrepresentable transitions are errors. Reusing rho
as the correlation of the OU innovation and a whole-interval spot Brownian
increment fails this covariance contract when k>0.

The asset update uses the start-of-step factor:

`v = L(t,f)^2 * exp(2*nu*X)`,
`f_next = f * exp(-v*h/2 + sqrt(v*h)*z1)`.

Every result must be finite and strictly positive. There is no LSV variance cap
or floor hidden in this path step. The effective Dupire target grid already
carries its declared Local-variance floor/cap.

## 3. Particle calibration and extrapolation

The calibration stream is Philox `RandomDomain::LsvCalibration=4`; existing
stream identifiers retain their values. At n time intervals, independent normal
dimensions are `[spot_0,...,spot_(n-1), orthogonal_0,...,orthogonal_(n-1)]`.
Both factors change sign for a pricing antithetic. QMC Brownian bridge is applied
to each independent normal block before the exact OU correlation map.

For fixed explicit log bandwidth b, use the unnormalized compact kernel

`w_i(x) = (1-((x_i-x)/b)^2)^2` for `abs(x_i-x)<b`, and zero otherwise.

The common normalization cancels in

`m_p(t,x) = sum_i(w_i*a_i^p) / sum_i(w_i)`, `p=2,3,4`,
`ESS = sum_i(w_i)^2 / sum_i(w_i^2)`,
`L(t,f0*exp(x))^2 = sigma_Dup(t,x)^2 / m_2(t,x)`.

This kernel is in log-f coordinates; the 2011 paper's displayed implementation
uses an f-coordinate bandwidth and cubic spatial splines. Our versioned policy
is `lsv-quartic-log-f-left-time-v1`. Bandwidth and support threshold are explicit
inputs, and no Silverman helper silently chooses them. Bandwidth must be refined
with particle count in a convergence study.

At t=0, m2=m3=m4=1 exactly. At nu=0 these moments are one at every time/node and
the leverage equals the effective Dupire target. Elsewhere, nodes whose ESS is
below the explicit threshold use the moments of the nearest supported log-node;
equal-distance ties go to the lower index. The node's actual ESS, donor identity
and fallback flag are retained. If no node is supported the calculation fails.
The local target variance at the query node is retained when using donor moments.

Particles are sorted by `(log(f/f0), original particle index)`. Regression sums
use compensated reduction in that order. Spatial leverage interpolation is
linear in squared leverage with flat extrapolation. Time interpolation is
left-constant; an exact knot uses its own row. Pricing must include every
calibration knot inside its horizon. The facade inserts dividend/payoff nodes
into calibration before marching and interpolates the original target at them.

## 4. Discrete reverse contracts

For a cached step, with outgoing asset adjoint fbar_next, define

`b = fbar_next * f_next`,
`vbar = b * (-h/2 + z1*sqrt(h)/(2*sqrt(v)))`,
`L2bar = vbar * a^2`.

Accumulate spatial interpolation weights against L2bar and propagate its
log-coordinate derivative with `dx/df=1/f`. The OU reverse adds the factor
contribution `2*nu*v*vbar` and the exact OU transition's two shock loadings.
Seeds may occur at any payoff observation; an observed Spot seed becomes B(t)
times that seed in f. The public target-grid risk holds S0 fixed.

The calibration VJP differentiates every row and all preceding particles:

`target_var_bar += L2bar/m2`,
`m2bar -= L2bar*target_var/m2^2`,
`dm2/df_i = w_i'(x_i-x)/(b*f_i) * (a_i^2-m2) / sum_j(w_j)`.

Here `w_i'` means the derivative of `(1-u^2)^2` with respect to u; it vanishes
outside support and at its boundary. Moment adjoints from fallback cells flow
to their donor node. The frozen support/fallback choices yield an almost-
everywhere derivative; crossing an ESS selection boundary is non-smooth.
The VJP holds factor parameters, f0, grid axes, bandwidth and seed fixed.
The time-refinement interpolation is also transposed, returning adjoints on the
original **effective** target grid (after its construction-time floor/cap).

The method label is `lsv-discrete-particle-vjp-v1`. This is a derivative of the
realized finite algorithm; it includes calibration feedback but does not claim
the continuum/infinite-particle sensitivity in Theorem 4.2.

## 5. Normalization gates for future market-IV VegaKT

Two printed formulas need an explicit convention check before direct reuse:

- Differentiating (2.4) with respect to the additive leverage `sigma_abs(t,f)`
  introduces a `1/f` factor relative to differentiation with respect to our L.
  The right-hand side printed in (4.5) has the relative-leverage form. Our
  executable log-Euler chain rule is validated directly against finite differences.
- Under the variance-functional convention in (1.2), the LV limit gives
  `g_absvar(t,K) = 0.5 * mu(t,K) * Gamma(t,K)` in continuous time, and
  `g_absvar_row = 0.5 * mu * Gamma * delta_t` for a time-row parameter.
  Consequently `Gamma = 2*g_absvar_row/(mu*delta_t)`. The denominator `2*mu*delta_t`
  printed in (4.7) does not match that normalization. This discrepancy must be
  resolved with the LV limit, (3.1) and a unit-controlled reference fixture before
  connecting the approximate operator to reported buckets.

For our relative variance coordinate,
`g_relvar_row = K^2 * g_absvar_row`, and conversion to relative local-volatility
adjoints multiplies by `2*sqrt(target_var)`. Density conversion between strike
and log-moneyness additionally requires the strike Jacobian. No market-IV Vega
is produced by applying only this volatility/variance conversion.

## 6. Uncertainty and replay

Calibration particles interact; their payouts must not be treated as independent
pricing samples. Pricing always uses independent streams after calibration.
Pseudo-MC antithetic pairs are one sampling unit. RQMC uncertainty uses the
independent scramble means, not pointwise variance. Calibration uncertainty is
excluded from the returned standard errors and is explicitly labelled.

Fixed block sizes and deterministic vector reductions provide same-platform
worker-count reproducibility. `calibrate_bergomi_lsv_parallel` and the pricing plan
spread each calibration time step over the execution policy's workers in fixed
4,096-particle tasks. Every particle update and every node's kernel sum is
computed and reduced in the same order as in the sequential
`calibrate_bergomi_lsv`, so the calibrated surface is bit-identical for any
worker count. Price-only evaluation uses `BergomiLsvPlan::evolve_states`, which
writes the same f states as `evolve_path` without the reverse-mode step records. The LSV plan fingerprint extends the base plan
with the factor parameters, particle count, seed, bandwidth, ESS threshold,
trace policy and realized leverage. Stable LSV JSON migration and retained
native-platform replay fixtures are later acceptance work.

## References

- [Dupire, *Pricing with a Smile*](https://www.risk.net/derivatives/equity-derivatives/1500211/pricing-with-a-smile), for the call-surface-to-local-volatility target.
- [Bergomi, *Smile Dynamics II*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1493302), for the mean-reverting Bergomi volatility factor and factor correlations.
- [Guyon and Henry-Labordère, *The Smile Calibration Problem Solved*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032), for the particle calibration of local-stochastic volatility and affine dividends.
- [Hamdouche and Henry-Labordère, *Vega KT for LSV Models: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4304114), for the LSV path-adjoint and calibration-sensitivity context.
- [Jourdain and Zhou, *Existence of a calibrated regime switching local volatility model and new fake Brownian motions*](https://arxiv.org/abs/1607.00077), for the conditional-expectation and interacting-particle calibration context.

The quartic log-coordinate kernel, finite-grid fallback rules and discrete VJP
are repository-specific policies; the cited results do not imply those exact
discretization choices.
