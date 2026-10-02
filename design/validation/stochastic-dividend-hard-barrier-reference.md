# Independent hard two-date Barrier price and Delta

The [smoothing-width panel](stochastic-dividend-rough-lsv-barrier-smoothing.md)
uses finite bumps of hard prices only as diagnostics. This panel supplies an
independent hard price and analytic physical-Spot Delta in an exact lognormal
limit of rough residual LSV. It includes the effect of moving both monitoring
boundaries, without differentiating a sampled hard indicator.

## Exact-limit scope and inputs

Set rough eta and dividend mean reversion kappa to zero and the residual
local-variance target to a flat 0.04. Then squared leverage is 0.04 for any
calibration sample. Normalized residual equity f and dividend multiplier Y
are correlated geometric Brownian motions with sigma=0.2, nu=0.35 and
correlation rho=-0.25. H does not affect this limiting law.

The up-and-in call has Spot 100, strike 80, barrier 105, observations at
`t1=182/365`, `T=364/365`, and payment at `456/365`. Cash means 5/3/12 occur
at t1/T/1.4. Monitoring includes pre- and post-cash stock at both dates;
terminal intrinsic value uses post-cash stock. Carry growth is
`G(t)=(0.98/0.95)^t`; discount to payment is `0.95^(456/365)`.

All inputs and reference values are retained in the
[fixture](../../fixtures/stochastic-dividends/rough-barrier-reference.json).
Production calibration, paths, payoff helpers, random numbers and risk
implementations are not called by the Python reference.

## Conditional bivariate-lognormal calculation

Let funded residual equity be `F0=S0-sum(q_i/G(t_i))`. At kappa=0,
post-cash stock is `S_t=A_t*f_t+B_t*Y_t`, where `A_t=F0*G(t)` and
`B_t=sum_{t_i>t} q_i*G(t)/G(t_i)`. Pre-cash stock adds the date's `q*Y_t`.

Condition on the two dividend Brownian increments. The conditional logs of
f at t1 and T have means `m_t=-sigma^2*t/2+sigma*rho*W_Y(t)`, standard
deviations `s_t=sigma*sqrt((1-rho^2)*t)` and correlation `r=sqrt(t1/T)`.
Define normalized cutoffs:

```text
U1 = [barrier - (B_t1 + q_t1)*Y_t1] / A_t1
U  = [barrier - (B_T  + q_T )*Y_T ] / A_T
L  = [strike  - B_T*Y_T] / A_T
a = (log U1 - m_t1)/s_t1
b = (log U  - m_T )/s_T
c = (log L  - m_T )/s_T
```

The in-the-money knock-out region is `f_t1<U1` and `L<f_T<U`. Nonpositive
cutoffs use the exact lower lognormal limit, and empty regions contribute
zero. Its probability is `Phi2(a,b;r)-Phi2(a,c;r)`. Its f_T-weighted moment
is `exp(m_T+s_T^2/2)` times the same difference with the first cutoff shifted
by `-r*s_T` and the terminal cutoffs by `-s_T`. These give the conditional
knock-out call value. Subtract it from the conditional vanilla call to obtain
the knock-in value.

Physical Spot changes A_t with derivative G(t); normalized f/Y are unchanged.
Every finite standardized cutoff has derivative `-1/(F0*s_t)`. Differentiate
the bivariate probabilities using
`dPhi2/da=phi(a)*Phi((b-r*a)/sqrt(1-r^2))` and the symmetric derivative in b.
This retains both the first and terminal barrier boundary terms. The vanilla
part uses its analytic conditional lognormal Delta. No Spot bump is used to
generate the reference Delta.

For Phi2, integrate its correlation derivative from r=0, starting at the
product of two univariate CDFs, with Gauss-Legendre quadrature. Integrate the
two remaining dividend normals with tensor Gauss-Hermite quadrature.

## Independent controls

The [reference implementation](../../tests/python/rough_dividend_barrier_reference.py)
and [Python checks](../../tests/python/test_rough_dividend_barrier_reference.py)
verify:

- Bivariate zero-threshold probabilities against the arcsine identity,
  exact infinite-cutoff limits, and both CDF partial derivatives.
- Dividend orders 32/64/96/128 and CDF orders 32/64/96 agree with the retained
  fixture within 2e-8. This is observed numerical agreement, not a rigorous
  quadrature error certificate.
- A separate price method integrates the first equity normal explicitly,
  splitting at its monitoring boundary, and integrates only the terminal
  equity increment analytically. It uses no bivariate CDF. Orders 24/32 for
  dividends and 64/96 for the inner integral agree within 2e-8. Truncating
  that first normal to [-10,10] leaves negligible Gaussian/lognormal tails
  at these parameters.
- Analytic Delta agrees with central differences of that second price method
  at Spot 95/100/105 and bumps 0.005/0.0025 within 3e-8. The bumps are a
  control on the analytic derivative, not its definition.

The retained hard reference is price **11.636503795605742** and Delta
**0.9573395571545552**. Run the reference checks with:

```sh
python -m unittest discover -s tests/python -p test_rough_dividend_barrier_reference.py -v
```

These NumPy-only checks are also discovered by the existing Python wheel
smoke suite. They do not need the production Python extension to run locally.

## Rust comparison and predeclared gates

The public Rust panel uses H=0.1 with two steps and H=0.3 with four steps,
valuation seeds 193/877, and hard or width-2/1/0.5 payoffs. Both grids simulate
the exact limiting law; the comparison has no discretization-bias allowance.
Each scenario uses 32 independent RQMC scrambles of 65,536 points, antithetic
sampling and Brownian bridge, totaling 4,194,304 paths. SE is computed across
the 32 scramble means.

For hard price and width-0.5 price/Delta, require
`abs(value-reference)+4*SE+2e-8 < 0.01` and `SE < 0.002`, separately for
each H/grid/seed. Earlier smoothing widths are diagnostic. The 2e-8 term
retains the quadrature agreement tolerance in the comparison budget.
All 28 rows are reported before any numerical-gate failure is raised.

A fast test checks constant leverage, required dates, and rejection of hard
Spot risk without affecting hard prices. Run the numerical panel with:

```sh
cargo test --locked --release -p pricing --test stochastic_dividend_barrier_reference -- --include-ignored --nocapture
```

The three-OS Barrier smoothing CI job retains the output as
`stochastic-dividend-hard-barrier-reference.log`. Source archives require the
fixture, both Python files, Rust test, documentation and CI gate.

### Initial precision observation

The initial 16,384-point/32-scramble panel passed both price gates but failed
the width-0.5 Delta gate. At seed 193, Delta minus reference was 0.003738009,
with SE 0.002705430 and comparison bound 0.014559750. At seed 877, the gap
was -0.005218131, SE 0.002611114 and bound 0.015662608. Both H/grid cases
gave the same values up to rounding in this exact limit with Brownian bridge.
This run was not a pass. Points per scramble were increased fourfold while
retaining the seeds, scramble count, contracts, widths, reference values and
all acceptance limits. No numerical policy or pricing algorithm was changed.

The 65,536-point panel passed all gates locally, retaining all 28 output rows.
Across both seeds and H/grid cases, the largest comparison bounds were
0.002897838 for hard price, 0.001736047 for width-0.5 price, and 0.006081533
for width-0.5 Delta. The largest width-0.5 Delta SE was 0.001178187.
These are observations from the stated finite sample, not worst-case bounds
over seeds, widths or parameters.

This adds validation, not hard-Barrier risk support. Public unsmoothed
pathwise Delta remains rejected. Agreement is limited to this two-date,
zero-eta/zero-kappa, flat-target case; it does not bound hard Delta bias for
nonzero rough vol-of-vol, other monitoring schedules or continuous barriers.
