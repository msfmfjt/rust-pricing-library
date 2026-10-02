# Conditional hard Barrier reference with nonzero rough volatility

The [exact-limit reference](stochastic-dividend-hard-barrier-reference.md)
sets eta and kappa to zero. This panel enables eta=0.6 and kappa=0.7, a
nonflat calibrated leverage surface, and H=0.1/0.3. It independently integrates
the hard two-date up-and-in call price and physical-Spot Delta on the same
**two-step discrete law** used by the Rust plan. It retains both moving
monitoring boundaries. It is not a continuous-time reference or a validation
of general time grids.

## Retained inputs and scope

The [fixture](../../fixtures/stochastic-dividends/rough-conditional-barrier-reference.json)
contains both calibrated surfaces, model correlations, calibration settings,
target variances, quadrature settings, and reference values. Its market and
contract link retains the exact-limit fixture: Spot 100, strike 80, barrier
105, cash means 5/3/12, t1=182/365, T=364/365, and payment at 456/365.
Monitoring includes pre- and post-cash stock; the terminal call uses post-cash
stock. Cash is positive, so pre-cash stock determines an up-barrier hit.

The three target rows are `[0.045,0.04,0.035]`, `[0.05,0.045,0.04]` and
`[0.055,0.05,0.045]` at log nodes -0.5/0/0.5. Calibration uses 64 particles,
seed 42, bandwidth 0.35 and minimum effective sample size 5. The Rust test
reconstructs and checks every retained leverage value within 2e-13. Only
rows at time zero and t1 drive the two steps. The terminal row is retained
and checked as calibration evidence, but does not drive an extra step.

The reference conditions on this retained surface. It does not independently
validate the calibration algorithm or calibration-sampling uncertainty. Spot
changes reanchor the surface to funded residual equity, leaving normalized
f/Y dynamics invariant, as in the existing public LSV Spot-risk convention.

## Conditional Gaussian law

Let D1 and D2 be standardized dividend Brownian increments. Write the first
Volterra value as `X1=t1^H*(d*D1+sqrt(1-d^2)*V)`, where V is an independent
standard normal, `c_H=sqrt(2H)/(H+1/2)` and `d=c_H*rho_DV`. The first hybrid
cell has variance `t1^(2H)` and covariance with an equity/dividend Brownian
normal equal to `c_H*rho*t1^H`.

Given D1 and V, the first equity Brownian normal has mean
`rho_SD*D1+q*V` and residual variance `1-rho_SD^2-q^2`, where
`q=(c_H*rho_SV-rho_SD*d)/sqrt(1-d^2)`. Thus the first normalized equity
is `f1=exp(m1+s1*Z)`, with one remaining independent normal Z. No future
Volterra history is required: the last step uses the first node's history,
not the history at expiry.

Given f1 and X1, the second-step equity volatility is
`sqrt(L2(t1,log(f1)))*exp(eta*X1/2-eta^2*t1^(2H)/4)`. Interpolation of the
retained squared leverage is linear in log(f1), with constant wings.
Given D2, terminal f is lognormal. Its conditional mean M and log deviation s
are obtained from this second-step volatility and rho_SD.

## Dividend split and stock reconstruction

For each step, let `a=exp(-kappa*dt/2)`. The symmetric dividend split is

```text
Y_next = a * [a*Y + (1-a)*(alpha*f+1-alpha)]
           * exp(nu*sqrt(dt)*D - nu^2*dt/2)
         + (1-a)*(alpha*f_next+1-alpha).
```

Given D1, Y1 is affine in f1. Given f1/D1/D2, terminal Y is affine in terminal
f. Carry-funded stock coefficients are reconstructed directly from all future
cash means, including the one beyond expiry; no production reserve or payoff
helper is used. Consequently, first pre-cash stock is affine in f1, and
terminal pre/post-cash stock is affine in terminal f. The first hit boundary
is a single cutoff in Z.

For a missed first observation, terminal exercise requires terminal f to
exceed the larger of zero, the strike cutoff and the pre-cash barrier cutoff.
For a hit first observation, only the strike cutoff is needed. For cutoff l,
terminal post-cash call value is
`Q*M*Phi(d1)+C*Phi(d2)`, with `d2=[log(M/l)-s^2/2]/s` and `d1=d2+s`.
A nonpositive cutoff uses the full lognormal moment.

## Analytic Delta, including both boundaries

Stock's coefficient of f has physical-Spot derivative G(t), while the
normalized path law is unchanged. Differentiate the truncated terminal
moment analytically. The exercise boundary has zero intrinsic payoff, but
the hard-barrier boundary generally has a nonzero payoff and contributes an
additional density term.

Integrate the first normal on separate hit/miss intervals, also splitting at
every leverage knot. If its barrier cutoff is z*, its derivative is
`-G(t1)/(Q_pre1*s1)`, where Q_pre1 is the first pre-cash coefficient of f1.
The first-boundary Delta contribution is

```text
[conditional_hit_price(z*) - conditional_miss_price(z*)]
    * phi(z*) * G(t1)/(Q_pre1*s1).
```

This term is added explicitly; differentiating only the terminal conditional
payoff would miss it. The negative control which drops only this term changes
Delta from 0.951196149 to 0.726632080 at H=0.1, and from 0.952000873 to
0.732869803 at H=0.3, while leaving price unchanged.

The [NumPy implementation](../../tests/python/rough_dividend_conditional_barrier.py)
uses tensor Gauss-Hermite quadrature for D1/V/D2 and split Gauss-Legendre
quadrature for Z. It calls no production calibration, evolution, payoff or
risk routine. Its normal CDF helper is shared with the independent exact-limit
Python reference.

## Numerical controls and Rust gates

The [Python tests](../../tests/python/test_rough_dividend_conditional_barrier.py)
check outer orders 12/24/40, inner orders 24/32/40, and first-normal truncation
at 10/12 standard deviations against the retained references within 2e-7.
This is observed numerical agreement, not a certified quadrature error bound.
The eta=kappa=0 flat-surface limit agrees with the separate bivariate-lognormal
reference within 2e-8 for both H values. Analytic Delta agrees with central
price differences at Spot 95/100/105 and bumps 0.005/0.0025 within 3e-8;
these bumps control the derivative and do not define it.

| H | Hard price | Hard Spot Delta |
| --- | ---: | ---: |
| 0.1 | 12.16960301932176 | 0.9511961490766794 |
| 0.3 | 12.280436981731082 | 0.9520008726520396 |

The Rust panel uses seeds 193/877, 65,536 points per scramble, 32 independent
RQMC scrambles, antithetic sampling and Brownian bridge: 4,194,304 paths per
scenario. Hard price and width-0.5 price/Delta must independently satisfy
`abs(value-reference)+4*SE+2e-7 < 0.01` and `SE < 0.002` for each H/seed.
Width 1 is diagnostic. All 20 rows are printed before any numerical-gate
failure is raised. The initial run at these settings passed without changes
to sample count, seeds, widths or acceptance limits.

The largest observed bounds were 0.004789134 for hard price, 0.004791939 for
width-0.5 price, and 0.007590499 for width-0.5 Delta. Maximum width-0.5 Delta
SE was 0.001820977. These finite-sample results do not bound other seeds or
parameters.

```sh
python -m unittest discover -s tests/python -p 'test_rough_dividend_*barrier*.py' -v
cargo test --locked --release -p pricing --test stochastic_dividend_barrier_reference -- --include-ignored --nocapture
```

The existing three-OS Barrier CI step runs both references and retains 48 rows
(28 exact-limit and 20 nonzero-eta/kappa rows) in
`stochastic-dividend-hard-barrier-reference.log`. The Python wheel smoke suite
discovers the new NumPy-only controls. Source archives require both new Python
files, the new fixture and this document.

Public unsmoothed Barrier Spot risk remains rejected. Extending independent
hard-price/Delta validation to more evolution steps, finer grids and other
monitoring schedules remains open. This panel establishes a nonzero-eta/kappa
finite-grid check and does not remove the time-discretization or calibration
limitations of the existing rough-LSV validation programme.
