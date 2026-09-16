# Hull–White cash-dividend numerical contracts

Date: 2026-09-13. Status: experimental price and coordinate contracts.
Base: [HW contracts](hull-white-numerical-contracts.md), PR #48.
First-order risk is specified separately in the [AAD contracts](hull-white-aad.md).

## Explicit model and API

Fixed cash amounts are known in advance and paid at their ex-times. At event i,
`S_i+ = (1-beta_i)*S_i- - D_i`. The existing request's `DividendEvent` types
represent fixed cash, proportional, or mixed payouts. The entire supplied
schedule matters, including cash payments after option expiry. Amount uncertainty
and a separate dividend payment date are not modeled here.

Use Rust `HullWhiteEquityPricingPlan::compile_bs_with_cash_dividends` or
`compile_lsv_with_cash_dividends`. In Python, keep the existing compiler and add
`cash_dividend_model="escrowed"`. `None` retains the previous behavior and its
cash-dividend rejection. Plan and price metadata identify
`escrowed-hw-bonds-v1`; `plan.risky_spot` reports the initial residual equity.
The [Python example](../../examples/python/hull_white_lsv.py) covers this mode.

This choice is a model change, not a deterministic spot/strike adjustment. In
BS mode the supplied volatility applies to the residual equity. In LSV mode the
input smile describes the continuous deterministic escrow coordinate defined
below. Neither interpretation is an unadjusted Black implied volatility on
physical Spot. Existing deterministic dividend, LV and LSV entry points are
unchanged. The stable JSON stores the payout schedule but not this compile flag.

## Reserve and physical Spot

Let P(t,u) be the HW bond, P0(t) the initial discount factor, Q0(t) the
deterministic continuous-dividend discount factor, and
`B(t)=product_{t_i<=t}(1-beta_i)`. Define the post-event reserve

```text
A(t,x) = sum_{t_j>t} D_j / product_{t<t_i<=t_j}(1-beta_i)
                    * P(t,t_j) * Q0(t)/Q0(t_j).
A0(t)  = the same sum with P(t,t_j) replaced by P0(t_j)/P0(t).
U0     = S0 - A0(0-),
g0(t)  = Q0(t)/P0(t),
c(t)   = B(t)*U0*g0(t)/S0.
```

Require U0 > 0. An ex-time zero event is included in `A0(0-)`; S0 is the
pre-event input and the first stored observation is post-event, matching the
schedule convention. The residual equity follows

```text
dU = (r-q)*U dt + L(t,F)*a_v*U dW_S,
R  = S0*U/(U0*g0(t)),
S  = c(t)*R + A(t,x).
```

BS uses constant L; Bergomi uses `a_v=exp(nu*X_v)`. R is the positive normalized
equity stored in the existing hybrid state. The reserve is a portfolio of bonds
with deterministic carry weights, so it has drift `(r-q)*A` between events.
B jumps proportionally and A obeys the affine payout jump, giving S its
prescribed jump while U and R stay continuous.

For HW mean reversion a and instantaneous rate volatility sigma_r(t), its rate
loading is

```text
v_A(t,x) = -sigma_r(t) * sum_j reserve_term_j(t,x)*B_HW(a,t_j-t).
```

Bond loadings and deterministic coefficients are precompiled per time node;
each path only evaluates the state-dependent exponentials. Rate and rate-integral
sampling is unchanged. No new Gaussian coordinate is used at a dividend event.
For pricing, the contractual graph receives physical S directly, including
pre-event S where requested. The old affine mapping is not applied again.
The existing conditional bond discount for product payment lags is retained.

## LSV target coordinate and calibration

Define a continuous coordinate using the *deterministic* reserve and scale:

```text
F = (S-A0(t))/c(t) = R + (A(t,x)-A0(t))/c(t),
h = A0(t)/c(t),       zeta = v_A(t,x)/c(t),
y = r-f0,            eta = a_v*R/F.
```

Between events, A0 and c have the same deterministic growth, so h is constant.
At an event, the numerator and denominator in F transform by the same factor;
F has no jump. Its dynamics are

```text
dF = y*(F+h)dt + L*a_v*R dW_S + zeta dW_r.
```

The paired input grid describes normalized calls
`C_F(t,K)=E[Dbar(t)*(F(t)-K)+]`, where `Dbar=D/P0`, initial F=S0, and the
target density is `p_log=K*p_F^T`. To use market options at physical strike K_S,
convert to `K_F=(K_S-A0(t))/c(t)` and divide the discounted call price by c(t).
Construct a consistent smooth target in that coordinate. Simply feeding a
spot-IV eSSVI surface unchanged is not that conversion. Quote conversion and
fitting across ex-dates remain the caller's responsibility; the target
constructor's no-repair policy still applies.

Discounted Ito/Tanaka for the F call has rate drift
`(K+h)*E[Dbar*y*1(F>K)]`. Accordingly the required relative conditional variance
is

```text
V_req = sigma_Dupire^2 - 2*(1+h/K)*q_hat/p_log.
M2 = E_D[eta^2 | F=K],
M1 = E_D[eta*(zeta/F) | F=K],
M0 = E_D[(zeta/F)^2 | F=K],
M2*L^2 + 2*rho_Sr*M1*L + M0 = V_req.
```

Here E_D is a discounted conditional expectation. `q_hat` uses the existing
empirically centered digital-rate estimator, now conditioning on F; its
finite-particle ratio bias remains. The existing quartic log-F kernel and ESS
policy estimate M2, M1 and M0 jointly. Dividing particle loadings by the
particle's F preserves the no-bond limit exactly in the moment calculation.
At t=0, use the analytic residual ratio one and known bond loading at each
strike; the rate correction is zero. At deterministic rates and nu=0, leverage
is exactly the target local variance.

Choose the upper positive root of the quadratic. With `b=rho_Sr*M1`,
`d=b^2+M2*(V_req-M0)`, use `(sqrt(d)-b)/M2`; for b>=0 rationalize the
subtraction. Negative discriminants, nonpositive roots and non-finite output
are errors. The zero-bond case uses the original variance quotient directly.
Unsupported nodes inherit all three moments and the rate correction from the
same nearest supported node, retaining their own local-variance target. There
is no independent clipping of the covariance or bond variance.

Rust calibration results retain `conditional_cross_moments` and
`conditional_rate_variances`, as well as M2 and the drift corrections.
The method is `lsv-hw-escrowed-quadratic-quartic-v1`. Discounted-equity diagnostics
now describe F; its discounted mean is S0 because `E[Dbar*y]=0`.

## Domain, grid and uncertainty

- All in-horizon dividend ex-times must be on the grid. BS compilation inserts
  them. LSV requires the paired target grid to contain them exactly, including
  zero/expiry collisions; no interpolation of the initial singular density.
  The existing request compiler requires the effective variance grid to end
  exactly at product expiry, even when its dividend schedule extends beyond it.
- F must be strictly positive for the log-coordinate LSV scheme. Although S
  and residual U are positive in this funded model, extreme rate paths can make
  F nonpositive. Such a calibration/pricing path errors; it is never clamped.
- U0 and scale must be positive and all reserve coefficients representable.
  The existing request/payoff compiler's positive-forward validation also applies;
  this opt-in does not relax those pre-existing gates.
- A dividend reserve plan must use the same HW parameters, initial Spot and
  time grid as the evolution/calibration plan. Mismatches are rejected.
- Pricing and calibration use the same reconstructed F to query leverage, but
  independent RNG domains. Four-block antithetics and Brownian bridge remain.
- The effective reserve coefficients, event data, initial residual equity and
  model identifier enter the plan fingerprint, including future dividends.
- Pricing SE excludes calibration noise, kernel bias and log-Euler time bias.
  The explicit AAD extension includes initial-curve DV01 and paired target
  sensitivities. [Quote-node VegaKT](hull-white-vegakt.md) is available for
  converted escrow F IV inputs. Physical quote conversion adjoints, Gamma and
  dividend-amount risk remain unsupported.

## Focused evidence and acceptance boundary

Tests cover the reserve's rate loading against a finite difference; continuity
at a cash event including a future mixed payout; exact BS+HW terminal pricing
after the last cash payout; independent LSV vanilla repricing while a dividend
reserve is still present at expiry; worker replay; cash-driven barrier hits;
valuation-date and expiry-date events; the exact expiry-grid contract;
no-cash/deterministic-factor limits;
missing nodes, insufficient initial funding and an impossible leverage
discriminant. Python tests exercise opt-in selection, immutable metadata and
the typed boundary. These tests establish the initial implementation, not broad
market-smile calibration acceptance. H3/H4 still need multiple calibration seeds,
cash schedules, bandwidth/time/particle refinement and stress cases.

Local Linux validation with Rust 1.98.1 and CPython 3.12: 313 Rust workspace
tests, 3 native Python-extension tests, statistical acceptance, 48 Python tests,
four wheel-installed examples, typed/runtime wheel contracts, formatting,
Clippy, Rust documentation and reference/schema/dependency/link checks passed.

## References and specialization

- Buehler, [Volatility Modelling with Cash Dividends and Simple Credit Risk](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1141877),
  equity decomposition with a future-dividend reserve. Credit/default modeling
  from that work is outside this implementation.
- Henry-Labordère, [Equity modelling with local stochastic volatility and
  stochastic dividends](https://www.risk.net/media/download/991346/download),
  particle calibration with an additional dividend diffusion and a quadratic
  leverage equation. That article uses deterministic rates and stochastic
  dividend amounts; here amounts are fixed and their HW bond values fluctuate.

The coordinate, discounted drift correction and numerical policies above are
our specialization. The cited papers do not establish acceptance of this code.
