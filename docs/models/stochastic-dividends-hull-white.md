# Stochastic cash dividends with Hull–White

Experimental, single-asset, constant residual-equity volatility, price-only.
Use Rust `StochasticDividendHullWhitePricingPlan::compile_bs` or Python
`StochasticDividendHullWhitePlan.compile_bs`. This is a separate plan: existing
BS/1F/2F/rough stochastic-dividend plans and their AAD/Gamma remain unchanged.
No stochastic-volatility/HW combination or inherited fixed-rate risk is silently
used. The request must specify Black–Scholes initial residual volatility and
price-only risk flags. American and continuous barriers remain unsupported.

## Measure and cash input

Let Q be the domestic money-market pricing measure. With f(0)=Y(0)=1,

```text
df = sigma_f f dW_f,
dY = kappa (alpha f + 1-alpha-Y) dt + nu_D Y dW_D,
dx = -a x dt + sigma_r(t) dW_r,   x(0)=0,
r = fitted HW deterministic shift + x,
D_i = mean_i Y(T_i).
```

The nonnegative `fixed_cash` schedule amounts **remain Q means**:
`E^Q[D_i]=mean_i`. In particular they are NOT the collateral-forward dividend
amounts when dividend uncertainty and discounting are correlated. This API does
not reinterpret a dividend-forward curve as Q means or calibrate that conversion.
Ex-date and payment date coincide for cash dividends; all ex-times are strictly
future. All supplied means, including post-option-expiry cash, are funded.

The rate input is the existing constant-reversion, piecewise-constant-volatility
`HullWhite1Factor`. All volatility knots are included in rate covariances and
conditional cash forecasts, including knots after the option's expiry. Three
independent correlation inputs specify the symmetric matrix of (W_f,W_D,W_r).
Full PSD is checked even at zero loadings. These are Brownian driver correlations,
not correlations between physical-stock returns and observed dividend revisions.

## Conditional discounted dividend value

Define B_a(z)=(1-exp(-a z))/a, with B_0(z)=z, and

```text
L(t,T) = integral_t^T sigma_r(v) B_a(T-v) dv.
c_f = sigma_f rho_fr,    c_D = nu_D rho_Dr.
```

Variation of constants for Y followed by a Gaussian exponential tilt gives

```text
Pi_i(t) = E_t^Q[exp(-integral_t^T r) D_i]
        = mean_i P(t,T) [A(t,T) f_t + B(t,T) Y_t + C(t,T)],
B(t,T)  = exp[-kappa(T-t) - c_D L(t,T)],
A(t,T)  = kappa alpha integral_t^T
            exp[-kappa(T-u) - c_f(L(t,T)-L(u,T)) - c_D L(u,T)] du,
C(t,T)  = kappa(1-alpha) integral_t^T
            exp[-kappa(T-u) - c_D L(u,T)] du.
```

The f increment before u and the multiplicative Y increment after u occupy
disjoint intervals. Their Gaussian cross-covariance is zero, explaining why
rho_fD does not appear explicitly in these coefficients. It still affects the
joint path distribution and option prices. Rate/dividend correlation enters B/C,
and rate/equity correlation also enters A. The formulas above are derived for
this adapter; they are not claimed to be a general closed form for Buehler with
stochastic residual volatility.

At t=T the coefficients are (0,1,0), so the claim settles to the actual cash.
At kappa=0, Pi=mean P Y exp[-nu_D rho_Dr L]. At zero rate volatility the original
fixed-rate forecast is recovered. At alpha=nu_D=0 the cash is deterministic and
Pi=mean P, independent of residual-equity/rate correlation.

Python properties `initial_dividend_claim_values` and
`initial_dividend_forwards` report Pi_i(0) and Pi_i(0)/P(0,T_i) in `cash_times`
order. These are collateral values, not the repo-funded reserve itself.

## Escrow and repo spread

Let Q0(t)=exp(-integral_0^t s) be the deterministic repo-spread factor; the
legacy Rust accessor is `dividend_curve()`. Between cash events,

```text
A_t = sum_{T_i>t} [Q0(t)/Q0(T_i)] Pi_i(t),
U0  = S0 - A0,  require U0>0,
U_t = U0 exp(integral_0^t(r-s)) f_t,
S_t = U_t + A_t.
```

Each carry-weighted claim and residual U has drift (r-s) times its value. Thus
`exp(integral_0^t s) D(0,t) S_t` plus similarly discounted/carry-weighted paid
cash is a Q martingale in continuous time. At an ex-date the reserve drops by
`mean_i Y_i`, giving `S_plus = S_minus - D_i` without a paid-cash stock process.
Expiry on an ex-date uses post-dividend Spot. The finite splitting scheme can
have a discretization error in this martingale identity; the continuous-time
reserve formula does not remove that error.

## Simulation and numerical integration

The existing HW transition samples the joint vector
`(Delta W_f, Delta W_D, rate innovation, integrated-rate innovation)`. Its second
OU reversion is set to zero to obtain a Brownian dividend driver. All **four**
independent coordinates remain allocated at deterministic or singular limits.
The existing positive Buehler split advances f/Y using the first two independent
normals; the correlated rate factor and its integral use the same Gaussian
vector. Brownian bridging acts on independent coordinates before correlation.

Reserve coefficients and bond loadings are compiled outside the path loop.
Their scalar integrals use adaptive Simpson integration, split at rate-volatility
knots, with absolute target 1e-12, relative target 1e-11 and maximum depth 20.
Failure to meet the estimator's target is an error, not silent acceptance.
These are estimated quadrature tolerances, not rigorous error bounds.
Compiled node/cash pairs are limited to 1,000,000. Positive residual states and
finite cash values are checked; no new clipping or covariance projection occurs.

Only to expiry is simulated. For payment U>=T the shared contractual payoff
already contains P0(U); multiply by
`D(0,T) P(T,U)/P0(U)`. Conditional payment discounting uses the terminal HW state,
including volatility breakpoints outside the simulation horizon. Do not discount
the shared graph a second time by P0(U).

MC uncertainty uses independent units, with antithetic averaging. RQMC uncertainty
uses scramble means. `standard_error` excludes time-step, reserve quadrature,
smoothing, calibration and model uncertainty. The initial sigma is NOT the
physical-stock market IV. No market-IV calibration, VegaKT, AAD or Gamma is
available for this separate HW plan.

## References and examples

The underlying conventions are specified in [stochastic cash dividends](stochastic-dividends.md),
[Hull–White](hull-white-calculation-specifications.md) and
[fixed-cash HW escrow](hull-white-cash-dividends.md). The conditional-cash formulas
are the derivation above, independently checked through their backward equation.
See the [Python example](../../examples/python/stochastic_dividend_hull_white.py),
[decision](../../design/adr/0022-stochastic-dividend-hull-white.md), and
[validation protocol](../../design/validation/stochastic-dividend-hull-white.md).
