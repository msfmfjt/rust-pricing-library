# Stochastic discrete cash dividends

[Component index](components/README.md) · [Library guide](../library/README.md) ·
[Python example](../../examples/python/stochastic_dividends.py)

## Scope and entry points

`pricing::stochastic_dividends::StochasticDividendPricingPlan::compile_bs` and
Python `StochasticDividendPlan.compile_bs` add a **separate price-only** Buehler
cash-dividend model at deterministic rates. The request supplies constant
volatility of normalized residual equity, not implied volatility of physical
stock. The fixed-cash amounts in this entry point are risk-neutral mean amounts.
Existing entry points retain deterministic cash semantics and numerical results.

The component uses the existing single-asset contractual payoff graphs, including
European calls/puts, digital, Asian, discrete lookback and discrete barriers.
American exercise and continuous barriers are rejected by the shared compiler.
Greeks in the request are rejected rather than silently computed under fixed
cash. There is no `evaluate_aad` for this plan. Public price accuracy tests in this
change cover European calls and the discrete-barrier dividend jump; the new
stochastic model has no dedicated broad exotic accuracy panel yet.

This is **not** an implementation of stochastic-dividend LSV, pure SV/HW hybrids,
multi-asset dividends, dividend derivatives or dividend-option calibration.
Those require their own pricing, covariance, calibration and reverse contracts.
The same model object must not be passed to an existing fixed-cash calibrator.

## Model and market meaning

Under the pricing measure, with initial states `f=Y=1`,

\[
 df_t=\sigma f_t\,dW_t^f,\qquad
 dY_t=\kappa(\alpha f_t+1-\alpha-Y_t)dt+\nu Y_t\,dW_t^D,
 \qquad d\langle W^f,W^D\rangle_t=\rho\,dt.
\]

The validated parameter ranges are `kappa,nu >= 0`, `0 <= alpha <= 1` and
`-1 <= rho <= 1`. All inputs must be finite. The correlation is between the
**internal Brownian drivers**, not an observed stock/realized-dividend return
correlation. Physical stock also has exposure to the dividend factor.

At a strictly future ex-date `t_i`, the cash amount is `D_i = mean_i * Y(t_i)`.
Ex-date and payment date of this dividend are identified in this initial scope;
there is no separate declaration or payment-lag state. Proportional dividends
and cash/proportional mixtures are rejected in this entry point. Use explicit
zero-volatility settings to represent fixed cash; do not mix two meanings of the
same market amount inside one plan. Annual totals must already be allocated to
the supplied ex-dates; neither historical forecasts nor annual futures quotes
are automatically converted into risk-neutral per-date cash means.

The conditional factor forecast, used **before** the dividend is realized, is

\[
 M(t,u;f,Y)=e^{-\kappa(u-t)}Y+
 (1-e^{-\kappa(u-t)})(\alpha f+1-\alpha).
\]

A future simulated `Y(u)` must never be used in the current reserve. The
forecast process for a fixed future date is a martingale even though `Y` has
mean-reverting drift. The entire supplied future cash schedule is a common
market horizon; a shorter option maturity does not discard later cash.

## Carry-funded escrowed stock

Let `D_r(t)` be the deterministic collateral discount factor and `D_s(t)` the
carry-factor curve already held by the equity market. For the repo-spread
convention `s=r-r_repo`, define

\[
 G(t)=D_s(t)/D_r(t),\qquad
 x_0=S_0-\sum_i \frac{\overline D_i}{G(t_i)} >0.
\]

The existing legacy API field `dividend_curve` continues to name `D_s`; this
extension does not rename that wire/Python field and does not combine repo
spread with cash dividends. At a grid node immediately after any ex-date,

\[
 S_t=G(t)\left[x_0 f_t+
 \sum_{t_i>t}\frac{\overline D_i}{G(t_i)}M(t,t_i;f_t,Y_t)\right]
     =a(t)f_t+b(t)Y_t+c(t).
\]

The coefficients are deterministic and nonnegative, with `a(t)>0`. The path
compiler precomputes them. Every supplied future amount contributes to `x_0`,
including cash beyond the option expiry. Only ex-dates at or before expiry are
inserted into pricing paths; funding later dividends does not extend simulation.

This is a **carry-funded reserve**, not the collateral value of a tradable
strip of dividend claims when repo spread is nonzero. In this deterministic-rate
model a claim to dividend `D_i` has collateral-discounted conditional value
`D_r(t_i)/D_r(t) * mean_i * M(t,t_i;f,Y)`. That valuation is conceptually separate
and no dividend-claim product API is introduced here. With stochastic rates,
conditional expectations of discount factors and dividends cannot generally be
factorized; the present formula must not be reused unchanged in HW.

At an ex-date the internal states are continuous. Stock drops by `mean_i*Y`,
and same-date expiry uses the post-dividend stock. The payoff adapter also gets
the pre-dividend stock for discrete-barrier jump tests. Positive internal states
and the funded residual prevent negative post-dividend stock by construction.
The low-level path compiler rejects a grid omitting a required ex-date.

## Calculation specifications

The scheme identifier is `buehler-cash-positive-split-v1`. For an interval `h`,
let `a=exp(-kappa*h/2)`, `b=-expm1(-kappa*h/2)`, and `T(f)=alpha*f+1-alpha`:

\[
 Y_{1/2}=aY+bT(f),\quad
 f'=f\exp(-\sigma^2h/2+\sigma\sqrt h\,Z_f),
\]
\[
 Y_* =Y_{1/2}\exp(-\nu^2h/2+\nu\sqrt h
 [\rho Z_f+\sqrt{(1-\rho)(1+\rho)}Z_D]),\quad
 Y'=aY_*+bT(f').
\]

This positive drift/diffusion/drift split is a repository discretization, not
an exact continuous-time simulation for nonzero mean reversion. It preserves
conditional first moments of both factors, and therefore the affine forecast
semigroup, up to floating-point rounding. Nonlinear prices and second moments
still have discretization error. No floor/cap or full-truncation Euler is used.
Nonfinite, nonpositive underflowed diffusion states and nonfinite prices fail
explicitly instead of being clipped.

`alpha=0, nu=0, Y0=1` gives fixed dividends exactly at the state level. Merely
setting `nu=0` does not remove randomness when `alpha>0`. At `kappa=0` both
factors are correlated geometric Brownian motions; this has an independent
conditional-lognormal price reference. Zero equity volatility is allowed.

There are always two independent normal coordinates per interval, including
zero loading and perfect-correlation cases. Unbridged coordinates are
step-major, and bridged RQMC coordinates are bridge-rank-major, equity then
dividend. The Brownian bridge is applied separately to independent factors
**before** correlation. This is a new fingerprint domain; no existing layout,
fixture, schema or fingerprint domain changes.

Pseudo-MC reports uncertainty across independent units (an antithetic pair is
one unit). RQMC reports uncertainty across scramble means, not individual points.
Results expose `uncertainty_scope="pricing_only"`: this excludes timestep bias,
parameter uncertainty and model risk. A deterministic payment discount is already
in the shared payoff and is not applied twice. Compare smaller `maximum_step`
values separately from increasing paths or scrambles.

## Reference

Hans Buehler, [*Volatility and Dividends II — Consistent Cash Dividends*](https://quantitative-research.de/dl/Volatility%20and%20Dividends%20II%20-%20Consistent%20Cash%20Dividends.pdf),
section 3.1, system (S), for the mean-reverting factor and its conditional
expectation. The finite supplied schedule, deterministic repo-carry extension,
positive splitting, public support limits and validation policy above are
repository-specific choices rather than assertions of complete paper coverage.
