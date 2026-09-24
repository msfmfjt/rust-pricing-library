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
cash. First-order risk is requested explicitly through `evaluate_aad`; see below. Public price accuracy tests in this
change cover European calls and the discrete-barrier dividend jump; the new
stochastic model has no dedicated broad exotic accuracy panel yet.

The pure 1F/2F Bergomi factories below extend this scope. This is **not** an
implementation of stochastic-dividend LSV, stochastic-rate SV/HW hybrids,
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


## Pure 1F / 2F Bergomi coupling

Rust `StochasticDividendPricingPlan::compile_bergomi` and
`compile_bergomi_two_factor` take the existing Bergomi factor objects and an
explicit dividend/volatility correlation (an ordered pair for 2F). The request
still supplies `sigma0` through its Black-Scholes volatility field.
Python exposes the same factories on `StochasticDividendPlan`. Its
`dividend_mean_reversion` is distinct from the Bergomi `mean_reversion(s)`.
See the [example](../../examples/python/bergomi_dividends.py).

For weighted OU state `Z=sum_j w_j X_j`, `dX_j=-k_j X_j dt+dW_j`,

\[
 v_t=\sigma_0^2\exp(2\nu_B Z_t-2\nu_B^2\operatorname{Var}[Z_t]),
 \qquad df_t=f_t\sqrt{v_t}\,dW_t^f.
\]

Weights and log-volatility convention match the
[pure-SV model](pure-stochastic-volatility.md). The flat initial forward variance
is not a physical-stock market-IV calibration. The Buehler forecast/reserve and
same-date dividend semantics above are unchanged. The discrete scheme preserves
conditional means using left-frozen variance; exact continuous-time martingale
properties require the usual integrability conditions and are not certified
by the finite-step tests.

Order Brownian drivers as `(f,D,V1[,V2])`. The full instantaneous correlation
matrix must be PSD. `rho_DVj` is independent input, not `rho_fD*rho_fVj`.
For interval `h`, set `k_f=k_D=0`. The joint innovations have covariance

\[
 \operatorname{Cov}(I_i,I_j)=\rho_{ij}B(k_i+k_j,h),\quad
 B(k,h)=\begin{cases}(1-e^{-kh})/k,&k>0,\\h,&k=0.\end{cases}
\]

The compiler validates the Brownian matrix first, then integrates and factorizes
this covariance, retaining all three/four independent Gaussian columns even
at singularity or zero loading. Exponential kernel integrals use stable zero-k
limits. Weighted-OU variance is computed as a sum of squared loadings to avoid
cancellation. The Buehler positive split freezes `sqrt(v)` at the left endpoint;
OU state updates affect the **next** interval, not the current equity return.
No variance floor, calibration or leverage approximation is introduced.

Schemes are `buehler-bergomi-1f-joint-ou-positive-split-v1` and
`buehler-bergomi-2f-joint-ou-positive-split-v1`. Unbridged coordinates are
step-major `(f,D,V1[,V2])`; bridged coordinates are rank-major with the bridge
applied to each independent column before correlation. All supplied model
parameters affect the fingerprint, including correlations of unused factors.
Existing BS stochastic-dividend fingerprints and paths are unchanged.

The exact Gaussian OU transition is only one part of the algorithm. Nonlinear
prices retain time-discretization bias. New independent two-step reference tests
are finite-algorithm checks, not a continuous-time convergence certificate.
HW, LSV, rough, multi-asset, Greeks, dividend-derivative calibration, American
exercise and continuous barriers remain unsupported in these factories.

## First-order risk

`evaluate_aad()` returns `StochasticDividendAadRisk` for BS/1F/2F plans.
It differentiates the shared compiled payoff and both Buehler drift halves,
dividend diffusion, all future reserves and their initial funding. Raw labels are
Spot, initial residual volatility, dividend mean reversion, equity linkage,
dividend volatility, cash means in event order, discount log-DF pillars, and
repo-spread log-DF pillars. Both curve grids include fixed time-zero anchors
whose sensitivities are zero. A cash bucket after expiry is generally nonzero.

The initial-volatility loading is computed without dividing by sigma0. At zero
sigma0 or other parameter bounds the reported derivative is the inward derivative.
The equality shortcut in convex drift blending retains its mathematical
state/parameter derivatives. Correlations, Bergomi parameters, dates, time grid,
and smoothing width are fixed. There is no Gamma, market-IV VegaKT or
recalibration derivative. The method name explicitly records the fixed-correlation
scope. Raw derivative arrays and sampling standard errors have matching labels.

`initial_volatility_vega_per_vol_point` and
`dividend_volatility_vega_per_vol_point` multiply the corresponding raw derivative
by 0.01. `cash_mean_adjoints` are price per cash-amount unit.
`discount_node_dv01` and `repo_spread_node_dv01` equal `-1e-4 * t * dPrice/dlogDF`
at each curve pillar, holding the other curve fixed. The legacy market field
`dividend_curve` still supplies the repo-spread curve; no wire rename is made.

Discontinuous payoffs require explicit payoff smoothing before AAD. The risk is
then of the smoothed price, with the smoothing width fixed. Constructors retain
their price-only request contract; request risk flags are not silently ignored.
Existing American and continuous-barrier restrictions remain.

MC standard errors use independent antithetic pair averages when enabled; RQMC
uses scramble averages, never individual Sobol points as independent samples.
They exclude timestep and model uncertainty. Price-only and AAD evaluations use
the same paths, reductions and price fingerprint; risk has a separate method label.
The price-only numerical implementation and previous price fingerprints remain.

See the [risk example](../../examples/python/stochastic_dividend_risk.py).
The [validation record](../../design/validation/stochastic-dividend-risk.md)
distinguishes full-recompile derivative checks, continuous-time BS moment checks
and common-noise finite-grid price comparisons. The finest grid is not exact.
