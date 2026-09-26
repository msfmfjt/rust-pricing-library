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

The pure 1F/2F Bergomi, residual-equity LSV, rough Bergomi and Hull-White
factories below extend this scope. The LSV entry point is deliberately a
calibration of the **funded residual-equity coordinate**, not a direct
physical-stock local-volatility fit. Multi-asset stochastic dividends, dividend
derivatives and dividend-option calibration remain separate scopes with their
own pricing, covariance, calibration and reverse contracts.

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
The plain Bergomi factories do not themselves perform calibration. Hull-White,
residual-equity LSV and rough Bergomi use the dedicated entry points described
below. Multi-asset stochastic dividends, dividend-derivative calibration,
American exercise and continuous barriers remain unsupported here.

## Residual-equity LSV coupling

Rust `StochasticDividendPricingPlan::compile_bergomi_lsv`,
`compile_bergomi_two_factor_lsv` and `compile_rough_bergomi_lsv`, with
matching Python factories on `StochasticDividendPlan`, combine the Buehler
cash-dividend state with the existing particle-calibrated Markovian or rough
Bergomi LSV machinery.

The request model must be `LocalVolatility`, but its target has a narrower
meaning than in the ordinary single-stock Local Volatility engine. Let

\[
F_t^{res}=x_0 f_t
\]

be funded residual equity before deterministic carry reconstruction. The target
grid is interpreted as local variance of \(F^{res}\), with log-moneyness
\(\log(F_t^{res}/x_0)\). The existing particle calibration computes

\[
L^2(t,k)=
\frac{\sigma_{loc,res}^2(t,k)}
{\mathbb E[A_t^2\mid \log(F_t^{res}/x_0)=k]},
\]

where \(A_t\) is the 1F/2F Bergomi multiplier or the finite-grid rough
Bergomi volatility multiplier. Pricing then uses

\[
\frac{dF_t^{res}}{F_t^{res}}
 =L(t,F_t^{res})A_t\,dW_t^f.
\]

The Buehler dividend factor \(Y\) is added only to the joint pricing system. Its
correlations with equity and volatility factors do not change the marginal
\((F^{res},A)\) calibration problem as long as the equity/volatility marginal
law is unchanged. The full pricing Brownian matrix is still validated. Markovian
OU cross-covariances use the exact integrated law; rough paths use the same
finite-grid Volterra history and newest-cell construction in calibration and
pricing.

This is **not** a physical-stock Dupire calibration. Physical stock is

\[
S_t=a(t)f_t+b(t)Y_t+c(t),
\]

so its instantaneous diffusion contains both residual-equity and dividend-factor
exposures. Matching a local-volatility surface quoted directly on \(S\) therefore
requires a different conditional-moment calibration with the \(Y\) state in the
target; the current API does not claim that fit.

The execution grid contains all Local Volatility target knots, contractual
observation times and dividend ex-dates, and is further refined by
`maximum_step`. Calibration and pricing use the same grid. The plan exposes
`lsv_time_nodes`, `lsv_log_moneyness_nodes`,
`lsv_squared_leverage` and `lsv_initial_residual_equity` for audit.
Scheme identifiers are
`buehler-bergomi-1f-residual-lsv-joint-ou-positive-split-v1`,
`buehler-bergomi-2f-residual-lsv-joint-ou-positive-split-v1` and
`buehler-rough-bergomi-residual-lsv-joint-hybrid-positive-split-v1`.

Local-variance risk is opt-in through
`evaluate_local_variance_risk()`. Compile the LSV factory with
`retain_reverse_trace=true`; otherwise the risk call fails before valuation
sampling. The reverse first maps payoff seeds on reconstructed physical Spot
through the Buehler positive split to every squared-leverage node, including
the dependence of leverage interpolation on the residual-equity state. It then
uses the existing finite-particle calibration VJP to return sensitivities to the
**original requested residual-equity Local-variance grid**, including particle
motion and kernel-regression feedback.

The method label is
`buehler-residual-lsv-path-and-discrete-particle-vjp-v1`. RQMC node standard
errors are computed across scramble-level recalibrated gradients; pseudo-MC
returns node gradients without a separate gradient standard error. In both
cases the calibration seed, particle count, bandwidth, fallback decisions,
time grid and model parameters are fixed. Calibration sampling/model uncertainty
is therefore excluded.

Market-IV reporting on the **same residual-equity coordinate** is available
through `evaluate_vega_kt()` when the request contains a VegaKT configuration
and the `LocalVolatility` target contains a reporting-IV basis. The
Local-variance VJP is first converted nodewise to Local-volatility sensitivity,

\[
\frac{\partial P}{\partial \sigma_{loc,res}}
=
2\sigma_{loc,res}
\frac{\partial P}{\partial \sigma^2_{loc,res}},
\]

and is then passed through the shared density gate and first-order VegaKT
projection. This ordering is important: particle leverage is recalibrated
*before* the market-IV projection, so VegaKT includes the finite-particle
calibration response instead of freezing the leverage surface.

For RQMC, each scramble-level recalibrated gradient is projected separately;
bucket sample variances, price/bucket covariances and optional full bucket
covariance therefore have the same independent-unit meaning as elsewhere in
the library. For pseudo-MC the expensive calibration VJP is applied once to
the aggregate gradient, so the result is a point VegaKT estimate and its bucket
sampling variances/covariances are absent. The projection remains conditional
on the fixed calibration seed and particle cloud.

This VegaKT is **not physical-stock \(S\) VegaKT**. Reporting maturities,
log-moneyness nodes and quoted implied volatilities must describe the
\(F^{res}\) market surface used to define the residual-equity Dupire target.
Using a physical-stock reporting surface would mix coordinates and is outside
the current model contract.

Physical-Spot Delta is available separately through
`evaluate_lsv_spot_risk()`. This does **not** freeze the calibrated LSV
surface. The calibration is homogeneous in its initial residual equity
\(F_0^{res}\): calibration particles scale with \(F_0^{res}\), while both
the kernel and leverage lookup use \(\log(F/F_0^{res})\). A Spot bump
therefore re-anchors the surface `initial_f` but leaves every calibrated
squared-leverage value unchanged. With that re-anchoring, the normalized
\(f/Y\) pricing dynamics are Spot-independent, so the finite-algorithm Delta
is exactly the reverse of the Buehler physical-Spot reconstruction
coefficients.

The method label is
`buehler-residual-lsv-scale-invariant-spot-reverse-v1`. MC standard error is
computed over independent antithetic units when enabled; RQMC uses scramble
means. Unlike Local-variance/VegaKT risk, this Spot calculation does not need
`retain_reverse_trace=true`, because no particle-calibration VJP is required.
Validation compares the analytic Delta with a full up/down recompile: the
recompiled plans have shifted residual-equity anchors and unchanged leverage
values.

Residual-LSV Spot Gamma is available through `evaluate_lsv_gamma()`.
It uses the same half/base/double Spot-bump ladder as the non-LSV Gamma API, but
every shifted scenario rebuilds the Buehler physical-Spot coefficients and
re-anchors the LSV surface `initial_f` to the shifted funded residual equity.
The calibrated squared-leverage values are unchanged. Because leverage lookup
uses the relative coordinate \(\log(F/F_0^{res})\), normalized \(f/Y\) states
are identical across the six Spot scenarios and can be reused with common
valuation random numbers.

Gamma is the central difference of the exact scale-invariant LSV Spot Delta,
not a second reverse pass. The result uses the existing
`StochasticDividendGammaRisk` ladder diagnostics and has method label
`buehler-residual-lsv-scale-invariant-aad-delta-gamma-v1`. Sampling errors
are computed on the paired Gamma estimators. Bump differences remain numerical
convergence diagnostics rather than certified error bounds.

The dedicated method works for both 1F and 2F residual-LSV plans and does not
require `retain_reverse_trace=true`. Validation compares every ladder entry
with independently recompiled up/down plans and their
`evaluate_lsv_spot_risk()` Deltas while checking that calibrated leverage
values remain unchanged. The legacy `evaluate_gamma()` continues to reject
LSV plans so the fixed-leverage non-LSV contract cannot be selected
accidentally.

Spot, fixed-cash mean and deterministic curve risk can be requested together
through `evaluate_lsv_market_risk()`. The requested residual-equity
Local-variance grid is held fixed. Cash amounts and discount/repo-spread curves
change funded residual equity \(F_0^{res}\), the affine physical-Spot
reconstruction and, for the discount curve, the payment discount. Re-anchoring
the residual LSV surface to the new \(F_0^{res}\) leaves the relative-coordinate
leverage values and normalized \(f/Y\) dynamics unchanged, so these market
sensitivities are exact pathwise coefficient reverses rather than finite bumps.

The result reports physical-Spot Delta, cash-mean adjoints in event order,
discount-curve and repo-spread-curve log-DF adjoints, sampling standard errors
for each quantity, and DV01 helpers. Time-zero curve anchors remain zero.
Dividends after option expiry can still have nonzero cash-mean risk because they
participate in funded residual equity. The method label is
`buehler-residual-lsv-scale-invariant-market-reverse-v1`.

This market reverse works for both 1F and 2F residual-LSV plans and does not
require `retain_reverse_trace=true`. Full-recompile validation bumps cash
amounts and nonzero curve log-DF pillars, checks that calibrated leverage values
remain unchanged up to floating-point scale effects, and verifies the reported
adjoints against common-random central differences.

Buehler dividend-model parameter risk is available through
`evaluate_lsv_dividend_model_risk(...)` for dividend mean reversion, equity
linkage, dividend volatility and equity/dividend Brownian correlation. None of
these four parameters enters the marginal residual-equity/Bergomi particle
calibration. The production estimator therefore holds the calibrated leverage
surface fixed exactly, rebuilds the Buehler/joint pricing path for each up/down
scenario, and uses common-random central differences.

The mean-reversion and dividend-volatility bumps must stay in the nonnegative
domain, the equity-linkage bump inside `[0,1]`, and the correlation bump inside
`[-1,1]`; the bumped full joint Brownian matrix must also remain admissible.
There is no one-sided fallback or adaptive bump. Both 1F and 2F residual-LSV
plans use the same API. The method label is
`buehler-residual-lsv-common-noise-fixed-calibration-dividend-model-v1`.

This is deliberately a fixed-calibration finite-bump model-parameter risk, not
the Local-variance calibration VJP and not the fixed-volatility
`evaluate_aad()` contract. It does not require `retain_reverse_trace=true`.
Validation independently recompiles all four bumped models and requires every
resulting calibrated squared-leverage surface to be identical to the base
surface before comparing the common-random central differences.

1F Bergomi model-parameter risk is available through
`evaluate_lsv_bergomi_parameter_risk(mean_reversion_bump=..., vol_of_vol_bump=...)`.
This is intentionally **not** the fixed-leverage Bergomi AAD used by pure-SV
plans. Four scenarios are built for
\(k\pm h_k\) and \(\nu\pm h_\nu\). Each scenario reruns the finite-particle
LSV calibration using the same calibration seed, particle count, bandwidth,
fallback rules and grid, then reprices with common valuation random numbers.
The reported derivatives therefore include both the direct pricing response
and the induced leverage-surface response of the finite calibration algorithm.

Only central bumps are supported. Both down-bumped parameters must remain in
the nonnegative Bergomi domain; there is no one-sided fallback or adaptive bump.
The Spot/Bergomi correlation and dividend-volatility/Bergomi correlation are
held fixed. The method currently applies only to the 1F Bergomi residual-LSV
factory; 2F parameters and correlation risk are separate extensions.

The method label is
`buehler-residual-lsv-common-noise-full-recalibration-bergomi-1f-v1`.
MC standard errors are calculated on paired pathwise central differences; RQMC
uses scramble-level paired differences. This is a forward full-recalibration
risk and therefore does not require `retain_reverse_trace=true`. Validation
also performs independent full recompiles at each bumped parameter and checks
that their central differences reproduce the dedicated risk result.

1F correlation risk is available through
`evaluate_lsv_correlation_risk(equity_volatility_correlation_bump=...,
dividend_volatility_correlation_bump=...)`. The two correlations have different
calibration semantics. The equity/Bergomi correlation `rho_fV` belongs to the
marginal residual-equity/Bergomi system used by the particle calibration, so its
up/down scenarios rerun the full LSV calibration. The dividend/Bergomi
correlation `rho_DV` appears only in the joint Buehler pricing Brownian matrix;
it does not change the marginal `(F_res, A)` calibration, so its up/down
scenarios reuse the calibrated leverage surface exactly and rebuild only the
joint pricing kernel.

Both risks use central correlation bumps, common valuation random numbers and
paired MC/RQMC error estimates. There is no one-sided fallback: both bumped
correlations must remain inside the valid correlation domain and the full joint
correlation matrix must remain admissible. The calibration seed and all
non-bumped model inputs are fixed. The method label is
`buehler-residual-lsv-common-noise-bergomi-1f-correlation-v1`, and the method
does not require `retain_reverse_trace=true`.

Validation uses independent full recompiles for both correlations. For
`rho_DV`, those recompiles are also required to reproduce the base calibrated
squared-leverage surface exactly, making the no-recalibration boundary explicit.
The current method applies only to 1F Bergomi residual LSV; 2F cross-factor
correlations remain a separate extension.

2F Bergomi model-parameter risk is available through
`evaluate_lsv_bergomi_two_factor_parameter_risk(mean_reversion_bumps=...,
vol_of_vol_bump=..., mixing_weight_bump=...)`. It reports derivatives with
respect to the ordered factor mean reversions `k1`, `k2`, common
log-volatility vol-of-vol and the raw mixing weight `theta`. Every up/down
scenario reruns the finite-particle residual-LSV calibration with the same
calibration seed and reprices with common valuation random numbers, so the
reported derivatives include the induced change in normalized factor weights,
conditional moments and calibrated leverage.

The mixing-weight bump must stay inside `[0,1]`; mean reversions and
vol-of-vol must remain nonnegative. There is no one-sided fallback or adaptive
bump. Spot/vol correlations, factor correlation and both dividend/vol
correlations are fixed in this method. The method label is
`buehler-residual-lsv-common-noise-full-recalibration-bergomi-2f-v1`.
As with the 1F forward-recalibration risk, `retain_reverse_trace=true` is not
required.

2F correlation risk is available through
`evaluate_lsv_bergomi_two_factor_correlation_risk(...)`. The two spot/volatility
correlations and the volatility-factor correlation belong to the marginal
residual-equity/Bergomi system, so their up/down scenarios rerun the full
finite-particle LSV calibration. The two dividend/volatility correlations affect
only the joint Buehler pricing Brownian matrix; their scenarios reuse the
calibrated leverage surface exactly and rebuild only the joint pricing kernel.

All five derivatives use central bumps with common valuation random numbers and
paired MC/RQMC sampling errors. Each scalar bump must stay inside the individual
correlation range and every bumped full correlation matrix must remain
admissible. The method label is
`buehler-residual-lsv-common-noise-bergomi-2f-correlation-v1`.
This is a forward selective-recalibration risk and does not require a retained
calibration reverse trace.

Existing `evaluate_aad`, Bergomi/correlation AAD and common-noise Gamma remain
rejected on LSV plans: those APIs report a different risk contract and would
freeze calibrated leverage if reused unchanged. The dedicated LSV methods now cover Local-variance risk,
residual-surface VegaKT, physical-Spot Delta/Gamma, cash-mean and
deterministic-curve risk, Buehler dividend-model parameter risk, 1F Bergomi
parameter/correlation risk, and 2F Bergomi parameter/correlation risk.

## First-order risk

`evaluate_aad()` returns `StochasticDividendAadRisk` for BS/1F/2F/rough plans.
It differentiates the shared compiled payoff and both Buehler drift halves,
dividend diffusion, all future reserves and their initial funding. Raw labels are
Spot, initial residual volatility, dividend mean reversion, equity linkage,
dividend volatility, cash means in event order, discount log-DF pillars, and
repo-spread log-DF pillars. Both curve grids include fixed time-zero anchors
whose sensitivities are zero. A cash bucket after expiry is generally nonzero.

The initial-volatility loading is computed without dividing by sigma0. At zero
sigma0 or other parameter bounds the reported derivative is the inward derivative.
The equality shortcut in convex drift blending retains its mathematical
state/parameter derivatives. Correlations, Bergomi/rough parameters, dates, time grid,
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


## Bergomi parameter risk

For a 1F/2F plan, `evaluate_bergomi_aad()` returns the same risk type as
`evaluate_aad()`, preserving every existing label and derivative as a prefix.
Appended raw derivatives are `bergomi_mean_reversion[0]`, and for 2F
`bergomi_mean_reversion[1]`, then `bergomi_vol_of_vol`, and for 2F
`bergomi_mixing_weight`. The basic method still holds those parameters fixed.
The extended risk method is `buehler-bergomi-parameter-reverse-fixed-correlation-v1`.
Neither method recalibrates market IV or provides market-IV VegaKT.

Mean-reversion risk includes exact-OU covariance/factorization and decay
sensitivities. Mixing risk includes normalized weights; all extra risks include
lognormal centering and the existing stochastic reserve/physical-stock payoff
chain. Correlations, grid and smoothing width are fixed. The extra coefficient
Jacobians are built once per risk evaluation; per-path OU adjoints use no bumps.

The extended method requires integrated normalized correlation Cholesky pivots
and 2F weight variance greater than `1e-10`. Singular/near-singular cases reject
before sampling without changing price or basic AAD support. Nonnegative model
parameters at zero and mixing-weight endpoints use inward derivatives. A BS
plan has no Bergomi parameters and rejects this extra method.

```python
risk = plan.evaluate_bergomi_aad()
for label, value, se in zip(risk.parameter_labels, risk.derivatives, risk.standard_errors):
    if label.startswith("bergomi_"):
        print(label, value, se)
```

See the [runnable example](../../examples/python/stochastic_dividend_bergomi_risk.py),
[decision](../../design/adr/0016-stochastic-dividend-bergomi-risk.md) and
[validation record](../../design/validation/stochastic-dividend-bergomi-risk.md).
The reported uncertainties are sampling errors, not timestep/model uncertainty.

## Correlation risk

`evaluate_correlation_aad()` returns all existing basic-AAD results for BS or all
Bergomi-parameter-AAD results for 1F/2F (H/eta AAD for rough) as an exact prefix, then appends raw
Brownian-correlation entry partials. It is opt-in; older risk methods still hold
correlations fixed. The method label is `buehler-joint-correlation-reverse-v1`.

| Appended order | BS | 1F | 2F |
| --- | --- | --- | --- |
| `equity_dividend_correlation` | Yes | Yes | Yes |
| `spot_volatility_correlation[0]` | - | Yes | Yes |
| `spot_volatility_correlation[1]` | - | - | Yes |
| `dividend_volatility_correlation[0]` | - | Yes | Yes |
| `dividend_volatility_correlation[1]` | - | - | Yes |
| `volatility_factor_correlation` | - | - | Yes |

Each partial varies one symmetric off-diagonal pair while holding all other raw
correlations fixed. Values are price per unit correlation, not per percentage
point; `0.01 * derivative` is a one-percentage-point linear approximation only
within the valid domain. The calculation includes dividend-driver rotation,
integrated OU covariance/factorization, and (2F) normalized weights and centering.
It is not the risk of a PSD-projected or recalibrated market model.

Both the instantaneous and integrated normalized-correlation Cholesky pivots,
and 2F weight variance, must exceed `1e-10`. BS requires `1-rho_SD^2 > 1e-10`.
Outside this numerical domain the new method rejects before sampling. It does
not clip or regularize correlations, and existing price/basic/model-risk methods
retain their own domains. Even if OU integration makes the covariance nonsingular,
an instantaneous singular matrix does not admit independent raw entry partials.
The guard applies even when some factors have zero loading. Near the boundary,
large sampling errors are possible despite a successful domain check.

```python
risk = plan.evaluate_correlation_aad()
for label, value, se in zip(risk.parameter_labels, risk.derivatives, risk.standard_errors):
    if "correlation" in label:
        print(label, value, se)
```

See the [example](../../examples/python/stochastic_dividend_correlation_risk.py),
[decision](../../design/adr/0017-stochastic-dividend-correlation-risk.md) and
[validation protocol](../../design/validation/stochastic-dividend-correlation-risk.md).
Dates, simulation grid and payoff-smoothing width are held fixed. The reported
standard errors cover sampling only, not timestep/model uncertainty. There is no
new continuous-time convergence, HW/rough/LSV/multi-asset or VegaKT claim.

## Gamma with common-noise AAD Delta bumps

`evaluate_gamma(GammaConfig)` in Rust, or Python
`evaluate_gamma(gamma_relative_bump=0.01)`, evaluates

\[
\Gamma_h = \frac{\Delta_{\mathrm{AAD}}(S_0+h)-\Delta_{\mathrm{AAD}}(S_0-h)}{2h}.
\]

Specify exactly one `gamma_absolute_bump` or `gamma_relative_bump` in Python.
Relative bumps resolve against the initial physical Spot, not funded residual
Spot. The result always includes the half/base/double ladder, in that order:
`spot_bumps`, `gamma_estimates`, `gamma_standard_errors`. `gamma` and
`standard_error` select the base bump. `delta_change_per_one_percent_spot` is
`0.01 * S0 * Gamma_h`, a linearized Delta change, not Gamma price P&L.
`bump_differences` and `bump_difference_standard_errors` report half-minus-base
and base-minus-double estimates and their paired errors. No extrapolation or
adaptive bump selection is performed.

All bumps hold the cash-mean schedule, model parameters, correlations, curves,
contract constants, grid and smoothing width fixed. Each shifted reserve is
rebuilt, including the post-expiry funding. Normalized state paths can be reused
because they do not depend on initial Spot in these BS/pure-Bergomi models.
All six shifted Spots must be representable and leave positive funded residual
Spot. Unsmoothed discontinuous payoffs reject before sampling; explicit smoothing
returns Gamma of the smoothed payoff. Vanilla calls/puts may use bumped Delta.

The `StochasticDividendGammaRisk` result includes the original price and Delta
with their existing sampling errors, plus a separate `risk_fingerprint` that
identifies the method and bump convention/ladder. `price.evaluated_paths` counts
simulated state paths including antithetics; `payoff_evaluations` is seven times
that count. No second reverse is implied by the method label
`buehler-common-noise-aad-delta-gamma-v1`.

Gamma SE is estimated from the paired Delta difference, not two independent
Delta errors. MC uses independent antithetic units and RQMC uses scramble means.
Neither sampling error nor adjacent-ladder gaps bound finite-bump, timestep,
smoothing or calibration error. In a small sample, tiny bumps near vanilla kinks
can give zero Gamma/SE simply because no paths cross the strike. The ladder is a
diagnostic, not a convergence certificate. The original price and AAD methods,
including their singular-covariance domains, are unchanged.

```python
risk = plan.evaluate_gamma(gamma_relative_bump=0.01)
print(risk.gamma, risk.standard_error)
print(risk.spot_bumps, risk.gamma_estimates)
print(risk.bump_differences, risk.bump_difference_standard_errors)
```

See the [example](../../examples/python/stochastic_dividend_gamma.py),
[decision](../../design/adr/0018-stochastic-dividend-gamma.md) and
[validation protocol](../../design/validation/stochastic-dividend-gamma.md).

## Rough Bergomi price composition

Rust `StochasticDividendPricingPlan::compile_rough_bergomi` and Python
`StochasticDividendPlan.compile_rough_bergomi` combine the same Buehler reserve
with a Riemann--Liouville rough variance driver. The corresponding
`compile_rough_bergomi_lsv` factories calibrate the same finite-grid rough
driver to the funded residual-equity Local-variance target described above.
The direct rough factory reads sigma0 from a Black-Scholes request; the rough-LSV
factory instead requires a `LocalVolatility` request and particle-calibration
configuration. The parameter domain is
`0 < hurst <= 0.5`, `vol_of_vol >= 0`, with finite inputs and a PSD joint
Brownian matrix for residual equity, dividend and variance drivers. Specify
`correlation` for f/variance, `equity_dividend_correlation` for f/dividend and
`dividend_volatility_correlation` for dividend/variance.

The continuous driver and finite-grid variance convention are

\[
X_t=\int_0^t\sqrt{2H}(t-s)^{H-1/2}\,dW_s^v,\qquad
v_i=\sigma_0^2\exp\!\left(\eta X_i-\tfrac12\eta^2 V_i\right).
\]

Here `vol_of_vol` is **eta in log variance**, unlike nu in the 1F/2F log-volatility
convention. H=1/2 corresponds to zero-mean-reversion 1F Bergomi with nu=eta/2.
The request's BS volatility supplies sigma0, not physical-stock market IV.

The hybrid scheme integrates the newest power-kernel cell exactly in distribution
and uses the L2-average power kernel on older cells of the actual, possibly
nonuniform, simulation grid. `V_i` is the variance of this discrete driver, not
`t_i^(2H)`. Variance is frozen at the left endpoint of each positive f/Y split.
The direct newest-cell residual standard deviation is
`dt^H*(0.5-H)/(H+0.5)`; it is zero at H=1/2. Four independent normals are always
reserved per step: equity, orthogonal dividend, orthogonal variance, and
newest-cell residual. Brownian bridging uses independent factors before
correlation. No coordinate is dropped at zero loading or singular correlation.

```python
plan = rp.StochasticDividendPlan.compile_rough_bergomi(
    request, hurst=0.1, vol_of_vol=0.6, correlation=-0.4,
    dividend_mean_reversion=0.7, equity_linkage=0.6, dividend_volatility=0.35,
    equity_dividend_correlation=-0.25, dividend_volatility_correlation=0.15,
    maximum_step=1/32, worker_threads=2, reduction_block_size=64,
)
result = plan.evaluate()
```

**The direct rough-dividend factory supports prices, basic/H-eta AAD, Gamma and
correlation AAD in the instantaneous SPD interior.** Rough residual-LSV plans
instead use the dedicated LSV risk contract: recalibrated Local-variance risk
and VegaKT, scale-invariant Spot Delta/Gamma, market risk and Buehler
dividend-model risk are available, while fixed-calibration AAD remains rejected.
Full-recalibration rough-parameter risk is available through
`evaluate_lsv_rough_bergomi_parameter_risk(hurst_bump=..., vol_of_vol_bump=...)`.
The H and eta up/down scenarios rerun the finite-particle rough-LSV calibration
with the same calibration seed and common valuation random numbers.

Rough correlation risk is available through
`evaluate_lsv_rough_bergomi_correlation_risk(equity_volatility_correlation_bump=...,
dividend_volatility_correlation_bump=...)`. The equity/rough-volatility
correlation belongs to the marginal residual-equity calibration and therefore
reruns that calibration for both bumped scenarios. The dividend/rough-volatility
correlation affects only the joint Buehler pricing covariance, so its bumped
scenarios reuse the calibrated leverage surface exactly. Both derivatives use
central bumps and paired MC/RQMC sampling errors; all scalar and joint
correlation-domain checks still apply. The 1F/2F-specific Bergomi
parameter/correlation risk methods continue to reject rough plans explicitly.
Shared payoff graphs can still price discrete path-dependent contracts, but the
new acceptance tests cover terminal calls and cash-event/path construction, not
broad rough-dividend exotic accuracy.
American exercise, continuous barriers, proportional cash mixtures, stochastic
rates and multiple assets are not added by these rough-dividend factories.

History evaluation and compiled storage are O(N^2); a new explicit limit of 4096
steps avoids unbounded dense-history allocation (~64 MiB of scalar weights).
No FFT or Markovian surrogate is used. The new scheme label is
`buehler-rough-bergomi-joint-hybrid-positive-split-v1`. Old fingerprints and
arithmetic are unchanged; the new identity includes H, eta and both vol-driver
correlations, even when a loading is zero.

Sampling SE excludes hybrid/equity/dividend time-grid error and model uncertainty.
Two-step quadrature tests certify the finite algorithm at selected inputs, not
continuous-time price convergence. See the
[executable example](../../examples/python/rough_dividends.py),
[decision](../../design/adr/0019-rough-stochastic-dividends.md),
[validation](../../design/validation/rough-stochastic-dividends.md), and
Bennedsen, Lunde and Pakkanen,
[Hybrid scheme for Brownian semistationary processes](https://arxiv.org/abs/1507.03004).


## Rough-dividend AAD and Gamma

For a rough plan, `evaluate_aad()` differentiates Spot, initial residual-equity
volatility, the three dividend-model parameters, cash means and original
curve pillars while holding H, eta and all correlations fixed. The causal
rough history is replayed in the same summation order as pricing to provide
all left-endpoint volatility loadings. No normalized-state dependency on
initial physical Spot is introduced.

`evaluate_rough_aad()` preserves that entire basic-risk prefix exactly and
appends `rough_hurst` and `rough_vol_of_vol` (eta), with matching sampling errors.
Its method label is `buehler-rough-hybrid-parameter-reverse-fixed-correlation-v1`.
It differentiates the actual hybrid history, exact newest power-integral cell
and finite-grid variance centering. With

\[
L_i=\exp\left(\frac{\eta X_i}{2}-\frac{\eta^2 V_i^{\rm grid}}4\right),
\]

its local derivatives include

\[
\partial_\eta L_i=L_i(X_i/2-\eta V_i^{\rm grid}/2),\qquad
\partial_H L_i=L_i(\eta\,\partial_H X_i/2-\eta^2\partial_H V_i^{\rm grid}/4).
\]

For the newest cell, write `J=A*dW_v+B*z_residual`, where
`A=sqrt(2H)*dt^(H-1/2)/(H+1/2)` and
`B=dt^H*(1/2-H)/(H+1/2)`. Then
`dA/dH=A*(1/(2H)+log(dt)-1/(H+1/2))` and
`dB/dH=B*log(dt)-dt^H/(H+1/2)^2`.
At H=1/2, B vanishes but its inward H derivative does not. Older-cell weight
and centering derivatives are compiled once, then their history rows are
transposed per path. Correlation factorization remains fixed. Fixed singular
PSD correlations need no regularization for this risk scope.

H=1/2 means the inward left derivative; eta=0 means the inward right derivative.
There is no division by eta or sigma0. These are raw derivatives at fixed
other inputs, not quoted-vol/recalibrated sensitivities or continuous-time
risk-convergence guarantees.

```python
risk = plan.evaluate_rough_aad()
print(risk.delta, risk.initial_volatility_vega_per_vol_point)
print(risk.parameter_labels[-2:], risk.derivatives[-2:])
gamma = plan.evaluate_gamma(gamma_relative_bump=0.01)
print(gamma.spot_bumps, gamma.gamma_estimates, gamma.gamma_standard_errors)
```

Gamma uses the existing common-noise half/base/double Delta-bump ladder at
fixed H/eta and correlations. All Spot scenarios share immutable rough kernel
coefficients; the O(N^2) dense history is not copied six times. Per-path time
and compiled storage remain O(N^2); extended H risk needs an additional
triangular coefficient-Jacobian array. The 4096-step cap still applies.
Dates, grid and smoothing width are held fixed; unsmoothed discontinuous risk
still rejects. All uncertainty and small-bump caveats in the Gamma section
apply here too. H/eta AAD and Gamma do not add HW, LSV, VegaKT or multiple assets.

See the [example](../../examples/python/rough_dividend_risk.py),
[decision](../../design/adr/0020-rough-stochastic-dividend-risk.md) and
[validation protocol](../../design/validation/rough-stochastic-dividend-risk.md).


## Rough-dividend correlation AAD

`evaluate_correlation_aad()` on a rough plan returns the entire H/eta-risk
prefix, then `equity_dividend_correlation`, `spot_volatility_correlation[0]`,
and `dividend_volatility_correlation[0]`. The common method label remains
`buehler-joint-correlation-reverse-v1`; the price scheme and fingerprint identify
the rough family. The raw derivatives are per unit correlation. Multiply by
0.01 for a linearized +1 correlation percentage-point move, provided the scenario
remains admissible. These are Brownian-driver, not physical-stock return,
correlations. No market recalibration or PSD projection is performed.

With fixed H, the hybrid driver at node i is

\[
X_i=A_{i-1}\Delta W^v_{i-1}+B_{i-1}z_{i-1,3}
    +\sum_{j<i-1}w_{ij}\Delta W^v_j.
\]

All correlation dependence enters the Brownian factor row
`dWv_j = sqrt(dt_j) * sum_k L_vk z_jk`. The reverse transposes each history row
into every earlier increment, then through the analytic Cholesky derivatives.
The direct dividend-driver rotation supplies an additional equity/dividend
term. The fourth independent newest-cell residual has no correlation derivative.
The volatility Brownian marginal variance stays fixed, hence
`d V_grid / d rho = 0` and `d loading / d rho = loading * eta/2 * dX/d rho`.

Every pivot of the instantaneous 3x3 Brownian matrix must exceed `1e-10`, even
at zero loadings. Singular/near-singular input rejects this scope before
sampling, without changing fixed-correlation price/basic/H-eta/Gamma support.
At H=1/2 the fourth near-cell residual vanishes; this does not preclude
correlation risk if the 3x3 Brownian matrix is SPD. No separate nonsingularity
condition is imposed on the augmented newest-cell covariance. All four random
coordinates are retained, and no extra time discretization is introduced.

```python
risk = plan.evaluate_correlation_aad()
for label, value, se in zip(risk.parameter_labels, risk.derivatives, risk.standard_errors):
    if "correlation" in label:
        print(label, value, se)
```

Existing H/eta and basic risks keep other inputs fixed for each partial and
are returned unchanged. Raw correlation risk adds nine prepared coefficients
and O(N) path workspace; its history transpose is O(N^2). The combined scope
also retains the H-derivative coefficient table. All existing grid/resource,
payoff smoothing and sampling-uncertainty caveats remain. No continuous-time
risk-convergence, stochastic-rate/HW, LSV/VegaKT or multi-asset support is implied.

See the [example](../../examples/python/rough_dividend_correlation_risk.py),
[decision](../../design/adr/0021-rough-stochastic-dividend-correlation-risk.md) and
[validation protocol](../../design/validation/rough-stochastic-dividend-correlation-risk.md).

## Stochastic-rate extension

The separate [BS/Buehler/Hull–White price plan](stochastic-dividends-hull-white.md)
uses conditional discounted cash forecasts and explicit rate correlations.
Existing deterministic-rate plans and their risk methods keep their domains.
