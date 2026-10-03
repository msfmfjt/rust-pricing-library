# Additional rough-volatility families

[Model index](README.md) · [Components](components/README.md) ·
[Python example](../../examples/python/rough_volatility_families.py) ·
[Validation status](../../design/validation/rough-volatility-families.md) ·
[Finite-grid refinement](../../design/validation/rough-volatility-refinement.md) ·
[Factor-price validation](../../design/validation/lifted-heston-factor-prices.md)

**Experimental, price-only extension.** Six families have compiled Rust/Python
implementations with executable boundary, sampling and finite-grid reference
tests. These are not accepted production models: the tested two-step prices
and finite 64/128-step comparisons do not establish continuous-time accuracy
or general parameter-range robustness. Positive-lag lift-kernel checks are not
option-price error bounds. The linked validation records separate
local evidence, CI gates and remaining acceptance work.

## API and support boundary

Rust exports the model types and `RoughVolatilityPathPlan` /
`RoughVolatilityPricingPlan` from `pricing::rough_volatility`. Python exports
`ForwardVarianceCurve`, `RoughVolatilityModel`, `RoughVolatilityPathPlan`,
`RoughVolatilityPath` and `RoughVolatilityPlan`. Models are immutable, plans own
their parameters, and returned Python vectors are copies.

The pricing adapter takes an existing **price-only BlackScholes request** as a
carrier for market data, contractual payoff, dates and sampling settings. The
explicit rough model supplies **all** volatility parameters; the carrier's
BlackScholes sigma is not used in the rough dynamics. Stable JSON model tags and
existing engines are unchanged. This is an additive model-specific entry point,
not automatic routing of a stable request.

The adapter reuses the existing single-asset hybrid payoff graph, deterministic
rate/repo curves and escrowed fixed/proportional dividend treatment. Future cash
reserves include dividends beyond option expiry; dividend events before expiry
are included in the grid. Physical pre- and post-dividend values are supplied to
the payoff graph. Discounting occurs once, at the contract's payment date.
European vanilla, digital, arithmetic Asian, discrete fixed-strike lookback and
discretely monitored Barrier payoff routes are inherited from that graph;
coverage executed specifically for this extension is narrower and is listed in
the validation record. American exercise, continuous Barriers, smoothing-width
ladders and all requested Greeks are rejected by the extension or shared graph.

This MC adapter does **not** implement Fourier pricing (a separate
[forward-only API](rough-heston-fourier.md) does), model calibration (the separate
[Heston price-calibration API](heston-calibration.md) does), VIX contracts, SSR calculations, AAD/Greeks, leverage-function
calibration, stochastic rates, stochastic dividends or multi-asset composition.
A model's support elsewhere in the library does not imply that combination is
available through this adapter. In particular, using the existing
`HullWhitePrice` result shape does not imply stochastic-rate support.

All times are in years. Fractional models accept `0 < H <= 1/2`, with the Brownian
boundary exposed explicitly. Valid parameter ranges and finite arithmetic are
necessary but do not prove existence, uniqueness or martingality for arbitrary
model/parameter combinations.

## Rough Heston

With `alpha = H + 1/2`, the fractional kernel is

```text
K(t) = t^(alpha-1) / Gamma(alpha)
V(t) = v0 + integral K(t-s) kappa(theta-V(s)) ds
          + integral K(t-s) nu sqrt(V(s)) dW(s)
dF(t) = F(t) sqrt(V(t)) dB(t),   d<B,W>(t) = rho dt.
```

The implementation stores the raw Volterra variance and uses its positive part
in drift/diffusion coefficients and asset evolution. Negative raw nodes remain
visible in `latent_states` and `negative_variance_nodes`. This is hybrid
full-truncation discretization, not an exact positivity-preserving simulator.
Drift cell integrals are exact. The newest singular stochastic cell is jointly
Gaussian with the corresponding Brownian increment; older cells use averaged
kernel weights. The limit `H=1/2` is the full-truncation Euler Heston recursion.

El Euch and Rosenbaum derive the characteristic function through a fractional
Riccati equation [1]. A separate [Fourier forward API](rough-heston-fourier.md)
now provides that calculation without changing this MC discretization.

## Lifted Heston

For positive `weights[i]` and nonnegative `rates[i]`,

```text
K_m(t) = sum_i w_i exp(-x_i t)
V(t) = v0 + sum_i w_i U_i(t),   U_i(0) = 0
dU_i = -x_i U_i dt + kappa(theta-V) dt + nu sqrt(V) dW.
```

Every factor shares the **same** variance Brownian motion. The scheme makes the
linear factor decay implicit and truncates the diffusion variance:

```text
U_i(next) = [U_i + kappa(theta-V_positive) dt
                 + nu sqrt(V_positive) dW] / (1+x_i dt).
```

`LiftedHeston::from_rough` / `lifted_heston_from_rough` builds positive weights and
rates using geometric-bin quadrature of the fractional kernel's Laplace measure
[2]. For bin `[l,u]` and `alpha=H+1/2`,

```text
w = (u^(1-alpha)-l^(1-alpha)) / [Gamma(alpha) Gamma(2-alpha)]
x = (1-alpha)/(2-alpha)
    * (u^(2-alpha)-l^(2-alpha))/(u^(1-alpha)-l^(1-alpha)).
```

Both Laplace tails are truncated. Factor count and geometric ratio are explicit
approximation parameters, **not a tolerance certificate**. Finite factor counts
can be inadequate, particularly near `H=1/2`; kernel, price and Greek errors need
separate study over the intended time range. Exactly `H=1/2` uses the single
`w=1, x=0` factor. A finite lift is Markovian and is not mathematically rough.

For a refinement experiment, factor count and geometric ratio must be considered
together: keeping the ratio fixed does not shrink the log-bin width. A
[test-only sequence](../../design/validation/rough-volatility-refinement.md)
uses `r_n = exp(3/sqrt(n))`, so the log-bin width shrinks while both tails expand.
The retained check samples five positive lags between `1/128` and `1` year;
it is neither a tolerance-certified production factory nor a price-error bound.
Time-grid refinement holds the lift fixed at 20 factors and ratio 2.5. Do not
interpret finite-factor and finite-time errors as interchangeable.

A separate [factor-price panel](../../design/validation/lifted-heston-factor-prices.md)
holds 64 time steps fixed and compares 8-256 factors against the infinite-Laplace-
measure limit of the **same semi-implicit scheme**. It covers H=.1/.3, .25/1-year
horizons and three strikes, with paired sampling errors and raw-truncation/forward
moment diagnostics. The target is not the continuous-time model, nor the existing
hybrid rough-Heston simulator at the same grid. Consequently, the observed small
256-factor price differences do not bound time or truncation bias.

## Quadratic rough Heston

The implemented pure-feedback subfamily is [3]

```text
V(t) = a (Z(t)-b)^2 + c
Z(t) = z0 - lambda integral K(t-s) Z(s) ds
           + lambda eta integral K(t-s) sqrt(V(s)) dB(s).
```

Here `a=quadratic`, `b=shift`, `c=variance_floor`, and the asset and `Z` use the
same Brownian motion. `mean_reversion=lambda` multiplies **both** the mean
reversion and feedback diffusion. General time-dependent input functions are
not provided. Variance is a quadratic function of the feedback state; this is
not the square of Heston variance. `a=0` gives constant variance `c`.
The same hybrid fractional-kernel scheme is used, with `latent_states=Z`.

## Mixed rough Bergomi

The supplied `ForwardVarianceCurve` is `xi0(t)`. With one shared normalized
Gaussian Volterra factor,

```text
X(t) = sqrt(2H) integral (t-s)^(H-1/2) dW(s)
V(t) = xi0(t) sum_i w_i exp(eta_i X(t) - eta_i^2 Var[X(t)]/2).
```

Weights are nonnegative and sum to one; only tiny floating roundoff in the sum
is normalized. This is a **mixture of variance components**, not a mixture of
option prices. The implemented subfamily shares `H`, `W` and asset correlation
across components; independent component drivers are not exposed. In the finite
grid scheme the compensator uses the **actual discrete Gaussian variance**, not
`time^(2H)`, so the discrete forward variance is centered consistently. A single
component is the `beta=1` Rough SABR variance specification below. See [4].

The curve supports constant variance, piecewise-linear interpolation with flat
tails, and an exponential form. Positive exponential values that overflow or
underflow are rejected rather than silently made deterministic. Negligible
individual mixture components may round to zero, but losing every positive
component is an error.

## Rough SABR

The forward-variance formulation [5] is

```text
dF(t) = sqrt(xi_t(t)) F(t)^beta dB(t)
dxi_t(u) = xi_t(u) eta sqrt(2H) (u-t)^(H-1/2) dW(t).
```

`vol_of_vol=eta` is a **log-variance** coefficient. At `H=1/2`, classical SABR
with `d alpha = nu alpha dW` requires **both**

```text
eta = 2 nu
xi0(t) = alpha0^2 exp(nu^2 t).
```

A flat initial forward variance does not produce the same classical SABR
volatility process. The exponential curve allows this boundary without
interpolation error.

The asset scheme is log Euler for `beta=1`, signed additive Euler for `beta=0`,
and explicit absorbing Euler for `0<beta<1`. Negative proposals in the latter
case become zero, and zero is absorbing. `absorbed_forward_steps` counts first
absorptions. No exact CEV sampling or convergence-rate claim is made.

In the equity/dividend pricing adapter, the CEV power applies to the **normalized
funded-forward coordinate**, not physical Spot. With `beta=0` this is a normal
approximation that may produce negative physical values. The direct path API
also accepts negative initial forwards only for `beta=0`.

## RFSV: stationary fractional OU log volatility

The implementation uses stationary fractional OU, rather than a renamed rough
Bergomi process [6]:

```text
X(t) = m + nu integral_{-infinity}^t exp(-kappa(t-s)) dB_H(s)
sigma(t) = exp(X(t)),   V(t) = exp(2 X(t)).
Var[X] = nu^2 Gamma(2H+1) / (2 kappa^(2H)).
```

`initial_log_vol=None` samples the stationary law. A supplied value conditions
on **X(0) only**, not the complete past. At `H=1/2` this reduces to the exact
finite-grid stationary OU covariance. At `nu=0` the process is constant at `m`;
an incompatible requested initial value is rejected.

The asset Brownian motion is **independent** of the fractional log-volatility
field. No ad hoc fractional leverage correlation is added. Historical RFSV
parameter estimates do not automatically define risk-neutral pricing parameters.

The compiler builds a dense covariance factor. At `z=kappa*abs(t-s)` and `q=2H`
the normalized covariance is evaluated through

```text
rho(z) = integral_0^infinity exp(-u)
         [abs(u-z)^q + (u+z)^q - 2 z^q] du / (2 Gamma(q+1)).
```

A globally adaptive Simpson rule, a cusp split and endpoint transformation are
used. The infinite tail is cut at 64 and the target absolute integration error
is `2e-12` before normalization. This is a numerical error **target**, not a
rigorous posterior bound. Negative long-lag correlations are valid for `H<1/2`.
Failed quadrature, nonpositive-definite or numerically singular covariance
matrices return errors; no diagonal jitter or covariance clipping is hidden.
Gaussian finite-grid sampling is exact in distribution only up to the numerical
covariance and floating-point factorization. Asset evolution remains discretized.

## Sampling, complexity and reproducibility

Rough Heston and Quadratic rough Heston reuse each left-node diffusion
coefficient within a path. This [performance-only cache](../../design/validation/rough-diffusion-cache.md)
preserves the hybrid schemes and summation order; it does not remove time-grid
or truncation bias.

The new plans reuse Philox, scrambled Sobol, antithetic pairs and deterministic
block reduction. Their coordinate layout is separate from existing engines.
Only actual Brownian blocks receive Brownian-bridge transformation; hybrid
near-cell residuals and fOU Gaussian level coordinates do not. Asset increments
always use left-node variance, preventing volatility lookahead.

An MC antithetic pair is one independent unit. RQMC standard error is calculated
across independent scramble means, not across the individual Sobol points.
Reported error excludes time discretization, truncation/absorption, lift-kernel
error, covariance integration, calibration uncertainty and model error.

| Family | Grid-step cap | Main path cost | Random dimension for n steps |
| --- | ---: | --- | ---: |
| Rough Heston, mixed rough Bergomi, Rough SABR | 2,048 | O(n^2) convolution | 3n |
| Quadratic rough Heston | 2,048 | O(n^2) convolution | 2n |
| Lifted Heston | 65,536 | O(n * number of factors) | 2n |
| RFSV | 256 | O(n^2), after O(n^3) covariance factorization | 2n+1 |

These are allocation guards, not recommendations for accuracy or speed. The
Sobol backend can impose a tighter dimension limit. Single-platform replay and
bitwise price/standard-error equality across worker counts are tested. Plan
fingerprints intentionally include execution policy and therefore change when
the worker count changes; numerical equality does not mean identical metadata.

## References

1. El Euch, O. and Rosenbaum, M. (2016), [The characteristic function of rough Heston models](https://arxiv.org/abs/1609.02108).
2. Abi Jaber, E. (2018/2019), [Lifting the Heston model](https://arxiv.org/abs/1810.04868).
3. Gatheral, J., Jusselin, P. and Rosenbaum, M. (2020), [The quadratic rough Heston model and the joint S&P 500/VIX smile calibration problem](https://arxiv.org/abs/2001.01789).
4. Bourgey, F., De Marco, S. and Gobet, E. (2022), [Weak approximations and VIX option price expansions in forward variance curve models](https://arxiv.org/abs/2202.10413).
5. Fukasawa, M. and Gatheral, J. (2021), [A rough SABR formula](https://arxiv.org/abs/2105.05359).
6. Gatheral, J., Jaisson, T. and Rosenbaum, M. (2014), [Volatility is rough](https://arxiv.org/abs/1410.3394).

## Separate Fourier calculation

Rough Heston and Lifted Heston additionally expose an experimental
[continuous-time-model Fourier forward pricer](rough-heston-fourier.md).
It uses a separate positive-forward vanilla contract, not the MC pricing
request or its cash-dividend adapter. Other families remain MC-only here.
