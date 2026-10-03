# Rough and Lifted Heston: Fourier forward prices

## Status and contract

This is an **experimental continuous-time-model pricing method**, evaluated by
numerical Riccati integration and a truncated Fourier integral. It is not a
closed-form answer, an automatically calibrated model or a certified error bound.
The companion [Monte Carlo families](rough-volatility-families.md) use different
time-discretized laws. See the [validation record](../../design/validation/rough-heston-fourier.md).

Only `RoughHeston` and `LiftedHeston` are accepted. All four other family tags
return `FourierError::UnsupportedModel`; there is no fallback to a wrong model.
No schema/model-tag changes, stochastic rates, LSV, stochastic dividends,
Greeks/AAD, VIX, SSR or calibration are added.

The contract is a vanilla call and put on a **positive forward**, with payoff
`(F_T-K)+` or `(K-F_T)+` at the stated maturity. Forward, strike and discount
factor must be finite and strictly positive. Discounting is provided explicitly;
this API does not derive a forward from spot, reconstruct an escrow reserve,
apply a cash jump or handle a later payment date. In particular, inserting a
physical spot into the forward argument does not implement fixed-cash dividends.
The caller must use a forward/measure consistent with the model and discounting.
Deterministic carry can be incorporated in that forward outside this API.

## Continuous-time model and transform

Let `X_t=log(F_t/F_0)` and let `dF_t/F_t=sqrt(V_t)dB_t`, with
`d<B,W>_t=rho dt`. The parameters are the existing family parameters:

```text
V_t = v0 + integral_0^t K(t-s) kappa(theta-V_s) ds
           + integral_0^t K(t-s) nu sqrt(V_s) dW_s.

R(z,psi) = (z*z-z)/2 + (rho*nu*z-kappa)*psi + nu*nu*psi*psi/2.
psi(t) = integral_0^t K(t-s) R(z,psi(s)) ds.

log E[exp(z X_T)]
  = v0 integral_0^T R(z,psi(t)) dt
    + kappa*theta integral_0^T psi(t) dt.
```

Rough Heston has `alpha=H+1/2`, `K(t)=t^(alpha-1)/Gamma(alpha)`.
The initial-variance term is **not `v0*psi(T)`** except in the Heston boundary
`H=1/2`. Equivalently it is `v0*I^(1-alpha)psi(T)`; since `psi=I^alpha R`,
this equals `v0*integral R`. This matches the fractional-Riccati transform in
[El Euch and Rosenbaum](https://arxiv.org/abs/1609.02108). Their convention
with diffusion `lambda*nu` must be mapped to this library's diffusion `nu`;
mean reversion must not be multiplied into vol-of-vol a second time.

Lifted Heston has `K(t)=sum_i w_i exp(-x_i*t)`. Its transform states obey
`psi_i'=-x_i psi_i+R(z,sum_i w_i psi_i)` and start at zero. The variance
initial curve corresponding to the constant-initial-variance model is
`g0(t)=v0+kappa*theta*integral_0^t K(s) ds`. The affine exponent
`integral R(z,psi(s))*g0(T-s) ds` reduces, by convolution/Fubini, to the same
formula above; see [Abi Jaber](https://arxiv.org/abs/1810.04868).

`log_transform(real+i*imag)` returns this normalized log transform, without
multiplying by `F_0^z`. Only the closed moment strip `0<=real<=1` is exposed,
with finite `|imag|<=1e6`. `characteristic_function(u)` is `exp(log_transform(iu))`.
The exact identities at exponents zero and one are returned directly. These
identities are mathematical inputs, not empirical martingale tests of an MC run.

## Riccati discretization

For the fractional kernel, the implementation integrates the piecewise-linear
interpolant of `R` against the power kernel exactly. The endpoint is implicit.
It solves a scalar quadratic at each step using its rationalized small root,
continuous at zero vol-of-vol. This is **implicit linear product integration**;
it is not a claim to implement the predictor-corrector iteration from the paper.
The affine exponent is then integrated by the trapezoidal rule. Complexity is
`O(time_steps^2)` per Fourier node.

For a finite lift, exponential decay and the integrals of each linear hat
function are evaluated analytically (small-argument series avoid cancellation).
The common implicit endpoint again reduces to one quadratic. The integrator
handles large positive rates without imposing an explicit-Euler rate-step
restriction; it does not eliminate all integration error. Work is
`O(factors*time_steps)` per Fourier node. Zero-rate factors and splitting a
factor into identical-rate parts are supported.

Quadratic residuals, finite values, the transform's necessary moment-strip
bound and option bounds are checked. Failure returns an error; prices are not
clipped into arbitrage bounds and failed roots are not silently repaired.
Passing those checks is not proof of grid convergence or correct root selection
in every extreme parameter regime. Refine grids and compare independent methods
before admitting a new parameter regime.

## Fourier inversion and diagnostics

For `M(z)=E exp(z X_T)`, the half-moment representation is

```text
call = discount * (F - sqrt(F*K)/pi * integral_0^infinity
                  Re[exp(i*u*log(F/K))*M(1/2+i*u)]/(u*u+1/4) du).
```

The code subtracts a Black control transform with integrated variance `v0*T`,
adds its analytic price, and applies composite Simpson quadrature on `[0,cutoff]`.
Put pricing uses the same correction to the Black put, preserving put-call
parity without subtracting a large in-the-money call. A compiled plan caches
all half-moment nodes, so reusing it at other positive forwards or strikes does
not rerun Riccati integration. The model parameters and maturity stay fixed.

`HestonFourierPrice` exposes two **diagnostics, not error estimates**:

- `quadrature_difference`: absolute price difference between the full and
  every-other-node Simpson grids, at the same cutoff and Riccati grid.
- `tail_indicator`: the envelope integral of the absolute control-variate
  integrand on the last half of the retained interval. It does **not** bound the
  omitted integral beyond the cutoff; control-variate cancellation can make it
  small without controlling other errors.

Neither quantity includes Riccati time error, model error or MC uncertainty.
There is no reported Monte Carlo standard error. Separately vary time steps,
frequency spacing and cutoff. A small frequency diagnostic is not a reason to
skip time refinement. The default `512/512/128` is a starting setting, not an
accuracy guarantee or a universally sufficient configuration.

Allocation/work guards are `2..8192` time steps, `8..8192` frequency intervals
(divisible by four), `0<cutoff<=10000`, and an estimated arithmetic work cap of
8 billion kernel operations per compilation. These are resource guards, not
recommended workloads. Unsupported, invalid or numerically unstable inputs
return errors; huge maturities/model scales may therefore be rejected.

## Public API

Rust exports under `pricing::rough_volatility` include `HestonFourierConfig`,
`HestonFourierPlan`, `HestonFourierPrice`, `FourierError` and `Complex64`.
Complex arithmetic itself lives in the finance-independent numerics crate.

```rust
use pricing::rough_volatility::{HestonFourierConfig, HestonFourierPlan, RoughHeston};
let model = RoughHeston::new(0.1, 0.04, 0.7, 0.055, 0.18, -0.65)?;
let config = HestonFourierConfig::new(1024, 512, 128.0)?;
let plan = HestonFourierPlan::compile(model.into(), 1.0, config)?;
let prices = plan.price(100.0, 100.0, 0.97)?;
```

The Rust fragment assumes an enclosing function returning a compatible error.
Python uses frozen `HestonFourierPlan` and `HestonFourierPrice` classes:

```python
import rust_pricing as rp
model = rp.RoughVolatilityModel.rough_heston(
    hurst=0.1, initial_variance=0.04, mean_reversion=0.7,
    long_run_variance=0.055, vol_of_vol=0.18, correlation=-0.65,
)
plan = rp.HestonFourierPlan.compile(
    model, 1.0, time_steps=1024, integration_intervals=512, cutoff=128.0,
)
prices = plan.price(100.0, 100.0, 0.97)
print(prices.call, prices.put, prices.quadrature_difference, prices.tail_indicator)
print(plan.characteristic_function(2.0))  # (real, imaginary)
```

See [the runnable example](../../examples/python/heston_fourier.py).
Invalid inputs map to `ValidationError`; unsupported model/numerical failures
map to `PricingError`. The compiled plan releases the Python GIL during
numerically expensive calls. No new runtime dependency is added to the wheel.

## Fixed-model forward sensitivities

`price_and_greeks` adds [Forward Delta and Gamma](rough-heston-fourier-greeks.md)
without changing the existing `price` method. These are not physical-Spot,
recalibrated, or model-parameter Greeks.
