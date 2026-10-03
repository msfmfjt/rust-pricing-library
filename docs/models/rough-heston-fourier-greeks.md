# Heston Fourier: Forward Delta and Gamma

The experimental [rough/lifted Heston Fourier plan](rough-heston-fourier.md)
provides `price_and_greeks(forward, strike, discount)`. It returns
`HestonFourierGreeks`, whose `price` is exactly the existing `price` result.
The model and cached transforms are unchanged by evaluating this method.

These are discounted **forward** derivatives with the model parameters,
maturity, strike and discount factor fixed. They are not physical-Spot Delta or
Gamma, an implied-volatility sticky-strike risk, a recalibrated risk, or an AAD
result. Model-parameter Greeks, VegaKT, Theta and discount-curve risk are not
provided. No Spot/cash-dividend, LSV or stochastic-rate mapping is inferred.

## Calculation specifications

Let `M(z)` denote the transform of `log(F_T/F)` in the underlying affine model,
`w = v0*T` the existing Black-control integrated variance, and

```text
x = log(F/K)
A = D*sqrt(F*K)/pi
B(u) = [exp(-(u*u+1/4)*w/2) - M(1/2+i*u)] / (u*u+1/4)
J(x) = integral_0^cutoff Re[exp(i*u*x)*B(u)] du
```

The finite-grid price is the Black control plus `A*J`. Differentiating with
respect to F while leaving the cached model and the control variance fixed gives

```text
call Delta_F = D*N(d1)
             + (A/F) integral Re[(1/2+i*u)*exp(i*u*x)*B(u)] du
put  Delta_F = -D*N(-d1) + the same correction
Gamma_F     = D*n(d1)/(F*sqrt(w))
             - (A/F^2) integral (u*u+1/4)*Re[exp(i*u*x)*B(u)] du
```

Thus the Gamma multiplier cancels the denominator of B. Call and put have the
same Gamma and their Deltas differ by D. The code differentiates the **cached
finite quadrature**, without price bumps, calibration or additional Riccati
solves. It reuses the same composite-Simpson nodes and compensated sums.
Differentiating a truncated quadrature does not certify the corresponding
continuous-model Greek. Gamma is particularly sensitive to the high-frequency
part of the transform, so price convergence alone is not a Greek accuracy test.

The affine transform and numerical solver are specified in the parent model
reference, based on El Euch--Rosenbaum, *The characteristic function of rough
Heston models* (arXiv:1609.02108), and Abi Jaber, *Lifting the Heston model*
(arXiv:1810.04868). The forward-derivative identities above follow by direct
differentiation; they are not an additional fractional-Riccati model assumption.

## Return values and diagnostics

Rust exposes `HestonFourierGreeks` through `pricing::rough_volatility`. Python
exposes a frozen class with read-only properties:

| Property | Meaning |
| --- | --- |
| `price` | Unchanged `HestonFourierPrice` result |
| `call_forward_delta` | Partial derivative of discounted call with respect to F |
| `put_forward_delta` | Partial derivative of discounted put with respect to F |
| `forward_gamma` | Common second derivative with respect to F |
| `delta_quadrature_difference` | Absolute Delta difference, Simpson N versus N/2 |
| `gamma_quadrature_difference` | Absolute Gamma difference, Simpson N versus N/2 |
| `delta_tail_indicator` | Delta correction envelope on `[cutoff/2, cutoff]` |
| `gamma_tail_indicator` | Gamma correction envelope on `[cutoff/2, cutoff]` |

Neither the grid differences nor the retained-interval envelope integrals are
omitted-tail bounds or total numerical error estimates. They exclude Riccati
error, model error and calibration uncertainty; there is no sampling SE. They
must not be used to claim an IV-basis-point error guarantee.

## Domain and failure behavior

Forward, strike and discount must be finite and strictly positive. Only the two
affine families accepted by `HestonFourierPlan` are supported. The existing
price checks run first and any price failure propagates.

For zero maturity, or an identically zero variance process (`v0=0` and
`kappa*theta=0`), derivatives away from ATM are intrinsic: the call Delta is D
above strike and zero below, the put Delta is minus D below strike and zero
above, and Gamma is zero. At the **exact ATM kink**, Delta is undefined and
Gamma is not an ordinary finite function. The method returns `InvalidInput`
(`ValidationError` in Python), rather than inventing Delta=0.5 or Gamma=0.

For a non-degenerate law with `v0=0`, the existing Black control has zero
variance and its separate derivatives are singular. This first Greek API
explicitly rejects that case, even though `price` may succeed. It does not
silently replace the control or change existing prices. Underflow of a positive
control variance to zero is also rejected. Inputs that overflow computed
sensitivities cause an explicit numerical failure.

Non-finite derivatives and material violations of `0 <= call Delta <= D`,
`-D <= put Delta <= 0` or nonnegative Gamma cause numerical failure. A tolerance
of `1e-10*D` is used for Delta and for the dimensionless check `F*Gamma`; no value
is clamped. These checks are necessary shape conditions, not accuracy guarantees.

## Example

```python
import rust_pricing as rp

model = rp.RoughVolatilityModel.rough_heston(
    hurst=0.1, initial_variance=0.04, mean_reversion=0.7,
    long_run_variance=0.055, vol_of_vol=0.18, correlation=-0.65,
)
plan = rp.HestonFourierPlan.compile(
    model, 1.0, time_steps=512, integration_intervals=512, cutoff=128.0,
)
g = plan.price_and_greeks(100.0, 100.0, 0.97)
print(g.price.call, g.call_forward_delta, g.forward_gamma)
```

The executable [Python example](../../examples/python/heston_fourier_greeks.py)
includes both rough and lifted models. See the
[validation protocol](../../design/validation/rough-heston-fourier-greeks.md)
for the independent references, exact scope and numerical evidence.

## Related parameter risk

[Fixed-kernel scalar parameter sensitivities](heston-parameter-risk.md) are
available separately from Forward Delta/Gamma and are not calibrated market risk.
