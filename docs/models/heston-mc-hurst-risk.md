# Rough Heston: MC Hurst risk

Experimental **Pure-SV** Hurst sensitivity through the implemented hybrid
full-truncation scheme. It supplements [fixed-kernel scalar MC risk](heston-mc-parameter-risk.md)
and does not change the existing price, random coordinates, or five-scalar API.

## Contract

`RoughVolatilityPlan.evaluate_hurst_risk()` returns `RoughHestonMcHurstRisk`:
unchanged `price`, `hurst_sensitivity`, sampling `standard_error`, `method`, and
`risk_fingerprint`. Python additionally exposes the coordinate
`rough_heston_hurst_fixed_scalar_parameters`. Derivative units are price per one
absolute H unit. For a small H change of .01, multiply the derivative by .01.
All properties are read-only. Scalar model parameters, physical Spot, curves,
complete fixed/proportional dividend schedule, dates, grid and Gaussian samples
remain fixed. No parameter bump, model recalibration or extra particle calibration
is performed. This is not IV Vega, LSV Hurst risk or a Fourier derivative.

Rust additionally exposes `RoughHestonMcHurstPlan::compile(&path_plan)`, its
`evolve_path(initial_forward, normals)` and a recorded-path `reverse` accepting
fixed cotangents on forward **and diffusion-variance** observations.
Kernel derivatives are cached once per plan, not recomputed for each MC sample.
Raw latent-variance observation seeds are not supported.

Only power-kernel Rough Heston is accepted. Lifted Heston, including a lift made
by a rough-model factory, has no unique H derivative at fixed coefficients and
is explicitly rejected. The parent domain requires v0>0 and |rho|<1. Exact zero
preterminal raw variance is rejected; strictly negative raw variance has zero
truncation derivative. Terminal raw zero is rejected only when seeded.
No variance floor or path removal is introduced. Positive near-zero variance
can lead to noisy derivatives.

## Calculation specifications

Let a=H+1/2 and let C be the average of the unnormalized kernel
`u^(a-1)/Gamma(a)` over a lag cell. The drift weight is C times the cell width.
Differentiate its analytic cell integral, including Gamma's logarithmic
`digamma(a)` derivative. Older-cell diffusion uses `C*dW`.

For a newest cell of width dt, the unnormalized innovation is

```text
Cnear*dW + Rnear*z_res
Cnear = dt^(H-1/2) / [(H+1/2)*Gamma(H+1/2)]
Rnear = dt^H*(1/2-H) / [sqrt(2H)*(H+1/2)*Gamma(H+1/2)].
```

Differentiate **both** loadings, holding the existing independent normals fixed.
At H=1/2 the API returns the **left derivative**. Although Rnear is zero there,
its left derivative is `-sqrt(dt)` and must not be discarded. The finite-H
Gaussian coupling is the implementation's reparameterization; it is not a claim
of the exact joint continuous Brownian-integral law for every different H.

The existing reverse of the complete variance history supplies row adjoints.
At row i, accumulate `barV_i*(kappa*(theta-V_j)*dt*dC_dH +
nu*sqrt(V_j)*dInnovation_dH)` from all preceding cells. Propagation through
previous variance states remains the existing full-truncation reverse.
The real-arithmetic kernel function is differentiated at floating primals;
compensation branches and the machine representation are not economic kinks.
Work remains O(n^2) per rough path; derivative-cache storage is O(n^2).

The hybrid representation is related to [Bennedsen, Lunde and Pakkanen](https://arxiv.org/abs/1507.03004).
The derivative formulas here are direct differentiation of this library's
specific finite scheme, not a method attributed to that paper. Gamma/digamma
identities follow [NIST DLMF chapter 5](https://dlmf.nist.gov/5).

## Statistics and limitations

MC uses independent paths or antithetic pairs; RQMC uses independent scramble
means. Standard error excludes time/full-truncation/smoothing/model bias and
is not a total-error bound. A zero derivative in a constant-variance limit has
zero standard error, not a fabricated positive number. Hard discontinuities
require explicit smoothing. No universal differentiation-under-expectation or
finite second-moment guarantee is claimed, especially near zero variance.

Finite bump agreement tests discrete derivatives, not continuous-time accuracy.
The independent deterministic-variance Black test uses the exact integrated
variance of **the finite scheme**, not the continuous fractional mean-reversion
law. Small-H MC bias is not removed. No 5 IV bp guarantee, second-order AAD,
other-family parameter risk, or LSV Hurst recalibration risk is added.

See the [validation protocol](../../design/validation/heston-mc-hurst-risk.md)
and [runnable example](../../examples/python/heston_mc_hurst_risk.py).
