# Rough and Lifted Heston: Riccati/Fourier pricing

[Model index](README.md) · [Six model families](rough-volatility-families.md) ·
[Python example](../../examples/python/heston_fourier.py) ·
[Validation record](../../design/validation/heston-fourier.md)

## Scope and coordinates

`pricing::rough_volatility::HestonFourierPlan` and Python `HestonFourierPlan`
provide an additive **experimental European call/put** pricing route for
`RoughHeston` and `LiftedHeston`. The transform describes the continuous-time
model, evaluated by deterministic numerical integration. It is not the transform
of either existing full-truncated finite-grid Monte Carlo scheme. The other four
rough model families are rejected; they are not silently approximated by Heston.

The caller supplies a positive martingale forward, a nonnegative strike, maturity
in years, and a nonnegative deterministic payment discount. The payoff is
`max(+/-(F_T-K),0)`. A delayed payment can use its own deterministic discount, while
variance evolves only until the option fixing. There is **no automatic conversion
of stock/cash-dividend payoffs**, no stochastic-rate/dividend or LSV composition,
no model calibration, Greeks/AAD, VIX/SSR or path-dependent payoff support in this
route. Feeding a raw spot/forward with fixed cash dividends into a multiplicative
model is not a substitute for the required escrow-coordinate payoff conversion.
Existing request JSON tags, MC pricing, sampling layouts and fingerprints are
unchanged.

## Transform convention and continuous-time equations

The model parameters and kernels are those of the six-family contract:

```text
V = v0 + K * [kappa(theta-V) dt + nu sqrt(V) dW]
dF = F sqrt(V) dB,  d<B,W> = rho dt
rough: K(t) = t^(alpha-1)/Gamma(alpha), alpha=H+1/2
lifted: K(t) = sum w_i exp(-x_i t)
```

Define `X=log(F_T/F_0)` and `z=d+i*u`. The public `transform(d,u)` is
`M(z)=E[exp(z X)]`, exposed only for `0<=d<=1`. `d=0` is the characteristic
function. Python returns `(real, imaginary)`, not a Python complex instance.
For the affine Volterra transform [1–3],

```text
R(z,h) = (z*z-z)/2 + (rho*nu*z-kappa)*h + nu*nu*h*h/2
h = K * R(z,h), h(0)=0
log M(z) = v0 integral_0^T R(z,h(t)) dt
           + kappa*theta integral_0^T h(t) dt.
```

The last formula is equivalent by Fubini to integrating `R(z,h(t))` against
`g0(T-t)`, where `g0(t)=v0+kappa*theta*integral_0^t K(s)ds`. For the rough kernel,
`integral R = I^(1-alpha)h(T)`. This parameterization uses `nu` as the actual
square-root diffusion coefficient, not `kappa*nu` as in the original version
of [1]. The original paper's correlation restriction is not used as an assertion
of general validity; [2–3] give the affine-Volterra/martingale-strip framework.

## Calculation specifications

The Riccati solver integrates a **piecewise-linear interpolant of R** against
its kernel. For rough Heston this is a uniform-grid implicit fractional
trapezoidal product integration. The implicit quadratic is solved at every
node; it is **not** the explicit Adams predictor/corrector used in the independent
reference. Its root uses the principal square root and cancellation-aware
algebraic forms. Residual and non-finite checks return errors, not fallback prices.

For Lifted Heston, exponential decay and its linear hat integrals are exact
within each interval, including zero-rate and very stiff modes. The common
nonlinear endpoint is solved by the same scalar quadratic. This integrates the
continuous-time finite-factor Riccati ODE, not the semi-implicit MC factor law.
`H=1/2` uses the equivalent single zero-rate factor without a history convolution.
The final two ordinary integrals use the trapezoidal rule. Constant-variance,
absorbing-zero and zero-maturity limits are handled directly.

For European pricing use Lewis's half-moment contour [1, section 5.2; 4] and a
Black control variate. Choose control total variance `Q=-8 log M(1/2)`, matching
its half moment. This stays nonzero when v0 is zero but variance immigration is
positive; Q is **not claimed to be expected integrated variance**. The call is

```text
C = discount * [BlackCall(F,K,Q) + sqrt(F*K)/pi * integral_0^U
    Re{exp(i*u*log(F/K)) [exp((z*z-z)*Q/2)-M(z)]}/(u*u+1/4) du],
z=1/2+i*u.
```

The same correction added to the Black put preserves call/put parity. Composite
Simpson quadrature is used on `[0,U]`. Transform samples are cached per maturity
and reused across strikes. The OTM Black leg is evaluated first, then parity.
Non-finite outputs and prices outside intrinsic/upper bounds are errors; there
is no silent clipping. These checks do not establish accuracy of an in-bounds
price.

## Explicit resolution and diagnostics

`FourierConfig` defaults to 512 Riccati steps, 512 even integration intervals,
and cutoff 128. Rust accepts 2–8192 time steps, 4–8192 even integration intervals,
and a finite cutoff in `(0,10000]`. An estimated 2.5 billion kernel-operation
cap rejects oversized plans before allocation. These are resource guards, **not
accuracy targets**. A plan fingerprint covers the model, maturity, grids and
scheme, not the forward/strike/discount passed to a later price call.

`refinement` returns four prices, with configurations

```text
(N,M,U), (2N,M,U), (2N,2M,U), (2N,4M,2U).
```

The three signed adjacent changes isolate time refinement, Fourier mesh
refinement, and cutoff extension at a fixed Fourier mesh. They are not standard
errors, certified tail bounds or proof of convergence. The method does not
silently increase resolution until a desired answer is obtained and does not
return a convergence boolean. Very short maturity, small variance, extreme
strikes and stressed parameters may require much larger cutoffs or time grids.

## References

1. El Euch and Rosenbaum, [The characteristic function of rough Heston models](https://arxiv.org/abs/1609.02108), theorem 4.1 and sections 5.1–5.2. Fractional Riccati transform and fractional trapezoidal/Adams construction.
2. Abi Jaber, [Lifting the Heston model](https://arxiv.org/abs/1810.04868), equations (2.6), (2.10)–(2.12). Finite-factor transform and admissible input curves.
3. Abi Jaber, Larsson and Pulido, [Affine Volterra processes](https://arxiv.org/abs/1708.08796), *Annals of Applied Probability* 29 (2019), 3155–3200. Affine transform framework.
4. Lewis, [A Simple Option Formula for General Jump-Diffusion and Other Exponential Levy Processes](https://doi.org/10.2139/ssrn.282110), 2001. Fourier contour inversion. The subtraction of a Black control and the numerical policies above are implementation choices, not performance claims from this paper.
