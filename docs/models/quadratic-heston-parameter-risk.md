# Quadratic rough Heston model-parameter risk

Experimental first-order manual reverse-mode derivatives of the existing
finite-grid, fixed-driver quadratic model. This adds neither a new pricing law
nor a continuous-time accuracy guarantee.

## API and coordinates

Rust: `RoughVolatilityPricingPlan::evaluate_quadratic_heston_parameter_risk(include_hurst)`
and `RoughFamilyLsvPricingPlan::evaluate_quadratic_heston_parameter_risk(include_hurst)`.
Python: both corresponding plans expose
`evaluate_quadratic_heston_parameter_risk(*, include_hurst=False)`.

Results are `QuadraticHestonMcParameterRisk` and `QuadraticHestonLsvParameterRisk`.
The parameter order is `initial_state`, `mean_reversion`, `vol_of_vol`,
`quadratic`, `shift`, `variance_floor`, optionally followed by `hurst`.
These are derivatives per absolute parameter unit. The model's `variance_floor`
is the additive variance constant c, not a numerical clipping floor or IV Vega.
Hurst uses its left derivative at H=1/2.

Both results return the unchanged nested `price`, `parameter_names`,
`parameter_adjoints`, `standard_errors`, `method` and `risk_fingerprint`.
LSV additionally returns `direct_adjoints` and `calibration_adjoints`; their sum
is the total derivative, up to floating-point reduction order. Python properties
are read-only and array getters return copies. The optional Hurst flag changes
the risk method and fingerprint, not the underlying price identity or the six
scalar entries. The QRH model has no independently variable price/volatility
correlation: its feedback Brownian motion is the stock Brownian motion.

Low-level Rust provides `QuadraticHestonMcRiskPlan`, `QuadraticHestonMcRecordedPath`
and `QuadraticHestonMcAdjoints`. Fixed cotangents seed every forward and diffusion
variance observation, not latent Z observations. The calibration-only API is
`CalibratedRoughFamilyLsv::reverse_quadratic_heston_parameters(seeds, include_hurst)`.

## Fixed model and LSV convention

The implemented model is

```text
V_i = a (Z_i-b)^2+c
Z_i = z0 + sum(j<i)[-kappa Z_j D_ij + kappa nu sqrt(V_j) J_ij].
```

Here D integrates the unnormalized fractional kernel over the cell and J is
its hybrid Brownian cell approximation. Near-cell integrals retain their joint
Gaussian law with the stock increment. The extra residual draw has index n+j
in the existing two-block QRH layout, not the three-block Heston layout.

LSV applies Leverage to the asset only. **Leverage is not inserted into Z's
feedback equation**. This preserves the existing fixed-driver model definition;
it is not a model driven by the leveraged stock return.

All market curves, physical Spot, the complete fixed/proportional dividend
schedule (including post-expiry reserve), payoff dates/strikes/smoothing, grids
and random inputs remain fixed. For LSV, relative Local Variance target values
and axes, calibration seed/bandwidth and retained support/donor/interpolation
branches also remain fixed. Leverage is recalibrated as model parameters change.
This is not refitting the stochastic parameters to market quotes, market-IV
Vega, sticky-absolute-strike Spot risk, or second-order AAD.

## Reverse calculation

The payoff and log-Euler asset reverse seed diffusion variances. Each reversed
variance polynomial contributes to a, b, c and the latent Z cotangent. Every
Volterra row has its own z0 source. Future drift and diffusion terms then seed
earlier Z and V nodes, kappa and nu. In particular, kappa multiplies **both**
the drift and the feedback diffusion.

The a=0 primal shortcut does not remove the derivative `(Z-b)^2` with respect
to a. No production parameter bumps are used. Optional Hurst derivatives use
the existing cached unnormalized-kernel derivative, including Gamma-function
normalization, cell drift weights and the newest independent residual loading.
At H=1/2 that loading is zero, but its left derivative is `-sqrt(dt)`.

LSV separately reverses conditional moments, particle weights/positions, earlier
Leverage rows and initial moments using the existing particle-variance transpose.
Those cotangents then traverse the QRH variance history. Variance-only reverse
does not simulate an unused unlevered asset that could underflow or overflow.

The history work remains O(n^2); no claim of new simulation acceleration is made.
Hurst loadings are cached once per reverse plan, not once per path. Existing
maximum time-grid and particle-cell resource limits remain in force.

## Domain and uncertainty

The new risk requires strictly positive **preterminal diffusion variances**.
An exact zero there is rejected rather than divided by, floored or skipped.
Terminal zero variance is allowed because this scope differentiates the smooth
variance polynomial and no following square root is needed. Negative/nonfinite
inputs or adjoints fail explicitly. The original pricing domain is unchanged.
At nonnegative scalar parameter boundaries, the returned polynomial/recurrence
derivative is interpreted inward whenever this variance domain holds. The full
risk API conservatively rejects preterminal zero even when a smaller selected
parameter subset might admit a derivative.

Hard discontinuous payoffs require explicit smoothing. Other payoff active-set
semantics are inherited. Particle support/donor/interpolation switching is not
included in the derivative, and a retained calibration reverse trace is required.

Pure MC standard errors use independent paths or antithetic pairs. RQMC uses
independent scramble means. LSV combines direct and recalibration contributions
**inside each scramble before estimating the total gradient's marginal error**;
contribution errors must not be combined as independent. Pseudo-MC LSV errors
are `None`, not zero. Full parameter covariance is not returned.

Errors omit calibration-seed variation, grid/bump-reference/roundoff/smoothing
bias and model error. Differentiation-under-expectation for arbitrary parameters
is not guaranteed by finite-path tests. A two-step reference checks that finite
law only; it does not certify continuous-time or five-IV-basis-point accuracy.

See the [validation protocol](../../design/validation/quadratic-heston-parameter-risk.md)
and [complete Python example](../../examples/python/quadratic_heston_parameter_risk.py).
