# Mixed rough Bergomi parameter risk

Experimental manual first-order reverse-mode derivatives of the shared-driver
variance mixture, for Pure SV and particle LSV. This is not the existing
one-/two-factor Markovian Bergomi API.

## Fixed inputs and coordinates

The new `evaluate_mixed_bergomi_parameter_risk()` method is available on Rust
`RoughVolatilityPricingPlan` / `RoughFamilyLsvPricingPlan`, and Python
`RoughVolatilityPlan` / `RoughFamilyLsvPlan`.

Parameters are `vol_of_vol[0]`, ..., `vol_of_vol[m-1]`, `correlation` in that order.
Eta is log-variance volatility. Derivatives are per absolute parameter unit, not
market-IV Vega. Hurst, convex mixture weights, the entire Forward Variance Curve,
Spot, discount/repo curves, cash/proportional dividends (including future cash),
contractual fixings/strikes/payment/smoothing, time grid and Gaussian draws are fixed.
No parameter bump or stochastic-model refit is used in production.

For LSV the relative Dupire Local Variance target values and axes are fixed, while
Leverage is recalibrated. Particle counts, calibration seed, bandwidth, support,
donor and interpolation active sets are fixed. A retained reverse trace is required.
This is not a sticky-absolute-strike market-IV convention.

## Finite-scheme reverse

At positive time nodes the implemented law is

    V_i = sum_j c_ij,
    c_ij = xi(t_i) w_j exp(eta_j X_i - eta_j^2 Q_i / 2),
    dV_i/deta_j = c_ij (X_i - eta_j Q_i),
    dV_i/drho = sum_j c_ij eta_j dX_i/drho.

`Q_i` is the variance of the actual discrete hybrid Gaussian driver, not a
silently substituted continuous-time variance. Correlation differentiates the
Brownian component `rho*z_asset + sqrt(1-rho^2)*z_independent`, holding the asset
normal fixed. The newest-cell independent residual draw and its loading are fixed
in this eta/rho contract. Hurst derivatives are not added here.

The Pure SV reverse propagates payoff cotangents through the affine escrow map and
log-Euler asset steps into variance nodes, then into each mixture component and
shared rho. LSV retains both the direct valuation contribution and the parameter
contribution through all earlier particle states and Leverage rows. The existing
particle variance transpose is shared without changing its arithmetic. The LSV
variance-only reverse does not generate an unused unlevered asset path.

## Results and uncertainty

`MixedBergomiMcParameterRisk` returns the unchanged nested `price`,
`parameter_names`, `parameter_adjoints`, marginal `standard_errors`, `method` and
`risk_fingerprint`. Python coordinate is
`mixed_bergomi_eta_rho_fixed_kernel_weights_and_xi`.

`MixedBergomiLsvParameterRisk` additionally returns `direct_adjoints` and
`calibration_adjoints`; their sum is the total `parameter_adjoints`, up to reduction
roundoff. Python coordinate is
`mixed_bergomi_eta_rho_fixed_relative_local_variance_target`.

Pure SV MC errors use independent paths/antithetic pairs; RQMC errors use independent
scramble means. For LSV each scramble's direct and calibration contributions are
added **before** estimating total marginal errors, retaining their covariance.
Pseudo-MC LSV errors are `None`, not zero. Marginal errors do not specify a parameter
covariance matrix or the error of an arbitrary linear combination.

All errors exclude calibration-seed uncertainty, support changes, time-discretization,
smoothing and model error. Agreement with finite differences is not a
continuous-time or IV-5bp accuracy guarantee.

## Boundaries and ownership

Only Mixed rough Bergomi is accepted. Correlation must be strictly inside (-1,1),
even when a degenerate input makes its derivative unnecessary. Existing price-only
APIs retain their wider domain. Eta=0 uses the smooth inward derivative; it is not
assumed zero. Zero-weight components have zero eta adjoints. Zero xi nodes are
identically zero under these fixed-xi parameter perturbations and are allowed by
the Pure SV reverse. Existing LSV denominator/support checks still apply.

Invalid/nonfinite seeds, source/plan mismatch, missing trace and unsupported hard
discontinuous payoffs fail explicitly. Digital/Barrier pathwise risk requires the
existing explicit payoff smoothing. No floor, clipping or path dropping is introduced.
Python result properties are frozen and arrays are returned as owned copies.

The result's separate risk fingerprint includes the method and unchanged price-plan
fingerprint, which already covers model parameters, weights and xi. No existing
price or risk fingerprint is changed.

See the [validation protocol](../../design/validation/mixed-bergomi-parameter-risk.md)
and [complete Python example](../../examples/python/mixed_bergomi_parameter_risk.py).

An explicit [Hurst extension](mixed-bergomi-hurst-risk.md) appends H to these risks.
