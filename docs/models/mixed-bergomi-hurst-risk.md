# Mixed rough Bergomi MC and LSV Hurst risk

Experimental first-order reverse of the implemented finite Gaussian hybrid
scheme. This adds Hurst to [component eta/shared rho risk](mixed-bergomi-parameter-risk.md)
without changing the old API, primal arithmetic or its risk identity.

## API and coordinates

Rust `RoughVolatilityPricingPlan` and `RoughFamilyLsvPricingPlan`, and Python
`RoughVolatilityPlan` and `RoughFamilyLsvPlan`, provide
`evaluate_mixed_bergomi_parameter_risk_with_hurst()`.
It returns the existing MixedBergomiMcParameterRisk or MixedBergomiLsvParameterRisk
with eta[0],...,eta[m-1],correlation followed by `hurst`. The old
`evaluate_mixed_bergomi_parameter_risk()` keeps its shape, method and fingerprint.
Rust also exposes a cached `MixedBergomiMcHurstPlan` for recorded-path reverse and
`CalibratedRoughFamilyLsv.reverse_mixed_bergomi_parameters_with_hurst(seeds)`.

All derivatives are per one absolute parameter unit. Mixture weights, the
entire Forward Variance Curve, Spot, curves, full cash/proportional dividends,
fixings, payments, smoothing, time grid and Gaussian draws remain fixed. Only
Mixed rough Bergomi with interior correlation is accepted. H=0.5 returns its
left derivative. No finite lift, Markovian Bergomi or other family is silently
substituted. Zero fixed xi nodes and zero eta/weight components retain the
existing contract. No variance floor or discarded path is introduced.

For LSV, relative Local Variance target nodes/axes, bandwidth, calibration draws
and retained support/donor/interpolation branches are fixed, but Leverage is
recalibrated. `parameter_adjoints` includes `direct_adjoints` plus
`calibration_adjoints`. This does not refit the stochastic parameters to market
quotes and is not market-IV Vega. A retained particle reverse trace is required.

Python coordinates:
- Pure: `mixed_bergomi_eta_rho_hurst_fixed_weights_and_xi`.
- LSV: `mixed_bergomi_eta_rho_hurst_fixed_relative_local_variance_target`.

Methods are `mixed-bergomi-mc-parameter-hurst-vjp-v1` and
`mixed-bergomi-lsv-fixed-target-parameter-hurst-vjp-v1`, respectively. Each has
its own risk fingerprint while retaining the existing price fingerprint.

## Derivative of the actual discrete law

Let alpha=H+1/2. Newest-cell Brownian and independent-residual loadings are

    C = sqrt(2H) dt^(H-1/2) / alpha
    R = dt^H (1/2-H) / alpha
    dC/dH = C [1/(2H) + log(dt) - 1/alpha]
    dR/dH = dt^H [(1/2-H) log(dt)/alpha - 1/alpha^2].

At H=1/2, R vanishes but dR/dH=-sqrt(dt); it must not be discarded.
Older weights are the normalized power-kernel interval averages; their analytic
H derivatives use stable log1p/expm1/series evaluation for nearly equal lags.
The finite driver variance is

    Q_i = dt_near^(2H) + sum_j W_ij^2 dt_j,
    dQ_i/dH = 2 log(dt_near) dt_near^(2H) + sum_j 2 W_ij dW_ij/dH dt_j.

For c_ij = xi_i weight_j exp(eta_j X_i - eta_j^2 Q_i/2),

    dc_ij/dH = c_ij [eta_j dX_i/dH - eta_j^2 dQ_i/dH / 2].

The centering derivative uses Q_i, not the continuous t_i^(2H). Kernel
derivatives are compiled once per risk evaluation, not per path. Work/cache
remain O(n^2), within the parent maximum 2048 Volterra steps. The recorded
floating primals define the real-arithmetic discrete reverse; this is not a
claim about differentiating rounding branches or a unique cross-H continuous
Brownian-integral coupling.

## Uncertainty and limitations

Pure MC errors use independent paths/antithetic pairs; RQMC uses independent
scramble means. LSV adds direct and calibration effects in each scramble before
computing total-gradient marginal errors, preserving their covariance. Pseudo-MC
LSV errors remain None. No parameter covariance matrix is returned.

These errors exclude calibration-seed uncertainty, support/donor switching,
time-grid/smoothing/model bias and errors of the continuous model. Hard
payoff discontinuities still require explicit smoothing. The independent
nondegenerate expectation is a two-step law, not an arbitrary-maturity LSV
accuracy certificate or a 5 IV bp guarantee.

Mixture-weight/Forward Variance Curve risks, other families' parameters, second
order AAD and continuous VegaKT projection are outside this increment. See the
[validation protocol](../../design/validation/mixed-bergomi-hurst-risk.md) and
[complete example](../../examples/python/mixed_bergomi_hurst_risk.py).
