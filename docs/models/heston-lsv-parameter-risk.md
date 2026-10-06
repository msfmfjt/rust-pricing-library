# Heston LSV parameter risk at a fixed Local Variance target

Experimental first-order, finite-scheme model-parameter adjoints for Rough Heston
and Lifted Heston with particle Leverage recalibration. This is distinct from
[Pure-SV parameter risk](heston-mc-parameter-risk.md),
[Pure-SV Hurst risk](heston-mc-hurst-risk.md), and
[market-IV quote adjoints](rough-family-market-iv.md).

## Contract

Rust: `RoughFamilyLsvPricingPlan::evaluate_heston_parameter_risk(include_hurst)`.
Python: `RoughFamilyLsvPlan.evaluate_heston_parameter_risk(*, include_hurst=False)`.
The result is `HestonLsvParameterRisk` in both APIs.

The parameter order is `initial_variance`, `mean_reversion`,
`long_run_variance`, `vol_of_vol`, `correlation`. With `include_hurst=True`,
`hurst` is appended for power-kernel Rough Heston only. Finite lifts, including
factory-generated ones, reject that option: weights and rates remain fixed.
The left H derivative is used at H=0.5, including the newest-cell Gaussian
residual-loading derivative, exactly as in the Pure-SV Hurst contract.

Hold physical Spot, curves, the entire fixed/proportional cash schedule,
contract dates/strikes/smoothing, relative Local Variance target values and axes,
initial funded-forward anchor, grids, random inputs, particle count, bandwidth,
minimum-support rule, donor indices and interpolation branches fixed. Change
model parameters and recalibrate Leverage to the **same target**. This does not
refit model parameters to market prices, differentiate a Dupire/IV surface, or
provide a sticky-absolute-strike Spot convention. Derivatives are per absolute
parameter unit; the v0 derivative is a variance sensitivity, not IV Vega.

The returned fields are:

| Field | Meaning |
|---|---|
| `price` | Unchanged existing LSV price, error, seed and price fingerprint |
| `parameter_names` | Five names, optionally followed by Hurst |
| `parameter_adjoints` | Total derivative after particle Leverage recalibration |
| `direct_adjoints` | Valuation derivative with the Leverage values fixed |
| `calibration_adjoints` | Effect through the parameter-dependent calibration |
| `standard_errors` | Total-gradient marginal RQMC SEs; `None` for pseudo MC |
| `method` | `heston-lsv-fixed-target-parameter-particle-vjp-v1` |
| `risk_fingerprint` | Includes parent price fingerprint and Hurst-selection flag |

Python additionally returns
`coordinate="heston_parameters_fixed_relative_local_variance_target"`.
Properties are read-only; array getters return copies. Total equals direct plus
calibration contribution up to floating-point reduction order. Diagnostic
contribution fields are not separately calibrated risk conventions.

## Discrete reverse

Write v[i,r] for each model's diffusion variance and ell[r,j] for squared
Leverage. For one particle, q = ell(t,f) v and the asset update is

    f_next = f exp(-q dt/2 + sqrt(q dt) z).

Payoff seeds first propagate through this update to both ell and v. Squared
Leverage's spatial derivative is included in the backward state recursion. The
variance seeds are then passed through the existing full-truncation Heston
history reverse; correlated normals are differentiated holding the asset normal
fixed. No parameter bumps are used in the production path.

For calibration, M2_j = sum_i w_ij v_i / sum_i w_ij and ell_j = a_j/M2_j.
Thus the M2 adjoint is -bar(ell_j) a_j/M2_j^2. It contributes both

    bar(v_i) += bar(M2_j) w_ij / sum_i w_ij,
    bar(f_i) += bar(M2_j) (v_i-M2_j) (dw_ij/df_i) / sum_i w_ij.

The latter term propagates through the **earlier** calibration asset steps and
Leverage rows; it must not be dropped. Extrapolated nodes route moment adjoints
to the retained donor node. At time zero the primal uses empirical initial
moments, so each particle receives sum_j bar(M2_j)/particle_count rather than a
kernel-regression derivative. Third/fourth moments are diagnostics and do not
enter the squared-Leverage formula in this implementation.

After reversing the calibration time march, regenerate each calibration
variance history with its original `LsvCalibration` random domain, and reverse
all diffusion-variance seeds into scalar/Hurst parameters. No unused unlevered
stock path is generated; its overflow cannot spuriously reject an LSV risk.
Reverse plans are compiled separately from the valuation and calibration
histories' recorded grids. H-dependent loadings are cached once per plan, not
once per path. The current high-level compiler uses a shared calibration and
valuation grid.

## Uncertainty and boundaries

For each RQMC scramble, reduce the valuation direct gradients and Leverage
seeds, pass the latter through the complete particle-calibration reverse, and
**add the two contributions before** computing the total-gradient sample SE.
They are correlated. Neither a sum of marginal SEs nor independent quadrature
of contribution SEs gives the total-gradient error. Antithetic pairing occurs
before the point reduction. Pseudo MC currently exposes no parameter SE: it
returns `None`, not zero. RQMC marginal errors are not a full parameter covariance
matrix or a combined-direction error.

All sampling errors condition on one calibration sample; they exclude
calibration-seed variation, finite-grid/full-truncation bias, smoothing and model
error. Tiny positive variances can generate unstable pathwise sensitivities.
There is no general finite-moment or derivative/interchange guarantee.

A retained particle trace is required. The inherited Heston reverse requires
v0>0 and |rho|<1, rejects exact zero preterminal raw variance, and assigns zero
truncation derivative to strictly negative raw variance. Exact terminal zero
is allowed only when its variance is unseeded. Nonnegative scalar-parameter
boundaries retain the parent's inward-extension convention. No variance floor or path dropping
is added. Invalid/nonfinite seeds, unsupported models, and hard discontinuous
risk payoffs without explicit smoothing are errors. Support/donor/interpolation
switches are not differentiated: results describe the retained local branch.

The additional particle-variance adjoint table is capped at 8,000,000 doubles,
matching the existing calibration cell limit. The reverse is sequential in time
and particle identity; worker policy does not change numerical reduction order.
There is one calibration pullback per RQMC scramble, or one on the aggregate in
pseudo MC, not a fresh calibration per valuation path. Rough history work is
quadratic in time steps and lifted-history work is linear in steps times factors.

## Example and scope

See the [complete Python example](../../examples/python/heston_lsv_parameter_risk.py)
and [validation protocol](../../design/validation/heston-lsv-parameter-risk.md).

This connects Heston model-parameter changes through Leverage recalibration.
Other rough-family scalar/covariance adjoints, finite-lift factory/weight/rate
sensitivities, direct second-order AAD and paper-style continuous VegaKT projection
remain separate work. The existing small-H pricing bias is not removed by
correctly differentiating the discrete calculation.
