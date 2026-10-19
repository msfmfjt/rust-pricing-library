# Mixed rough Bergomi weight and Forward Variance Curve risk

Experimental first-order reverse of the finite model, not market-IV Vega or a
stochastic-parameter refit. The new no-argument method
`evaluate_mixed_bergomi_shape_risk()` is available on Rust/Python Pure SV and LSV
pricing plans. It returns the existing `MixedBergomiMcParameterRisk` or
`MixedBergomiLsvParameterRisk` with a separate method, coordinate and fingerprint.
The existing eta/rho and eta/rho/Hurst methods retain their contracts.

## Coordinates and domain

For m normalized mixture weights, return m-1 coordinates
`weight_transfer[i,m-1]`: increase weight i by epsilon and decrease the LAST
weight by epsilon, with all other weights unchanged. This fixes the sum at one;
it is not an unconstrained derivative or a logit/softmax/raw-weight derivative.
The last component is a coordinate reference, not an omitted risk. One component
has no weight coordinates. All weights must be strictly positive for this scope.
An arbitrary sum-zero perturbation is represented by the first m-1 components;
its price change is the dot product with these transfer derivatives.

Curve coordinates follow the transfers. A constant curve has a single
`forward_variance[0]`; a piecewise-linear curve has one `forward_variance[i]`
per ORIGINAL input knot, with all knot times fixed. Linear interpolation and flat
extrapolation are differentiated exactly. A knot after the final observation can
still influence earlier values through interpolation and is not silently dropped.
An exponential curve returns `forward_variance_initial` and
`forward_variance_growth` for xi(t)=initial*exp(growth*t). The initial/knotted
variances must be positive; at most 4096 combined coordinates are accepted.
Existing prices and eta/rho/Hurst risks retain their wider zero-weight/zero-xi
contracts. Since rho is fixed here, its endpoints remain allowed.

Units are price per absolute normalized-weight or variance/growth unit, as named.
A weight transfer of 0.01 is one percentage point of mixture weight. A variance
node bump is not a volatility bump. Hold Hurst, component etas, rho, Spot, all
curves/dividends and contractual dates/strikes/smoothing fixed.

## Discrete reverse

Let E_j=exp(eta_j*X-eta_j^2*Q/2), using the existing finite-grid Q. For V=xi*sum(w_j*E_j),

    dV/d transfer_i = xi*(E_i-E_last)
    dV/d xi = sum(w_j*E_j).

At time zero V=xi(0) exactly and all weight-transfer derivatives vanish. The
reverse reuses the recorded log-Euler asset/payoff cotangents and transposes the
original curve's interpolation. It does not bump parameters in production.

For LSV, the relative Local Variance target, axes, random inputs, bandwidth and
retained support/donor/interpolation branches are fixed. Leverage is recalibrated
in the derivative. The direct and calibration contributions are returned
separately, and total includes both. Earlier particle positions, moment weights,
initial empirical variance and earlier Leverage rows remain in the reverse.

On the high-level shared calibration/valuation time grid, deterministic xi scales
factor out of the conditional variance estimate. They cancel against squared
Leverage, so total xi risk vanishes (up to roundoff), even at nonzero eta. The
implementation actually reverses both terms; it does not hard-code this zero.
The identity is not asserted for independently interpolated off-grid Leverage,
other target conventions or stochastic-model refits. Mixture transfers generally
do not cancel because they change the stochastic variance distribution.

## Uncertainty and limitations

Pure MC errors use independent paths/antithetic pairs; RQMC errors use independent
scramble means. LSV adds direct and calibration gradients per scramble before
computing marginal total-gradient errors, preserving their covariance. Pseudo-MC
LSV errors remain `None`, not zero. These are not a full parameter covariance
matrix. Calibration-seed uncertainty, active-set changes, discretization,
smoothing and model biases are excluded. A retained particle trace and an
explicitly supported/smoothed payoff are required. No paths are discarded, and
no variance floor or numerical tolerance was added to manufacture a derivative.

[Validation protocol](../../design/validation/mixed-bergomi-shape-risk.md).
[Executable example](../../examples/python/mixed_bergomi_shape_risk.py).
