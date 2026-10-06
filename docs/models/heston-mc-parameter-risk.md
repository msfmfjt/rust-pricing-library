# Heston MC fixed-kernel parameter adjoints

**Experimental Pure-SV first-order reverse**, separate from
[Fourier parameter risk](heston-parameter-risk.md),
[market-IV LSV risk](rough-family-market-iv.md), and
[finite-bump Gamma](rough-family-gamma.md).

`RoughVolatilityPlan.evaluate_heston_parameter_risk()` returns the unchanged
nested price and five `parameter_adjoints`, ordered by `parameter_names`:
initial variance v0, mean reversion kappa, long-run variance theta, vol-of-vol nu,
and correlation rho. Rust exports `HestonMcParameterRisk` and
`HESTON_MC_PARAMETER_NAMES`. Python returns a frozen class; arrays are copies.

These are price derivatives **per one absolute parameter unit** at fixed Spot,
contract, curves, the full cash/proportional dividend schedule, Hurst, kernel,
time grid and Gaussian draws. In particular, dPrice/dv0 is not volatility Vega.
The asset normal remains fixed; rho differentiates the correlated variance normal.
Only Rough Heston and Lifted Heston are supported. Every lift weight/rate is fixed,
including lifts generated from Hurst. No parameter bumps are used in production.

This increment does **not** differentiate Hurst, a lift factory, kernel weights,
calibration, LSV model parameters, RFSV covariance, or the other four families.
Existing LSV Local Variance and market-IV APIs continue to fix model parameters.
A direct MC-model parameter derivative is not a recalibrated market sensitivity.

## Discrete reverse

The primal is the unmodified [rough-family full-truncation scheme](rough-volatility-families.md).
With raw variance y and v=max(y,0), reverse the log-Euler spot recursion through
all variance history nodes, then reverse the Volterra or factor recurrence.
For a seeded next forward with adjoint b, the spot contribution to variance is

```text
bar_v[j] += b * F[j+1] * (-dt/2 + sqrt(dt)*z_spot[j]/(2*sqrt(v[j])))
```

For Rough Heston, reverse every history term in

```text
y[i] = v0 + sum_j d[i,j]*kappa*(theta-v[j])
          + sum_j fractional_scale*nu*sqrt(v[j])*innovation[i,j].
```

The exact singular newest-cell innovation and the older averaged-cell weights
are retained. Propagate rho sensitivity through correlated Brownian increments
and their newest-cell loading. The loading and the independent newest-cell
residual are rho-independent at fixed Hurst.
For Lifted Heston, reverse

```text
U_l[j+1] = (U_l[j] + common[j])/(1 + rate_l*dt)
y[j+1] = v0 + sum_l weight_l*U_l[j+1].
```

Fixed rates/weights permit an O(factors) backward factor vector without a dense
factor-state tape. Rough reverse is O(n^2); lifted reverse is O(n*factors).
The original path and post-bridge Gaussian inputs are retained. Compensation
branches in floating sums are not treated as economic model discontinuities;
the reverse is of the real-arithmetic discretization at the recorded primal.

Rust's borrowed `HestonMcRecordedPath` supports seeds for every forward and
**diffusion variance** observation, plus an initial-forward output adjoint.
It does not accept raw latent-variance seeds. Its immutable plan and normal
slice must outlive the record.

## Domain and uncertainty

Require **v0>0 and |rho|<1** for this five-direction API. At rho=+/-1 the chosen
Gaussian coordinate map has a singular derivative. Strictly negative raw nodes
have zero truncation derivative; no variance floor is introduced. Exact zero
raw variance before the final node is rejected because square-root/truncation
has no finite ordinary derivative there. A terminal zero is rejected only if
its diffusion variance is seeded. Failed paths or seeds are not dropped.
At nonnegative kappa/theta/nu boundaries, derivatives are the inward extension
of the discrete formula. Payoff kinks retain the existing tape convention;
hard discontinuous risks require explicit supported smoothing.

MC standard errors use independent paths or antithetic pairs; RQMC uses
independent scramble means. `standard_errors` are **marginal sampling errors**,
not a covariance matrix or an error for a basket of parameter risks. They exclude
finite-grid/full-truncation, smoothing and model bias. Near-zero positive
variance can generate large pathwise derivatives and poor estimator variance.
No universal moment, differentiation-under-expectation, or continuum guarantee
is implied by valid finite derivatives. No clipping, failed-sample deletion or
standard-error floor is used. Nonfinite derivatives fail explicitly.

The price-plan fingerprint is unchanged. A separate risk fingerprint includes
the fixed five-direction method identifier. No stable JSON request tag changes.

[Complete example](../../examples/python/heston_mc_parameter_risk.py) ·
[Validation protocol](../../design/validation/heston-mc-parameter-risk.md)
