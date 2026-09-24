# 0017: Optional raw correlation risk with stochastic dividends

## Decision and scope

Add `StochasticDividendPricingPlan::evaluate_correlation_aad()` and the matching
Python `StochasticDividendPlan` method for single-asset, deterministic-rate
BS / 1F / 2F Bergomi plus Buehler cash dividends. Use the existing risk result type.
BS preserves the complete basic-AAD prefix; Bergomi preserves the complete
Bergomi-parameter-AAD prefix. Append correlations in the order documented in the
[model reference](../../docs/models/stochastic-dividends.md#correlation-risk).
The new risk method label is `buehler-joint-correlation-reverse-v1`.

A raw partial varies one symmetric off-diagonal Brownian-correlation pair while
holding all other pairs and model/market inputs fixed. It is not an independent
upper/lower matrix entry derivative, a projected-matrix derivative, or a
recalibrated/market-IV risk. Values are price per unit correlation; multiply by
0.01 for a one-percentage-point linear approximation within the admissible domain.

## Calculation specification

Keep the existing price, basic-risk and model-risk algorithms. Add prepared
coefficient Jacobians and a reverse OU pass, using the existing payoff and
Buehler-split adjoints. No finite-difference production code is introduced.
For normalized kernels `a_i = k_i dt`, the integrated correlation is
`C_ij = rho_ij * ou_kernel_correlation(a_i,a_j)`, with k=0 for equity/dividend
Brownian increments. Its raw-entry tangent is the kernel correlation itself,
including at rho=0. Differentiate the same unpivoted Cholesky recursion as the
primal; scale OU rows by the unchanged marginal standard deviations.

Include the direct Buehler dividend-driver tangent
`d(rho*z0 + sqrt(1-rho^2)*z1)/d rho = z0-rho*z1/sqrt(1-rho^2)`.
This contribution alone is insufficient when Bergomi OU loadings also depend on
that correlation. Add the indirect loading contribution from the reverse OU pass.
For 2F factor correlation, also differentiate the normalization of both weights:
`dw_i/d rho_12 = -w_i theta(1-theta)/norm2`. Differentiate the actual
`nu*Var[Z_t]` centering, including both weight and OU factorization effects.

## Numerical domain

Require **instantaneous and integrated** normalized correlation Cholesky pivots
and 2F weight variance greater than 1e-10, at the step intervals and positive
left-endpoint horizons used for centering. BS requires `1-rho_SD^2 > 1e-10`.
Reject before path generation otherwise. An instantaneous singular matrix with
unequal OU kernels can have a positive-definite integrated matrix; that does not
give an open domain for independent raw instantaneous-correlation perturbations.
No projection, clipping, rank change, tolerance relaxation or arbitrary zero
Greek substitutes for rejection. Existing price/basic/model risk methods retain
their own domains, independent of this opt-in method. Zero-load factors still
participate in full-domain validation. A domain passing this guard is not a
promise of low estimator variance near the boundary.

## Compatibility and validation

No constructor, request schema, frozen replay, price fingerprint, price arithmetic,
random dimension/order or old method label changes. No HW, rough, LSV, multi-asset,
Gamma or VegaKT support is implied. Dates, grid and smoothing width remain fixed.
These are derivatives of the finite simulation algorithm with independent normals
fixed, not a proof of continuous-time risk convergence. Standard errors use the
existing antithetic-unit or scramble-average interpretation.

See the [prespecified tests](../validation/stochastic-dividend-correlation-risk.md).
