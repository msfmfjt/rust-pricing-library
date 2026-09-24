# 0023: Explicit basic AAD for stochastic-dividend Hull–White

Status: proposed. Parent: [0022](0022-stochastic-dividend-hull-white.md).

## Decision and scope

Add `StochasticDividendHullWhitePricingPlan::evaluate_aad()` and the same Python
method. Reuse `StochasticDividendAadRisk` and its existing basic parameter order:
Spot, initial residual-equity volatility, dividend mean reversion/linkage/volatility,
schedule Q cash means, discount-curve log-DF pillars, repo-spread log-DF pillars.
This is an opt-in method on a price-only request, not fixed-rate dispatch.

Hold HW mean reversion/volatility knots and values, all driver correlations,
contract constants, dates, grid and smoothing width fixed. No Gamma, HW-parameter
or correlation AAD, fixed-forward dividend calibration or market-IV VegaKT.
The conditional cash distribution and initial funded residual are recomputed in
this partial derivative; model Q cash means are not collateral-forward quotes.

## Calculation and boundaries

Reverse contractual payoff seeds through each post/pre-cash physical Spot, the
full conditional reserve and its initial funding, and the existing positive f/Y
split. Exact x and integrated-x paths and the relative payment adjustment are
independent of these active inputs. The fitted discount curve still contributes
through all bonds/growth ratios and payment-date P0(U); it is not frozen.

Differentiate A/B/C analytically and integrate their signed partial integrands
with the existing absolute/relative Simpson policy. Derivative panels are selected
independently of primal panels. This is a numerical approximation to the analytic
conditional-claim derivative, not differentiation through adaptive branch choices.
Never divide by zero cash, sigma, kappa, alpha or nu. Retain valid-side derivatives
at boundaries, including derivatives absent from a primal shortcut's formula.

Fixed singular Brownian matrices retain their price domain: no Cholesky derivative
is taken. Unsmoothed discontinuous payoff risk rejects before sampling. No new
clipping, PSD projection, extrapolation rule, seed or numerical budget is introduced.

## Compatibility, replay and resource impact

Price arithmetic, RNG coordinates, primal fingerprints, previous stochastic-
dividend plans, JSON request schemas and Python constructors are unchanged. Only
one new method and a crate-local Python result-field visibility change are added.
Numerical results/SE replay across workers; execution-policy fingerprints retain
separate identities. Derivative coefficient storage is O(nodes * cash), using the
existing one-million node/cash cap. Market coefficients are built once per risk
call, never in the path loop; sampled state and gradient workspaces are local.

## Verification

See the [prespecified protocol](../validation/stochastic-dividend-hull-white-risk.md)
and [calculation specifications](../../docs/models/stochastic-dividends-hull-white.md).
Focused passes do not certify full platform/legacy acceptance or continuous-time
risk convergence. Existing gate and legacy 4 bp/5 bp decisions remain separate.
