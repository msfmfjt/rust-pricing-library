# Rough stochastic-dividend risk

## Decision

Extend the deterministic-rate single-asset rough/Buehler plan with basic
`evaluate_aad()`, paired Delta-bump `evaluate_gamma()`, and the opt-in
`evaluate_rough_aad()` scope. The latter appends `rough_hurst` and
`rough_vol_of_vol` to the exact basic-risk prefix. Hold all three Brownian
correlations fixed. Reject `evaluate_correlation_aad()` and
`evaluate_bergomi_aad()` on rough plans, before sampling.

Reuse the existing payoff and positive-split reverse. Replay the exact causal
hybrid history to obtain each left-endpoint volatility loading. For H/eta risk,
transpose the loading seeds into every history coefficient, exact near-cell
coefficient, and the finite-grid centering variance. Compile coefficient
Jacobians once per risk evaluation; there are no production finite differences
in first-order risk. H=1/2 has an inward left derivative, including the nonzero
derivative of the disappearing near-cell residual. Eta=0 has an inward right
derivative; there is no division by eta, sigma0 or (1/2-H).

Gamma remains a central difference of AAD Delta, not second-order AAD. All six
bumped escrow reserves share the same normalized f/Y/history path. An immutable
Arc shares the dense rough coefficients between the six Spot scenarios; the
4096-step limit and primal arithmetic are unchanged. H risk adds one triangular
coefficient-Jacobian array. Storage and per-path time remain O(N^2).

## Compatibility and scope

No constructor/wire/schema or dependency changes. Add one Rust/Python method on
the existing plan and reuse the existing immutable risk type. Price-only
fingerprints, schemes, all old BS/1F/2F results and frozen replay fixtures stay
unchanged. The H/eta scope identifies itself as
`buehler-rough-hybrid-parameter-reverse-fixed-correlation-v1`.

Previously rejected rough basic AAD and Gamma now succeed where payoff and
numerical domains permit; update those support tests without altering price
budgets or seeds. Fixed singular PSD correlations remain supported for these
risks because no derivative of their factorization is needed.

Results differentiate the finite simulation algorithm. No claim of continuous-
time risk convergence, market-IV VegaKT, recalibration, stochastic-rate/HW,
multiple-asset support, rough-correlation risk or improved rough discretization
is made. Gamma diagnostics and all sampling errors exclude grid, bump,
smoothing and model uncertainty.

See the [validation protocol](../validation/rough-stochastic-dividend-risk.md)
and [model calculation specifications](../../docs/models/stochastic-dividends.md).
