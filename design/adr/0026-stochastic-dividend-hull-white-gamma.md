# 0026: Hull–White stochastic-dividend Gamma from paired AAD Delta

Status: proposed. Parents: [0018](0018-stochastic-dividend-gamma.md) and
[0025](0025-stochastic-dividend-hull-white-correlation-risk.md).

## Decision

Extend the dedicated BS/Buehler/Hull–White plan with explicit `evaluate_gamma`.
Use the existing `GammaConfig`/`SpotBump` and `StochasticDividendGammaRisk` types.
Python accepts exactly one absolute or relative bump keyword. Constructors keep
their price-only request contract. Compute central differences of AAD Delta at
half/base/double Spot widths, without a second-order reverse or extrapolation.

Keep Q cash means, all model parameters/correlations, curves, dates, grid and
smoothing fixed. In this constant residual-volatility model, normalized equity,
dividend and rate states have no initial-Spot dependence. The conditional cash
claims and initial cash reserve are likewise invariant. Reuse one state path,
one payment discount, and the existing claim coefficients; recompute the funded
risky Spot for every scenario by summing the original time-zero cash reserve in
compile order. Future cash beyond expiry remains funded. Do not copy or rebuild
the whole claim grid for a Spot-only bump.

Evaluate the original contractual payoff and its Spot adjoints seven times per
path. The Spot-only reverse uses the full basic AAD's reverse node order and
arithmetic, including discounting each pre/post seed before combining them.
This retains exact baseline price/Delta/SE identity. It is valid for this BS
family only; a future local-volatility or LSV process requires its own analysis.

## Domain and diagnostics

Validate all six shifted Spots and their positive funded residuals before
sampling. Reject unrepresentable, nonfinite, nonpositive or unfunded ladders;
do not clamp or adapt. Fixed singular correlations and zero rate variance are
supported because Gamma holds the covariance law fixed. Vanilla kinks are
handled by the Delta bump; discontinuous contracts require explicit smoothing.

Calculate Gamma and adjacent-ladder differences per common-noise sampling unit,
averaging antithetics before MC statistics and reducing RQMC to scramble means.
Never infer paired SE from independent marginal Delta errors. Ladder gaps are
diagnostics, not bias bounds. Small bumps can produce zero crossing paths and
misleading zero estimates/SE. Sampling SE excludes bump, grid, smoothing,
quadrature, calibration and model error.

## Identity and compatibility

Use method `buehler-bs-hw-common-noise-aad-delta-gamma-v1`. Risk identity hashes
the original price-plan fingerprint, method, bump convention and absolute ladder.
Equivalent absolute/relative widths have identical estimates but distinct risk
fingerprints. The public immutable result and its unit conversions are reused.
Price arithmetic, request schemas, RNG coordinates, existing first-order risk,
legacy seeds/budgets and replay fixtures remain unchanged.

See the [validation protocol](../validation/stochastic-dividend-hull-white-gamma.md)
and [model reference](../../docs/models/stochastic-dividends-hull-white.md).
