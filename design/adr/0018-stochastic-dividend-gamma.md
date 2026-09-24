# 0018: Stochastic-dividend Gamma from paired AAD Delta bumps

## Decision

Extend the BS/1F/2F deterministic-rate Buehler plans from ADRs
[0013](0013-stochastic-cash-dividends.md) through
[0017](0017-stochastic-dividend-correlation-risk.md) with opt-in `evaluate_gamma`.
Use the existing `GammaConfig`/`SpotBump` absolute/relative convention. Constructors
still accept price-only requests. The method is central differencing of AAD Delta,
not second-order AAD. It computes half/base/double bumps without extrapolation.

Cash means (including post-expiry cash), all curves, model parameters,
correlations, dates/grid, contract constants and payoff-smoothing width stay fixed.
The normalized f/Y/OU evolution does not depend on initial Spot in these models,
so evolve it once. Rebuild the funded escrow coefficients at all six shifted
Spots on the identical grid. Apply the contractual payoff reverse at each Spot,
then the Spot-only slice of the existing reverse in its original node order.
A future local-volatility/LSV extension cannot reuse this Spot-independent shortcut.

## Domain and uncertainty

Require finite, positive, representable bumps and valid funded residual equity
for every shifted Spot before sampling. Fail rather than adapt the ladder, clamp
Spot or switch to one-sided bumps. Ordinary vanilla kinks may use bumped Delta;
unsmoothed discontinuous payoffs retain the explicit-smoothing requirement.
Smoothing results describe the fixed-width smoothed payoff. Gamma does not require
the stricter covariance domain for differentiating correlation or OU parameters.

Calculate each Gamma sample and each adjacent-ladder difference with common noise.
Pseudo-MC uses independent units, averaging antithetic pairs before estimation.
RQMC uses scramble means. Do not combine marginal Delta SEs as independent errors.
Sampling SE excludes finite-bump, grid, smoothing, model and calibration errors.
Ladder gaps are diagnostics, not bounds or a claimed convergence order.

## Compatibility and identity

Existing price and AAD arithmetic, RNG coordinates, fingerprints, requests, schemas,
legacy budgets and dependencies are unchanged. The baseline price/Delta and their
SEs must exactly match basic AAD. The new risk fingerprint includes the original
plan fingerprint, method, bump convention and resolved ladder. A result reports
one simulated path per sign and seven payoff evaluations per simulated path.
Add a separate immutable Rust/Python result type; existing result layouts remain.
See the [validation protocol](../validation/stochastic-dividend-gamma.md).
