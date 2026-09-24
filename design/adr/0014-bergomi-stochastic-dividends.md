# ADR 0014: Pure Bergomi with stochastic cash dividends

Status: implemented candidate; validation gates remain explicit.

## Decision

Extend ADR 0013's separate deterministic-rate, single-asset, price-only facade
with 1F and 2F Bergomi. Use the existing validated Bergomi factor definitions,
Buehler reserve/evolution, payoff compiler, MC/RQMC executor and PSD policy.
A private const-generic kernel owns exact joint OU innovations and left-endpoint
variance centering. Select the concrete 1F/2F kernel before the path loop.

Accept explicit dividend/each-volatility Brownian correlations. Validate the
entire instantaneous matrix even for zero-weight/zero-volatility factors, then
factorize its integrated Brownian/OU covariance. The correlation of an OU
innovation with a Brownian increment is not the instantaneous correlation.

Retain the full supplied cash-mean schedule, including beyond option expiry.
The request's BS volatility supplies sigma0, the flat initial forward variance
scale; do not imply calibration to physical-stock implied volatility.

## Compatibility and numerical impact

Add Rust/Python factories without changing existing signatures or serialized
schemas. BS stochastic-dividend paths retain their two coordinates, scheme,
fingerprint and evaluation order. New paths reserve three/four coordinates in
order equity, dividend, vol1, optional vol2; no rank/zero-loading compression.
They use new scheme/fingerprint domains, including all model parameters.
No dependency, workspace boundary, replay fixture or acceptance budget changes.

Exact OU endpoint laws do not make the joint nonlinear equity/dividend evolution
exact. It uses left-frozen variance and the positive Buehler split. Sampling SE
excludes discretization and model uncertainty. Risk, HW, LSV, rough, multi-asset,
American, continuous barriers and dividend derivatives remain outside scope.

## Acceptance

See the [validation record](../validation/bergomi-stochastic-dividends.md),
[model specifications](../../docs/models/stochastic-dividends.md) and
[runnable example](../../examples/python/bergomi_dividends.py).
