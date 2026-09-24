# 0022: Stochastic cash dividends with Hull–White, constant equity volatility

Status: experimental implementation, child of the rough-correlation work.

Add a separate BS/Buehler/HW price plan. Keep Buehler dynamics in the domestic
money-market measure and keep cash inputs as Q means. Discounted cash forecasts
are computed by a Gaussian tilt, not by multiplying an undiscounted forecast by
a bond. Explicit rate/equity and rate/dividend correlations complete the existing
equity/dividend correlation. Require full PSD and positive initial funding.

Reuse the existing exact four-coordinate HW transition at zero second-factor
reversion and the existing positive Buehler split. Reconstruct physical equity
with a carry-funded reserve of conditional cash claims. Discount payment lags
conditionally from expiry, without an extra post-expiry simulation grid.

API impact: additive Rust types and one immutable Python plan. No request schema
or existing signature changes. The new class deliberately has no AAD/Gamma:
fixed-rate reverse formulas are not valid for a state-dependent HW reserve.
Existing BS/1F/2F/rough plans retain all previous risk scopes.

Reproducibility impact: new scheme and fingerprint domain, with rate volatility
knots/values, all correlations, dividend inputs, grid and existing execution
provenance. Keep all four random coordinates at boundaries. No dependency,
existing RNG/scheme/fingerprint, seed, old threshold, or replay fixture changes.

Numerical scope: nonzero Buehler mean reversion, piecewise HW volatility and
repo-spread term structures, all future cash, fixed-observation contractual
payoffs. The cash integrals use explicitly bounded adaptive quadrature.
Continuous-time no-arbitrage construction and discrete-time simulation accuracy
are separate. Broad exotic accuracy and stochastic-volatility coupling remain
unaccepted. This slice does not add multi-asset, HW AAD, calibration or VegaKT.

See the [model calculation specifications](../../docs/models/stochastic-dividends-hull-white.md)
and [validation protocol](../validation/stochastic-dividend-hull-white.md).
