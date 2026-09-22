# ADR 0013: Buehler stochastic discrete cash dividends

Date: 2026-09-23. Status: implemented first-stage candidate; acceptance pending.
Base: PR #83, `cc06779bcfbc892f754669018f697bac8d346a7e`.

## Decision

Add the Buehler dividend factor as an independent financial component, retaining
escrowed simulation and separate deterministic repo carry. Introduce a separate
single-asset constant-residual-volatility, deterministic-rate, price-only plan.
Do not reinterpret existing cash amounts or change existing pricing paths.
All supplied future cash means are funded, even beyond product expiry.

The [model specification](../../docs/models/stochastic-dividends.md) defines the
SDE, conditional cash forecast, carry-funded stock reconstruction and positive
split. Public source/model/state definitions remain in `pricing`; execution
stays under the private engine. No crate or runtime dependency is added.

## Compatibility and numerical impact

New Rust component/plan/result types and Python plan/result classes are additive.
`MonteCarloError` gains a variant; the enum is already non-exhaustive. Stable
request/result JSON schemas and all frozen fixtures are unchanged. In the new
factory only, the existing fixed-cash schedule supplies Q-mean cash amounts.
Proportional mixtures, nonconstant volatility and requested Greeks fail.
American exercise and continuous barriers remain unsupported by this adapter.

Both random coordinates are reserved even at zero loading; bridge-rank-major
ordering precedes correlation. A new scheme/fingerprint domain identifies the
new finite algorithm. Existing requests do not route through this code and their
operation/reduction order is unchanged. Pricing SE excludes timestep/model error.

## Staged extension boundaries

1. This change: factor/state, positive split, funded reconstruction, deterministic
   rate pricing through shared payoffs, Rust/Python entry points, independent
   finite-step and limiting-price tests, explicit support limits.
2. Pure 1F/2F/rough SV: joint dividend/volatility/spot driver covariance and typed
   composition; no automatic correlation substitutions.
3. LSV: conditional **total physical-stock variance**, including dividend and
   covariance terms; rederive and test particle calibration before exposure.
4. HW: measure-consistent dividend claim and carry reserves with discount/dividend
   covariance; do not multiply a bank-account Q-mean by a bond as a shortcut.
5. Reverse/AAD and VegaKT: explicit component pullbacks and fully recompiled or
   recalibrated differences; no frozen-cash risk fallback.
6. Multiple assets, dividend instruments/calibration, American state features
   and continuously monitored stochastic boundaries require separate validation.

These are deferred scopes, not already supported combinations. No main push,
merge, paid-cash reintroduction, Bos–Vandermark addition or inherited accuracy
budget adjustment is authorized by this implementation.

## Evidence

See the [validation record](../validation/stochastic-dividends.md). Numerical
moment preservation does not certify nonlinear continuous-time prices.
