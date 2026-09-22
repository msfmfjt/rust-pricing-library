# ADR 0012: Expose pure SV independently of particle calibration

Date: 2026-09-22. Status: implemented; validation recorded separately.
Baseline: PR #82, tree `0473f0a5627912f04224c3f5d32eac03e5f5f086`.

## Context and decision

The user requested pure-SV support after identifying that only rough Bergomi
exposed a calibration-free pricing entry. Add pure one-/two-factor Bergomi to
the shared hybrid evaluator and a deterministic-rate facade for all three.
Driver selection is independent of the optional calibrated leverage surface.
This request extends the no-new-production-model scope of the preceding
static-boundary work; it does not resume the deferred optimization tasks.

Normalize Markovian variance explicitly to preserve flat xi0=sigma0^2.
Reuse exact joint OU/rate transitions, pricing, dividends and path reverse.
Keep historical uncentered LSV multipliers and rough finite-grid centering.
The [model specification](../../docs/models/pure-stochastic-volatility.md)
defines the numerical and API contract.

## Impact and validation

- API: additive Rust/Python compilers and a deterministic-rate facade. The
  public low-level `HybridEquityVolatility` enum gains two variants; external
  exhaustive matches must handle them. Existing constructors remain valid.
- Wire: no schema change; the base BS request supplies initial volatility.
- Numerics: centering applies only to new pure Markovian paths. Existing
  BS, rough and calibrated-LSV arithmetic, factor layouts and fingerprints
  retain their contracts. Pure variants have distinct parameter tags/schemes.
- AAD: sigma0, Spot and initial curves are active; model parameters and payout
  quotes remain fixed. No false calibration trace or market-IV VegaKT.
- Performance: one immutable O(time_steps) centering array per new Markovian
  plan. Deterministic pricing retains unused rate coordinates. No performance
  improvement is claimed; no generic per-step trait-object dispatch is added.
- Checks: independent OU covariance/path identities, BS/1F/rough limits,
  public MC/RQMC AAD versus recompilation, curves and escrowed dividends,
  parallel replay, finite zero-sigma right derivative, independent two-step
  Gaussian quadrature prices, validation/fingerprints and wheel API conformance.
- Limits: flat xi0, single asset, existing hybrid product boundary; parameter
  calibration/Greeks, nonflat xi0, reduced random layout and multi-asset pure
  SV remain separate extensions. Existing accuracy budgets are unchanged.

See the [validation record](../validation/pure-stochastic-volatility.md).
