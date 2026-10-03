# ADR 0027: Experimental multi-marginal Bass local volatility

Date: 2026-10-03

## Decision

Implement the Brownian multi-marginal Bass-LV construction from the supplied
Conze/Henry-Labordere paper (SSRN 3853085, May 25, 2021), Sections 1-3, as an
additive experimental Rust/Python API, with implied-surface marginal projection
and the mapping sensitivities in Sections 5.2-5.4. A separate market-IV quote
adapter adds central-difference VegaKT through the complete recalibration.
`ModelSpec::BassLocalVolatility` also integrates this pipeline with shared
requests, payoff graphs, market coordinates, sampling and risk reports.
The user authorized this integration.
Frozen MVP requirements and existing acceptance baselines are unchanged.

Domain inputs/errors live in `models::bass_lv`; calibration, compiled sampling
and pricing live in the private engine. `pricing::bass_lv` is the public
facade. Python adapts Rust semantics and releases the GIL for long operations.

Use finite-support continuous marginals with exact moment/call integration and
convex-order checks. Use Gaussian integrals of linear interpolation bases.
Keep convergence, tail-domain and propagated marginal errors distinct. Fail
on nonconvergence and out-of-grid simulation.

## Impact

- API: new immutable model, marginal, configuration, plan, diagnostic and
  estimate objects. Existing API semantics are preserved.
- Wire: additive `bass_local_volatility` tag in current schema v3, with explicit
  IV/grid/bump parameters. v1/v2 reject this tag; historical schemas and golden
  outputs remain unchanged. Older v3 readers cannot read the new model tag.
  Plans are rebuilt from requests, not serialized.
- Numerics: finite-grid approximation with diagnostics; lognormal inputs are
  truncated tabulated conveniences.
- Replay: addressable Philox valuation coordinates, one dimension per
  augmented date interval, deterministic statistics. Common requests also
  support RQMC, antithetics, Brownian bridge ordering and worker-count replay.
- Validation: convolution identities, distribution constraints, BS recovery,
  conditional martingale, multi-expiry MC, boundary continuity, skewed
  marginals, refinement, Python parity and failure paths.
- Dependencies: no new crates or Python packages.
- Surface adapter: differentiate forward calls including the skew term; report
  truncation, mean correction and source repricing error. Retain strict
  convex-order validation after projection.
- Mapping risk: compile explicit triangular hats; reverse the price maps,
  boundary inversions and fixed-spot initial inversion. Transport CDF tangents
  for deterministic vanilla sensitivities. Both have independent bump APIs.
- Coordinates: common requests calibrate normalized residual equity with mean
  one and reconstruct physical spot from existing affine dividend coordinates.
  Spot bumps keep cash amounts fixed. Source IVs must already use residual
  log-forward moneyness; raw physical-spot quote conversion is not automatic.
- Shared risks: central CRN spot differences and recalibrated IV differences;
  sticky log moneyness only. VegaKT reports the original quote axes and full
  covariance when requested, without density filtering or risk allocation.
- Deferred: raw market quote-coordinate conversion, a vanilla-hedge portfolio
  solver, full-calibration AAD, continuous barriers, early exercise and Bass2
  extensions.
- Market-IV risk: preserve the existing natural-cubic total-variance surface
  convention. Rebuild projection and calibration for every quote bump, with
  common Brownian streams and paired-difference errors. Report method/step and
  every scenario's diagnostics. This is distinct from mapping AAD; no claim
  of differentiation through the fixed-point solver is made. Independent
  parallel shifts and covariance-aware bucket sums provide a comparison.

See the [specification](../../docs/models/bass-local-volatility.md) and
[validation record](../validation/bass-local-volatility.md).
