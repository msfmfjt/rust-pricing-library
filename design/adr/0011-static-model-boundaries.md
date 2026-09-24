# ADR 0011: Extend models through compiled, static boundaries

Date: 2026-09-21. Status: direction accepted; incremental implementation.
Baseline: escrowed dividends from ADRs 0009–0010 and PR #72.

## Context

The user requested an architecture review before performance tuning, with
generality limited by the requirement to preserve performance. The existing
one-/two-factor Bergomi trait already shares deterministic-rate calibration and
reverse kernels. Hull–White combinations, rough history and multi-asset driver
offsets still expose model-specific assumptions in several execution modules.

## Decision

Keep the three-crate workspace and concrete compiled plans. Separate five
responsibilities: volatility evolution, rate evolution, random/correlation
layout, calibration/reverse capabilities, and composition configuration.
Introduce a boundary only when an existing implementation can use it.
Use static dispatch inside numerical kernels; keep enum selection at the
compilation or asset/path boundary. Existing compilation-time surface trait
objects do not require removal.

Markov factor steps and rough full-path history preparation remain distinct
operations. Rate state, its integral, bond valuation and covariance with equity
innovations must be represented explicitly. A model without an implemented
calibration or reverse capability must fail validation rather than substitute
zero sensitivities or frozen calibration.

Begin with a private multi-asset `DriverLayout`, shared by random dimension
validation, public factor counts and volatility-driver slicing. This metadata
does not change covariance construction, Cholesky permutations, Sobol ordering
or numerical transitions. Preserve the base layout during local-correlation
calibration and apply endpoint-block duplication only where required.

See the [implementation roadmap](../roadmaps/static-model-boundaries.md) for
code ownership, the remaining stages and their acceptance gates.

## Impact

- **API and wire:** retain current Rust/Python entry points, aliases, errors and
  JSON schemas. New internal descriptors are not serialized. A later composition
  API must adapt existing constructors before changing public contracts.
- **Numerics and risk:** preserve escrowed funding, event order, interpolation,
  calibration estimators, supported domains and finite-algorithm AAD. Keep
  existing Gamma and unsupported-capability distinctions explicit.
- **Reproducibility:** preserve independent coordinate counts, including
  zero-loading/singular directions, bridge ordering, seeds, reductions and
  fingerprints. Any intentional numerical-policy change needs its own decision.
- **Performance:** no per-step trait-object dispatch or new per-path layout
  allocation. The first extraction adds immutable per-asset ranges at compilation.
  Runtime and memory equivalence must be measured before claiming performance
  neutrality for subsequent kernel refactors.
- **Verification:** retain the existing sampling, model, recalibrated AAD and
  replay gates. Add focused descriptor tests and cross-check that compiled QMC
  dimensions agree with executed driver counts.
- **Scope:** no new production model, paid-cash support or Bos–Vandermark
  implementation. Broader timing/memory instrumentation and optimizations follow
  the structural work; existing deferred accuracy thresholds remain unchanged.
