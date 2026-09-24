# Design and development records

This directory contains requirements, architecture, decisions, implementation
plans and acceptance evidence. The [library guide](../docs/library/README.md),
[product reference](../docs/products/README.md) and
[model reference](../docs/models/README.md) describe the usable APIs, product
contracts and calculation specifications.

## Baselines and change process

- [Frozen MVP requirements](requirements-v1.0.md)
- [Architecture](architecture-v0.1.md)
- [Requirements change template](requirements-change-template.md)
- [ADR template](adr/0000-template.md)
- [Release readiness](release-readiness-v0.1.md)
- [Contributor workflow and checks](../CONTRIBUTING.md)

Requirements and ADRs retain their recorded scope and chronology. Older records
may describe earlier crate names, defaults or extension boundaries. Follow the
model contracts and their explicit superseding decisions for current behavior;
the [three-crate migration guide](../docs/library/three-crate-migration.md)
documents the public API migration.

## Architecture decisions

| ADR | Decision |
| --- | --- |
| [0001](adr/0001-hull-white-equity-hybrid.md) | Equity/Hull–White hybrid |
| [0002](adr/0002-hull-white-cash-dividends.md) | Hull–White cash dividends |
| [0003](adr/0003-hull-white-aad.md) | Hull–White AAD |
| [0004](adr/0004-hull-white-vegakt.md) | Hull–White market-IV VegaKT |
| [0005](adr/0005-rough-bergomi.md) | Rough Bergomi |
| [0006](adr/0006-integrate-completed-baseline.md) | Integration of completed baselines |
| [0007](adr/0007-affine-paid-cash-dividends.md) | Historical paid-cash default (superseded by 0009) |
| [0008](adr/0008-deterministic-rate-rough-lsv.md) | Deterministic-rate rough-LSV |
| [0009](adr/0009-escrowed-simulation-and-iv-conversion.md) | Escrowed simulation (IV conversion superseded by 0010) |
| [0010](adr/0010-remove-paid-cash.md) | Remove paid-cash support; defer Bos–Vandermark |
| [0011](adr/0011-static-model-boundaries.md) | Compiled static boundaries for model extensions |
| [0012](adr/0012-pure-stochastic-volatility.md) | Pure SV independently of particle calibration |
| [0013](adr/0013-stochastic-cash-dividends.md) | Stochastic discrete cash dividends, staged integration |

| [0014](adr/0014-bergomi-stochastic-dividends.md) | Pure 1F/2F Bergomi with stochastic cash dividends |
| [0015](adr/0015-stochastic-dividend-risk.md) | Buehler first-order reverse and timestep refinement |

| [0016](adr/0016-stochastic-dividend-bergomi-risk.md) | Optional Bergomi parameter reverse with stochastic dividends |
| [0017](adr/0017-stochastic-dividend-correlation-risk.md) | Optional raw Brownian-correlation reverse with stochastic dividends |

## Implementation roadmaps and acceptance

The [Bergomi parameter-risk record](validation/stochastic-dividend-bergomi-risk.md)
tracks fixed-correlation model-risk differentiation and its numerical domain.

The [stochastic-dividend record](validation/stochastic-dividends.md) distinguishes
finite-split and limiting-price tests from the deferred hybrid/calibration scope.
The [Bergomi coupling record](validation/bergomi-stochastic-dividends.md) adds
exact joint OU covariance and independent finite-step price references.

The [model-boundary extension/comparison record](validation/model-boundary-extension.md)
defines the test-only extension exercise and paired native timing, heap and RSS
evidence for stages S1–S6 of the [static-model roadmap](roadmaps/static-model-boundaries.md).
The [Gaussian HW reference correction](validation/hw-escrowed-gaussian-reference.md)
tracks the separate funded-strike reporting mismatch found in extended CI.
The [local-correlation scaling record](validation/local-correlation-scaling.md)
starts S7 with independent workload axes and cold/warm AAD allocation attribution.
The first measured candidate is [reverse-moment buffer reuse](validation/local-correlation-moment-workspace.md),
with exact outputs, lower first-AAD allocation churn and an explicit peak-heap follow-up.
The [repeated-AAD heap study](validation/local-correlation-heap-followup.md)
checks the full peak-stack difference and fresh-process reproducibility.

| Stage | Roadmap | Acceptance evidence |
| --- | --- | --- |
| European Black–Scholes | [G0–G8](roadmaps/european-bs-roadmap-v0.1.md) | [Conformance report](validation/european-bs-conformance-v0.1.md) |
| Local Volatility / VegaKT | [L0–L8](roadmaps/local-vol-vegakt-roadmap-v0.1.md) | [Conformance report](validation/local-vol-vegakt-conformance-v0.1.md) |
| Path dependence | [P0–P8](roadmaps/path-dependence-roadmap-v0.1.md) | [Conformance report](validation/path-dependence-conformance-v0.1.md) |
| Early exercise | [E0–E8](roadmaps/early-exercise-roadmap-v0.1.md) | [Conformance report](validation/early-exercise-conformance-v0.1.md) |
| Bergomi LSV | [Implementation and remaining acceptance](roadmaps/lsv-roadmap-v0.1.md) | Experimental status and evidence recorded in the roadmap |
| Hull–White | [Implementation and acceptance](roadmaps/hull-white-roadmap-v0.1.md) | Experimental status and evidence recorded in the roadmap |
| Extended-model accuracy | [Panel and remaining scope](validation/extended-model-accuracy.md) | Independent repricing, multiple calibration seeds and refinement gates |
| Extended-model refinement and stress | [Cases and measurements](validation/extended-model-refinement-stress.md) | Original 73/74 result and joint 2F sampling-error failure; subsequent correction linked |
| Multi-asset Sobol sampling | [Attribution and correction](validation/multi-asset-sobol-attribution.md) | Bridge coordinate correction; original 2F stress gate passes locally, native CI pending |
| Extended-model AAD and VegaKT | [Sensitivity acceptance](validation/extended-model-risk-accuracy.md) | Multi-seed, multi-bump comparison with full recalibration through public plans |
| Static model boundaries | [S0–S7](roadmaps/static-model-boundaries.md) | Driver metadata, typed volatility/rate inputs and explicit reverse capabilities; native CI and later stages tracked in the roadmap |

The [three-crate validation report](validation/three-crate-validation.md) links
the preserved [raw evidence](validation/three-crate/). Captured JSON and the
compressed run archive retain their original bytes, including paths and command
output from the revisions they validated. Those historical paths are not a
directory map for the current checkout.

## Where new documentation belongs

| Content | Location |
| --- | --- |
| Usage, serialization, migration and benchmarking | `docs/library/` |
| Product definitions, payoff behavior and event conventions | `docs/products/` |
| Model equations, calibration, simulation, risk and model diagnostics | `docs/models/` |
| Requirements, architecture and release planning | `design/` |
| Architecture decisions | `design/adr/` |
| Implementation plans and remaining work | `design/roadmaps/` |
| Conformance reports and captured validation evidence | `design/validation/` |

Keep calculation specifications linked from the library or model index, even when a
design decision introduces them. Update the relevant index when adding a page.
Repository-local links in both `docs/` and `design/` are checked by
`python3 scripts/check_markdown_links.py`.

Escrowed simulation and IV conversion: [validation record](validation/escrowed-dividends.md).

- [Stochastic-dividend risk and refinement](validation/stochastic-dividend-risk.md)

The [correlation-risk validation protocol](validation/stochastic-dividend-correlation-risk.md)
separates raw-entry derivatives from the instantaneous/integrated numerical domain.
