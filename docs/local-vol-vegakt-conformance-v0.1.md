# Local Volatility and VegaKT conformance report v0.1

Status: accepted
Evidence date: 2026-09-09
Requirements: `requirements-v1.0.md`
Roadmap: `local-vol-vegakt-roadmap-v0.1.md`

## Decision

The Local Volatility/VegaKT slice satisfies Gates L0-L8 and is accepted as the
v0.1 Local Volatility/VegaKT baseline. Pull-request CI for this report's commit
passes on Apple Silicon macOS, Windows x86-64, and Ubuntu development runners,
and retains the supported wheel, replay, and benchmark artifacts.

No known correctness failure is waived by the current performance results.
Current limitations are recorded explicitly and do not change the frozen
European Black-Scholes baseline.

## Acceptance evidence

| Area | Evidence | Result |
|---|---|---|
| Equation fixtures | `fixtures/local-vol/reference-cases-v0.1.json` checked by `scripts/check_local_vol_reference_fixture.py` | Standard SSVI, eSSVI, Dupire, equation (11), hat projection, and transition-cell references pass independent checks |
| Market surfaces | Unit and reference tests in `pricing-market` | SSVI/eSSVI values, analytic derivatives, density factors, Local-variance repairs, node helpers, and affine dividends pass |
| Path engine | Unit tests in `pricing-mc` and the public `pricing` facade | Non-uniform Brownian bridge, Local Volatility Log-Euler, Price/Delta/Gamma/Vega/VegaKT MC/RQMC evaluation, event substeps, boundary diagnostics, reverse interpolation, and dividend reverse caches pass |
| VegaKT | Unit tests in `pricing-risk` | Active domain, equation (11), transition operator, reporting-IV basis, projection residuals, bucket estimates, and covariance layouts pass |
| Public API | Rust wire tests, Python smoke tests, and Python examples | Discrete dividends, explicit Local Volatility grids, optional reporting-IV bases, eSSVI helpers, standard SSVI helpers, VegaKT requests, and VegaKT result reports serialize through public facades; Python exposes typed VegaKT result accessors when a report is attached |
| CI | Latest pull request CI for this report's commit | Formatting, Clippy, Rust tests, statistical acceptance, supported wheels, Python smoke tests, dependency checks, and benchmark harness pass |

## Requirements Section 13.1 status

| Criterion | Current status |
|---|---|
| Local Vega calculated by Monte Carlo and AD | Covered by Local Volatility reverse tests and public risk orchestration tests; public Local Volatility evaluation returns Price/Delta/Gamma/Vega for MC/RQMC and populates VegaKT result reports when the model carries a matching reporting-IV basis; Apple Silicon macOS and Windows same-platform replay fixtures are frozen |
| Paper-defined VegaKT decomposition | Covered by projection, equation (11), transition, reporting-basis, and bucket tests |
| Coordinates, units, and bucket ordering | Covered by `VegaKtBucketCoordinate`, unit accessors, row-major shape checks, and report tests |
| CRN bump-and-revalue agreement | Scalar and basis-bump validation paths exist; Apple Silicon macOS and Windows risk replay fixtures are frozen |
| Reproducibility diagnostics | Estimator, bump conventions, Local-variance interpolation, clamp records, active domain, residual, and covariance layout are observable |
| Runtime and peak-memory benchmarks | CI runs the baseline benchmark harness and emits a retained Local Volatility/VegaKT Rust benchmark covering Price-only, AAD Local Vega, VegaKT decomposition, and selected CRN bump workloads; peak memory and allocation counts remain explicitly unavailable under the portable harness |

The additional conservation requirements are covered by unit tests for hat-kernel
projection, cell-integrated transition probability, reporting-basis partition of
unity, bucket-plus-residual reconciliation, and equation (11) refinement
diagnostics.

## Frozen Artifacts

- Frozen equation-level artifact: `fixtures/local-vol/reference-cases-v0.1.json`.
- Frozen Apple Silicon macOS Price-only and Delta/Gamma/Vega/VegaKT replay
  artifact:
  `fixtures/replay/local_volatility-macos-aarch64.json`.
- Frozen Windows x86-64 Price-only and Delta/Gamma/Vega/VegaKT replay artifact:
  `fixtures/replay/local_volatility-windows-x86_64.json`.
- Frozen numerical policy: `docs/local-vol-vegakt-numerical-contracts-v0.1.md`.
- Diagnostics catalogue: `docs/local-vol-vegakt-diagnostics-v0.1.md`.
- Python valuation example: `examples/python/local_vol_vegakt.py`.
- Retained Local Volatility/VegaKT benchmark artifact:
  `benchmark-results/local-volatility-rust.json` from CI.
- CI evidence: latest pull-request workflow run for this report's commit.

## Remaining Work

- No L8 acceptance item remains open for the v0.1 Local Volatility/VegaKT
  baseline.

## Known limitations

- Same-platform bitwise replay is guaranteed only for the pinned supported
  platform, toolchain, configuration, and library version represented by each
  fixture. Cross-platform bitwise equality is not required.
- Portable peak-memory and allocation counters remain `null` in benchmark
  metadata for the same reason documented by the European benchmark baseline.
- The public Python Local Volatility helpers materialize explicit grids before
  request compilation; they do not preserve calibrated surface parameters as a
  separate runtime model path. The calibrated-surface helpers now retain an
  explicit reporting-IV basis, and public VegaKT evaluation requires requested
  reporting nodes to match that retained basis.
- SSVI parameter Greeks are intentionally outside the MVP VegaKT definition.
