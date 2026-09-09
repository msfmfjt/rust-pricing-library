# Local Volatility and VegaKT conformance report v0.1

Status: candidate, pending final acceptance
Evidence date: 2026-09-09
Requirements: `requirements-v1.0.md`
Roadmap: `local-vol-vegakt-roadmap-v0.1.md`

## Decision

The Local Volatility/VegaKT slice has implementation and CI evidence for Gates
L0-L7 and partial L8 evidence. It is not yet marked accepted because complete
same-platform Local Volatility replay fixtures and full benchmark artifacts for
the AD/decomposition-versus-bump workload still need to be frozen as retained
release evidence.

No known correctness failure is waived by the current performance results.
Current limitations are recorded explicitly and do not change the frozen
European Black-Scholes baseline.

## Acceptance evidence

| Area | Evidence | Result |
|---|---|---|
| Equation fixtures | `fixtures/local-vol/reference-cases-v0.1.json` checked by `scripts/check_local_vol_reference_fixture.py` | Standard SSVI, eSSVI, Dupire, equation (11), hat projection, and transition-cell references pass independent checks |
| Market surfaces | Unit and reference tests in `pricing-market` | SSVI/eSSVI values, analytic derivatives, density factors, Local-variance repairs, node helpers, and affine dividends pass |
| Path engine | Unit tests in `pricing-mc` | Non-uniform Brownian bridge, Local Volatility Log-Euler, event substeps, boundary diagnostics, reverse interpolation, and dividend reverse caches pass |
| VegaKT | Unit tests in `pricing-risk` | Active domain, equation (11), transition operator, reporting-IV basis, projection residuals, bucket estimates, and covariance layouts pass |
| Public API | Rust wire tests and Python smoke tests | Discrete dividends, explicit Local Volatility grids, eSSVI helpers, standard SSVI helpers, and VegaKT requests serialize through public facades |
| CI | Pull request CI run 143 for commit `8947908` | Formatting, Clippy, Rust tests, statistical acceptance, supported wheels, Python smoke tests, dependency checks, and benchmark harness pass |

## Requirements Section 13.1 status

| Criterion | Current status |
|---|---|
| Local Vega calculated by Monte Carlo and AD | Covered by Local Volatility reverse tests and risk orchestration tests; full same-platform replay fixture still pending |
| Paper-defined VegaKT decomposition | Covered by projection, equation (11), transition, reporting-basis, and bucket tests |
| Coordinates, units, and bucket ordering | Covered by `VegaKtBucketCoordinate`, unit accessors, row-major shape checks, and report tests |
| CRN bump-and-revalue agreement | Scalar and basis-bump validation paths exist; complete Local Volatility replay fixture remains pending |
| Reproducibility diagnostics | Estimator, bump conventions, Local-variance interpolation, clamp records, active domain, residual, and covariance layout are observable |
| Runtime and peak-memory benchmarks | CI runs the baseline benchmark harness; peak memory and allocation counts remain explicitly unavailable under the portable harness |

The additional conservation requirements are covered by unit tests for hat-kernel
projection, cell-integrated transition probability, reporting-basis partition of
unity, bucket-plus-residual reconciliation, and equation (11) refinement
diagnostics.

## Frozen and candidate artifacts

- Frozen equation-level artifact: `fixtures/local-vol/reference-cases-v0.1.json`.
- Frozen numerical policy: `docs/local-vol-vegakt-numerical-contracts-v0.1.md`.
- Candidate diagnostics catalogue: `docs/local-vol-vegakt-diagnostics-v0.1.md`.
- Candidate CI evidence: pull request run 143 on commit `8947908`.

## Remaining acceptance work

- Freeze Local Volatility Price, Greek, and VegaKT replay fixtures for Apple
  Silicon macOS and Windows x86-64 under the same same-platform byte-comparison
  contract used by the European baseline.
- Retain benchmark artifacts that separately identify the AD/decomposition
  workload and its selected and aggregate CRN bump validations.
- Add a Python example that starts from calibrated SSVI/eSSVI input and requests
  Local Volatility pricing plus VegaKT output.
- Promote this report from candidate to accepted only after the retained replay
  and benchmark evidence is attached to the final acceptance PR.

## Known limitations

- Same-platform Local Volatility replay fixtures are not yet frozen.
- Portable peak-memory and allocation counters remain `null` in benchmark
  metadata for the same reason documented by the European benchmark baseline.
- The public Python Local Volatility helpers materialize explicit grids before
  request compilation; they do not preserve calibrated surface parameters as a
  separate runtime model path.
- SSVI parameter Greeks are intentionally outside the MVP VegaKT definition.
