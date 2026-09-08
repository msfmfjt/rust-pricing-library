# European Black–Scholes implementation conformance report v0.1

Status: accepted
Evidence date: 2026-09-08
Requirements: `requirements-v1.0.md`
Roadmap: `european-bs-roadmap-v0.1.md`

## Decision

The European Black–Scholes vertical slice satisfies Gates G0–G8 and is accepted
as the infrastructure baseline for the Local Volatility/VegaKT slice. No known
correctness issue is waived by benchmark performance. Public European product,
result, RNG-addressing, deterministic-reduction, and Python ownership contracts
are frozen subject to the requirements change process.

## Acceptance evidence

| Area | Evidence | Result |
|---|---|---|
| Analytical | Frozen 18-case call/put covering grid across moneyness, maturity, rate, dividend yield, and volatility | Price/Delta/Gamma/Vega pass binary64 reference tolerances |
| Deterministic replay | Full-risk Pseudo-MC and 16-scramble RQMC fixtures on Apple Silicon macOS and Windows x86-64 | Exact same-platform UTF-8/bit-pattern comparison passes |
| Pseudo-MC uncertainty | 16 fixed seeds, 4,096 independent antithetic units per seed | 15/16 95% coverage; RMS z 0.965989; maximum absolute z 2.667271 |
| RQMC uncertainty | 8 fixed seeds, 256 points × 16 independent scrambles | 8/8 95% coverage; RMS z 0.732658; maximum absolute z 1.298212 |
| Risk | Analytical, AAD, bumped-AAD Gamma, and paired CRN validation tests | Delta/Gamma/Vega reconcile under sampling-aware bounds and explicit bump ladders |
| Reduction | Fixed logical blocks, padding, task-order permutation, and worker-count replay tests | Scheduling and worker count do not alter covered result bits |
| Rust/Python | Public examples, typed builders/results/diagnostics, JSON parity, and GIL-release tests | Linux integration and both supported private wheels pass |
| Packaging | CPython 3.12 clean-environment wheel tests | `aarch64-apple-darwin` and `x86_64-pc-windows-msvc` pass |
| Performance | CI-retained Rust and installed-wheel Python reports with host/compiler metadata | Baseline recorded; no latency SLA asserted |

CI run 96 exercised formatting, Clippy, unit/integration tests, the explicit
multi-seed statistical suite, dependency-direction checks, both native wheels,
benchmarks, and replay generation. The replay fixture intentionally stores
platform-specific low-bit differences in CRN difference diagnostics.

## Frozen replay configurations

Both cases use Spot/Strike 100, one-year expiry, continuously compounded rate
5%, continuous dividend yield 2%, volatility 20%, antithetic variates, two
workers, reduction block 256, AAD tile 128, and checkpoint interval 16.

- Pseudo-MC: 10,003 independent units and seed `0x0123456789abcdef`.
- RQMC: 1,024 Sobol points per scramble, 16 scrambles, Brownian bridge, LMS plus
  Digital shift, and seed `0xfedcba9876543210`.

The fixtures preserve normalized Request/Result JSON, Request/Plan/payoff
fingerprints, binary64 estimator and validation fields, effective execution
policies, and direction/scramble checksums.

## Diagnostics and warnings

The complete successful-result diagnostic surface and the two currently emitted
warning codes are frozen in `european-bs-diagnostics-v0.1.md`. Errors remain
fail-fast and do not produce partial pricing objects.

## Performance interpretation

The benchmark workload uses 16,384 independent antithetic units (32,768 paths),
two workers, and reduction block 256. It reports Price-only, the current
integrated AAD-plus-CRN-validation kernel, an independently executed five-plan
Price-only CRN bump workload, and installed-wheel Python calls. Peak memory and
allocation count remain `null` with explicit reasons because portable collection
requires a platform-specific process harness or instrumented allocator.

## Known limitations accepted for v0.1

- Bitwise replay is guaranteed only for the same platform, pinned configuration,
  binary/toolchain, and library version; cross-platform bitwise identity is not
  promised.
- The full-risk execution currently combines AAD, bumped-AAD Gamma, and CRN
  validation. A separate five-plan bump benchmark is available, but pure AAD-only
  wall time is not inferred by subtracting noisy measurements.
- Allocation count and peak memory are not yet collected; their absence is
  explicit in benchmark artifacts.
- The analytical oracle is validation infrastructure, not a public production
  pricing engine.
- Discrete dividends, Local Volatility, VegaKT, discontinuous payoffs, multiple
  assets, and early exercise remain outside this slice.

These limitations do not weaken a correctness criterion of the European slice
and do not require changing its frozen public contracts before the next roadmap.

## Next stage

The next implementation roadmap starts with calibrated SSVI/eSSVI inputs,
Dupire Local variance with explicit Floor/Cap, Log-Euler simulation in the
continuous martingale `f` coordinate, Local Vega AAD, and the paper-defined
VegaKT decomposition.
