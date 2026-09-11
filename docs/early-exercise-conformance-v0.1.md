# Early Exercise conformance report v0.1

Status: candidate; Gates E7 and E8 are not yet accepted

Evidence date: 2026-09-12

Requirements: `requirements-v1.0.md`

Roadmap: `early-exercise-roadmap-v0.1.md`

## Decision

The Early Exercise slice is not yet accepted. Gates E0-E6 have direct local
evidence and the locally executable E7 evidence is implemented. Two retained
release items remain open:

1. the Windows x86-64 Early Exercise replay artifact has not yet been reviewed
   and frozen;
2. the candidate commit has not yet completed the supported Apple Silicon
   macOS and Windows x86-64 CI and clean-wheel matrix.

These items are not waived as known limitations. E8 may be accepted only after
both are obtained, tested, and retained.

## Gate status

| Gate | Current evidence | Status |
|---|---|---|
| E0 | Frozen numerical contract, Decimal reference fixture, independent checker | Satisfied locally |
| E1 | American domain, schedule materialization, event-order and deterministic exercise tests | Satisfied locally |
| E2 | Polynomial basis, scaling, CPQR, rank, resource-limit, and reference tests | Satisfied locally |
| E3 | Backward training, domain isolation, zero-ITM policy, immutable model, and fingerprint tests | Satisfied locally |
| E4 | Independent valuation, stopping records, Pseudo-MC/RQMC uncertainty, bounds, and worker replay tests | Satisfied locally |
| E5 | Fixed-policy AAD/Gamma/Vega/VegaKT, CRN validation, risk labels, and stopping-index identity tests | Satisfied locally |
| E6 | Rust/JSON/Python API, v3 schemas and golden files, complete result round trips, examples, and clean-wheel smoke coverage | Satisfied locally |
| E7 | Call/Put grids, independent Bermudan tree, in/out-of-sample evidence, Mac replay, and benchmark harness | Incomplete: Windows replay is not frozen and candidate CI is pending |
| E8 | Candidate diagnostics catalogue, this requirement map, README, and release status | Incomplete until E7 is accepted |

## Requirement evidence

| Included requirement | Direct evidence | Result |
|---|---|---|
| Explicit increasing American Call/Put exercise schedule containing expiry | `AmericanVanillaSpec` validation and product tests | Covered |
| Calendar-aware Weekend-plus-Custom-holidays schedule materialization | schedule construction and adjustment-collision tests in `pricing-product` | Covered |
| Post-dividend exercise at a collision | `exercise_events_record_post_dividend_collision_order` and compiled `dividend_collisions` diagnostics | Covered |
| Strict exercise and ITM comparisons with equality continuing | independent Decimal fixture and `pricing-mc::lsm` decision tests | Covered |
| Deterministic total-degree basis and original-order coefficients | basis fixture, exponent tests, regression model diagnostics | Covered |
| Training-ITM scaling and pure-Rust deterministic CPQR | Decimal reference checker plus scaling, pivot, rank, and residual tests | Covered |
| Checked dimensions, tolerances, and allocations | LSM error tests and configured matrix-element limit | Covered |
| Independent Pseudo-MC training and valuation domains | random-domain tests, phase metadata, and result diagnostics | Covered |
| Independent randomized Sobol training and valuation scrambles | RQMC checksum and seed-change tests | Covered |
| Frozen out-of-sample policy with stopping indices | policy valuation tests and American worker-count replay tests | Covered |
| Pseudo-MC/RQMC Call and Put acceptance against an independent reference | `early_exercise_acceptance.rs` deterministic/statistical grids and independent binomial tree | Covered locally; candidate CI pending |
| In-sample versus out-of-sample bias evidence | eight independent training replicates in `independent_training_replicates_bound_in_sample_optimism` | Covered locally; candidate CI pending |
| Fixed-strategy Delta/Vega, bumped-AAD Gamma, and Local Vega/VegaKT | American fixed-policy facade tests for Black-Scholes and Local Volatility | Covered |
| CRN validation retains policy, path identity, and stopping index | fixed-policy risk tests and typed method metadata | Covered |
| Complete Rust, v3 JSON, and typed Python replay state | request/result golden files, strict schemas, rich result round trip, Python example and wheel smoke | Covered |
| Same-platform replay on both supported targets | Mac fixture and replay checker; Windows CI artifact workflow | Not covered until the Windows artifact is reviewed and frozen |
| Training, valuation, AAD/CRN, and peak-memory benchmark evidence | `benchmark_early_exercise.rs`, benchmark runner/report checker, retained CI metadata | Harness covered; candidate CI artifacts pending |

## Current retained evidence

- numerical policy: `docs/early-exercise-numerical-contracts-v0.1.md`;
- diagnostics catalogue: `docs/early-exercise-diagnostics-v0.1.md`;
- independent reference artifact:
  `fixtures/early-exercise/reference-cases-v0.1.json`;
- Apple Silicon macOS replay artifact:
  `fixtures/replay/early_exercise-macos-aarch64.json`;
- Rust/Python example: `examples/python/american_lsm.py`;
- acceptance suite: `crates/pricing/tests/early_exercise_acceptance.rs`;
- replay generator: `crates/pricing/examples/replay_early_exercise.rs`;
- benchmark workload: `crates/pricing/examples/benchmark_early_exercise.rs`.

## Required evidence before acceptance

1. Push the candidate stack and obtain green Apple Silicon macOS and Windows
   x86-64 CI, including clean-wheel smoke and Early Exercise statistical
   acceptance.
2. Review the generated Windows `early-exercise-replay.json`, promote it as
   `fixtures/replay/early_exercise-windows-x86_64.json`, and require exact
   same-platform replay on both CI targets.
3. Re-run the complete locked validation suite and retained source-archive
   check, then change this report, README, and release-readiness status only
   when every E0-E8 item has current evidence.

## Preserved baselines

The candidate work does not change the accepted European Black-Scholes or
Local Volatility/VegaKT decisions. Path Dependence remains a separate candidate
with its own unresolved Windows replay requirement. All prior fixtures and
reports remain mandatory regression evidence.
