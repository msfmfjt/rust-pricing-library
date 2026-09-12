# Path Dependence conformance report v0.1

Status: accepted

Evidence date: 2026-09-12

Requirements: `requirements-v1.0.md`

Roadmap: `path-dependence-roadmap-v0.1.md`

## Decision

The Path Dependence slice satisfies Gates P0-P8 and is accepted as the v0.1
Path Dependence baseline. Pull-request CI for this report's commit passes on
Apple Silicon macOS, Windows x86-64, and Ubuntu development runners, and
retains the supported wheel, replay, source, and benchmark artifacts.

## Gate status

| Gate | Current evidence | Status |
|---|---|---|
| P0 | `path-dependence-numerical-contracts-v0.1.md`, Decimal reference fixture, independent checker | Satisfied locally |
| P1 | `pricing-product` smoothing and graph tests cover kernels, extrema, reverse rules, constant folding, limits, and fingerprints | Satisfied locally |
| P2 | Digital exact/smoothed unit tests, CRN risk validation, public diagnostics, Python smoke coverage | Satisfied locally |
| P3 | Discrete Barrier state, parity, rebates, endpoint/jump handling, reverse checks, and collision tests | Satisfied locally |
| P4 | Exact and compact-C2 continuous bridge formulas, Black-Scholes and Local Volatility reverse, diagnostics, refinement, parity, and random-coordinate tests | Satisfied locally |
| P5 | Asian/Lookback validation, fixed-state diagnostics, zero-risk fixed products, bounds, refinement, Local Volatility and Python coverage | Satisfied locally |
| P6 | Rust/wire/Python smoothing API, Schema v2 migration, typed diagnostics, examples, and Width ladder tests | Satisfied locally |
| P7 | Deterministic/statistical acceptance, supported-platform replay, benchmark harness, and full-suite CI | Satisfied |
| P8 | Accepted diagnostics catalogue, requirement map, README, and release status | Satisfied |

## Requirement evidence

| Included requirement | Direct evidence | Result |
|---|---|---|
| Digital cash/asset Calls and Puts with explicit terms and equality convention | `DigitalSpec` validation and `digital_builder_executes_exact_cash_and_asset_payoffs` | Covered |
| Arithmetic Asian explicit weighted observations, known fixings, validation-date classification, and payment ordering | `ArithmeticAsianSpec`, `PricingRequest` validation tests, graph tests, and path-state diagnostics | Covered |
| Fixed Lookback declared discrete monitoring and required historical extremum | `FixedLookbackSpec`, request validation tests, graph tests, and path-state diagnostics | Covered |
| Known Asian and Lookback state is fixed under market bumps | `fully_fixed_asian_and_lookback_report_exact_zero_market_risks` and `partially_fixed_history_is_invariant_under_all_market_risks` | Covered |
| Observation/dividend collisions use post-dividend Spot | `asian_and_lookback_dividend_collisions_observe_post_jump_spot` | Covered |
| Discrete Barrier Up/Down, KnockIn/KnockOut, inclusive touch, and expiry rebate | Barrier graph tests and `continuous_barrier_in_out_parity_covers_directions_and_rebates` | Covered |
| Affine-dividend Barrier jumps use pre/post Spot without random draws | `barrier_dividend_collision_uses_pre_and_post_spot_without_an_extra_dimension` and bridge collision tests | Covered in exact and smoothed modes |
| Continuous bridge transformed barriers, trapezoidal variance, log accumulation, and full reverse | `pricing-mc::barrier_bridge` tests and continuous Barrier facade tests | Covered in exact and smoothed modes |
| Continuous Local Volatility bridge convergence and constant-variance limit | `continuous_barrier_local_vol_*` refinement and limit tests | Covered |
| Compact C2 Indicator/Maximum/Minimum formulas and reverse rules | independent Decimal fixture checker plus `pricing-product::smoothing` and graph tests | Covered |
| Smoothed Price and Greeks share one surrogate payoff | valuation diagnostics and Digital/Barrier risk tests | Covered |
| Smoothed continuous Barrier endpoint and dividend-jump predicates | independent Decimal fixtures, bridge primitive tests, facade finite differences, Local Volatility limit tests, and diagnostics | Covered locally |
| Width ladder is explicit, ordered, non-adaptive, and uses common random coordinates | `PricingPlan::evaluate_width_ladder`, request validation, wire/Python tests, and `width_ladder_preserves_order_primary_and_common_random_coordinates` | Covered |
| Pseudo-MC/RQMC statistical acceptance for all four products | ignored CI tests in `path_dependence_acceptance.rs` with sampling-error bounds | Covered locally and on supported CI runners |
| Digital and Barrier AAD versus CRN Delta/Gamma/Vega | unit tests and ignored acceptance test `smoothed_digital_and_barrier_aad_agree_with_crn_validation` | Covered locally for Digital and discrete/continuous Barrier |
| Same-platform replay on both supported targets | Frozen macOS and Windows fixtures plus mandatory supported-wheel replay checks | Covered |
| Runtime and peak-memory benchmark evidence | `benchmark_path_dependence.rs`, benchmark suite/report checker, and retained CI metadata | Covered |
| Rust, Schema v2 JSON, and typed Python parity | wire round trips, schemas/golden fixtures, Python examples, stubs, and clean-wheel smoke tests | Covered locally |

## Current retained evidence

- numerical policy: `docs/path-dependence-numerical-contracts-v0.1.md`;
- diagnostics catalogue: `docs/path-dependence-diagnostics-v0.1.md`;
- independent reference artifact:
  `fixtures/path-dependence/reference-cases-v0.1.json`;
- Apple Silicon macOS replay artifact:
  `fixtures/replay/path_dependence-macos-aarch64.json`;
- Windows x86-64 replay artifact:
  `fixtures/replay/path_dependence-windows-x86_64.json`;
- Rust/Python example: `examples/python/path_dependence.py`;
- acceptance suite: `crates/pricing/tests/path_dependence_acceptance.rs`;
- replay generator: `crates/pricing/examples/replay_path_dependence.rs`;
- benchmark workload: `crates/pricing/examples/benchmark_path_dependence.rs`.

## Remaining work

- No P8 acceptance item remains open for the v0.1 Path Dependence baseline.

## Preserved baselines

The candidate work does not change the accepted European Black-Scholes or
Local Volatility/VegaKT conformance decisions. Their existing fixtures and
reports remain mandatory regression evidence while the open Path Dependence
items are completed.
