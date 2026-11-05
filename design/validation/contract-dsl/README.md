# Contract DSL Phase 0–4 verification

Implementation baseline: main `75902772ea615b7cf551f8f849db7cb7636ef87c`.
Pricing implementation commit: `a7f3d62`; subsequent commits retain benchmark
and validation evidence without changing pricing logic.
Validation date: 2026-11-05. Local environment: Darwin 27.0.0 arm64,
Rust 1.98.1, CPython 3.14.2, release wheel built with maturin 1.16.0.

## Review sequence

These are stacked, independently reviewable changes. None has been merged and
main has not been pushed. Review/merge in dependency order and retarget each
successor only after its predecessor lands.

| Phase | PR | Scope |
|---|---|---|
| 0 | [#207](https://github.com/msfmfjt/rust-pricing-library/pull/207) | Current-state investigation, ADR and migration boundary |
| 1 | [#208](https://github.com/msfmfjt/rust-pricing-library/pull/208) | Typed source, validation, analysis and strict v1 JSON |
| 2 | [#209](https://github.com/msfmfjt/rust-pricing-library/pull/209) | Event replay, reduction, unpaid payments and checkpoints |
| 3 | [#210](https://github.com/msfmfjt/rust-pricing-library/pull/210) | Native graph operations, lowering, task adapters and versioned wire extensions |
| 4 | [#211](https://github.com/msfmfjt/rust-pricing-library/pull/211) | Builders, Python, workspace reuse, benchmarks and validation evidence |

## Requirements and evidence

| Requirements | Implementation / acceptance evidence |
|---|---|
| FR-01–04, FR-09–10 | Typed numeric/Boolean/date expressions, complete observation identity, currency/date validation, analysis; `contract_source` tests and strict schema rejection |
| FR-05, FR-17–19 | Forward, European, Digital, Swap, Asian, discrete Barrier and Autocall builders; `contract_builders` native-path comparisons, inclusive hits and early-redemption cancellation |
| FR-06–08 | `contract_lifecycle`: duplicate/conflicting IDs, future-event exclusion, replay/checkpoint equality, explicit settlement, termination and original/residual scenario equivalence |
| FR-11–13 | Existing SourceGraph/CompiledPayoff, constant folding, exact CSE, observation deduplication, lazy branches and reusable buffers; `contract_mc` and native Payoff conformance |
| FR-14–16 | Existing single/multi-asset, SV/LSV, correlated and stochastic-dividend task adapters; same-seed PV/SE/Greeks/AAD parity and unsupported-estimator/model rejection |
| FR-22–24 | Rust and immutable Python construction/lifecycle/task APIs; strict source/checkpoint JSON, preserved legacy fingerprints, graph v2/specification v3/task v2 guards |
| NFR-01–05 | Recorded compilation/path/MC timing and workspace size; stable seeds, source/plan identity, explicit cashflow IDs, mutation/property/regression tests |
| FR-20–21 | Explicitly unsupported in Contract v1; Phase 5 was excluded from this request |

The source property suite uses 1,000 deterministic generated cases. The mutation
suite injects six targeted semantic faults: inclusive comparison, payment sign,
event idempotency, future look-ahead, nested redemption gating and coupon
notional. All must cause actual Rust test failures; compiler failures and zero
selected tests do not count as killed mutants. This is curated mutation testing,
not exhaustive mutation coverage. Raw results are in [mutations.json](mutations.json).

## Local checks

Final counts and hosted status are retained in [results.json](results.json).
Rust: **1,155 passed, 91 ignored** plus **7 pricing-python crate tests**.
Installed Python: **44 passed**, including eight Contract integration tests.
All six curated mutants were killed. CI is recorded separately in the
[dated commit-specific snapshot](ci-snapshot.json); it is still incomplete.
The following commands are reproducible from the repository root:

```sh
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features --exclude pricing-python
cargo test --locked -p pricing-python
cargo check --locked --workspace --all-targets
cargo check --locked -p pricing --no-default-features --all-targets
cargo doc --locked --workspace --all-features --no-deps
python scripts/check_contract_mutations.py --report-directory /tmp/contract-mutations
cargo run --locked --release -p pricing --example benchmark_contract
```

The installed release wheel passes `test_contracts`, `test_product_graph`,
`test_pricing_task`, `test_pricing_specification`, `test_typed_pricing_sources`,
`test_wheel_runtime_contract` and `test_multi_asset` in isolated processes.
The independent frozen API checker verifies all 368 callable signatures.
Archive layout, metadata, RECORD hashes, dependency SBOM, ABI and exact stub
contents are checked through the existing wheel-validation routines.

Additional gates cover current/legacy JSON schemas and downgrade rejection,
three-crate dependency direction, Markdown links, source archive contents,
wheel-runner/shard/failure-propagation tests, and the new example.
The normal Rust run does not execute ignored release acceptance panels; hosted
CI retains those cross-platform and extended numerical checks. A local pass
must not be used to relabel queued hosted checks as successful.

## Performance

[benchmark.json](benchmark.json) records the retained release measurements.
The executable compares the existing named products with generated contracts on
the same paths, and checks exact PV/standard-error agreement within 1e-11 under
seed 7, 16,384 antithetic sampling units and two workers. Timing uses one warmup
and five repetitions per native/contract MC pair. Five complete invocations are
retained. Native PricingPlan and native PricingTask are both reported so API
reporting overhead is visible. Path evaluation uses 100,000
paths; compilation uses 200 repetitions. The retained sample reports are
machine-specific measurements, not universal performance thresholds.

The raw Contract payoff retains individual potential cashflow outputs; the
native named barrier has one net output. Their path timings therefore include
different reporting work. Native deterministic aggregation folds complementary
same-date Contract outputs into one lazy choice while preserving source IDs and
signed-zero behavior. Existing legacy graph evaluation order is unchanged.

Explicit reusable workspaces show no capacity growth after the exercised branch
shapes are warmed. Values occupy one slot per surviving native opcode;
CSE removes duplicate observation/subexpression state. The byte metric includes
value/visited/traversal/output vector capacities, not model path storage or
process-global allocator counts. Native lazy calls also retain at most 1 MiB of
scratch capacity per thread and allocate an owned result vector; explicit
`evaluate_reusing` avoids that output allocation. No AST is evaluated per path.

The generic lazy barrier's isolated payoff remains slower than the specialized
legacy barrier. Short MC timings vary substantially across invocations, so the
aggregate median alone does not establish a performance pass. Broader portfolios,
longer schedules and larger sample counts require further benchmarking before
setting an acceptance threshold. Advanced register liveness compression is Phase 5.

Retained MC totals (median across five invocations, milliseconds):

| Product | Native plan | Native task | Contract task | Contract/native task range by invocation |
|---|---:|---:|---:|---:|
| European | 3.139 | 3.528 | 2.359 | 0.67–0.99× |
| Asian | 5.646 | 4.237 | 4.239 | 1.00–1.29× |
| Discrete barrier | 5.671 | 6.019 | 5.971 | 0.74–1.42× |

## Explicit limits and remaining work

* Hosted CI results remain pending until the submitted heads actually complete.
  Keep all PRs draft while those gates or performance acceptance remain unresolved.
* Contract v1 preserves but cannot simulate intraday/timezone observations or
  future rate fixings. Swap floating legs can be reduced from published rate
  fixings; future floating-rate pricing fails explicitly.
* Native single-asset stochastic rates reject distinct payment dates. The native
  multi-asset adapter retains existing per-payment pathwise discounting.
* Currency groups are separate. A convenience task rejects mixed currencies;
  callers must create explicit native tasks per group with appropriate markets.
* Hard comparisons reject unsupported pathwise/AAD estimators. Explicit common
  random-number MC bumps remain available through repricing; no LR estimator,
  smoothing, continuous-barrier correction or exercise policy is silently chosen.
* Autocall v1 uses absolute levels, non-memory coupons and at most 20 periods.
  Rebate amounts are absolute cash paid at maturity. General exercise and
  continuous monitoring are Phase 5; existing legacy APIs remain unchanged.
