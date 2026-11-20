# Contract DSL Phase 0–4 verification

Implementation baseline: main `75902772ea615b7cf551f8f849db7cb7636ef87c`.
Pricing implementation: `aafdcf2c2169ed007aedbb66d930ec9ec6049d1b`.
The compiled lazy schedule, constant branch shortcuts and deterministic analysis
are part of Phase 4. Performance acceptance follows the retained
[fixed protocol](performance-protocol.json).
Validation dates: 2026-11-05–06 UTC. Local environment: Darwin 27.0.0 arm64,
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

The source property suite uses 1,000 deterministic generated cases. Native
program tests additionally compare 100 generated shared DAGs on 20 scenarios
each and verify a 5,000-level shared dependency chain without recursive traversal. The mutation
suite injects six targeted semantic faults: inclusive comparison, payment sign,
event idempotency, future look-ahead, nested redemption gating and coupon
notional. All must cause actual Rust test failures; compiler failures and zero
selected tests do not count as killed mutants. This is curated mutation testing,
not exhaustive mutation coverage. Raw results are in [mutations.json](mutations.json).

## Local checks

Final counts and hosted status are retained in [results.json](results.json).
Rust: **1,164 passed, 91 ignored** plus **7 pricing-python crate tests**.
Installed Python: **44 passed**, including eight Contract integration tests.
The final implementation also passes the deterministic analysis check through
the installed Python wheel. All six curated mutants were killed. CI is recorded separately in the
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

[benchmark.json](benchmark.json) retains all ten complete release invocations:
five with two observation dates and five with twelve. Each uses 1,000,000
payoff paths, 262,144 antithetic MC sampling units, seed 7 and two workers.
The benchmark uses the frozen Black-Scholes request (volatility 0.2); the
European remains a one-date control in both schedule settings.
One warmup precedes seven measured repetitions. Native PricingPlan, native
PricingTask and Contract PricingTask run in alternating order, with raw paired
timing samples retained. Every repetition checks PV and standard error agreement
within 1e-11; compilation timings average 200 compilations.

The protocol was fixed before these retained measurements: the median across
five invocations must be at most 1.15 times native runtime for both complete MC
and reused payoff evaluation. Contract compilation must remain below 1 ms,
with zero warmed workspace capacity growth and no more path workspace bytes
than the native comparator. All representative cases pass unchanged criteria.
These are machine-specific acceptance measurements, not a universal SLA.

| Dates | Product | MC / native task | Reused payoff / native | Contract compile, µs | Workspace / native |
|---:|---|---:|---:|---:|---:|
| 2 | European | 1.002 | 1.005 | 10.01 | 1.000 |
| 2 | Asian | 0.978 | 0.818 | 14.83 | 0.813 |
| 2 | Discrete barrier | 0.947 | 0.676 | 21.88 | 0.880 |
| 12 | European | 1.003 | 1.004 | 10.06 | 1.000 |
| 12 | Asian | 0.947 | 0.767 | 48.42 | 0.768 |
| 12 | Discrete barrier | 0.922 | 0.637 | 65.27 | 0.858 |

Autocall has no legacy named-product comparator in this API. Its two/twelve-date
median compilation times are 40.66/278.64 µs, reused payoff times 74.68/363.83 ns,
and workspace sizes 290/1,260 bytes. Both show zero warmed capacity growth.
The twelve-date barrier uses 446 workspace bytes and 1,200 immutable program
bytes; Autocall uses 5,280 immutable program bytes. Program storage is shared
across paths and reported separately from path state.

Contract compilation includes semantic validation, analysis and lowering,
whereas native compilation begins with an existing graph. This extra fixed
cost is outside the path loop. The raw Contract barrier preserves individual
potential cashflow outputs; native named barriers have one net output. The
native deterministic aggregation combines complementary same-date payments
into one lazy choice while preserving source identities and signed-zero bits.

Unique dependencies are inlined into a bounded execution schedule; shared
expressions use memoized calls. Literal slots are prepared once per tape and
workspace. Constant branches use conditional copy/jump shortcuts, avoiding
repeated Select interpretation while preserving inactive observations/errors.
A prior larger-schedule run before these shortcuts failed the 1.15 MC criterion
for twelve-date barriers at 1.199. Its complete
[pre-shortcut evidence](benchmark-before-shortcuts.json) is retained; the
threshold and workload were not loosened to obtain the final pass.

Values occupy one slot per surviving native opcode; CSE deduplicates observation
and expression state. Workspace bytes include value, visited, call-stack and
output capacities. They exclude model path storage and process-wide allocator
counts; unsafe allocator interception is prohibited by workspace policy.
The explicit reuse API makes no capacity-growing allocations after branch shapes
are warmed. Native lazy calls retain at most 1 MiB of scratch capacity per thread
and allocate an owned output vector; `evaluate_reusing` avoids that output
allocation. No Contract AST is evaluated per path. Advanced register liveness
compression and model-wide buffer redesign remain Phase 5.

Reproduce the measurement inputs after building the release example:

```sh
cargo build --locked --release -p pricing --example benchmark_contract
for periods in 2 12; do
  for invocation in 1 2 3 4 5; do
    CONTRACT_BENCH_PERIODS=$periods target/release/examples/benchmark_contract > "benchmark-$periods-$invocation.json"
  done
done
```

## Explicit limits and remaining work

* Hosted CI results remain pending until the submitted heads actually complete.
  Keep all PRs draft until required hosted checks actually complete. The local
  performance acceptance is complete.
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
