# Standalone AAD crate: P0 baseline and P1 architecture (draft)

Status: proposed; no implementation or performance claim yet.
Base: `main` at `f82a8b2c3fb2231959ae4104ec30778cdba683a1` (2026-11-20).
Objective: produce a finance-independent Rust AAD crate with a long-term **measured** objective of matching or exceeding MatLogica AADC on representative derivative-pricing workloads. XAD is an architectural reference, not an implementation dependency.

## Boundaries and current state

- `engine/aad/workspace.rs` provides `AadTilePolicy`, `CheckpointPolicy`, aligned per-slot buffers and `SoaWorkspace`. These do **not** by themselves constitute a generic AD graph engine.
- `engine/payoff/{compile,program,tape}.rs` compiles a finance-specific source graph to an immutable tape, evaluates it with branch-aware lazy instructions, and differentiates a selected scalar output in reverse. `CompiledOpcode` still refers to `UnderlyingId`, `Date`, `GraphComparison`, `CompactC2Smoothing` and `GraphError`.
- Forward path evaluation supports reusable `PayoffWorkspace`; existing scalar reverse evaluation allocates values, visited flags, and adjoints per invocation. Do not claim allocation-free reverse before a benchmark proves it.
- `engine/processes/capabilities.rs` implements model-specific `PathReverse`; `engine/calibration/capabilities.rs` implements `CalibrationReverse`. Keep their model/finance semantics in `pricing`, connecting them by externally seeded VJPs.
- `engine/risk/hull_white/aad.rs` aggregates pathwise seeds and applies the calibration VJP once per pseudo-MC mean or once per independent RQMC scramble; preserve this ordering.

## Target dependency direction

```
pricing-python --> pricing --> pricing-aad --> pricing-numerics (only if required)
                       \----> pricing-numerics
```

`pricing-aad` is a proposed workspace package name, **not verified as available on crates.io**. No dependency from `pricing-aad` back into `pricing`, `product`, `market`, or Python. Keep `pricing` as the finance-specific owner of the payoff source language and domain-model adjoints. Initially `publish = false`; publication is a separate explicit release decision.

## P1 public API contract (proposal)

- `InputId`, `NodeId`: bounded, opaque graph-local numeric handles; no financial identifiers.
- `GraphBuilder`: f64 constants, input placeholders, add/sub/mul/div/neg, exp/log, min/max, comparisons, conditional select. Preserve deterministic node order, explicit validation, output handles and finite-value failure policies.
- `CompiledGraph`: immutable `Send + Sync` execution artifact, with explicit `workspace_layout()`, input/output counts and stable versioned fingerprint. Graph compile must validate topology and memory limits.
- `Workspace`: independent mutable allocation owned per worker; `evaluate(inputs)` and `vjp(inputs, output_seeds, input_adjoints)` (exact signature TBD). No allocation on the hot path after preparation unless documented. Multiple output seeds and accumulated input adjoints must be possible without re-recording.
- `BatchWorkspace` and `Backend`: internal evolution points for scalar interpreter, batched SIMD and optional future JIT/AOT. Do not expose backend-specific representations in the stable graph interface.
- `CustomVjp`: future extension via stable primitive interfaces / explicit callbacks; do not force model-specific path records into the scalar tape. Error propagation and ownership/lifetimes must be explicit.
- `CheckpointPolicy`: reusable scheduling interface, with the precise recomputation/memory contract defined before extraction.

### Branches and discontinuities

Current payoff semantics must not silently change. A branch-aware graph must avoid evaluating inactive domain-invalid branches; simple eager `select` is not equivalent. Record active-path information per invocation for reverse, reset it on workspace reuse, and specify equality/tie rules for min/max and comparison. A hard indicator has zero pathwise adjoint almost everywhere; never market that as a correct digital/barrier Greek. Financial smoothing/bridge estimators stay in `pricing`; generic smoothing primitives may move after separate tests.

### Floating point contract

- f64 baseline, finite-input/output checks and diagnostic errors; preserve signed zero where the existing interface relies on it.
- Determinism is scoped to same executable, target and execution policy. Different CPU/OS or SIMD/JIT backends must meet numerical tolerances, not assumed bitwise identity.
- Baseline builds retain `#![forbid(unsafe_code)]` for the pure-Rust interpreter. Any eventual JIT FFI/unsafe should be isolated in a separate backend crate and audited.

## P0 benchmark contract

Report separately: graph construction/compile, initial workspace allocation, warm forward-only, warm forward+VJP, cold re-record+VJP if relevant, allocated bytes/op, peak resident memory, and throughput. **Never compare an optimized JIT price+Greeks numerator against an unoptimized price-only denominator without labeling both.**

Baseline workload matrix:
1. Branch-free European/Asian graph, 1e4, 1e5 and 1e6 paths, varied observation counts and graph nodes.
2. Branch-heavy barrier/autocall graph, exercised in three path distributions (rare, balanced, frequent branch) with lazy/invalid inactive branches.
3. Model-path reverse under 1F/2F Bergomi, 120 steps, fixed random draws; compare existing hand-coded VJP separately from generic graph cost.
4. Calibration reverse (LSV particles) and Hull–White + LSV **as application integration benchmarks**; do not treat them as generic tape benchmarks.
5. Small graph with 1, 10, 100, 1000 independent inputs and several outputs/seeds; evaluate per-input gradient scaling.

Measure: price value, every named native-coordinate first derivative, any standard errors, same-platform replay metadata, build mode/compiler/CPU features/threads and median plus spread (>= 10 warmed repetitions); force identical deterministic input vectors across frameworks. Run on available representative Apple Silicon macOS, Windows x86-64/MSVC and Linux. Use pinned toolchains and record exact sha.

Comparison targets: baseline Pricing, extracted interpreter, echidna, CppAD; XAD JIT/AADC only when licensed and executable on identical hardware. Publish vendor results strictly as **third-party figures**, not head-to-head measurements.

## Gates / acceptance

P0:
- [ ] Freeze exact input fixture hashes, graph shapes, pricing/risk outputs and compile flags.
- [ ] Land benchmark harness and CI artifact schema, without altering pricing logic.
- [ ] Collect baseline Rust release-mode timing and memory evidence on at least one supported host.

P1:
- [ ] Implement numeric-only graph and compiled execution with reusable primal+adjoint workspace.
- [ ] Support arbitrary multi-output VJP seeds and input adjoint mapping.
- [ ] Demonstrate lazy branch equivalence including inactive-domain and tie tests.
- [ ] Exercise thread isolation and independent workspace reuse.
- [ ] Adapt existing `pricing` payoff graph to numeric inputs, retaining its public API through an adapter/re-export as needed.
- [ ] Pass existing Rust/Python tests, statistical acceptance, replay fixtures, `cargo fmt`, clippy and documentation.
- [ ] Produce A/B price/Greeks comparisons with unchanged conventions; performance regressions require explicit justification.
- [ ] Do **not** publish the crate or merge a performance claim without reproducible local evidence.

## Implementation sequence

1. Add an independent P0 benchmark harness to record current payoff forward and adjoint cost and allocations, without changing runtime code.
2. Introduce new internal `pricing-aad` crate with only `GraphBuilder`, `CompiledGraph`, `Workspace`, and generic f64 reverse ops; keep `publish=false`.
3. Add an adapter mapping `(UnderlyingId, Date, pre/post)` to `InputId` and generic adjoints back to financial observations.
4. Move branch-aware evaluation and custom primitives behind well-defined generic internal interfaces.
5. Compare pre/post profiles; later optimize with SoA, SIMD, optional code generation and checkpoint plans, one independent measured change per PR.

## Explicit non-goals for first extraction

- Rewriting Bergomi/Hull–White/LSV calibration derivatives with a generic tape.
- Claiming to beat AADC without same-machine, same-workload and same-computation measurements.
- Moving `pricing` risk labels, curve conventions, smoothing policies or payoff product types into a public numeric crate.
- Enabling unsafe or adding nightly compiler dependencies to the core crate.
