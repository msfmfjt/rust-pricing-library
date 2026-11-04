# Numerical execution and allocation-error review — 2026-10-22

## Scope and baseline

This continues the [model and calibration review](hw-market-iv-boundary-review-2026-10-22.md)
on PR #173, commit `168f16c30ca18b69d1c58f62640efbd1f5a5617c`.
Main remains `45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The review branch is
`fix/executor-allocation-errors`, based on #173. The independent mutation-testing
PR #172 remains based on #170 and is outside this change.

The review covered the calculation-owned executor, fixed reduction blocks,
scalar/vector/tiled/AAD reductions, Philox coordinates, Sobol table and point
validation, Brownian-bridge dimensions, LSM scaling/replay and CPQR error checks,
and analytic/Fourier output validation. No numerical-formula change is proposed.
This is a scoped review, not a proof that every allocation or extreme floating
point input is safe across the library.

## P2: accepted execution sizes escape as Rust panics

`PseudoMcConfig` accepts the full positive `u64` sampling-count domain.
`ExecutionPolicy` likewise accepts positive `u64` reduction-block sizes. The
executor checked whether `ceil(sampling_units / block_size)` fit `usize`, then
used `Vec::with_capacity` for the block table. A count can fit `usize` while its
`Range<u64>` allocation exceeds the allocator's layout limit. Allocation failure
also remained infallible at this boundary.

Reproduction: set `independent_sampling_units = 18446744073709551615` and
`reduction_block_size = 1` on a valid task. Parsing and compilation succeed;
evaluation panics with `capacity overflow`. This reproduces through the standard,
multi-asset, and stochastic-dividend task adapters. Python receives a PyO3
`PanicException`, which ordinary `except Exception` / `except PricingError`
handlers do not catch. Rust callers likewise receive a panic instead of the
declared `Result` error.

The existing requirement in [Rust API requirements, section 8](../requirements-v1.0.md#8-rust-api-requirements)
requires typed errors instead of normal validation panics. The common executor
owns allocation of execution metadata; model/source adapters should not each
implement their own capacity limits.

The fix reserves the block table with `try_reserve_exact` and maps failure to
`ExecutionError::ReductionBlockAllocationFailed { count }`. The existing
`TooManyReductionBlocks` path remains for a count that does not fit `usize`.
Both failures occur before callbacks, reductions or AAD workspace initialization.
All five public reduction entry points use the same helper.

## Compatibility and numerical impact

- `ExecutionError` was already `#[non_exhaustive]`; the new variant is additive.
  Function signatures and existing valid configuration domains are unchanged.
- Existing error adapters turn the failure into `PricingError`. Standard and
  stochastic-dividend tasks preserve `resource_limit` / `reduction_blocks`.
  Multi-asset currently retains its existing numerical-error wrapper.
- Block boundaries, callback order, random coordinates, reduction trees,
  AAD tiling, covariance accumulation, calibration, numerical tolerances, wire
  schemas and fingerprint domains are unchanged.
- A maximum block size with a small sampling count is still valid. The fix
  checks the required allocation, without introducing an arbitrary count cap.
- Fallible reservation is not a universal memory budget: operating-system
  overcommit and later allocations can still fail. No broad OOM-safety claim
  is made.

## Regression evidence

Before the fix, the new Rust suite produced two capacity-overflow failures and
one passing valid-boundary test. The Python reproduction failed on each of five
distinct source/operation combinations (the initial run also repeated the
standard operation). No large allocation or long simulation is needed: the
requested layout itself is impossible.

The regression tests exercise scalar, fallible scalar, dynamic vector, tiled
array and AAD-workspace reduction entry points; no callback may execute on
allocation failure. They also check executor reuse and exact moments for small
workloads using the maximum block size. Python tests cover common-task price
and risk entry points, repeated failures and unrelated compiled-plan reuse.

Twenty-four complete serialized reports were captured from the parent wheel:
the six task fixtures and eighteen cases varying worker count (1, 3) and block
size (1, 7, `u64::MAX`) across standard, basket and stochastic-dividend risk.
All twenty-four reports are byte-for-byte identical after the fix, including
prices, Greeks, standard errors, diagnostics and replay fingerprints. No
comparison tolerance is used. The before/after report collections have SHA-256
`2c3094faa01b3d11bc6676e1bc46821a8bff513ac8346b6a542a60540effc36c`.

Local validation:

- `cargo test --locked --workspace --all-features --exclude pricing-python`:
  1,074 passed, 0 failed, 91 intentionally ignored acceptance tests.
- Native Rust binding tests: 5 passed. Explicit statistical, path-dependence
  and early-exercise acceptance targets: 5 passed in total.
- Formatting, all-target workspace check, all-target/all-feature Clippy with
  `-D warnings`, all-feature documentation and dependency-direction checks pass.
- The 18 required static/reference/schema/link scripts and 34 replay/wheel
  checker tests pass.
- Release CPython 3.12 wheel builds. Package/ABI/metadata/SBOM/clean-install
  checks pass, including the 341-callable runtime-signature contract.
- Python installed-wheel validation: 153 tests passed, covering the new resource
  regression, facade smoke tests, typed sources, task/report replay and reuse,
  structured errors, graph products, HW/LSV/rough/basket/local-correlation models,
  calibration reports, backtest native contracts and installed-wheel contracts.

The tests are [executor resources](../../crates/pricing/tests/executor_resources.rs)
and [Python execution resources](../../tests/python/test_execution_resources.py).
The full Python discovery suite and release-only extended model/risk panels are
delegated to hosted CI; the ordinary Rust command does not run ignored panels.
An additional local run of `test_vega_kt_rqmc_replay` was interrupted during its
long native backtest calculation; it is not counted as a pass or failure here
and remains part of hosted Python CI.

## Outstanding issues

The experimental multi-asset adapter converts several underlying errors,
including execution failures, to `Numerical(String)`, losing the typed source
chain and specific resource code. Fixing that deserves a separate compatibility
review: unlike `ExecutionError`, `MultiAssetError` is currently exhaustive.
This change preserves its existing Rust and Python classification contracts.

The hosted CI chains for #170, #171 and #173 were still running or queued when
this batch began; no new failed checks were present. Older Windows failures
on #164/#165 still lack retrievable logs and remain unresolved. Final CI status
will be recorded separately from local validation. No PR is merged by this work.
