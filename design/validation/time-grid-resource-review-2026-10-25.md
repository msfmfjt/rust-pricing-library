# Time-grid resource review — 2026-10-25

## Scope

This follows PR #180 (`1bdbda8c009268c9ddb3562289848f3a043c18b9`) on
`fix/time-grid-allocation-errors`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`; the independent mutation-testing
branch #172 is not included, and no merge is authorized.

The review continues the source/Task/Model/Numerical Engine boundary, focusing
on the common event/substep grid used by native single-asset, basket and
stochastic-dividend compilers. Source documents can retain any finite positive
maximum step; only the native compiler knows the complete event horizon and
required grid size. Resource validation belongs at that shared boundary.

## Finding and correction

P2: `LocalVolTimeGrid::compile` checked each interval's floating-point step count
against `usize::MAX`, then repeatedly pushed nodes into growing vectors. Two
gaps remained:

- On 64-bit platforms, `usize::MAX as f64` rounds up to 2^64. A strict `>`
  comparison therefore accepts 2^64, and the following float-to-integer cast
  saturates to `usize::MAX`. A safe private-helper regression reproduces this
  on the parent: it returns `Ok(18446744073709551615)` instead of
  `Err(StepCountOverflow)`. No huge allocation is needed for that reproduction.
- Even representable interval counts can overflow when summed or require an
  impossible vector byte layout. The old implementation did not preflight the
  complete size or reserve the output buffers fallibly before generation.

The compiler now checks the complete count, including the initial node, before
generating any interval. It reserves the node and event-index buffers with
`try_reserve_exact`. Arithmetic/count failures retain `StepCountOverflow`;
reservation failures return the additive non-exhaustive variant
`LocalVolError::GridAllocationFailed { nodes }`. All native adapters preserve
this typed source. The existing Python error classifier maps it to
`resource_limit` / `time_grid`, with `stage == "compile"` through the common
Specification/Task entry points.

Every distinct event interval contributes at least its endpoint, even when
`interval / maximum_step` underflows to zero. This keeps the exact prior nodes
for valid extremely short intervals while making the capacity count agree with
generation. The ordinary interpolation arithmetic, event sorting/deduplication,
event index order and maximum-step value are unchanged.

There is no new arbitrary grid cap, minimum step, silent coarsening, numerical
tolerance, sample-count, RNG, worker/block, AAD or calibration-policy change.
No wire schema, signature or fingerprint domain changes. An allocator error is
not a universal memory budget: operating-system overcommit and other input,
model or evaluation allocations can still fail. This is not a fix for the
unresolved macOS SIGABRT/SIGTRAP described in the
[previous checkpoint](portable-wheel-crash-review-2026-10-25.md).

## Regression evidence

The new Rust tests cover the rounded integer boundary, a count fitting `usize`
whose `f64` buffer cannot fit the allocator layout, aggregate overflow across
multiple intervals, underflowing interval ratios, exact ordinary node bits and
event indices, typed error chains through basket and dividend compilers, and
continued exact reuse of healthy compiled plans. Impossible-layout tests are
rejected before an actual large allocation or node-generation loop.

Python regressions cover price/AAD compilation for basket, stochastic-dividend
BS and stochastic-dividend HW sources, repeated failures, unrelated healthy
plan reuse, and single-asset Bergomi/HW and rough-Heston source compilation.
These failures remain catchable with typed metadata, without parsing display
text. Specification compilation keeps its
historical `ValidationError`/`ValueError` class and `plan_compile_error` issue;
Task compilation keeps `PricingError`/`RuntimeError`. The rough-family's earlier
model-specific grid guard retains `invalid_hull_white_input` with field
`rough_time_grid_resource_limit`. The binding's native classifier test also
checks the new allocator error variant. An initial new Python assertion assumed
the wrong single-asset exception/guard contract; it was corrected to preserve
these existing behaviors, without changing the implementation or any numerical
acceptance condition.

The comparison baseline is #179's digest-verified CI Linux wheel; #180 changed
no native/library source. The baseline capture includes 36 complete task/report
pairs: all six task fixtures, three additional SV/HW/rough/LSV sources, eighteen
worker/block configurations, and nine step-width variations across basket and
dividend sources. All 36 candidate task/report pairs are byte-for-byte identical
to the parent, including prices, Greeks, standard errors, calibration diagnostics
and replay fingerprints. No comparison tolerance is used. Both captures have
SHA-256
`b857d64d5a940a75b4c50ba99280725343d0799b9bd234b786f267840361e6d0`.

Local validation:

- `cargo test --locked --workspace --all-features --exclude pricing-python`:
  1,091 passed, 0 failed, 91 existing ignored acceptance tests.
- Native binding tests: 6 passed. Explicit statistical, path-dependence and
  early-exercise acceptance targets: 5 passed in total.
- Formatting, all-target/all-feature Clippy with `-D warnings`, all-target
  workspace check, pricing's no-default-feature all-target check, all-feature
  documentation and dependency-direction checks pass.
- The 18 required reference/schema/link scripts and 47 replay/wheel checker
  tests pass. The final staged source archive also passes its checker.
- Release CPython 3.12 wheel builds; package/ABI/metadata/SBOM/install checks and
  all 341 runtime callable signatures pass. Candidate wheel SHA-256:
  `3ba7426072d1f1bfd26c5848be8a4628c4093babf91009616a93913fac91bd3a`.
- Installed-wheel validation: 220 unique tests across 34 modules pass, covering
  resource failures, facade/source/Task/Graph/report/error contracts, HW/LSV/rough/
  basket/local-correlation models, dividend risk and calibration. This includes
  the corrected four-test resource module rerun; the other 33 modules passed
  unchanged. The two new methods check 44 rejected compilation attempts.
- Full import-only discovery confirms 110 modules / 610 methods and the exact
  provider partition of 20 shards / 179 cases. Discovery is not an execution
  pass for those additional numerical cases.

The managed local runtime needed a symlink-backed venv interpreter because its
initial copied interpreter was not executable. Only environment creation was
adjusted in the validation invocation; package checks and repository code were
unchanged. Full Python numerical execution, default/minimal-feature test-suite
runs and release-only extended model/risk panels remain hosted CI gates. The
ordinary Rust command does not run ignored panels, except the five acceptance
tests explicitly executed above.

## CI follow-up

The complete Windows Rust job on #180 (`121170796684`) now passes, including
the smoke-runner/provider-shard checks, Clippy, default/minimal features, tests,
statistical acceptance, documentation and dependency direction. The new macOS
installed-wheel run was still queued during this checkpoint, so no new native
crash diagnosis is available.
The native macOS failures remain open merge blockers; a passing Linux run or
the resource fix above does not resolve them.
