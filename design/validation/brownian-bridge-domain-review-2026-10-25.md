# Brownian-bridge numerical-domain review — 2026-10-25

## Scope and finding

This follows PR #182, commit `0595479d6db66751f9dc748512608b231f23145f`,
on `fix/brownian-bridge-coefficient-validation`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The separate mutation-testing
branch #172 is excluded, and no merge is authorized.

P2: the public Rust `BrownianBridgePlan::compile` validated finite, strictly
increasing times but accepted an interior conditional variance of zero or
infinity. For the three-node grid `[0, s, 2*s]`, its fixed arithmetic evaluates
`(s*s)/(2*s)`. With `s = 2^-600` the product underflows to zero; with
`s = 2^600` it overflows to infinity. In both cases the input times and the
analytical conditional variance `s/2` are finite and positive. The parent
returns `Ok` with an interior standard deviation of zero or infinity.

Zero removes an independent Brownian component without an error. Infinity
contaminates both the forward transform and its reverse. Neither is a usable
compiled Brownian bridge. Two regressions failed against the parent before
the implementation changed, printing the accepted degenerate instructions.

## Correction and compatibility

The shared sampling compiler now checks every candidate's computed conditional
variance, including candidates in child intervals after the first split. A
nonpositive or nonfinite value returns the additive, non-exhaustive
`BrownianBridgeError::InvalidConditionalVariance` variant with the bracketing
indices, interior index and observed variance bits. Compilation returns no
partially usable plan.

The [frozen sampling requirements](../requirements-v1.0.md) prescribe the
conditional-variance formula, maximum-variance ordering and deterministic
tie breaks. This correction preserves that arithmetic and all successful
candidate comparisons. It does not reassociate the formula, rescale time,
clamp a coefficient, drop a node, consume a different normal, or change the
forward/reverse kernels. The bridge ABI identifier and valid plan/task
fingerprints remain unchanged. There is no new arbitrary minimum/maximum
time: terminal-only grids remain supported from the smallest positive `f64`
through `f64::MAX` because they need no conditional-variance product.

This guard rejects degeneracy in the existing floating-point calculation; it
does not extend that calculation to all mathematically valid scales or claim
uniform relative accuracy across the entire `f64` range. Supporting additional
scales with different arithmetic would require its own numerical/replay review.

The check belongs to the numerical sampling boundary, which owns the resolved
grid and coefficients, rather than duplicating limits in Product Graph, source,
Task, Python, or each model. No public Python signatures, serialized schemas,
sampling counts, seeds, worker/block policies, prices/Greeks/AAD/calibration
formulae or numerical acceptance tolerances are changed.

This is independent of the unresolved macOS SIGABRT/SIGTRAP. There is no
evidence that these extreme grids caused those failures. The
[native-backtrace follow-up](native-wheel-backtrace-review-2026-10-25.md)
remains the diagnostic path for that merge blocker.

## Regression coverage

The public-API tests cover underflow, overflow, smallest-subnormal inputs,
multiple factor counts, and failures arising only after valid root candidates
split into left/right child intervals. Positive controls retain extreme
terminal-only bridges and ordinary four-step bridges scaled by `2^-500`,
`1` and `2^500`. The latter preserve normalized shocks and reverse adjoints
bit-for-bit and check the transform's identity covariance deterministically.
The existing nonuniform coefficients, analytical uniform bridge, factor order
and adjoint-transpose regressions remain unchanged.

The installed-wheel replay comparison uses #181's digest-verified Linux wheel
as its native baseline (#182 changes only the diagnostic harness). It covers
54 complete task/report pairs: the earlier 36 source, worker/block and step
cases, plus explicit bridge enabled/disabled variants of all nine base sources.

Local validation:

- Workspace/all-feature Rust tests excluding the Python binding: **1,096
  passed, 0 failed**, with **91 existing ignored** acceptance cases. The five
  new public bridge tests are included in the passed count.
- Native Python binding tests: **6 passed**. Explicit statistical,
  path-dependence and early-exercise acceptance: **5 passed**.
- All-target/all-feature Clippy with `-D warnings`, workspace all-target check,
  pricing's no-default-feature all-target check, all-feature API docs and
  dependency-direction checks pass.
- All **18 reference/schema/link scripts** and **55 checker tests** pass.
  The **2 real macOS LLDB integration methods remain unexecuted locally**;
  their Linux skips are not passing evidence for debugger operation.
- A fresh CPython 3.12 Linux release wheel passes archive/metadata/SBOM/ABI,
  clean-install and all **341 runtime callable signature** checks. Its SHA-256 is
  `99aafa3d79daacf5a5fd19279261db1a3d5a33d6fec0377f89d68d68c2a2a8cc`.
- All **54 complete task/report pairs are byte-identical** to the parent,
  including prices, Greeks, standard errors, calibration diagnostics and replay
  fingerprints. Both captures have SHA-256
  `afc8da6768cc150553ca2da075807d9b981c37cb6683b6a3416cba905830a69d`.
  The installed parent native binary was also matched to the verified #181
  wheel, SHA-256
  `3ba7426072d1f1bfd26c5848be8a4628c4093babf91009616a93913fac91bd3a`.
- Installed-wheel execution: **220 tests across 34 modules** and all **42
  original examples** pass. Every module/example process record confirms
  completion and exit zero. Coverage includes source/Task/Graph/report/error
  contracts, HW/LSV/rough/basket/local-correlation models, dividend risk and
  calibration.
- Full import-only discovery confirms **110 modules / 610 test methods** and
  the exact provider partition of **20 shards / 179 cases**. This is inventory
  evidence, not execution of the additional heavy numerical tests.

The managed host used a symlink-backed venv interpreter, as in the
[time-grid review](time-grid-resource-review-2026-10-25.md). The native binding
linker also needed a temporary link to the runtime's existing `libpython3.12`
shared library. These environment-only adjustments did not change repository
build configuration, package gates, or test bodies. Full Python numerical
execution, default/minimal-feature suite runs and release-only extended
model/risk panels remain hosted CI gates; the ordinary Rust command does not
execute the ignored panels beyond the five acceptance tests listed above.

## CI follow-up

Before publication, #182 has 22 successful checks, 2 in progress, 63 queued
and 4 skipped, with no reported failure. Its macOS wheel build passes, but
the mandatory real LLDB integration methods and full macOS numerical wheel
validation are not yet confirmed. #180 has 109 successful checks, 13 in
progress, 24 queued and 4 skipped; its macOS Rust job now passes, while the
previously crashing numerical policies are still in progress or queued.

These snapshots do not resolve #179's native macOS SIGABRT/SIGTRAP. Passing
Linux regression and byte-exact replay evidence for this coefficient guard
does not establish a crash diagnosis or authorize a merge.
