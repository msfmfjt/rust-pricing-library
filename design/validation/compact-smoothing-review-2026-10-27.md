# Compact smoothing numerical review — 2026-10-27

## Reproduced defects

This continues the library review from PR #189, commit
`76375a4010bc31f6a50911cbb039c7fdca040627`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The review stack is unmerged;
the separate mutation-testing work in #172 is unchanged.

This restores the exact exterior and shared primal/reverse rules in
[requirements section 7.3](../requirements-v1.0.md#73-non-smooth-payoff-sensitivities)
and preserves the source/execution boundaries of
[ADR 0029](../adr/0029-product-graph-source.md). No normative policy changes.

P1: compact C2 min/max can lose the selected payoff outside their transition
band. At Spot 101 and half-width 2, the installed parent Python Task API prices
`smooth_min(1e16, Spot)` at **100**, while reversing the inputs returns **101**.
`smooth_max(Spot, -1e16)` likewise returns 100 instead of 101. AAD reports Delta 1
even though the forward value has been quantized by subtracting and re-adding
the large operand. Finite opposite-sign values can also overflow their
difference and incorrectly produce a nonfinite extremum.

The same primitive has three related range defects:

- A finite half-width whose double overflows makes the normalized transition
  coordinate zero or NaN; the indicator at zero returns 0 instead of 0.5.
- Squaring the width can overflow and erase representable subnormal curvature,
  or underflow and turn the exact zero midpoint curvature into NaN.
- Multiplying a subnormal width before the remaining polynomial terms can
  round twice; at width `4 * f64::from_bits(1)`, the positive part at zero
  incorrectly returns zero instead of one subnormal unit.

The seven initial Rust regressions compiled against the unchanged parent:
six failed and the ordinary-arithmetic control passed. All seven pass after
the correction. The three parent Python regressions also fail on the real
installed wheel, including future-path MC/RQMC values and basket AAD/Gamma.

## Implementation and responsibility boundaries

The shared `CompactC2Smoothing` primitive returns the selected original input
and its exact exterior derivatives at or outside the transition endpoints.
It no longer reconstructs an exterior value from the potentially inaccurate
operand difference. That difference may be infinite for finite inputs; its sign
still selects the correct finite exterior operand.

Inside the transition, the existing quintic polynomial and ordinary arithmetic
order remain. A bounded input/width ratio is used only when the doubled width
overflows. Curvature divides by the width twice only when the squared denominator
leaves the normal range. Positive-part scaling applies the width last only when
the earlier product would overflow or become subnormal/zero. A genuinely
nonrepresentable value or derivative is not clamped into the finite range.

Constant folding, interpreted graph evaluation, reverse AAD and native smoothed
payoff consumers already share this product primitive. The correction remains
there; no model-, task-, Python- or executor-specific workaround is introduced.
Public APIs, JSON schemas, fingerprint encodings, RNG streams, sample counts,
calibration algorithms and numerical tolerances are unchanged. The mathematical
C2 v1 policy is unchanged, so its policy version remains 1. Existing independent
limits on risk-request smoothing widths are retained.

Historical outputs affected by the arithmetic bugs necessarily change. Unaffected
interior points retain their original bits. The regression compares exact
exterior selection, exact dyadic polynomial values, and independent integer
ties-to-even rounding across 256 subnormal widths. Both terminal and pre-dividend
graph adjoints and constant folding are exercised. The added nonfinite-output
control retains genuine overflow behavior.

## Validation checkpoint

Completed on Linux x86-64 with the pinned Rust 1.98.1 and CPython 3.12:

- All-features workspace tests excluding the Python crate: 1,117 passed;
  91 intentionally ignored tests remain separate gates. Python crate tests:
  seven passed. Explicit statistical/path/early-exercise acceptance: five passed.
- Eight new Rust regressions passed, including under `--no-default-features`.
  Three new installed-wheel Python regressions and 48 related Python tests passed.
- Fresh release wheel archive/ABI/metadata/SBOM/installation checks passed,
  including all 341 installed callable signatures.
- Eighteen numerical-fixture/schema/link checkers and six checker test suites
  passed. The latter contain 62 passes and four physical-macOS skips on Linux;
  the skipped debugger probes remain mandatory on macOS.
- `cargo fmt`, workspace/all-target `cargo check`, all-feature/all-target Clippy
  with warnings denied, rustdoc, and dependency-direction checks passed.
- The freshly captured 70 complete Task reports are byte-identical to the
  unchanged parent, including price, Greeks, covariance, calibration diagnostics
  and fingerprints. SHA-256:
  `d1c94f2fe16e96f4a13c9385c20c492e701f4a2d47a5fda450dc665e9e8be816`.

- All four native replay generators were rebuilt from the candidate source.
  Their 16 cases (European, Local Vol, path dependence and early exercise) are
  byte-identical to freshly built parent captures, including smoothed digital
  and discrete/continuous barrier risks. Their four hashes are in the manifest.

The installed-wheel regressions cover both operand orders, MC and RQMC, one and
three workers, current/future observations, exact task JSON replay, basket AAD
and Gamma operation reselection, and finite results at very wide graph widths.
The ordinary-behavior comparison includes all 70 existing complete Task/risk/
covariance captures and all four existing native replay generators; these are
comparisons against the unchanged parent, not replacement frozen fixtures.
The [capture manifest](compact-smoothing-replay-2026-10-27.json) records source,
wheel, capture-script and result hashes. Linux same-host comparisons do not
replace the existing macOS/Windows frozen replay and benchmark gates; full
extended price/risk panels and every wheel numerical selection remain required
in candidate CI.

## CI and outstanding issues

At the 2026-10-27 23:40 UTC refresh, #189 had 85 successful, four running,
105 queued and four skipped checks. Its three Rust jobs, all wheel producers
and all extended-model jobs passed. Raw macOS and Windows wheel logs confirm all
four platform frozen replays and benchmark report gates. The generated 111
numerical selections remain required; queued or incomplete jobs are not passes.

The prior local Put fixed-lookback ledger has no completion record and no
remaining process. Its original log/running record is retained as incomplete
evidence. Both original macOS coupled-row-hats four-interval jobs on #187 passed
this time, while both Windows jobs were cancelled at the 330-minute limit. This
does not resolve the two earlier #186 macOS invalid frees. Their detection stacks
in libdispatch/pthread cleanup do not establish a corruption origin; this
smoothing fix does not claim to repair them. No merge is authorized.
