# Discount-curve mutation coverage — 2026-10-23

Base: `b4915f30908ad571699c3c21db6c11ffdf37f1b9`, following the
[execution-evidence work](mutation-reporting-2026-10-22.md) in draft PR #172.
This follow-up adds 11 unit tests to `market::curve`, taking its suite from five
to 16 tests. The production portion of the source file is byte-for-byte
unchanged. No API, serialized schema, dependency, RNG setting or existing
precision assertion changes.

## Contracts exercised

The tests strengthen the discount-curve behavior specified in
[requirements section 3.2](../requirements-v1.0.md) and
[architecture section 6.1](../architecture-v0.1.md):

- Preserve the supplied curve identity, pillar times and exact discount factors;
  normalize a negative-zero valuation anchor and expose the expected logarithms.
- Interpolate on nonuniform pillar intervals and extrapolate the final local
  log-discount slope. The analytical fixture uses times `0, 2, 5` and discounts
  `1, 1/4, 1/64`, so the two slopes differ. Eight pillar, interpolation and
  extrapolation anchors have exact rational discounts that are powers of two.
  The JSON retains independent Python `Fraction` checks and 80-digit `Decimal`
  logarithms; neither Python nor a numerical dependency is needed by Rust tests.
- Accept negative forward rates and a flat terminal segment. Test both default
  trait accessors, including propagation of evaluation errors.
- Reject invalid counts, anchors, ordering, nonfinite/negative times and
  nonpositive/nonfinite discount factors with the correct error variant, curve,
  index and original input bits. Query tests include negative subnormal times,
  both signed zeros, infinities and a NaN with a nondefault payload.
- Reject finite log discounts whose exponent overflows or underflows to zero,
  while accepting a positive subnormal discount. Separately reject infinite or
  NaN log discounts created by finite but extreme curve/query inputs, retaining
  the query context in the error.
- Keep extrapolation count and extrema independent of pillar/interpolation
  queries, including repeated and unordered extrapolation times. An overflow
  test seeds the reachable `u64::MAX` counter boundary directly and verifies that
  a rejected record leaves the previous diagnostics intact.

The existing equal-spacing fixture could not distinguish division by the
segment width from multiplication when that width was one. The nonuniform
fixture exposes this error without tightening tolerances. The extreme-value
tests distinguish each disjunction in the output-validation guard: an invalid
discount can coexist with a finite log discount.

The pinned cargo-mutants 27.1.0 visitor skips methods named `new` inside impls
and traits. `LogLinearDiscountCurve::new` and `CurveExtrapolationStats::new` are
therefore absent from this generated inventory. Constructor/domain assertions
are included explicitly; generated mutation coverage of these methods is not
claimed. No exclusion or tool configuration was added.

## Measured mutation results

Both diagnostic surveys select only `crates/pricing/src/market/curve.rs` and run
only the pricing unit tests matching `market::curve::tests::`. They do not run
the integration tests or numerical-crate tests for each mutation. The baseline
passes with five tests before and 16 after. The production code and all 71
mutation identities are unchanged.

| State | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 71 | 47 | 21 | 0 | 3 | 2 |
| After | 71 | 68 | 0 | 0 | 3 | 0 |

All 21 previously missed mutations become caught: six trait accessor returns,
12 pillar-array accessor returns, one division-to-multiplication change in the
segment weight, and two changes to output-validation disjunctions. All 47
previously caught mutations remain caught. The scoped diagnostic passes.

All three unviable mutations insert `Default::default()` for a type without a
`Default` implementation: two return `CurveId`, one returns `CurveEvaluation`.
Their compilation failures are reported separately and never counted as test
detections. They are not excluded.

The [JSON record](mutation-curve-2026-10-23.json) contains source hashes, actual
commands, timestamps, baseline phases, hashes of the complete raw inventories
and reports, and all 71 identity transitions. Each transition retains log/patch
paths and hashes; the after-state includes failed test names or compiler errors.
The runner's evidence checker reconciles identities, counters, execution phases,
package selection and exit statuses for both complete reports.

Reproduce on each corresponding source revision with a fresh output directory:

```bash
CARGO_PROFILE_TEST_DEBUG=0 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/curve.rs \
  --cargo-arg=--lib --cargo-test-arg=market::curve::tests:: \
  --baseline=run --no-shuffle --jobs=4 \
  --timeout=20 --build-timeout=600 --output=target/mutation/curve-survey
```

Repository configuration adds `--all-features` and `--locked`. Disabling debug
information reduces artifacts without changing optimization or precision policy.
The before run uses the clean base commit. The after run uses that same HEAD
with the test additions; the JSON's file hash pins the tested contents.

## Validation and remaining scope

- `cargo test --locked -p pricing -p pricing-numerics --all-features --lib`:
  **567 pricing and 55 numerical tests pass**, five existing pricing tests stay
  ignored, none fail.
- Workspace all-target/all-feature Clippy with warnings denied and formatting
  checks pass. Markdown links, the staged source archive and diff checks are
  verified for this change. Historical validation records are preserved.

This unit-only diagnostic is narrower than the normal `full`/`changed` runner,
which includes both Rust crates and their integration/doc tests. The previous
[normal changed-line gate](mutation-reporting-2026-10-22.md) remains a historical
result; it is not presented as a new run after these test additions.

The numerical survey still has **11 missed mutations, three timeouts and seven
unviable builds out of 952**. The date/calendar survey still has five timeouts
and eight unviable builds out of 135. This follow-up does not reclassify those
results or claim whole-library, ignored-acceptance or Python-wheel mutation
coverage. Remote CI is reported separately in the PR; no merge is authorized.
