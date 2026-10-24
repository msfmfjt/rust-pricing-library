# PCHIP mutation coverage — 2026-10-25

Base: `ccde971336ff46a31eb3d34d27d76d73d7695cb2`, following the
[CI encoding fix](mutation-ci-encoding-2026-10-24.md) in draft PR #172.
Nine new unit tests exercise `market::pchip`; the original five tests remain.
The production prefix of `pchip.rs` is byte-for-byte unchanged. Public APIs,
wire formats, dependencies, RNG settings and existing precision budgets are
unchanged.

## Contracts and independent expectations

The tests cover the non-decreasing C1 ATM total-variance interpolation and
explicit short/long extrapolation rules required by
[requirements section 3.1](../requirements-v1.0.md) and
[architecture section 6.2](../architecture-v0.1.md). Standard SSVI uses this
interpolator; eSSVI has a separate interpolation rule.

The independent reference has knots `(1,2)`, `(3,5)`, `(6,6)` and `(10,14)`.
Its interval widths are 2, 3 and 4; secants are `3/2`, `1/3` and 2. The endpoint
estimates and interior weighted harmonic means reduce to:

| Time | Total variance | PCHIP derivative |
| --- | ---: | ---: |
| 1 | 2 | `59/30` |
| 3 | 5 | `45/79` |
| 6 | 6 | `21/38` |
| 10 | 14 | `62/21` |

For each interval, exact rational Gaussian elimination solves the four endpoint
value/slope constraints for a cubic in local power basis. Horner evaluation of
those polynomials gives nine interior values and analytic derivatives. This
avoids copying the production Hermite-basis evaluation into the test oracle.
The companion JSON retains every rational coefficient and query result. The
new comparisons use absolute tolerance `5e-13` on these small fixed references;
existing finite-difference and pricing tolerances are not changed.

The tests also cover:

- Getters against independent input arrays, lengths and slope expectations.
- Two-knot linear interpolation with distinct short-end and terminal slopes.
- Fully constant and partially flat curves, including adjacent zero derivatives.
- Finite, very large time coordinates whose harmonic-weight sum would overflow.
  A flat adjacent interval must retain its exact-zero derivative before that
  computation, for either side of the plateau.
- Endpoint slope clipping at both ends of valid non-decreasing data, with
  monotonicity and interval-range checks.
- Immediate floating-point neighbors of each knot, region tags and C1 behavior
  inside the knot range. Boundary extrapolation slopes remain independent;
  global C1 continuity across the extrapolation joins is not required.
- Invalid queries with exact error bits, signed-zero normalization and the
  smallest positive subnormal query time.
- Precise constructor errors for lengths, counts, knot ordering, time/value
  domains and terminal slopes, while accepting equal theta and signed-zero
  terminal slopes.

The pinned tool skips impl/trait methods named `new`. Constructor tests are
explicit, but no generated mutation coverage of that method is claimed.

## Why the previous assertions missed arithmetic changes

Returning empty `times()` or `values()` slices made a zipped knot-check loop
vacuously pass. Explicit comparison with the supplied inputs closes that gap.
The new tests also observe the terminal-slope accessor and first-knot region.

The original central-difference queries were at scaled coordinate `1/3` in
the first and last intervals, where `h10'(u)=3u^2-4u+1` is zero. The middle
interval had width one. Those cases could miss replacing multiplication by
the width in the `h10` value term with division. Comparing independent values
at other coordinates and non-unit widths detects that error. A width computed
as the sum of endpoints can also retain local value/derivative consistency,
so finite-difference agreement alone is insufficient.

Two zero-secant guard mutations might appear redundant on ordinary data because
a harmonic mean with one zero input can still evaluate to zero. With finite
times `[1, 4e307, 8e307]`, however, its weight sum would overflow. For theta
`[1,1,2]` or `[1,2,2]`, bypassing the zero branch instead produces an
indeterminate infinity/infinity. The added large-time test detects both guard
mutations through the public constructor and evaluator; its expected flat
slope is exactly zero. These are valid constructor inputs, not unchecked
private-state fixtures.

## Measured survey

Both runs select the same 215 generated identities in `market/pchip.rs` and
run only pricing unit tests matching `market::pchip::tests::`. The unmutated
baselines contain five tests before and 14 after. These are scoped diagnostics,
not the normal two-crate `full`/`changed` runner.

| State | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 215 | 159 | 55 | 0 | 1 | 2 |
| After | 215 | 205 | 9 | 0 | 1 | 2 |

All 215 identities match. **Forty-six previously missed mutations become
caught**, and all 159 previously caught mutations remain caught. One mutation
still fails compilation because `ThetaEvaluation` has no `Default`
implementation. It is not counted as a detection. Both complete surveys exit
2; the nine residuals are not waived or excluded.

All remaining survivors are in the opposing-secant-sign endpoint limiter:

| Mutations | Count | Assessment |
| --- | ---: | --- |
| Changes to the second comparison/threshold or assignment at lines 232 and 234 | 7 | Under the current checked constructor, secants are nonnegative. Positive zero, positive finite values and positive infinity have signum `+1`, so the unmodified opposing-sign condition is false. The changed expressions are short-circuited or unreachable. This argument concerns single mutations and current construction, not arbitrary private helper inputs or future unchecked construction. |
| `&&` to `||` at line 232; `!=` to `==` at line 231 | 2 | These changes alter whether the limiter can be reached. In real arithmetic the endpoint estimate with nonnegative secants is at most twice the adjacent secant, below the three-secant limit. The tested finite cases do not distinguish them. That argument does not establish equivalence over floating-point overflow or the entire constructor-admitted domain; these two remain unresolved outside the tested cases. |

The final run uses two workers; the before run used four. Test selection and
the 20-second test / 600-second build limits are otherwise identical. Timing
is retained as execution evidence, not a performance comparison.

Reproduce the final diagnostic on this source:

```bash
CARGO_PROFILE_TEST_DEBUG=0 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/pchip.rs \
  --cargo-arg=--lib --cargo-test-arg=market::pchip::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/pchip-survey
```

Repository configuration adds `--all-features` and `--locked`. Debug information
is disabled to reduce artifacts; optimization and precision policy are unchanged.

## Validation and limits

- `cargo test --locked -p pricing -p pricing-numerics --all-features --lib -- --test-threads=2`:
  **594 pricing and 55 numerical tests pass**, five existing pricing tests stay
  ignored, none fail. The focused PCHIP baseline passes all 14 tests.
- Workspace all-target/all-feature Clippy with warnings denied, formatting,
  dependency-direction and diff checks pass. Markdown links and the staged
  source archive are verified before publication.

The [JSON record](mutation-pchip-2026-10-25.json) retains exact source hashes,
commands, baseline phases, full identity transitions, inventory/report hashes,
independent rational references and failure evidence. Only complete surveys
contribute to the before/after comparison. An initial after-run and a broad
unit-test attempt were interrupted to reduce memory pressure; neither is
included in the completed results.

The earlier numerical, calendar and carry surveys remain historical records,
including their missed mutations, timeouts and unviable builds. No additional
mutation exclusion is introduced. No whole-library or installed-Python-wheel
mutation coverage is claimed. PR #172 remains draft and unmerged; remote CI is
reported separately.
