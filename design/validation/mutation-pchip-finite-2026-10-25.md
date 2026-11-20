# PCHIP nonfinite-result validation — 2026-10-25

Base: `43329c4dca2af15825490e83a006dfa2fe290b3c`, following the
[PCHIP reference and mutation survey](mutation-pchip-2026-10-25.md) in draft
PR #172. Investigating its two unresolved endpoint-guard survivors exposed
successful construction and evaluation with nonfinite computed values, despite
finite input coordinates and theta values.

## Correction and compatibility

`ThetaPchip::new` now checks each computed knot derivative before storing the
curve. `evaluate` checks theta and its derivative before returning success in
every region. A failure uses the existing `MarketError::NonFiniteSurfaceValue`
variant with field `theta` or `theta_derivative`, the knot/query time bits and
the nonfinite value bits. Log-moneyness bits are zero because theta has no
log-moneyness coordinate. Theta is checked before its derivative. Query
negative zero continues to normalize to positive zero.

The PCHIP slope formulas, Hermite expressions and their arithmetic ordering are
unchanged. This correction rejects a nonfinite computed result; it does not
introduce a numerically stable formulation for the entire finite-input domain.
An extreme case can still return an error when its exact mathematical result
is finite. Computed NaN sign and payload are not frozen across platforms; the
tests check that the returned bits represent a NaN.

The correction supports the Standard SSVI theta/derivative contracts in
[requirements section 3.1](../requirements-v1.0.md) and
[architecture section 6.2](../architecture-v0.1.md). Public signatures, error
variants, serialized schemas, dependencies, RNG settings and numerical
precision budgets are unchanged. Previously successful nonfinite internal
states or evaluations become typed errors. No pricing algorithm or tolerance
is changed to improve the mutation score, and no requirements baseline is
revised.

## Reproductions

Four new regression tests fail against the original production source and
pass after the correction. The same test source is used on both sides: before,
14 pass and four fail; after, all 18 pass. The companion JSON retains the source
hashes, failure output and log hashes.

| Path | Finite inputs | Previous result | Required result |
| --- | --- | --- | --- |
| Knot slope construction | Times `[1,101,201]`, theta `[1,1e308,1.1e308]`; mirrored large final interval; adjacent floating-point times; very large harmonic weights | Infinite endpoint/secant slopes or a NaN interior slope stored in a successful curve | `theta_derivative` error at the offending knot |
| Short extrapolation | Times `[1e-308,1]`, constant theta `1e308`, query `0`, `-0` or `1e-309` | NaN/Inf theta and Inf derivative | `theta` error with the normalized query time |
| Interior interpolation | Times `[1e-308,1.5e-308]`, constant theta 1, query `1.25e-308` | Finite theta 1 but NaN derivative | `theta_derivative` error |
| Long extrapolation | Times/values `[1,2]`, terminal slope `f64::MAX`, query 4 | Infinite theta | `theta` error |

Positive controls retain successful finite knot results and a long-end query
immediately above the last knot with terminal slope `f64::MAX`. The existing
independent cubic references, plateau tests and knot-neighbor checks remain.

## Mutation evidence

The before state reuses the complete 14-test, 215-mutation survey retained in
the previous record. Its source hash matches this change's base commit.
The after diagnostic selects every generated operation in `market/pchip.rs`
and runs the 18 focused PCHIP unit tests. The new shared finite-value helper
adds two operations: returning `Ok(())` unconditionally and deleting the
negation of `is_finite()`.

Source positions move when evaluation is refactored to validate all branches.
The comparison therefore maps the 215 retained operations by function and
ordered generated description, verifying genre, replacement and original
source token for non-function-replacement mutations. Function replacements
retain their function/replacement identity, not an assertion that a changed
body is identical. Raw names on both sides are retained. Literal source-location
identities are not claimed to match.

| State | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before, retained complete 14-test survey | 215 | 205 | 9 | 0 | 1 | 2 |
| After, complete 18-test survey | 217 | 209 | 7 | 0 | 1 | 2 |

Both previously unresolved endpoint-guard mutations become caught. All 205
previously caught operations remain caught, and both new helper mutations are
caught. The one unviable replacement still fails compilation because
`ThetaEvaluation` does not implement `Default`; it is not a detection.
All 217 selected outcomes are reconciled, and all 217 completed test logs
(216 viable mutations plus the baseline) confirm the 18-test scope. The before
survey has 215 completed test logs with 14 tests each. Seven survivors remain,
so the final diagnostic still exits 2 rather than passing.

The seven endpoint-limiter operations previously classified as unreachable
remain in scope. Checked positive increasing times and non-decreasing positive
theta give nonnegative secants. Positive zero, positive finite values and
positive infinity have signum `+1`; the unchanged opposing-sign predicate is
false. These mutations affect its short-circuited second predicate or body.
This argument is limited to single mutations and the current checked
constructor, not arbitrary private helper inputs or future unchecked
construction. No exclusion or passing waiver is added.

The previous two unresolved operations change `&&` to `||` or the opposing-sign
comparison from `!=` to `==`. With the large endpoint fixture, the original
slope arithmetic produces infinity and must return a construction error.
Those mutations instead cap it to a finite, incorrect three-secant slope and
make construction succeed. The new constructor regression detects that
observable change; it does not assert equivalence over floating-point overflow.

Reproduce the diagnostic:

```bash
CARGO_PROFILE_TEST_DEBUG=0 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/pchip.rs \
  --cargo-arg=--lib --cargo-test-arg=market::pchip::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/pchip-finite-survey
```

Repository configuration supplies `--locked` and `--all-features`. Both
surveys use two workers and identical time limits. Debug information is
disabled to reduce artifacts, without changing optimization or precision.
The pinned cargo-mutants 27.1.0 omits methods named `new`; the complete generated
inventory is not coverage of every constructor expression.

Four additional manual removal checks use a separate checkout and the same
18-test suite. Removing the constructor validation loop, the short-extrapolation
validation block, the remaining theta check or the remaining derivative check
each makes its corresponding regression fail. The unmodified control passes
all 18 tests. These explicit block removals supplement the generated survey;
they are not added to its 217-operation denominator.

The original arithmetic layout is retained to limit incidental changed-line
selection. An initial version reindented the entire evaluation match, selecting
122 operations across the stacked PR; the final diff selects 13. Both versions
have 217 generated PCHIP operations. This changes source layout, not the
runner's filters or exclusions. The current 13-operation selection is list-only;
the normal changed-line gate on this final revision is left to CI. An incomplete
diagnostic and broad test attempt on the earlier layout were interrupted and
are excluded from the completed evidence below.

## Validation and limits

- `cargo test --locked -p pricing -p pricing-numerics --all-features -- --test-threads=2`:
  **1,158 pass, zero fail, 91 existing ignored, zero filtered** across two unit
  and 97 integration targets. Both doc-test targets succeed with zero cases.
  Unit totals are **598 pricing and 55 numerical tests**. Test debug information
  is disabled and Cargo build concurrency is two.
- Workspace all-target/all-feature Clippy with warnings denied, formatting,
  dependency-direction and diff checks pass. Markdown links and the staged
  source archive are verified before publication.
- All 22 static/script check commands pass: numerical-reference/recovery
  checks, schema checks, eight replay-audit tests, four encoding regressions,
  24 mutation-runner tests and 26 wheel-validation helper tests. The required
  `jsonschema` tooling dependency was installed in the test environment before
  running the three JSON Schema checks; project dependencies are unchanged.
- All 18 focused PCHIP tests and the four manual validation-removal checks
  pass their expected outcomes. Each removal builds successfully and fails
  its corresponding regression test with exit 101; the control exits zero.

The [JSON record](mutation-pchip-finite-2026-10-25.json) retains baseline phases,
all retained-operation transitions, new-operation outcomes, residual patches,
test failure evidence and source/inventory/report hashes. Historical records
remain unchanged. These are PCHIP unit diagnostics, separate from the normal
two-crate full/changed runner. Whole-library mutation coverage, ignored heavy
acceptance gates and installed-Python-wheel validation are not claimed.
PR #172 remains draft and unmerged; current-head CI is reported separately.
