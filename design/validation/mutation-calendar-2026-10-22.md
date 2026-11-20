# Date and calendar mutation coverage — 2026-10-22

Base: `641db5e4e0cd274f3faa6e8054222138594a4018`, following the
[evidence-integrity work](mutation-evidence-integrity-2026-10-22.md) in draft
PR #172. This follow-up adds 13 unit tests to `core::date` and `core::calendar`
and fixes a calendar behavior exposed by a new regression test.

## Calendar correction

`ModifiedFollowing` compared only the month number of the following business
day with the original date. An explicit holiday calendar can close all 365 days
of 2026. Starting at 2026-01-01, `Following` correctly returns 2027-01-01, but
`ModifiedFollowing` must return the preceding business day, 2025-12-31. The old
implementation incorrectly accepted 2027-01-01 because both dates are in January.

The regression test failed against the original production code (exit 101,
actual `Ok(2027-01-01)` versus expected `Ok(2025-12-31)`). The comparison now uses
the **year and month**. Tests retain the ordinary weekend, holiday, month-end,
year-end, unadjusted and date-range-error cases. The long-closure fixture uses an
explicit list of month lengths, not the date-step routine under test.

This is a production behavior correction for long closures. It changes no
public signature, serialized schema, dependency, RNG setting or precision budget.
Normal business-day schedule behavior is covered by the full pricing unit suite
and the selected integration tests below. No normative convention is changed.

## Added contracts and tool limits

The eight date tests cover component access, strict ISO text (including
noncanonical ASCII and multibyte UTF-8), invalid components, Gregorian century exceptions, all
month lengths, forward/backward steps, date limits, every weekday, signed day
counts and ACT/360 and ACT/365F year fractions. Fifteen ordinal/weekday anchors
were independently checked with Python's proleptic Gregorian `datetime.date`.
The [JSON record](mutation-calendar-2026-10-22.json) retains those values.

Five calendar tests cover empty/default calendars, explicit holiday membership
versus weekends, holiday runs, adjustment idempotence, date-range errors and the
cross-year regression. These are assertions about observable dates and errors.

The pinned cargo-mutants 27.1.0 source visitor skips methods named `new` inside
both impls and traits. The generated inventory therefore omits `Date::new` and
`Calendar::new`, although their files are selected. Constructor/domain tests are
included explicitly; a complete generated inventory is not coverage of every
expression or public API. No new exclusion was added.

## Measured mutation results

Both diagnostic surveys select the two production source files and run only
the pricing unit tests matching `core::`: **18 tests before, 31 after**. They do
not run pricing integration tests or numerical-crate tests for each mutation.
Both baselines pass, inventories and outcomes are complete, and the strengthened
runner checker reconciles identities, counters, phases, packages and exits.

| State | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 135 | 109 | 16 | 2 | 8 | 3 |
| After | 135 | 122 | 0 | 5 | 8 | 3 |

The inventory contains 119 date and 16 calendar mutations. There are **134 exact
identity matches** between the two runs. The other mutation negates the changed
calendar comparison: its location moves from column 37 to 56, and its operand
changes from the month to the `(year, month)` pair. Both versions are caught;
they are recorded separately rather than treated as the same mutation.

For the 134 common identities, 15 previously missed mutations become caught and
one becomes a timeout. Two previously caught mutations also become timeouts;
106 remain caught, two remain timed out and eight remain unviable. Thus zero
missed mutations does **not** make this survey pass.

The five timeouts alter day/month/year increments or decrements into operations
that do not make forward or backward progress. Calendar searches can then loop
indefinitely, including inside the new long-closure test. Their logs already
show failed direct date assertions, but the process cannot finish while another
test loops. The aggregate result remains `Timeout`, never `CaughtMutant`.

A separate diagnostic selects exactly these five identities and runs only
`core::date::tests::`, with the same source and 20-second limit. All **five are
caught**, each by a direct date assertion; its baseline passes and it exits zero.
This confirms that the assertions detect the broken steps when calendar loops
are absent. It does not overwrite the five timeouts in the complete survey.

All eight unviable mutations require a nonexistent `Default` implementation:
seven for `Date`, one for `Weekday`. Their build failures remain separate from
test detection. No timeout increase, exclusion or assertion weakening was used
to obtain the reported results. The JSON records every identity transition,
failure evidence and the separate diagnostic for the five timed-out mutations.

Reproduce the complete diagnostic on each source revision with a fresh output
directory and cargo-mutants 27.1.0 installed:

```bash
CARGO_PROFILE_TEST_DEBUG=0 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/core/date.rs \
  --file=crates/pricing/src/core/calendar.rs \
  --cargo-arg=--lib --cargo-test-arg=core:: \
  --baseline=run --no-shuffle --jobs=4 \
  --timeout=20 --build-timeout=600 --output=target/mutation/calendar-survey
```

Repository configuration adds `--all-features` and `--locked`. Debug information
is disabled to reduce build artifacts; optimization, tolerances and RNG behavior
are unchanged. The JSON retains source hashes, actual commands, timestamps and
hashes of the complete raw inventories and outcome reports.

## Validation and limits

- `cargo test --locked -p pricing --lib --all-features`: **556 passed**, five
  existing tests ignored, none failed.
- Pricing `facade_compatibility`, `pricing_task` and `product_graph` integration
  tests: **19 passed**, none failed.
- Pricing all-target/all-feature Clippy with warnings denied and workspace
  formatting checks pass.
- The changed-line inventory against `origin/fix/product-graph-task-review`
  selects two mutations: the changed comparison and its containing function's
  return replacement. This was **list-only**, not a passing changed-line run.
- Documentation links, staged source archive and diff checks are verified for
  this change; historical records remain intact.

The normal `full`/`changed` runner still tests both Rust crates and their
integration/doc tests. This diagnostic does not replace that broader scope.
Numerical production/test files are unchanged; the previous numerical survey
remains **931 caught, 11 missed, three timeouts and seven unviable out of 952**.
Whole-library mutation coverage and installed Python-wheel acceptance are not
claimed. Remote CI is reported separately in the PR; no merge is authorized.
