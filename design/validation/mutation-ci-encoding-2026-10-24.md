# Mutation CI source encoding — 2026-10-24

Base: `fde18e50e7ceedb7672d0d4973e7129ebb5a48af`, following the
[forward/dividend survey](mutation-carry-2026-10-23.md) in draft PR #172.
This change fixes a source-checking failure exposed by the earlier date mutation
tests. Rust source, mutation selection, numerical assertions and dependencies
remain unchanged.

## Observed Windows failure

The base revision's Mutation testing workflow succeeded. Its ordinary CI workflow
failed in [Rust / windows-2025, dependency direction](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39976239331/job/120492191245).
The Rust tests and Clippy steps in that job had already passed. Python 3.12.10
then decoded a Rust source file using the Windows `cp1252` default and raised
`UnicodeDecodeError` for byte `0x90` at position 8177.

The same byte and offset occur in `crates/pricing/src/core/date.rs`, in the
full-width invalid-date fixture `２０２４-02-29`. Rejecting non-ASCII date text
is intentional. The source-boundary checker must read that UTF-8 source correctly.

`check_dependency_direction.py` now explicitly decodes UTF-8 at all three source
read sites: the crate root, domain definitions and execution implementations.
Strict decoding is retained. Boundary predicates, comment handling and the
existing compatibility-re-export exception are unchanged.

## Regression evidence

Four new tests exercise these observable contracts:

- Valid UTF-8 Rust text is accepted with ASCII and Windows `cp1252` defaults,
  independently at each source read site.
- Unicode comments containing forbidden type names do not create false alarms.
- A public engine module, a domain importing execution state and an execution
  layer importing risk reporting each still produce their original error.
- Invalid UTF-8 is rejected at every read site, including under a `cp1252`
  default that would otherwise decode the invalid byte.

The tests simulate the default encoding only when text I/O has no explicit
encoding. They use real temporary source files and real decoders. Before the fix,
the suite fails with six decoding errors and three failed strict-decoding
assertions. After the fix all four tests pass. Removing each of the three UTF-8
arguments separately is also detected by the suite. These three targeted fault
injections are not a general Python mutation-coverage claim.

A separate real CLI check on the complete repository, with Python UTF-8 mode and
locale coercion disabled, fails before the fix and passes after it:

```bash
cargo metadata --locked --format-version 1 --no-deps > /tmp/pricing-metadata.json
LC_ALL=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 \
  python scripts/check_dependency_direction.py < /tmp/pricing-metadata.json
```

The regression suite also passes under those environment settings. This is a
Linux ASCII-locale reproduction and a simulated Windows default, not a claim
that a native Windows job has already rerun successfully.

CI runs the new suite on its existing three-platform Rust matrix, before Rust
builds. The contributor checklist and source-archive requirements retain both
the test file and its CI command.

## Current-source mutation gates

The normal changed-line runner uses the actual stacked parent branch
`origin/fix/product-graph-task-review`, whose merge base is
`376c92624f88083b46d86e1087d2b0f03e202b91`. Its baseline passes **1,145 tests**,
with 91 existing ignored, zero failed and zero filtered, across two unit and
97 integration targets. Both doc-test targets also pass with zero cases.
This includes the 29 curve/forward/dividend tests added since the earlier
[changed-line record](mutation-reporting-2026-10-22.md).

The smoke gate detects **all 15 selected mutations**, with zero missed,
timeouts or unviable builds and exit 0. It runs the five `core::number` unit
tests; it does not run all of the changed-line baseline tests for each mutation.

The changed-line gate finishes with **two selected, one caught, one unviable,
zero missed and zero timeouts**, exit 0. The year/month comparison mutation is
detected by three calendar tests. The other mutation requires a missing
`Date: Default` implementation and fails compilation; it is not counted as caught.

Reproduce the current-source gates after installing the pinned tool:

```bash
CARGO_PROFILE_TEST_DEBUG=0 python scripts/run_mutation_tests.py smoke --jobs 2
CARGO_PROFILE_TEST_DEBUG=0 python scripts/run_mutation_tests.py changed \
  --base origin/fix/product-graph-task-review --jobs 2
```

The existing mutation-runner suite passes all 24 tests. The installed-tool
fixture passes its four scenarios: downstream detection, deliberate survivors,
an intentionally failed baseline, and an empty shard. Formatting and Python
syntax checks also pass. Markdown links, the staged source archive and diff
checks are verified before publication.

The [JSON evidence](mutation-ci-encoding-2026-10-24.json) records the completed
gates, exact selection, baseline/test scopes, source and log hashes, the Windows
failure excerpt and the before/after encoding regressions.

## Limits

This fixes a validation-tool portability failure; no new Rust mutation scope is
claimed. The earlier numerical survey still has 11 missed mutations and three
timeouts; the calendar survey has five timeouts; the forward/dividend survey has
three missed mutations. Their separate unviable outcomes and historical records
are unchanged. Passing narrow CI gates does not make those surveys pass.
Whole-library mutation coverage and installed-wheel mutation coverage are not
claimed. PR #172 remains draft and unmerged; remote CI status is reported
separately.
