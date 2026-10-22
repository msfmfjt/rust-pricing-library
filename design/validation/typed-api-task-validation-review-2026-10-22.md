# Typed API and task validation review — 2026-10-22

## Scope

This continues the [Product Graph / Task review](product-graph-task-review-2026-10-22.md)
at #170 (`376c92624f88083b46d86e1087d2b0f03e202b91`). Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`; the existing PR chain is unchanged.
The detailed follow-up covers #161's typed risk coordinates and error adapter,
#162's typed Python source construction, and #164's shared compiled tasks.
It also inspects the report/calibration and backtest adapter boundaries introduced
by #165–#169. This is not certification of all numerical model combinations.

## P2: Task validation errors lose their source location

Reusable source readers reported domain-error pointers relative to their own
document even when parsing a nested task. The installed #170 wheel reproduced:

| Invalid input | Previous pointer | Correct task location |
| --- | --- | --- |
| Embedded execution policy | `/numerics` | `/source/specification/numerics` |
| Second basket Spot | `/market/spot` | `/source/markets/1/spot` |
| Second basket volatility | `/model/volatility` | `/source/models/1/volatility` |
| First marginal particle settings | `/model/equity/particles` | `/source/lsv/0/value/particles` |
| Dividend equity execution policy | `/numerics` | `/source/equity/numerics` |

The old pointers did not resolve in the submitted task. A client using structured
errors could not reliably identify the asset or calibration input to correct.
Three new Rust regression tests failed on the unmodified implementation.

The fix keeps path composition in the task wire adapter. Existing source readers
and domain constructors still perform validation. Nested specifications preserve
their source prefix; basket model/market/marginal/target readers retain indices;
correlation-entry dates retain their entry index. Basket composition failures
identify the actual flattened wire location, or `/source` when several model
inputs are involved. Task operation errors remain at `/operation`.

## Responsibility assessment

- Native risk producers supply coordinate kinds; the report adapter maps them
  to units without parsing display labels. Shape checks precede indexing.
- Frozen Python configuration objects own cloned native inputs. Complete source
  validation and compilation remain in Rust; constructors do not create a
  second financial validation implementation.
- Task operation selection shares an immutable source and realized backend.
  Source prerequisites and native preflight remain required; pricing workspaces
  are evaluation-local. Selection cannot manufacture a missing reverse trace.
- Common reports copy native estimates and diagnostics outside numerical loops.
  The backtest bridge checks currency, scalar shape, raw units and calibration
  convention, and retains pre-payment evidence with explicit payment scaling.
- Provider caches are bounded and snapshot-scoped; in-flight calls retain their
  original cache. Configuration is fixed for the provider lifetime and native
  API compatibility is checked before use.

No additional demonstrated numerical or ownership defect was found in these
reviewed paths. These observations do not extend the supported product/model
matrix or replace the independent numerical acceptance panels.

## Compatibility and validation

Only error locations change. Error variants, messages, exception classes, codes
and stages are preserved. Source/task/report schemas, serialization of accepted
inputs, fingerprints, prices, Greeks, AAD, calibration, seeds, reduction policy
and numerical tolerances are unchanged. Standalone source pointers retain their
existing scope. No numerical kernel is modified.

Before the fix, 24 installed-wheel typed-source, error, risk-metadata and task-reuse
tests passed; the new pointer regressions then exposed the missing coverage.

- The three new Rust tests pass after the fix and cover 24 rejected inputs.
  The Python regression failed on all six nested cases with the original wheel
  and passed with the newly built wheel. Standalone error locations are checked.
- Rust workspace/all-features: 1,067 passed, 0 failed, 91 ignored, excluding the
  Python binding crate. Its five native Rust tests also passed. Five ignored
  statistical/path-dependent/early-exercise acceptance tests passed separately.
- Installed-wheel Python: 65 passed across typed sources, structured errors,
  risk metadata, Graph/Task/specification/reuse, calibration reports, native
  provider compatibility and static/runtime wheel contracts.
- Six complete task reports from the original #170 wheel are exactly unchanged
  under the new wheel, including prices, Greeks, errors, calibration diagnostics
  and replay fingerprints. No numeric tolerance was used for this comparison.
- Companion backtesting: 330 passed. Wheel-checker suites: 26 passed.
- The fresh release wheel passed package/metadata/ABI/installation checks and
  runtime signature validation for 341 callables.
- Formatting, all-target check, all-target/all-feature Clippy with warnings
  denied, public Rust docs, dependency direction, schema checks and Markdown
  links passed. Full Python and heavy model panels remain hosted CI gates.

A cached object-file build failure was resolved by using a fresh
`CARGO_TARGET_DIR`; only the completed rebuild is counted above. Debug-symbol
settings were supplied through environment variables, without repository build
profile or numerical-policy changes.

## Outstanding CI

At the follow-up check, #170 had no failed checks; cross-platform jobs were still
queued/running. Windows extended-accuracy logs for #164/#165 were retried and
still returned `BlobNotFound`. Their numerical outcome remains unresolved.
The later #169 head (`aab90e5`) has a successful Windows extended-model
acceptance job (`119726340628`); this is evidence for that exact head, not a
substitute for #170 or this follow-up's checks. Earlier cancelled checks also
remain distinct from passes; all check-run pages were included in the review.
Current-head CI, including Windows Rust and independent model acceptance, must
complete before merge. No PR is merged without the user's explicit approval.
