# Model, calibration and HW quote-risk boundary review — 2026-10-22

## Scope and responsibility boundaries

This continues the [typed API/task review](typed-api-task-validation-review-2026-10-22.md)
from #171 (`b84a76abcb53fc10fb0805cdfdaa14e982868fb3`). Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The existing PR dependency chain
is unchanged; this correction is stacked on #171.

The review follows model definitions into particle calibration, retained
calibration reverses, Hull–White composition and native risk preflight. It also
inspects the financial-independent correlation, reduction and bounded
least-squares code. The current separation remains appropriate: model/source
objects describe inputs, private execution modules realize paths/calibrations,
and the numerics crate owns algorithms without financial model dependencies.
Python and task/report adapters consume those native contracts.
The correction follows [ADR 0030's shared native preflight](../adr/0030-pricing-task-capability-report.md)
and the [HW market-IV transpose contract](../../docs/models/hull-white-vegakt.md).

The retained-trace and HW primal-fingerprint checks protect calibration VJPs;
driver/correlation compilation checks marginal compatibility, and fixed random
coordinates/reduction order remain native execution concerns. No additional
demonstrated defect was found in those inspected paths. This is a scoped review,
not certification of every model/product combination or numerical panel.

## P2: HW quote risk accepts an undefined floor/cap derivative

`HullWhiteLsvTarget::from_market_iv` rejects a variance that needs repair, but
accepts an unrepaired variance exactly equal to its floor or cap. This is valid
for pricing. It is not an interior point for differentiation through the quote
to Dupire-variance map. Its two one-sided derivatives differ; a conventional
two-sided market-IV sensitivity does not exist there.

The original #171 wheel nonetheless returned market-IV adjoints for both
`floor = 0.0625` and `cap = 0.0625` with a flat 25% IV surface. Compiled tasks
also advertised AAD as supported. The same gap affected HW basket marginal
targets and the Local Correlation basket target. Rough-family market-IV risk
already rejected this boundary via the strict source validator.

New tests use binary-exact volatility/variance to expose the primal kink without
an equality tolerance. A positive and negative IV bump give one-sided variance
derivatives differing by approximately 0.5 at the 25% quote. Three Rust regression
tests failed before the fix. Installed-wheel Python checks failed in all eight
invalid native/task scenarios, including both bounds and the basket target.

## Correction and compatibility

- Share the existing strict variance-bound guard in the native market-IV layer.
- Invoke it from the paired target's public quote transpose and native AAD
  preflight for single assets, basket marginals and Local Correlation targets.
- Reuse that preflight for compiled task capability discovery and operation
  selection. Source-only discovery remains conditional until native compilation.
- Preserve pricing and effective-node risk without quote provenance. A failed
  operation selection leaves the original compiled price task reusable.

The boundary is exact: adjacent representable floor/cap values strictly outside
the target variance still permit AAD. No epsilon band or tolerance change is
introduced. Existing error mapping carries `market_iv_local_variance_bound_kink`;
no Python-specific financial validation is added.

No public signature, schema, accepted-input serialization, fingerprint domain,
calibration algorithm, derivative formula, RNG stream, reduction order, kernel
or numerical tolerance changes. Only the previously unsupported derivative at
an exact active-set transition is rejected earlier and consistently. Price-only
source construction is unchanged.

## Validation

The four new Rust tests pass after the fix. They cover the paired quote
transpose, independent one-sided primal differences, legacy native AAD, task
compile/selection/capability, basket marginal and Local Correlation targets,
price-task reuse, adjacent interior floating-point bounds and unquoted node risk.

- Installed-wheel Python: 104 passed, including HW/1F/2F/rough compositions,
  basket LSV, Local Correlation, market-IV risk, typed API, task/graph/report
  adapters, structured errors and wheel contracts. The four new tests pass;
  eight invalid subcases failed with the original #171 wheel.
- Thirteen complete JSON reports match the original #171 wheel exactly: six
  existing task fixtures, six boundary price cases (single asset, basket
  marginal and Local Correlation at both bounds), and an adjacent-interior AAD
  case. Prices, risks, standard errors, calibration diagnostics and fingerprints
  are compared as whole serialized reports, without numerical tolerance.
- A fresh release wheel passed package, metadata, ABI, clean-install and runtime
  signature checks for 341 callables. The 26 wheel-checker tests also passed.
- Numerical reference/checker scripts, all four source/task/graph schemas,
  Markdown links and eight replay-audit tests passed.

- Rust workspace/all-features: 1,071 passed, zero failed, 91 ignored, excluding
  the Python binding crate. Its five native Rust tests passed separately.
  Five explicitly selected statistical/path-dependent/early-exercise acceptance
  tests also passed without changing their thresholds.
- Formatting, all-target workspace check, all-target/all-feature Clippy with
  warnings denied, Rust docs and dependency direction passed.

The full Python discovery suite and heavy release model panels remain hosted
CI gates. Local checks used environment-only debug-symbol settings and separate
build output directories; repository build profiles are unchanged.

## Outstanding CI

At the review check #170 had 55 successful, two running and 36 queued checks
(four skipped); #171 had one running and 92 queued (four skipped). No failure
was reported at either head, but neither had completed all checks. The earlier
#164/#165 Windows extended-model failures still have unavailable logs; their
cause remains unresolved. The later #169 Windows extended-model job succeeded
at that exact head, which does not replace current-head checks.

Current-head cross-platform/numerical CI remains a merge gate. No PR is merged
without the user's explicit approval.
