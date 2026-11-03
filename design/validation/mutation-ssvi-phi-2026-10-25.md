# SSVI phi stability and mutation coverage — 2026-10-25

Base: `c6cdfe261583108401ce5818780183c7fac726a1`, following the
[PCHIP finite-result correction](mutation-pchip-finite-2026-10-25.md) in draft
PR #172. Ten added tests strengthen the SSVI shape-function contracts and expose
small-argument cancellation in the Heston-like derivative.

## Numerical correction

With `z = lambda * theta`, the Heston-like shape is

\[
\phi(z)=\int_0^1(1-u)e^{-zu}\,du
       =\frac{z-1+e^{-z}}{z^2},\qquad
\phi_z(z)=-\int_0^1u(1-u)e^{-zu}\,du.
\]

The old implementation switched from a sixth-degree series to the exponential
closed form above `z=1e-4`. The derivative numerator subtracts nearly equal
numbers and is of order `z^3`, causing cancellation near that boundary. At the
first binary64 value above `1e-4`, with `lambda=0.25`, the Rust regression gives
`-0.0416888745746746003` instead of the independent reference
`-0.0416645833958319417`: relative error **5.83e-4**, approximately **0.0583%**.
This is a shape-derivative error, not an option-price error in basis points.

For `0 <= z <= 0.5`, the correction evaluates the degree-14 polynomial
`sum(n=0..14) (-z)^n/(n+2)!` and its analytic derivative with differentiated
backward Horner evaluation. The two accumulators use the old value before
updating it. Alternating-series remainder bounds at `z=0.5` are below
**8.6e-20 for phi** and **2.6e-18 for phi_z**, before binary64 rounding. The
regular exponential branch above `0.5` is unchanged. Underflow of positive
`lambda*theta` to zero retains the limits `phi=1/2` and `phi_theta=-lambda/6`.

This implements the stable small-theta requirement in
[requirements section 3.1](../requirements-v1.0.md) and the shape/derivative
contract in [architecture section 6.2](../architecture-v0.1.md). It changes
floating-point arithmetic for `lambda*theta <= 0.5`; bitwise equality with the
old calculations is not claimed. Public signatures, serialized schemas,
dependencies, RNG settings and existing pricing/acceptance tolerances remain
unchanged. No requirements baseline is revised.

## Independent tests

The optional [generator](../../scripts/generate_ssvi_phi_references.py) uses
Python's standard-library `Decimal` at 100 digits and lifts binary64 inputs
exactly with `Decimal.from_float`. It integrates the exponential series term
by term for the two Heston-like integrals, independently of the production
polynomial and its evaluation order. A separate 100-digit exponential closed
form agrees within absolute `1e-68`. The companion JSON retains the full decimal
references and remainder bounds; Rust tests use rounded binary64 literals.

Heston-like tests cover 14 scaled arguments from `1e-8` through 16, including
both neighboring floats and the exact value at the old and new cutoffs. Three
power-of-two lambda values exercise derivative scaling without changing the
scaled arguments through rounding. Eight power-law references use logarithmic
evaluation and the independently simplified derivative
`-eta*(gamma+theta)/(theta^(gamma+1)*(1+theta)^(2-gamma))`. The new reference
comparisons use relative tolerance `5e-13`; existing assertions are unchanged.

Other tests cover:

- Exact error parameter names and bits for both families, theta queries,
  signed zero, NaNs, infinities and open gamma endpoints.
- Family syntax validation before query validation, and the distinction
  between standalone phi syntax and surface-level admissibility.
- Symmetric rho cases, inclusive gamma/wing/lambda limits, absolute-tolerance
  boundaries and their first rejecting floating-point neighbors. The first
  successor of binary64 `4/3` still rounds its wing product to exactly 2;
  the test accepts it and rejects the following successor.
- Overflow, unavoidable positive-value underflow, nonfinite derivatives with
  finite phi, and underflowed scaled theta. Existing typed errors are retained.

Against the original production source the same 15-test suite has **14 passes
and one failure** at the independent Heston reference. After the correction
all **15 pass**. The original five tests alone passed before the change.

Separate-checkout controls restore the old `1e-4` cutoff or shorten the new
series to degree six. Each produces exactly one failure in the independent
Heston reference test, with 14 passes and exit 101; the final unmutated source
passes all 15 tests first. These controls cover numerical constants that the
pinned tool does not mutate. Their source diffs and test results are retained
separately and do not enter the generated-mutation denominator.

## Mutation scope and comparison

The diagnostic selects `PhiSpec::{validate_syntax,validate,evaluate}`, the
positive/open-unit parameter validators, and `heston_phi_series` when present.
It runs pricing unit tests matching `market::ssvi::tests::`, five before and
15 after. It does not mutate the rest of the SSVI surface, density or call-price
implementation, or replace the normal two-crate full/changed runner.

| Source | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, five tests | 155 | 76 | 78 | 0 | 1 | 2 |
| Corrected, 15 tests | 111 | 110 | 0 | 0 | 1 | 0 |

Both runs completed, including successful unmutated baselines. Inventory names,
outcome counts, package selections and build/test phases reconcile. Every viable
mutation log contains the expected complete five- or 15-test summary; caught
cases have actual failed assertions. Neither run has a timeout or the unrelated
empty-object build failure encountered by the ordinary test run below.

The before source has 155 generated operations; the corrected source has 111.
**85 operations are retained, 70 old polynomial operations are removed, and
26 operations are introduced by the differentiated recurrence.** Removed
survivors are not counted as newly caught. Comparison anchors retain function,
generated description, genre, source-line text and token offset for operators;
function-value replacements retain the function/replacement identity rather
than claiming the changed body is identical. Both raw names and every unmatched
operation are retained in the JSON record.

Of the 85 retained operations, **20 change from missed to caught**, 64 remain
caught and one remains unviable. None regresses from caught to missed. The 70
removed operations comprise 58 missed and 12 caught cases; all 26 new recurrence
operations are caught. The one remaining unviable mutation replaces the whole
evaluation with `Ok(Default::default())`, which fails with Rust error E0277
because `PhiEvaluation` does not implement `Default`. It is retained as unviable,
not counted as test detection. No missed mutation is waived or excluded.

Reproduce the final diagnostic:

```bash
CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/ssvi.rs \
  --re='PhiSpec::|require_(positive|open_unit)_parameter|heston_phi_series' \
  --cargo-arg=--lib --cargo-test-arg=market::ssvi::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/ssvi-phi-survey
```

The pinned tool is cargo-mutants 27.1.0; repository configuration adds
`--locked` and `--all-features`. Both surveys use the same selection pattern,
worker/build concurrency and time limits. Test debug information is disabled,
without changing optimization or numerical tolerances. Tool inventory limits,
including omission of methods named `new` and numeric-constant replacements,
remain explicit; there are no additional exclusions.

## Changed-line CI and prior CI confirmation

On the parent commit, the normal changed-line CI run tested 13 operations in
37 minutes: **11 caught, two unviable, zero missed/timeouts**. Its two-crate
baseline took approximately 485 seconds of testing. The
[Mutation testing run](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40146133970)
succeeded. All three Rust jobs, including the Windows encoding check, succeeded
in the [ordinary CI run](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40146133996).
The complete ordinary workflow was not yet successful at the observation time;
some wheel groups were still running, and one macOS barrier job was cancelled.
These observations concern the parent commit, not this new revision.

The current stacked diff selects **41 operations**. The changed-line workflow
now uses two round-robin shards, with independent baselines and evidence, to
accommodate the larger selection within the existing 90-minute job budget.
The existing `Mutation testing / changed Rust lines` check name remains as an
aggregate requiring both jobs to succeed. Failure, cancellation, skipped or
missing results cannot produce aggregate success. The jobs do not cancel each
other on failure; each uploads a separate artifact.

Installed-tool fixture tests verify that the changed shards partition the
selection exactly once, retain a downstream-only observer, and return exit 2
when its assertion is removed. This extends the original caught/missed/failed
baseline/empty-shard scenarios to seven scenarios. On the actual source, the
two selected inventories contain **21 and 20 operations**, with disjoint sets
and a complete union of all 41. This actual-source check is list-only; full
changed-line execution on this revision is left to CI. Local aggregate-command
checks pass only for `success`, and fail for `failure`, `cancelled`, `skipped`
and an empty result.

## Validation and limits

The two Rust crates' normal test suite passes **1,168 tests**, with **91 existing
ignored**, zero failures and zero filtered cases, across two unit targets and
97 integration targets. The pricing unit target contributes 608 passes and
the numerical crate 55; both doc-test targets complete with zero cases.

An initial ordinary-suite build failed before any tests because a cached object
file had zero length (`memory map must have a non-zero length`). That attempt is
excluded from successful counts. Only the affected generated artifacts were
removed, and the successful rerun used `CARGO_INCREMENTAL=0`, one Cargo build
job, two test threads and disabled test debug information. No OOM event was
observed; the record does not assert a cause for the empty object.

Workspace Clippy for all targets/features with warnings denied, formatting,
dependency direction and 22 static/script check commands pass. These include
the numerical reference and schema checks, replay audit, four encoding
regressions, 24 mutation-runner tests and wheel-validation helpers. The
installed-tool fixture separately passes its seven scenarios. Markdown links,
staged source-archive contents and the final diff are checked before publication.
All 42 Heston and 40 power-law input/reference literals in Rust match the
independent generator's binary64 values exactly.

The [JSON record](mutation-ssvi-phi-2026-10-25.json) retains source/inventory/report
hashes, baseline phases, all retained-operation transitions, removed/new
operations, outstanding patches, red/green evidence and independent references.
Historical records remain unchanged. No entire-finite-domain accuracy guarantee,
whole-library mutation coverage or new installed-wheel validation is claimed.
In particular, the existing very-large-argument exponential arithmetic is not
changed by the small-argument correction. PR #172 remains draft and unmerged;
current-head remote CI is reported separately.
