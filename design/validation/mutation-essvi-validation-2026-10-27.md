# eSSVI slice validation and accessor mutation coverage — 2026-10-27

Base: `90f0d5aa8e755ec079dadad64871571c7946746e`, following the
[eSSVI variance tests](mutation-essvi-variance-2026-10-27.md) in draft PR #172.
Twelve tests strengthen the input, slice-admissibility and consistency contracts
in [calculation specification sections 3.1 and 3.4](../../docs/models/local-vol-vegakt-calculation-specifications.md).

Production Rust is byte-for-byte unchanged. Public signatures, error variants,
serialized data, dependencies, arithmetic order, RNG settings and existing
numerical/acceptance tolerances are unchanged.

## Exact boundary witnesses

The [reference generator](../../scripts/generate_essvi_validation_references.py)
uses Python's standard-library Fraction arithmetic. With `chi = rho_psi`, it
computes the independent residual expressions
`W = psi + abs(chi)`, `C = psi * (psi + abs(chi))`, and `L = 4*theta`,
without the production `rho = chi/psi` division. The effective tolerance is
`max(absolute, relative * max(1, abs(lhs), abs(rhs)))`.

Each row below tests the exact binary64 predecessor, boundary and successor.
All listed values and expected residuals are exactly representable. The generator
retains the corresponding rational residuals and effective tolerances. Its
finite witnesses do not model every intermediate floating-point operation over
the whole domain.

| Condition and varied input | Zero tolerance | Absolute tolerance | Relative tolerance | Accepted neighbors |
| --- | --- | --- | --- | --- |
| Strict wing: `W < 4`, vary `psi`, `chi=0` | `psi=4` | `a=1/8`, `psi=31/8` | `r=1/32`, `psi=31/8` | predecessor only |
| Curvature: `C <= L`, vary `theta`, `psi=1`, `abs(chi)=1/2` | `theta=3/8` | `a=1/8`, `theta=11/32` | `r=1/16`, `theta=45/128` | boundary and successor |
| Pair: `abs(delta chi) <= delta psi`, vary right `abs(chi)`, left `abs(chi)=1/4`, `delta psi=1/4` | right `abs(chi)=1/2` | `a=1/16`, right `abs(chi)=9/16` | `r=1/16`, right `abs(chi)=9/16` | predecessor and boundary |

The curvature relative tolerance scales by `3/2`, giving `3/32` of slack.
The pair relative tolerance uses the scale floor of one. Curvature and pair
cases run for both signs of correlation. Each rejected case checks the exact
condition, slice/pair index and diagnostic residual bits through the public
surface constructor. The strict-wing panel uses zero correlation; separate
nonzero-correlation cases freeze exact wing/curvature diagnostics for both signs.
When both bounds fail, the wing diagnostic takes precedence.

The literal audit compares all 27 tolerance/boundary inputs, three rejected pair
delta values, 20 diagnostic numerical literals and four condition labels with
the generator. Neighbor acceptance patterns are also checked. Rust tests use
frozen expectations and do not call the generator.

## Other public contracts

- Slice accessors preserve all four input values and the derived correlation,
  including signed zero and syntactically valid subnormal/large finite values.
  Syntactic slice validity does not imply admissibility in a surface.
- Nonfinite slice inputs preserve the field and original input bits, including a
  payload NaN. All finite-input checks precede positivity checks. Positive
  quantities reject either signed zero and negative values. Correlation is open
  at both `-1` and `1`, including adjacent inputs and quotient overflow.
- Surface accessors preserve slice order/data, custom tolerance and terminal
  slope bits. Accepting a large finite terminal slope at construction does not
  guarantee that every later extrapolation query is finite.
- Empty/one-slice inputs report their counts before checking terminal slope.
  Invalid slopes preserve their bits. Unsorted times identify the second pair
  correctly, including equal-time rejection.
- Theta and psi monotonicity remain exact despite a positive tolerance: even a
  one-ULP decrease is rejected, with theta diagnosed first if both decrease.
- All slice admissibility checks precede pair checks. A third invalid slice is
  diagnosed before an earlier unsorted pair, preserving its index.
- Finite inputs that produce an infinite wing or infinite `4*theta` limit are
  rejected, with zero and V1 tolerances. This checks fail-closed diagnostics,
  without claiming whole-domain accuracy or a separate reachable curvature
  overflow after the wing check has passed. No computed NaN payload is frozen.

## Matched mutation comparison

Both runs contain the same **44 identities, metadata and complete patches**:
21 in `validate_slice`, 17 in the five `EssviSlice` accessors, and six in the
three selected `EssviSurface` accessors.

| Focused eSSVI suite | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 16 tests | 44 | 24 | 18 | 0 | 2 | 2 |
| Expanded 28 tests | 44 | 42 | 0 | 0 | 2 | 0 |

**All 18 previously missed mutations become caught**, retaining all 24 existing
detections. Five affect the derived-correlation accessor; thirteen affect
admissibility arithmetic, logical guards or removal of the check. Every new
detection fails at least one added test. Seventeen fail the exact residual test;
the remaining logical guard mutation fails the nonfinite-limit test.

For example, the inner `|| -> &&` mutation on the curvature/limit finiteness
guard accepts a slice with `theta=f64::MAX`, `psi=1/2`, `chi=1/4` even though
`4*theta` overflows. The original sixteen tests pass. The new test requires the
existing curvature error with finite left residual `3/8`, infinite right limit
and slice index one. The production code already rejects it; only the tests change.

Two unviable replacements require `EssviSlice: Default` or
`SurfaceValidationTolerance: Default`, respectively. Both fail with E0277 and
retain their original classification. They are not detections, and no mutation
exclusion or waiver is added.

Both baselines pass. The normal runner's outcome validator reconciles selected
identities, completion, counts, packages, phases and statuses. Each survey has
43 complete test logs (baseline plus 42 viable operations) and two build-only
logs. Every viable log contains all sixteen or twenty-eight selected tests,
zero ignored tests and 642 filtered tests. Every caught operation has actual
test failures. The [JSON record](mutation-essvi-validation-2026-10-27.json)
retains the common full inventory/patches, log and patch hashes, failed-test
names, execution phases, references and the transition audit. Each actual patch
in both runs was compared byte-for-byte with the retained common patch.

Reproduce with Rust 1.98.1 and cargo-mutants 27.1.0:

```bash
CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/essvi.rs \
  --re='validate_slice|EssviSlice::|EssviSurface::(slices|terminal_theta_slope|tolerance)' \
  --cargo-arg=--lib --cargo-test-arg=market::essvi::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/essvi-validation
```

Repository configuration adds `--locked` and `--all-features`. The original
survey uses a detached checkout of the base above and identical options.
The pinned tool omits functions named `new`; a whole-file list contains 238
operations and no constructors. Thus constructor bodies, pair consistency and
validation order receive ordinary contract tests here, not generated mutation
coverage. Parameter interpolation and variance evaluation are separate historical
surveys. Constants, argument substitutions and metadata assignments absent from
the generated inventory are outside this denominator. A focused unit result
does not imply that the full baseline integration suite missed the same mutations.

## Other validation and remote evidence

- **720 unit tests pass**: 665 pricing and 55 numerical, five existing ignored,
  zero failed/filtered. All 28 focused eSSVI tests pass.
- Existing Standard SSVI and eSSVI integration reference targets both pass.
- Workspace Clippy across all targets/features with warnings denied, formatting,
  dependency direction and all 22 static/script checks pass, including 24
  mutation-runner tests. Markdown links, staged source archive and diff checks pass.
- The stacked changed-line inventory against
  `376c92624f88083b46d86e1087d2b0f03e202b91` remains 41 operations.
  This is list-only locally, not another changed-line mutation run.

At observation, the immediate parent's
[mutation workflow](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40529822987)
remained in progress. Smoke passed 24 runner tests, the installed-tool fixture
scenarios and 15/15 mutations. Changed-line shard 1 passed with 20 selected,
19 caught, one unviable and zero misses/timeouts. Exact head/base and completed
logs were checked. Shard 0 was still running; no aggregate or overall workflow
success is claimed. Artifact ZIP contents were not independently audited.

The parent's [ordinary CI](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40529822898)
was incomplete, with Rust jobs queued, wheel-validation groups queued and the
Windows stochastic-dividend risk/refinement job cancelled. Overall ordinary CI
success and the cause of that cancellation are not established. Job/run retrieval
covers the API's first page only; the JSON retains this observation separately
from completed local validation.

No new full integration suite, ignored heavy acceptance panel or installed-wheel
run is claimed. Historical failures/survivors, including the unresolved
fractional-OU guard, remain recorded. New-head CI is separate evidence.
