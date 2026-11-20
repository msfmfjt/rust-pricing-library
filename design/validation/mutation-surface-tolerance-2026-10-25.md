# Surface tolerance and butterfly-bound mutation coverage — 2026-10-25

Base: `65c7af6c14279ed32762747ae39fa33f23a61910`, following the
[forward-call tests](mutation-forward-call-2026-10-25.md) in draft PR #172.
Eleven tests strengthen the existing validation arithmetic specified in
[sections 3.1–3.2 of the local-vol calculation specification](../../docs/models/local-vol-vegakt-calculation-specifications.md).

Production Rust is byte-for-byte unchanged. Public signatures, error variants,
serialized data, dependencies, arithmetic order, RNG settings and existing
numerical/acceptance tolerances are unchanged.

## Validation contracts

The effective tolerance is

\[
\tau(l,r)=\max\{a,\;b\max(1,|l|,|r|)\}.
\]

The non-strict constraint accepts `l-r <= tau`; the strict constraint requires
`r-l > tau`. Tests cover zero tolerance, absolute tolerance and relative
tolerance, with boundary values and their immediately adjacent binary64
neighbors. Both signs and both operand orders exercise the magnitude scale.
Eight dyadic scale references are verified independently with Python's exact
`Fraction` arithmetic; their expected values use exact equality in Rust.

The configured values must be preserved, including signed zero and positive
subnormal values. V1 defaults remain exactly `1e-14` absolute and `1e-12`
relative. Negative values and nonfinite inputs retain their parameter names and
input bits in `InvalidSurfaceParameter`; absolute-parameter validation takes
precedence when both inputs are invalid.

For butterfly validation, the tested residuals are

\[
W=\theta\phi(1+|\rho|),\qquad
Q=\theta\phi^2(1+|\rho|).
\]

`W < 4` requires slack beyond the effective tolerance; `Q <= 4` includes the
tolerance boundary. Tests check adjacent values, both correlation signs,
which error wins when both bounds fail, and the reported residual/limit bits.
Three exact rational residual cases per correlation sign are retained in the
JSON record. Finite inputs whose products overflow must return the appropriate
typed violation, even if a relative-tolerance calculation also becomes infinite.

The direct helper tests use positive finite `theta`/`phi` and `abs(rho)<1`.
These local tuples need not arise from a phi family that has passed every
public-constructor check. They establish the validation-helper contract;
they do not prove public reachability of every mutation witness.

Two public `StandardSsvi::new` tests separately verify that accepted parameters
and tolerance are preserved, and that both butterfly error categories propagate
after family validation. These assertions do not add generated constructor
coverage: the pinned tool omits constructors named `new`. Numeric constant
replacements and changes to `max`/`abs` calls are also absent from this inventory.

## Matched mutation comparison

The regular expression selects `SurfaceValidationTolerance` methods and the
butterfly helper. It also selects `StandardSsvi::tolerance` through its return
type. All **44 operation identities and metadata** match, as do the actual
removed/added patch lines. Forty-three full patches are byte-identical. The
whole-helper replacement has different unchanged test-module context because
the new tests were inserted there; this does not change the mutation itself.
Both full patches are retained, without normalizing away the difference.

| SSVI suite | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 29 tests | 44 | 20 | 22 | 0 | 2 | 2 |
| Expanded 40 tests | 44 | 42 | 0 | 0 | 2 | 0 |

**All 22 previously missed mutations become caught**. The 20 existing
detections and two unviable builds retain their classifications; no detection
regresses. There are no missed operations or timeouts in the expanded survey.

Both runs have successful baselines. The standard runner's outcome validator
reconciles selected identities, completed operations, packages, phases and
counts. All viable logs contain the complete 29- or 40-test SSVI result, with
598 filtered cases. Each caught operation has actual test failures; every
newly caught operation fails at least one new test. This is a focused unit-test
comparison, not a claim that the original full integration suite missed the
same operations.

Two generated `Default::default()` replacements remain unviable: the V1
tolerance factory and `StandardSsvi::tolerance`. Both fail with E0277 because
`SurfaceValidationTolerance` does not implement `Default`. They are retained
without exclusions and are not counted as detections. Each survey has 43
complete test logs (baseline plus 42 viable mutations) and two build-only logs.

Reproduce the expanded diagnostic:

```bash
CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/ssvi.rs \
  --re='SurfaceValidationTolerance|validate_butterfly_bounds' \
  --cargo-arg=--lib --cargo-test-arg=market::ssvi::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/surface-tolerance
```

Tool versions are cargo-mutants 27.1.0 and Rust 1.98.1. Repository configuration
adds `--locked` and `--all-features`. Both runs use the same settings, limits and
worker/build concurrency. No score waiver or exclusion is added.

## Validation and limits

Both Rust crates' unit targets pass **688 tests**: 633 pricing and 55 numerical,
with five existing ignored and no failures or filtered cases. The focused SSVI
suite passes all 40 tests. The existing Standard SSVI and eSSVI integration
reference targets each pass their one test.

Workspace Clippy with all targets/features and warnings denied, formatting,
dependency direction and 22 static/script checks pass, including 24 mutation
runner tests. The stacked changed-line selection remains **41 operations**;
that check is list-only. Markdown links, the staged source archive and the final
diff are checked before publication.

At the start of this follow-up, the base revision's
[Mutation testing](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40240053891)
and [ordinary CI](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40240053906)
workflows were still queued. This observation is not a completed CI result and
does not establish the new revision's remote status.

The [JSON record](mutation-surface-tolerance-2026-10-25.json) retains references,
source/report hashes, full mutation patches, transitions, phase statuses,
failed test names and log hashes/summaries. Earlier records are unchanged.
No new full integration suite, ignored heavy acceptance or installed Python-wheel
run is claimed. Whole-library and installed-wheel mutation coverage remain
outside this survey. The historical PCHIP/forward-call survivors and
[unresolved fractional-OU guard](mutation-fou-guard-2026-10-25.md) are unchanged.
