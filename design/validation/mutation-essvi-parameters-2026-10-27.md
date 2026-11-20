# eSSVI parameter interpolation mutation coverage — 2026-10-27

Base: `028c3f81806ee9e59400fecf45ab3536b511bbf5`, following the
[Standard SSVI variance tests](mutation-ssvi-variance-2026-10-26.md) in draft PR #172.
Seven tests strengthen the eSSVI time-interpolation contract in
[calculation specification section 3.4](../../docs/models/local-vol-vegakt-calculation-specifications.md).

Production Rust is byte-for-byte unchanged. Public signatures, error variants,
serialized data, dependencies, arithmetic order, RNG settings and existing
numerical/acceptance tolerances are unchanged. The specification now describes
the existing right-sided eSSVI knot derivatives. Its stale Heston-like phi
paragraph is also synchronized with the already implemented degree-14, `0.5`
cutoff correction recorded in the [phi validation](mutation-ssvi-phi-2026-10-25.md).
That documentation correction introduces no further numerical change and does
not rewrite historical evidence.

## Reference panel and observable contracts

The new surface has unequal intervals and a first maturity different from one:

| Time | Theta | Psi | Rho times psi |
| ---: | ---: | ---: | ---: |
| 2 | 1/4 | 1/4 | -1/8 |
| 4 | 3/4 | 1/2 | 0 |
| 8 | 1 | 1 | -1/4 |

The terminal theta slope is `1/32`. These slices satisfy the existing
admissibility and pair-consistency checks through the public constructors.
The independent affine equations are:

| Time region | Theta | Psi | Rho times psi |
| --- | --- | --- | --- |
| `0 < t < 2` | `t/8` | `t/8` | `-t/16` |
| `2 <= t < 4` | `t/4 - 1/4` | `t/8` | `t/16 - 1/4` |
| `4 <= t < 8` | `t/16 + 1/2` | `t/8` | `1/4 - t/16` |
| `8 <= t` | `t/32 + 3/4` | `1` | `-1/4` |

Nine query times freeze all six values/derivatives and the reported region.
The [reference generator](../../scripts/generate_essvi_parameter_references.py)
uses Python's exact `Fraction` arithmetic on these independently solved
polynomials. All 54 expected literals, nine times and nine region labels match
the generated panel; the 12 slice inputs and terminal slope also match. Every
numeric reference is exactly representable as binary64, so the parameter test
uses exact equality. The Rust test does not invoke the generator.

Additional contracts cover:

- Every knot and its immediately adjacent binary64 times. At a knot the values
  reproduce its slice, the region is `Knot`, and the slopes come from the right.
  The final knot uses the terminal theta slope and zero psi/rho-psi slopes.
- Four changes of time unit: values and region are preserved while all three
  derivatives transform inversely with the time scale.
- Flat slices and zero terminal slope, which must retain constant parameters
  and zero derivatives after the first knot.
- Invalid time queries, including both signed zeros, a negative subnormal,
  infinities and a payload NaN. The exact error category, coordinate and input
  bits are preserved.
- Short-end smile scaling at fixed rho and phi, through public variance queries.
  ATM queries separately verify that variance and its time derivative use the
  selected theta and one-sided theta derivative. These additional regular-case
  identities use a new relative tolerance of `2e-14`; no existing tolerance changes.

These tests exercise public methods rather than calling interpolation helpers
directly. Exact regular-case references do not establish that every finite input
avoids intermediate overflow or underflow, nor that a centered maturity
difference across a slope discontinuity equals the reported derivative.

## Matched mutation comparison

Both runs contain the same **85 identities, metadata and complete patches**:
41 in `EssviSurface::parameters`, 37 in `interpolate` and seven in
`long_parameters`.

| eSSVI suite | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 3 tests | 85 | 54 | 28 | 0 | 3 | 2 |
| Expanded 10 tests | 85 | 82 | 0 | 0 | 3 | 0 |

**All 28 previously missed operations become caught**, and all 54 existing
detections remain caught. No exclusion or waiver is added. The three unchanged
unviable operations replace these functions with `Default::default()` results;
each fails with E0277 because `EssviParameters` does not implement `Default`.
They are not detections.

The newly detected operations expose specific gaps:

| Operations | Former gap | New observation |
| ---: | --- | --- |
| 1 | Weakened invalid-time predicate | Exact typed query errors |
| 1 | First knot classified as short extrapolation | Knot region and right-hand slopes |
| 2 | Reciprocal/scaling arithmetic hidden by first maturity 1 | First maturity 2 and changes of time unit |
| 6 | Short-end parameter derivatives unasserted | Exact slopes and public smile scaling |
| 9 | Knot-only match guards and slice indexes unexercised | First, internal and final knots plus neighbors |
| 2 | Interpolation weight changes hidden by interval width 1 | Unequal intervals of width 2 and 4 |
| 6 | Interpolation slope divisions hidden by interval width 1 | All three exact parameter derivatives |
| 1 | Long-end multiplication/division indistinguishable one year after the last knot | Queries two years beyond the last knot and the flat terminal case |

Both baselines pass. The normal runner's outcome validator reconciles all
identities, completed counts, packages, execution phases and statuses. Each
survey has 83 complete test logs (baseline plus 82 viable operations), and three
build-only logs. Every viable log contains all three or ten selected tests,
zero ignored tests and 642 filtered tests. Every newly caught mutation fails
at least one new test. Full patches and failure-test names are retained in the
[JSON record](mutation-essvi-parameters-2026-10-27.json).

Reproduce the expanded diagnostic with the pinned Rust toolchain and
cargo-mutants 27.1.0:

```bash
CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/essvi.rs \
  --re='EssviSurface::parameters|interpolate|long_parameters' \
  --cargo-arg=--lib --cargo-test-arg=market::essvi::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/essvi-parameters
```

The repository configuration adds `--locked` and `--all-features`. The original
survey uses a detached checkout of the base above with the identical options.
This is a focused unit observer, not a claim that the full baseline integration
suite missed these same operations. It does not mutate `validate_slice`,
variance evaluation or the complete eSSVI surface. Constructors named `new`,
numeric constants and argument/metadata substitutions absent from the generated
inventory are outside its denominator.

## Other validation and remote evidence

- **702 unit tests pass**: 647 pricing and 55 numerical, with five existing
  ignored tests and no failed/filtered tests. All ten focused eSSVI tests pass.
- Existing Standard SSVI and eSSVI integration reference targets both pass.
- Workspace Clippy across all targets/features with warnings denied, formatting,
  dependency direction and all 22 static/script checks pass, including 24
  mutation-runner tests. Markdown links, staged source archive and diff checks pass.
- The stacked changed-line inventory against
  `376c92624f88083b46d86e1087d2b0f03e202b91` still contains 41 operations.
  This local check is list-only and is not a new changed-line mutation run.

The immediate parent's [mutation workflow](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40296492094)
is completed successfully: smoke catches 15/15, both changed-line shards pass
(21/20 selected, 19/19 caught, two/one unviable, zero missed/timeouts), and the
aggregate passes. Exact head/base, completed job metadata and logs were checked;
artifact ZIP contents were not independently re-audited. Smoke also passes the
24 runner tests and installed-tool fixture scenarios; intentional misses and a
failed baseline inside those fixtures are expected failure-propagation controls.

The parent's [ordinary CI](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40296491921)
has overall status `cancelled`. Its three Rust platform jobs pass, but the macOS
custom-basis wheel validation aborts with SIGABRT in
`test_vega_kt_custom_basis_provider.py`; the available log does not establish the
root cause. Other jobs were cancelled. The separate
[Rough family MC Delta and LSV AAD workflow](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40296492191)
fails on a macOS artifact upload with `ENOTFOUND` after its numerical steps pass.
These are retained as unresolved remote failures, not reclassified as successful
overall CI. Job-list retrieval covers the API's first page only.

No new full integration survey, ignored heavy acceptance panel or installed
Python-wheel run is claimed here. Earlier full-suite evidence remains historical;
the unresolved fractional-OU guard and other recorded survivors are unchanged.
New-head CI is separate from this completed local and parent-head evidence.
