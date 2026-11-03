# Standard SSVI variance-derivative mutation coverage — 2026-10-26

Base: `494dbcc327a11fbf0eb0b4adcc4ca3966773fc06`, following the
[surface-tolerance tests](mutation-surface-tolerance-2026-10-25.md) in draft PR
#172. Seven new tests exercise `StandardSsvi::total_variance_derivatives`
through the public surface interface. They support the analytic surface
contracts in [requirements section 3.1](../requirements-v1.0.md) and the
[surface calculation specification](../../docs/models/local-vol-vegakt-calculation-specifications.md).

Production Rust is byte-for-byte unchanged. Public signatures, error variants,
schemas, dependencies, arithmetic order, RNG settings and existing
numerical/acceptance tolerances are unchanged.

## Independent value and derivative references

The optional [generator](../../scripts/generate_ssvi_variance_references.py)
evaluates only the scalar total-variance formula, using the expanded radicand
`1 + 2*rho*y + y*y`, where `y=phi(theta)*k`. Power-law phi is evaluated through
high-precision logarithms; Heston-like phi uses the independently integrated
exponential series. No production analytic variance-derivative formula is used.

Centered differences in log-moneyness and a local affine theta path provide
`w_k`, `w_kk` and `w_t`. Two calculations vary both decimal precision and step:
100 digits with `h=1e-20`, then 140 digits with `h=1e-30`. Their maximum absolute
difference over the 32 values is **1.82e-41**. This finite-panel agreement is not
a formal whole-domain derivative-error bound or a claim of 100-digit derivative
accuracy.

```bash
python scripts/generate_ssvi_variance_references.py
```

Eight cases cover both phi families, correlation signs, moneyness signs and
short extrapolation, knots, interpolation and long extrapolation. Rust compares
the four nonzero numerical fields with relative tolerance `5e-13`, and checks
theta, its derivative and region exactly. All 52 actual numeric input literals,
32 reference literals and eight region labels are audited against the generator;
its four unused Heston gamma placeholders are not counted as Rust literals.

The theta curve has knots `(1, 0.25)` and `(2, 0.5)` and terminal slope `0.125`.
At the last knot, the reported derivative is the interpolator's `0.25`; the
local reference follows that convention. It does not take a centered time
difference across the change to the long-extrapolation slope.

Additional tests check reflection under `(rho,k) -> (-rho,-k)`, ATM identities,
zero time derivatives on flat curves, and preservation of interpolation
metadata. Input tests require the exact query coordinate and input bits,
including signed zero, infinities and payload-bearing NaN, with time validation
taking precedence. Curve and phi errors propagate unchanged. Finite inputs
that overflow computed variance/time derivative, or underflow positive variance
to zero, retain the existing typed rejection and context; no whole-finite-domain
accuracy guarantee is introduced.

## Matched mutation comparison

Both inventories contain the same **71 identities, metadata and full patches**.
Only the Standard SSVI variance method is mutated, and the observer is the SSVI
unit module. Existing integration tests are outside this diagnostic denominator;
a baseline miss here does not establish a miss in the full normal suite.

| SSVI suite | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 40 tests | 71 | 67 | 3 | 0 | 1 | 2 |
| Expanded 47 tests | 71 | 70 | 0 | 0 | 1 | 0 |

All three missed operations become caught, each with an actual failure in a new
test. All 67 existing detections remain caught. The three cases illustrate why
independent reference levels and typed-error assertions supplement derivative
consistency checks:

| Mutation | Why the original SSVI tests miss it | New observation |
| --- | --- | --- |
| Time guard `||` becomes `&&` | Zero time reaches phi validation and still returns an error, satisfying a generic `is_err` check. | The exact `InvalidSurfaceQuery` coordinate and input bits are required. |
| `a = y + rho` becomes `y - rho` | The altered variance and its analytic derivatives remain mutually consistent, despite representing the wrong surface. | Independent values for `w`, `w_k`, `w_kk` and `w_t` distinguish it. |
| `1 + rho*y` becomes `1 * rho*y` in shape | Variance loses `theta/2` and its time derivative loses `theta_t/2` coherently. | Frozen levels and ATM identities expose the missing term. |

The single unviable replacement returns `Ok(Default::default())`. It fails with
E0277 because `TotalVarianceDerivatives` does not implement `Default`. It remains
in the inventory and is not a detection. There are no exclusions or score
waivers. Numeric constants, argument substitutions, metadata assignments and
constructors absent from this generated inventory are outside its coverage.

Both baselines pass. The standard outcome validator reconciles identities,
completion metadata, packages, phases and counts. Every viable log has its
complete 40- or 47-test result, with 598 filtered cases; caught cases have actual
failed tests. Each survey has 71 complete test logs (baseline plus 70 viable
mutations) and one build-only log.

Reproduce the expanded diagnostic:

```bash
CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/ssvi.rs \
  --re='total_variance_derivatives' \
  --cargo-arg=--lib --cargo-test-arg=market::ssvi::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/ssvi-variance
```

Both runs use cargo-mutants 27.1.0, Rust 1.98.1 and identical limits/concurrency.
Repository configuration adds `--locked` and `--all-features`.

## Validation and CI

Both Rust unit targets pass **695 tests**: 640 pricing and 55 numerical, with
five existing ignored and no failures or filtered cases. Focused SSVI tests
pass all 47 cases. Existing Standard SSVI/eSSVI integration reference targets
pass their two tests. Workspace Clippy with all targets/features and warnings
denied, formatting, dependency direction and 22 static/script checks pass,
including 24 mutation-runner tests. The reset environment initially lacked
`jsonschema`; installing the dependency specified by `CONTRIBUTING.md` resolved
the three affected checks without changing their code or assertions.

The stacked changed-line inventory remains **41 operations**, list-only locally.
Markdown links, staged source-archive contents and the final diff are checked
before publication. No new full integration suite, ignored heavy acceptance or
installed Python-wheel run is claimed.

The preceding revision's [mutation workflow](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40242563947)
has successful smoke and both changed-line shards on exact head `494dbcc3` and
base `376c92624f88083b46d86e1087d2b0f03e202b91`. Smoke catches 15/15 and passes the
24 runner tests plus installed-tool fixture scenarios. Changed shards select
21/20 operations and catch 19/19, with two/one unviable builds, no misses/timeouts
and successful baselines. Completed job metadata/logs were inspected; artifact
ZIP contents were not independently audited. The aggregate was still queued at
observation time, and ordinary CI was not complete, so overall workflow success
and new-revision CI success are not claimed here.

The [JSON record](mutation-ssvi-variance-2026-10-26.json) retains references,
hashes, patches, transitions, phase statuses, failed test names, log summaries
and prior-CI observations. Historical records and their outstanding items,
including the [fractional-OU guard](mutation-fou-guard-2026-10-25.md), are unchanged.
Whole-library and installed-wheel mutation coverage remain outside this survey.
