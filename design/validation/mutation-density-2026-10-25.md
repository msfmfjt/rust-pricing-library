# Durrleman density-factor mutation coverage — 2026-10-25

Base: `70b9cf6d16962fb1870e3699225368617e27ad13`, following the
[SSVI phi correction](mutation-ssvi-phi-2026-10-25.md) in draft PR #172.
Five new tests exercise the public `durrleman_density_factor` contract used by
SSVI/eSSVI and Dupire. They support the existing surface-derivative requirements
in [requirements section 3.1](../requirements-v1.0.md) and the
[surface architecture](../architecture-v0.1.md).

Production Rust is byte-for-byte unchanged. Public signatures, error variants,
schemas, dependencies, RNG settings, arithmetic order and numerical budgets
remain unchanged. Existing tests and tolerances are not weakened.

## Exact references and failure contracts

Write `w` for positive total variance, `s=w_k`, `c=w_kk`, and `k` for
log-forward-moneyness. For a local quadratic variance function with these
derivatives, the density-factor identity expands to

\[
g=1-\frac{ks}{w}+\frac{k^2s^2}{4w^2}
  -\frac{s^2}{4w}-\frac{s^2}{16}+\frac{c}{2}.
\]

Python's standard-library `Fraction` evaluates this expanded identity in exact
rational arithmetic, independently of the production binary64 evaluation order.
All inputs and results below are dyadic; the Rust assertions require exact
equality without a tolerance.

| `w` | `s` | `c` | `k` | Exact `g` |
| ---: | ---: | ---: | ---: | ---: |
| 1/4 | 1/8 | 1/16 | 1/2 | 799/1024 |
| 1/4 | 1/8 | 1/16 | -1/2 | 1311/1024 |
| 1 | -1/4 | 1/2 | 1/4 | 1325/1024 |
| 4 | 1/2 | -1/4 | 1 | 185/256 |
| 1/16 | -1/16 | 1/8 | -1/4 | 3327/4096 |
| 1/2 | 0 | 1/2 | 2 | 5/4 |

Twelve flat-variance cases also require `g=1`. These are pointwise mathematical
references; the local quadratics need not define globally arbitrage-free SSVI
surfaces.

The other four tests check observable failure and boundary behavior:

- Invalid total variance: positive and negative zero, negative finite values,
  both infinities and a payload-bearing NaN. The error field and original
  value/time/log-moneyness bits must match exactly.
- Nonfinite first and second derivatives, separately and together. Input
  validation precedes arithmetic, and the first derivative takes precedence
  when both are invalid.
- With zero slope, curvature at `-2` and its immediate neighbors produces
  zero, negative epsilon or positive epsilon/2. Nonpositive factors are
  rejected; the positive neighboring result is accepted.
- Finite inputs whose arithmetic overflows to positive infinity, negative
  infinity or NaN must return the existing density-factor error. Computed
  NaN sign/payload bits are not frozen across platforms.

The same unmutated production implementation passes the original 15-test SSVI
suite and the expanded 20-test suite. This is a test-coverage change, not a
production bug fix; failing mutated implementations provide the red evidence.

## Matched mutation scope

The diagnostic selects only `durrleman_density_factor`, using SSVI unit tests
matching `market::ssvi::tests::`. Both inventories contain exactly the same
38 generated names, metadata and patches. No operation is removed, introduced,
excluded or waived in this comparison.

| SSVI test suite | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 15 tests | 38 | 12 | 26 | 0 | 0 | 2 |
| Expanded 20 tests | 38 | 38 | 0 | 0 | 0 | 0 |

All **26 survivors become caught**, while the 12 existing detections remain
caught. No caught operation regresses. Each newly caught operation fails at
least one of the new density-factor tests. The improvements cover 24 arithmetic
replacements, the total-variance logical guard, and deletion of the final
finite-result negation.

Both surveys finish with successful unmutated baselines. The normal runner's
outcome validator reconciles inventory identities, complete counts, package
selection, build/test phases and exit statuses. All 39 logs per survey
(one baseline plus 38 mutants) contain a complete 15- or 20-test summary, with
598 filtered cases. Every caught mutation has an actual failed test, and no
build error or timeout is counted as test detection.

Reproduce the expanded diagnostic:

```bash
CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/ssvi.rs --re=durrleman_density_factor \
  --cargo-arg=--lib --cargo-test-arg=market::ssvi::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/density-survey
```

The pinned tool is cargo-mutants 27.1.0 on Rust 1.98.1. Repository configuration
adds `--locked` and `--all-features`. Both runs use the same worker/build
concurrency, timeouts and debug settings. This scope does not replace the
normal full/changed runner or claim coverage of the complete SSVI surface.

## Ordinary validation and limits

Both Rust crates' unit tests pass **668 cases**: 613 pricing and 55 numerical,
with five existing ignored and zero failures or filtered cases. The focused
SSVI suite passes all 20 cases. Workspace Clippy for all targets/features with
warnings denied, formatting, dependency direction and 22 static/script checks
pass, including all 24 mutation-runner tests.

The initial ordinary build failed linking `serde_derive` with undefined hidden
symbols, before any test ran. Only that dependency's generated artifacts were
removed with `cargo clean -p serde_derive`; the unchanged rerun passed. The cause
of the initial link failure is not established. Its log hash is retained, and
the attempt is excluded from successful test counts. Neither mutation survey's
successful baseline had this build error.

The normal stacked changed-line inventory remains **41 operations**, including
the tracked local test additions. This is a list-only check; new-head remote CI
is reported separately. Historical full-suite and CI results at the base commit
do not establish the new revision's CI status. No new integration, ignored heavy
acceptance or installed Python-wheel validation is claimed for this test-only
follow-up. Markdown links, staged source-archive contents and the final diff
are checked before publication.

The [JSON record](mutation-density-2026-10-25.json) retains exact rational
references, source/inventory/report hashes, per-mutation transitions, actual
failed test names, patches, build/test phases, and validation log hashes and
summaries. Historical records
remain unchanged. The separate
[fractional-OU investigation](mutation-fou-guard-2026-10-25.md) retains its
unresolved range guard; the bounded probe does not establish equivalence.
