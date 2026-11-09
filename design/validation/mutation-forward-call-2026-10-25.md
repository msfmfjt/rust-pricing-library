# Forward-call evaluation mutation coverage — 2026-10-25

Base: `98c30469a5c0b28992b85eb84bff8b2fdeaafff5`, following the
[density-factor tests](mutation-density-2026-10-25.md) in draft PR #172.
Nine new tests cover the default `ImpliedVarianceSurface::forward_call_evaluation`
and its private calculation helper. They support the existing analytic
surface/call-density contracts in [requirements section 3.1](../requirements-v1.0.md)
and the [surface architecture](../architecture-v0.1.md).

Production Rust is byte-for-byte unchanged. Public signatures, error variants,
schemas, dependencies, arithmetic order, RNG settings and existing numerical
tolerances remain unchanged.

## Independent price and density references

The optional [generator](../../scripts/generate_ssvi_forward_references.py)
uses Python's standard-library `Decimal`. It evaluates the Gaussian integral
by its Taylor series, with pi from Machin's arctangent identity. It does not
reuse the production rational/continued-fraction normal approximation.

For total variance `w`, log-forward-moneyness `k`, forward `F` and the
Durrleman factor `g`, the independent calculation uses

\[
K=F e^k,\qquad d_1=\frac{w/2-k}{\sqrt w},\qquad
d_2=-\frac{w/2+k}{\sqrt w},
\]
\[
C=F\Phi(d_1)-K\Phi(d_2),\qquad
C_{KK}=\frac{e^{-d_2^2/2}g}{\sqrt{2\pi}\,K\sqrt w}.
\]

The factor `g` uses the expanded exact identity from the preceding density
record. Eight local variance jets vary both signs of moneyness, forward scales,
variance, slope and curvature. They are pointwise references and need not
represent globally arbitrage-free SSVI parameterizations.

Binary64 inputs are lifted exactly with `Decimal.from_float`. A 100-digit run
and a 140-digit run agree within absolute **4.32e-95** over all 48 expected
values. All 40 input literals and 48 expected literals in Rust match the rounded
generator output. For example, flat variance `w=0.25`, `k=0`, `F=100` gives
undiscounted price `19.741265136584744` and density `0.007733362336056984`.

```bash
python scripts/generate_ssvi_forward_references.py
```

The Rust reference test uses relative tolerance `5e-13` on nonzero regular-case
prices, strikes, densities and `d1/d2`; all dyadic density factors and copied
metadata require exact equality. Existing assertions are unchanged. A separate
test checks homogeneity: scaling `F` scales strike and call price by the same
factor and divides strike density by that factor, while preserving `g`, `d1`,
`d2` and the supplied variance metadata.

## Error and boundary contracts

The remaining tests check:

- Invalid forwards, including signed zero, negative values, infinities and
  payload-bearing NaN, are rejected before surface-query validation.
- The default method forwards query coordinates and preserves a custom
  surface provider's error unchanged.
- Overflowed and underflowed strikes retain the existing strike error and
  query/value bits, before variance validation.
- Invalid supplied variance/derivatives and nonpositive density factors retain
  their exact typed errors through the forward-call method.
- Valid underflowed tails may return zero price and/or zero density.
- A positive subnormal strike times `sqrt(w)` can underflow to zero, producing
  infinite computed density; the existing nonfinite-density error is required.
- At `w=1e-32`, `k=1.12e-16`, `F=1`, `d1` and `d2` round to the same binary64
  value while the strike rounds above the forward. The resulting negative
  computed price must produce the price error, with a finite negative value and
  original query bits. The test does not freeze the magnitude of that rounding
  residual or introduce a whole-domain price-accuracy guarantee.

Both the original 20-test SSVI suite and the expanded 29-test suite pass without
mutating production code. Mutant assertion failures supply the red evidence;
this follow-up changes test coverage, not the pricing implementation.

## Matched mutation comparison

Only the default forward-call method and its calculation helper are selected.
Both inventories contain the same **42 identities, metadata and patches**:
four operations in the default method and 38 in the helper. The observer is the
SSVI unit-test module; other integration and whole-library tests are outside
this mutation denominator. A miss here is not a claim that the whole normal
suite would miss that operation.

| SSVI suite | Selected | Caught | Missed | Timeout | Unviable | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 20 tests | 42 | 4 | 36 | 0 | 2 | 2 |
| Expanded 29 tests | 42 | 39 | 1 | 0 | 2 | 2 |

**35 previously missed mutations become caught**, and all four existing
detections remain caught. Each newly caught operation fails at least one of
the new forward-call tests. No caught operation regresses.

Both surveys complete with successful baselines. Inventory identities,
completion metadata, package selection, phase statuses and counts reconcile
through the normal runner's outcome validator. Every viable mutation and
baseline log has a complete 20- or 29-test result, with 598 filtered cases;
all caught cases have actual failed tests. There are 41 complete test logs per
survey and two build-only unviable logs.

The two unviable replacements return `Ok(Default::default())` from the default
method or its helper. Both fail with E0277 because `ForwardCallEvaluation` does
not implement `Default`. They are not test detections.

One mutation remains missed: changing `<` to `<=` in the nested error-field
selector at the end of the helper. Its behavior is equivalent under the current
preceding checks:

1. Strike is finite and positive; accepted total variance and density factor
   are finite and positive.
2. The normal PDF is nonnegative. The strike/root-variance denominator is
   positive, positive zero or positive infinity. This arithmetic cannot produce
   a finite negative call density.
3. Nonfinite results are rejected before the final sign checks. Therefore
   density is nonnegative whenever execution reaches those checks.
4. Entering the final error branch consequently requires a strictly negative
   price. Both `< 0` and `<= 0` select the price field on every reachable entry;
   zero or signed zero cannot enter through a negative-density condition.

This is a single-mutation argument about the current helper and validation
order. The survivor remains in the raw missed count and the diagnostic still
exits 2. It is not a reason to remove the defensive guard or weaken validation.

Reproduce the expanded diagnostic:

```bash
CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 cargo mutants \
  --package=pricing --test-package=pricing \
  --file=crates/pricing/src/market/ssvi.rs \
  --re='forward_call_evaluation|evaluate_forward_call' \
  --cargo-arg=--lib --cargo-test-arg=market::ssvi::tests:: \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/forward-call
```

The pinned tool is cargo-mutants 27.1.0 on Rust 1.98.1. Repository configuration
adds `--locked` and `--all-features`. Both runs use the same debug settings,
worker/build concurrency and limits. No score waiver or exclusion is added.

## Validation and prior CI

Both Rust crates' unit tests pass **677 cases**: 622 pricing and 55 numerical,
with five existing ignored and zero failures or filtered cases. The focused
SSVI suite passes all 29 cases. Both existing Standard SSVI/eSSVI integration
reference targets pass their one test, with no ignored or filtered cases.

Workspace Clippy for all targets/features with warnings denied, formatting,
dependency direction and 22 static/script checks pass, including all 24
mutation-runner tests. The normal stacked changed-line inventory remains
**41 operations**, including the tracked local test additions; that check is
list-only. Markdown links, staged source-archive contents and the final diff
are checked before publication. No new full 97-target integration suite,
ignored heavy acceptance or installed Python-wheel run is claimed.

The preceding commit's [Mutation testing workflow](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40212412172)
completed successfully on exact head `98c30469a5c0b28992b85eb84bff8b2fdeaafff5`.
Its changed-line shards selected 21 and 20 operations: 19 caught in each, two
and one unviable respectively, zero missed/timeouts. Both baselines, artifact
uploads and the aggregate succeeded. Smoke caught all 15 operations and passed
24 runner tests plus the seven installed-tool fixture scenarios. Completed job
metadata and logs were inspected; artifact ZIP contents were not independently
re-audited. Ordinary CI still had queued work at observation time, so no overall
ordinary-CI success is claimed. These observations do not establish the new
revision's remote CI status.

The [JSON record](mutation-forward-call-2026-10-25.json) retains independent
references, source/inventory/report hashes, per-operation transitions, actual
failed test names, outstanding patches, phase statuses and log hashes/summaries.
Historical records remain unchanged. Whole-library or installed-wheel mutation
coverage is not claimed. The previously investigated fractional-OU range guard
remains [unresolved](mutation-fou-guard-2026-10-25.md).
