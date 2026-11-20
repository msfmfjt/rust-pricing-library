# Numerical mutation-testing follow-up — 2026-10-22

Base: `2513e9edbb6c0a218f0b007e4d24b8d9b0f691cb`, the initial mutation-testing
commit in PR #172. This extends the [initial pilot](mutation-testing-2026-10-22.md)
without overwriting its historical results.

## Scope and reproduction

The survey selects all **952 mutations across seven production files** in
`pricing-numerics`. It runs that crate's nonignored unit tests only, with Linux,
Rust 1.98.1 and cargo-mutants 27.1.0. It does **not** measure the additional
protection provided by downstream pricing integration tests. The ordinary
`run_mutation_tests.py full` command still includes both Rust crates.

The comparison uses the same selection, features, build/test limits and workers
before and after. Run from the appropriate checkout, using a fresh output path:

```bash
CARGO_PROFILE_TEST_DEBUG=0 cargo mutants \
  --package=pricing-numerics --test-package=pricing-numerics \
  --cargo-arg=--lib --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation-numerics-survey
```

The repository configuration adds `--locked` and all features. The diagnostic
20-second test limit bounds already-known infinite loops; it does not replace
the normal CI runner's timeout policy. The baseline and mutated builds use the
same test package here. No mutation is excluded to improve the results.

The [machine-readable record](mutation-numerics-2026-10-22.json) contains commands,
complete counts, per-file outcomes, baseline test commands, timestamps, source
hashes, remaining mutation identities and independent numerical references.
The before run uses the committed base; the after run uses the added tests,
identified by the recorded hashes. All production Rust text is unchanged.

## Test improvements

- Gaussian tails use relative error against independent `erfc` references.
  The previous absolute-only test could accept zero for tiny probabilities.
  Existing central-region precision checks remain in place.
- Complex arithmetic checks Cartesian reference results directly, both
  components' finiteness and the principal square-root branch, including signed
  zero. Self-consistency alone could miss a constant-return magnitude or an
  incorrect conjugation applied twice.
- Correlation factors preserve raw/canonical input, reconstruct the original
  four-dimensional matrix and expose exact diagnostics. Dyadic inputs test
  inclusive symmetry, diagonal, PSD and zero-pivot residual boundaries; each
  tolerance and independently overflowing threshold is checked.
- Fractional OU values use an independent 80-digit tanh-sinh quadrature on the
  infinite interval, without the Rust solver's endpoint substitution or tail
  cutoff. Polynomial/exponential integrals and heap ordering are checked
  separately. The [reference generator](../../scripts/generate_numerics_mutation_references.py)
  is optional and adds no runtime or Rust-test dependency.
- Least squares checks option/input limits before model evaluation, exact error
  categories, callback shape changes, a coupled three-variable fit, a known
  linear system, evaluation/iteration accounting, equal-cost/domain rejection,
  scaled step boundaries and termination after persistently rejected trials.

There are **47 numerical unit tests**, up from 23 at the base. No pricing API,
algorithm, production tolerance, RNG convention or library dependency changes.

## Measured outcomes and remaining work

| State | Selected | Caught | Missed | Timeout | Unviable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 952 | 741 | 201 | 3 | 7 |
| After | 952 | 912 | 30 | 3 | 7 |

The added tests detect **171 previously surviving mutations**. Every previously
caught mutation remains caught. Both surveys exit **3** because timeouts remain;
the after survey is not presented as a passing gate or complete test adequacy.

| Module | Selected | Caught after | Missed before → after | Timeout | Unviable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundation marker | 2 | 0 | 2 → 2 | 0 | 0 |
| Correlation | 160 | 158 | 32 → 0 | 0 | 2 |
| Normal distribution | 71 | 69 | 25 → 2 | 0 | 0 |
| Reduction | 181 | 179 | 0 → 0 | 1 | 1 |
| Fractional functions | 214 | 194 | 59 → 16 | 2 | 2 |
| Complex arithmetic | 99 | 98 | 18 → 0 | 0 | 1 |
| Least squares | 225 | 214 | 65 → 10 | 0 | 1 |

The 30 surviving identities are retained in the JSON record without exclusions.
They still need review: the legacy foundation marker has no unit assertion;
some branch changes preserve the tested mathematical result or remain within
the accepted floating-point error; adaptive quadrature limits/order and solver
regularization also need a separate robustness assessment. They are not all
declared equivalent. For example, choosing the other CDF branch at zero returns
the same 0.5, whereas a solver-internal change may require further analysis even
if convergence tests pass. Do not impose a 100% score by adding implementation
snapshots or weakening numerical precision checks.

The three timeouts are actual nonterminating mutations: the reducer keeps
processing its final singleton, or the digamma recurrence no longer advances
toward its exit when `x += 1` becomes `x -= 1` or `x *= 1`. Seven mutations are
unviable due to absent `Default` implementations/bounds or an undefined complex
remainder operator. Timeouts and unviable builds are not counted as caught.

## Evidence integrity and validation

An initial capture lacked completed JSON metadata and was discarded as evidence.
The retained comparison was rerun into isolated output directories and validated
for completion, identical mutation identities and reconciled counts. A console
summary alone is not accepted as a complete survey.

The runner now also reconciles aggregate counters with individual named outcome
records. Missing/duplicate records, unknown outcomes and multiple baselines
fail closed. Ten runner tests pass, and the installed-tool two-crate regression
still verifies real detection, missed-mutant exit 2 and failed-baseline exit 4.
The strengthened checker also accepts the retained real 15-mutant smoke result.
The current test-only Rust diff is explicitly `no_mutants` in the changed-line
runner; the separate smoke gate continues to run actual mutations.

All 47 numerical unit tests, numerical-crate Clippy with all targets/features
and warnings denied, formatting, documentation links and the staged source
archive are checked for this change. No full-library mutation, ignored numerical
acceptance or installed Python-wheel run is claimed here. Remote CI is reported
separately in the PR; no merge is authorized by this record.
