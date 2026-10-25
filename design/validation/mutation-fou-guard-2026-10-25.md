# Fractional-OU range-guard investigation — 2026-10-25

Base: `70b9cf6d16962fb1870e3699225368617e27ad13`, in draft PR #172.
This follows the unresolved fractional-OU survivor in the
[numerical mutation triage](mutation-triage-2026-10-22.md). No production code,
normal test assertion, tolerance or mutation exclusion is changed.

## Condition under investigation

The generated mutation changes the final guard in
`fractional_ou_correlation` from

```rust
result.is_finite() && result.abs() <= 1.0 + 1e-11
```

to the same predicates joined by `||`. For NaN and either infinity both
predicates are false, so these values do not distinguish the expressions.
A witness requires a finite computed result with absolute value strictly
greater than `1.0 + 1e-11`: the original returns `None`, whereas the mutant
returns `Some(result)`.

## Reproducible bounded search

The optional [probe](../../scripts/probe_fractional_ou_guard.py) copies the
actual numerical source and `NeumaierSum` implementation into a new directory,
changes exactly this one guard in a second source copy, and compiles a driver
that calls both implementations. The repository files are not edited.
It requires Python's standard library and the repository's Rust toolchain:

```bash
python scripts/probe_fractional_ou_guard.py --output /tmp/fou-guard-probe
```

The output directory must not already exist. It retains both source copies,
the driver, two binaries, raw CSV rows and `summary.json`. Each CSV row stores
the exact input and output binary64 bits, or `none`. The script rejects
duplicate input pairs and incomplete Cartesian grids before writing a summary.

The grid contains **161 Hurst values × 338 lags = 54,418 pairs**. Hurst values
cover subnormals, the least positive normal, tiny powers of two, a `0.005`
mesh, and values near the upper limit `0.5`. Lags cover zero, subnormal and
normal positive values, decimal exponents from -323 to 305 in steps of four,
quarter powers of two from `2^-20` to `2^20`, both neighboring floats at the
tail cutoff 64, and values through the largest finite float. Decimal powers
are parsed from strings to avoid intermediate overflow in negative-exponent
`powi`. The JSON retains every input bit pattern, so the sampled set is explicit.

Both compilation profiles use Rust 1.98.1 and edition 2024; the second adds
`-O`. The default and optimized runs each complete with:

| Outcome | Cases per profile |
| --- | ---: |
| Same finite output bits | 54,418 |
| Both return `None` | 0 |
| Different result | 0 |
| Range-guard witness | 0 |

The raw rows are also identical across the two profiles. The largest observed
absolute result is `1.0000000000000222` at the binary64 representation of
Hurst `0.03` and the smallest positive lag. It is within the existing `1e-11`
allowance. This observation does not justify tightening the guard to exactly 1.

## Generated mutation remains missed

The pinned cargo-mutants 27.1.0 separately runs the actual generated mutation
against all **55 numerical unit tests**, including a successful unmutated
baseline. Both build and test phases finish successfully, so the mutation is
still **missed**: selected 1, caught 0, missed 1, timeout 0, unviable 0, exit 2.
Inventory identity, completion metadata, package selection, counts and test
logs reconcile through the normal runner's outcome validator.

```bash
CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 cargo mutants \
  --package=pricing-numerics --test-package=pricing-numerics \
  --file=crates/pricing-numerics/src/fractional.rs \
  --re='fractional.rs:110:27: replace && with' --cargo-arg=--lib \
  --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation/fou-guard
```

The [JSON record](mutation-fou-guard-2026-10-25.json) retains the complete probe
summary, source/driver/script and raw-row hashes, exact grids, generated patch,
baseline/mutant phases, test summaries and log hashes. The ordinary numerical
unit run also passes all 55 tests.

This finite search provides **no witness, not an equivalence or unreachability
proof**. Comparing original and mutant outputs is not an independent accuracy
oracle. The probe does not enter the generated-mutation denominator, and this
survivor remains unresolved without a waiver. Historical survey counts remain
unchanged; whole-library mutation coverage is not claimed.
