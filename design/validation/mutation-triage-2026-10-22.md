# Numerical mutation survivor triage — 2026-10-22

Base: `d2d94586a1d328caa2dda73c7aae3559ddd45602`, the
[numerical follow-up](mutation-numerics-2026-10-22.md) in draft PR #172.
The earlier records remain historical evidence.

## Scope and measured result

This follow-up adds eight unit tests and three independent fractional-OU
reference cases. All **55 numerical unit tests** pass. The same **952 production
mutations** are tested using `pricing-numerics` unit tests only:

| State | Caught | Missed | Timeout | Unviable |
| --- | ---: | ---: | ---: | ---: |
| Previous follow-up | 912 | 30 | 3 | 7 |
| After survivor triage | 931 | 11 | 3 | 7 |

The added assertions detect **19 previously surviving mutations**. All 912
previously caught mutations remain caught; the timeout and unviable identities
are unchanged. Both runs exit **3**. This remains a failing diagnostic survey,
with no exclusions or conversion of timeouts/unviable builds into caught results.

| Module | Selected | Caught | Missed | Timeout | Unviable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundation marker | 2 | 2 | 0 | 0 | 0 |
| Correlation | 160 | 158 | 0 | 0 | 2 |
| Normal distribution | 71 | 69 | 2 | 0 | 0 |
| Reduction | 181 | 179 | 0 | 1 | 1 |
| Fractional functions | 214 | 202 | 8 | 2 | 2 |
| Complex arithmetic | 99 | 98 | 0 | 0 | 1 |
| Least squares | 225 | 223 | 1 | 0 | 1 |

The [JSON record](mutation-triage-2026-10-22.json) retains source hashes, commands,
complete counts, baseline commands, outcome transitions, the failed-test names
for each newly caught mutation and the rationale for every remaining survivor.
The previous run's file hashes match the committed base. The after run is pinned
by its file hashes; all seven production Rust prefixes are unchanged.

## Behavioral tests added

| Area | Independent expectation or contract | Newly caught |
| --- | --- | ---: |
| Foundation marker | The legacy public role remains `core` | 2 |
| Fractional OU | Infinite-interval reference values for a sharp cusp, a correlation just below one, and a negative far tail | 3 |
| Quadrature error budget | Simpson/Richardson integration of `1920*x^4` gives exactly 384 and needs no subdivision when the error equals the budget | 1 |
| Quadrature depth | A discontinuity at 1/3 fails when its required subdivision exceeds the depth budget | 2 |
| Equal-error panel order | A constant fourth derivative gives equal-error panels; earlier panels are refined first, including sibling ties | 2 |
| Bound activity | A zero-gradient coordinate can move inward through coupling, while an outward active coordinate cannot couple into the free solve | 3 |
| Damped least-squares step | With `J=diag(2,3)`, residual `(2,3)` and damping scale 9, the independent ridge solution gives the next point `(9/13,1/2)` | 4 |
| Failed factorization | Overflow in the damped matrix exhausts retries without a model trial or fabricated accepted/rejected evaluation | 2 |

The solver tests limit iterations/evaluations so later convergence cannot conceal
an incorrect first step. They check an analytical problem and explicit resource
accounting. Quadrature tests verify stopping and deterministic ordering as well
as the integral. They do not tighten public precision budgets to distinguish
arbitrary floating-point snapshots.

The [optional reference generator](../../scripts/generate_numerics_mutation_references.py)
now also covers `(H, lag) = (0.025, 0.1), (0.25, 1e-24), (0.25, 100000)`.
It uses mpmath 1.3.0 with 80 decimal digits and infinite-interval tanh-sinh
quadrature. The existing absolute error budget of `2e-12` is unchanged.

## Remaining eleven survivors

Locations below refer to the unchanged production source. Every entry remains
selected and reported as **missed**. Accuracy evidence does not imply bitwise
replay equivalence, and a finite sample is not a proof over all floating inputs.

| Location and mutation | Assessment and limitation |
| --- | --- |
| `normal.rs:15`, `<` → `<=` | Only the approximation boundary changes. At its negative floating-point value, relative errors against 80-digit `erfc` are about `3.61e-9` before and `2.89e-9` after, both below the existing `1e-8` tail budget. The outputs differ, so this is not bitwise equivalent. |
| `normal.rs:47`, `>` → `>=` | Only signed zero takes the other branch; both return exactly 0.5. Equivalent for this expression. |
| `fractional.rs:41`, `<` → `<=` | An extra digamma recurrence when the shifted argument equals 12 preserves the mathematical formula. On 1,501 inputs across `[0.5,2]`, both maximum absolute errors are about `1.001e-15`; the largest output change is `3.331e-16`. This supports precision adequacy on the grid, not global replay equivalence. |
| `fractional.rs:53`, `/` → `%` | Perturbs the coefficient of `x^-14`. Since shifted `x >= 12`, its real-arithmetic contribution is bounded by `(11/12)*12^-14 < 7.14e-16`, below the existing `3e-15` budget. This bound does not include every rounding interaction. |
| `fractional.rs:53`, delete unary `-` | The corresponding real-arithmetic perturbation is at most `(1/6)*12^-14 < 1.30e-16`. The same rounding qualification applies. |
| `fractional.rs:83`, `<` → `<=` | The two cusp expressions are algebraically equal at `u=lag`. Different floating-point evaluation can change roundoff; precision-preserving candidate, not an equivalence proof. |
| `fractional.rs:94`, `||` → `&&` | Removes explicit transformed-endpoint guards. The left Jacobian is exactly zero; the right uses floating `sin(pi)` and can introduce tiny roundoff. Precision-preserving candidate with no global error/replay proof. |
| `fractional.rs:104`, `<` → `<=` | At lag 64, adds a zero-width physical segment whose transformed integrand and integral are zero. Equivalent for the current valid input domain. |
| `fractional.rs:110`, `&&` → `||` | NaN/infinity still fail both conditions, but a finite out-of-range correlation would be accepted. A valid public input reaching this defensive guard has not been established. **Unresolved reachability**, not declared equivalent. |
| `fractional.rs:201`, `||` → `&&` | Midpoint collapse is unreachable for the production caller's `[0,1]` dyadic subdivisions before the depth-40 guard. The smallest eligible midpoint offset is `2^-40`, above the spacing on this interval. This reasoning does not cover arbitrary private-helper intervals or simultaneous mutations. |
| `least_squares.rs:209`, `||` → `&&` | The later finite scaled-width check still rejects any nonfinite bound with the same error, after independent finite-positive-scale validation. Redundant guard within the current input policy. |

Thus four entries have a local equivalence/redundancy or production-domain
unreachability argument, six have precision evidence or a precision-preserving
candidate rationale, and one remains an unresolved defensive guard. None is
waived. In particular, do not turn these labels into a passing mutation score.

The same three infinite-loop mutations still time out: reduction fails to finish
its singleton, or the digamma recurrence does not advance. Seven builds remain
unviable, as recorded in the earlier follow-up.

## Reproduction and validation

Use Linux, Rust 1.98.1 and cargo-mutants 27.1.0. From the appropriate checkout,
choose a fresh output path:

```bash
CARGO_PROFILE_TEST_DEBUG=0 cargo mutants \
  --package=pricing-numerics --test-package=pricing-numerics \
  --cargo-arg=--lib --baseline=run --no-shuffle --jobs=2 \
  --timeout=20 --build-timeout=600 --output=target/mutation-numerics-triage
```

Repository configuration adds all features and `--locked`. The 20-second
diagnostic limit contains known infinite loops; the normal runner's CI timeout
policy is unchanged. Compare the same mutation names, not just aggregate totals.
The retained reports have completion metadata, one successful baseline and
reconciled individual outcomes. The runner's existing evidence checker accepts
both reports with the explicitly narrower test-package scope.

Precision analysis compares the actual original and single-mutant Rust
expressions with mpmath at 80 digits. For the CDF, evaluate both signs of the
floating constant `7.071_067_811_865_475` and compare with `erfc(-x/sqrt(2))/2`.
For digamma, use `x = 0.5 + k as f64 / 1000.0`, `k=0..1500`, and compare with
`mp.digamma(mp.mpf(x))`, preserving the binary input rather than an ideal decimal
grid. The JSON retains the actual values, measured maxima and perturbation
bounds. These probes are diagnostic evidence, not additional acceptance tests.

All 55 numerical unit tests, all-target/all-feature numerical-crate Clippy with
warnings denied, formatting, documentation links, staged source archive and diff
checks pass. The prior runner/integration validation remains in its historical
record; those files and the production mutation configuration are unchanged.
No downstream, whole-library, ignored numerical acceptance or installed-wheel
run is claimed for this unit-only survey. Remote CI is reported separately;
PR #172 remains a draft and this record does not authorize a merge.
