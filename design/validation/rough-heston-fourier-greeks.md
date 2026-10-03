# Heston Fourier forward-Greek validation

This additive change implements fixed-model Forward Delta/Gamma for the two
affine Fourier families. The existing price method, MC path arithmetic,
parameters, RNG layout, schemas and Cargo dependencies are unchanged. There is
no Spot/AAD, recalibration, model-parameter risk or production-admission claim.
The [model specification](../../docs/models/rough-heston-fourier-greeks.md)
defines derivative and failure semantics.

## Independent reference and acceptance contract

`fixtures/rough-volatility/fourier-greeks.json` fixes inputs and budgets.
`scripts/check_heston_fourier_greeks.py` never imports the pricing extension.
It uses the independent reference module's ordinary-Heston closed transform and
three-factor-lift DOP853 transform. These transform routines are shared with the
previous independent Fourier fixture, not with production.

For each model, maturities 0.25/1 and strikes 80/100/120, forward 100 and discount
0.97, Delta is computed as the share-measure probability and Gamma as its log
return density, on the **Re(z)=1 contour** with adaptive quadrature. This is not
the production half-moment, Black-control, Simpson derivative formula. The 12
reference rows cross-check cutoffs 200 and 300. Centered differences of separate
P1/P2 prices at bump sizes 0.02/0.01 check both derivatives independently of
production. Twelve deterministic Black rows use SciPy's normal distribution,
including constant variance and ordinary-Heston mean-reverting variance.

Budgets fixed before Rust numerical runs:

| Check | Absolute tolerance |
| --- | ---: |
| Rust reference Delta | 0.00002 |
| Rust reference Gamma | 0.000002 |
| Independent reference cutoff change | 0.00000002 |
| Independent P1/P2 price-bump Delta/Gamma | 0.0000002 |
| Time/cutoff refinement Delta | 0.00005 |
| Time/cutoff refinement Gamma | 0.00001 |
| Production price-bump Delta | 0.000003 |
| Production price-bump Gamma | 0.000001 |

Rust's 12 stochastic comparisons use 1024 Riccati steps, 1024 intervals and
cutoff 192. The 18 refinement rows use Rough Heston H=0.1/0.3 and the explicit
three-factor lift, maturities 0.25/1, and the three strikes. Time refinement is
512 to 1024 steps at fixed cutoff 128 and 512 intervals. Cutoff refinement is
128 to 256 at fixed 1024 time steps and frequency spacing 0.25 (512 to 1024
intervals). This separates the two changes. All rows, not just ATM, are checked.
These are Greek units, not prices or implied-volatility basis points.

## Contracts and diagnostics

Four non-ignored Rust tests verify invalid inputs, zero maturity/variance and
ATM-kink rejection, explicit rejection of a non-degenerate zero-control case,
24 deterministic Black comparisons, unchanged price results, call-put Delta
parity, homogeneity under four forward/strike scale factors, discount scaling,
and production price bumps at two sizes. The price-bump comparisons validate
differentiation of a finite quadrature; they are not independent model oracles.

Frequency-grid differences are checked against separately compiled N/2 plans.
Both retained tail indicators are reconstructed using explicit scalar formulas
and independent transform queries, without reading the cached difference array
or using production quadrature/accumulator helpers. Nonzero envelopes ensure
these checks are not vacuous. Diagnostics are not treated as total-error bounds.

Python tests use the actual extension, independently retained Heston rows,
rough and 20-factor-lift price bumps, scaling/parity, validation errors and frozen
nested result properties. Stub mutation checks require the new class, method,
eight properties and their explicit signatures. The example is required by
the wheel smoke gate.

## Reference-generation correction before Rust comparisons

The first independent reference-generation guard failed at Heston T=0.25,
K=100 with a centered bump of 0.04: Delta differed by approximately 3.6764e-7,
above the unchanged 2e-7 finite-bump budget. This is finite-bump truncation, not
a Rust comparison. Reducing the reference bumps from 0.04/0.02 to 0.02/0.01
passed without altering the integration contour, model, reference-Greek formula
or tolerance. The failure log is retained in the delivery. Production-bump
checks keep their original 0.04/0.02 sizes and 3e-6/1e-6 budgets.

An initial Rust test-array type-inference error was fixed by explicitly naming
`RoughVolatilityModel`; it changed no financial or numerical expression.

## Evidence and limits

The dedicated read-only three-OS workflow runs fast tests in default/minimal
builds and explicitly includes the two ignored numerical tests in release.
`heston-fourier-greeks.log` is retained; a missing log fails artifact upload.
Source-archive checks and mutation tests require the workflow, new files and
wheel example. CI definition alone is not evidence of CI success.

Independent non-degenerate Greek references cover ordinary Heston and the
explicit three-factor lift, not a certified full rough-Heston Greek surface.
The rough-model evidence consists of derivative identities, exact deterministic
limits and finite time/cutoff comparisons. It does not bound the remaining
Riccati, frequency, omitted-tail or model errors. Short expiries, stressed
parameters, far wings, moment boundaries and the zero-initial-variance Greek
case need further work. A price-accurate configuration need not be Gamma-accurate.

## Executed numerical results (Linux x86_64, Rust 1.98.1)

The first Rust numerical run passed all six tests, including both explicitly
ignored acceptance tests. All 12 independent rows and all 18 refinement rows
met their unchanged budgets. Maximum observed absolute differences:

| Comparison | Delta | Gamma |
| --- | ---: | ---: |
| Independent reference | 6.343090053207e-09 | 7.300723602055e-10 |
| 512/1024 Riccati steps | 4.025288627973e-07 | 5.679037127365e-08 |
| Cutoff 128/256, fixed frequency spacing | 7.418868852582e-10 | 1.107497522509e-09 |

Column maxima need not come from the same scenario. These are finite
comparisons, not simultaneous confidence intervals or global error bounds.

The first full wheel smoke run exposed a packaging-guard issue: removing a
required member from the stub AST raised a raw `KeyError` before the intended
validation diagnostic. The checker now explicitly rejects missing required
members with `RuntimeError`, without removing any member/signature condition
or broadening the test's expected exception. Numerical Python tests were not
the cause; the full suite is rerun after fixing the checker.

The completed local workspace run reported **669 passed, 0 failed,
52 ignored** across 54 target summaries. The two new ignored numerical tests
were run explicitly in release; not all prior ignored panels were executed.
The optimized CPython 3.13 wheel passed **173 Python tests** and every existing
example plus the new forward-Greek example after the guard fix. Workspace
Clippy (all targets/features, warnings denied), formatting, schemas, dependency
direction, four new independent-reference guards and four parent Fourier
reference guards passed. The local offline wheel environment inherited system
NumPy 2.2.4; this is not a clean-install claim. Independent reference calculations
used NumPy 2.3.5, SciPy 1.17.0 and mpmath 1.3.0.
