# Rough Heston Fourier Hurst risk: validation protocol

## Definition and scope

This addition is stacked on parameter-risk PR #126 (03d870e). It adds an analytic
Hurst-only forward tangent of the normalized fractional kernel, the implicit
Riccati solution, and its Fourier integral. Existing price, Forward Greek and
five fixed-kernel parameter APIs keep their definitions and arithmetic order.
There is no path simulation, model calibration, new lift-factory sensitivity,
reverse-mode AAD, JSON change, or new Cargo dependency in this addition.

The result is currency per absolute H unit, at fixed F/K/D/T/v0/kappa/theta/nu/rho,
with time expressed in years. Call and Put share the derivative. H=1/2 uses the
inward limit. Fixed finite lifts are explicitly rejected; their factors alone do
not identify a differentiable H-to-kernel map.

## Independent references and frozen numerical criteria

`fixtures/rough-volatility/hurst-risk.json` records the protocol and references.
Normal execution of `scripts/check_heston_hurst_risk.py` recomputes them without
writing the fixture. Only an explicit `--generate` writes it. The reference does
not import the pricing extension, production kernel/Riccati/path/RNG helpers or
use production Fourier prices as target data.

All references use v0=.04, kappa=.7, theta=.055, nu=.18, rho=-.65, except the
explicit deterministic-variance price panel, which sets only nu=0. H values are
.01/.1/.3/.5; maturities .25/1; F=100; D=.97; strikes 80/100/120.

1. **24 complex Hurst derivatives**: 60-digit fractional power series, orders
   100/140 agreeing within 1e-25; high-precision differentiation with respect to
   alpha=H+.5 and an additional high-precision 1e-7 difference check (inward
   second order at H=.5). Exponents .5+.7i, 2i, 1+.7i. Production at 1024 Riccati
   steps must be within 5e-5 in complex absolute distance. This is selected
   low/moderate-frequency evidence, not an independent full strike surface.
2. **24 deterministic-variance prices/risks**: integrate the Mittag-Leffler
   mean-reverting variance using a 60-digit series, differentiate its total
   variance, and apply the independent Black total-variance derivative. Both
   prices and Hurst derivatives must be within 1e-4 at 1024 steps, cutoff256,
   1024 frequency intervals. Unlike the constant-volatility zero-risk limit,
   these scenarios have nonzero Hurst sensitivity, but no volatility randomness.
3. **36 kernel rows (108 derivative values)**: direct improper integrals of
   `K(t)*(log(t)-digamma(alpha))` against the initial/interior/endpoint hat
   functions at grid sizes128/8192 and lags1/2/16/128 plus8192 for the larger grid.
   Absolute tolerance3e-13. High-precision integration avoids the production
   binomial-tail derivative algorithm. Separate mass-derivative and digamma
   constant/recurrence tests cover boundary and large-lag behavior.
4. **12 nondegenerate price-risk refinement rows**: H=.1/.3, two maturities,
   three strikes. Compare Riccati512/1024 at cutoff128 with512 frequency
   intervals; compare cutoff128/256 at1024 steps, keeping frequency spacing.25.
   Both absolute differences must be <=1e-4. This is a finite refinement check,
   not a continuous-time risk bound or convergence-order proof.

These criteria were fixed before the first Rust numerical run. The first release
run passed all six tests, including both normally ignored numerical panels,
without changing grids, reference values, parameters or thresholds:

| Comparison | Largest absolute difference | Criterion |
| --- | ---: | ---: |
| Independent complex transforms | 9.751578291453e-7 | 5e-5 |
| Nonflat deterministic-variance price risks | 2.906549127463e-5 | 1e-4 |
| Riccati512/1024 price risks | 3.332649327251e-5 | 1e-4 |
| Cutoff128/256 price risks | 5.025513054313e-9 | 1e-4 |

The figures concern different quantities (log transforms versus currency risk),
and are not IV basis points. Logs retain individual rows rather than only maxima.

## Structural and regression tests

Four fast public Rust tests check two production bump sizes (1e-4/5e-5), Call and
Put, three complex arguments, the H=.5 inward derivative, zero maturity, constant
variance, identically zero variance, nondegenerate v0=0, moment strip/conjugacy,
finite-lift rejection, all price input fields, homogeneous Forward/Strike scaling,
discount scaling, and independently reconstructed nonvacuous tail envelopes and
coarse-frequency differences. A cross-partial check compares the H derivative of
existing kappa/nu/rho risk with the respective scalar derivative of Hurst risk.
This cross-partial check uses finite differences only in tests.

New private tests exercise the work guard before frequency allocation and the
36 independent kernel rows/mass identities. The numerics test checks digamma
special values, recurrence and invalid inputs. Four Python tests cover the public
bindings, selected independent deterministic references, boundary behavior and
mutation rejection of the new public classes/members. Stubs explicitly specify
frozen properties and method signatures.

Four reference guard tests recompute the references, reject modified parameters,
H order, tolerances, nonfinite/reference values and missing rows, and exercise the
actual source-archive gate with11 removed CI/evidence snippets. The executable
example and10 new archive members are mandatory.

The new read-only Linux/macOS/Windows CI explicitly includes the two release
numerical tests and retains `heston-hurst-risk.log`; missing artifacts fail upload.
Its successful definition is not a claim of remote CI completion. Normal wheel
CI continues to test clean installation. Local offline wheel testing may inherit
system packages; the delivery report records the actual environment and results.

## Limits

A small retained-interval Simpson difference or last-half derivative envelope is
not an omitted-tail bound or total-error estimate. Errors in solving Riccati,
normalization/rounding, remaining time/frequency bias and model error are not
covered by either diagnostic. Numerical shape checks are only necessary checks.

No nondegenerate full rough price-risk surface is independently certified. Broader
stressed regimes, very short maturities, far strikes, moment-boundary behavior,
calibration, lift-weight/rate derivatives, calibrated market risk, Hessians and
production admission remain separate work. Supporting v0=0 for this Hurst-only
route does not remove the other Greek routes' zero-control restrictions.
