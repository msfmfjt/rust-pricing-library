# Heston Fourier implementation and validation

[Design index](../README.md) · [Model contract](../../docs/models/heston-fourier.md)

## Change and compatibility

Add a deterministic, experimental European pricing and transform API for rough
and lifted Heston on top of PR #121 source `8750d13`. Domain model definitions,
MC formulas, legacy JSON/schema contracts, request routing, RNG layouts and
existing fingerprints are unchanged. The new plan has its own versioned
fingerprint; it does not claim complete price-request replay identity.

The implementation is in the private analytic engine, re-exported through
`pricing::rough_volatility`; finance-independent complex arithmetic resides in
`pricing-numerics`; Python conversion resides in `pricing-python`. No crate or
production dependency was added. Existing tests and acceptance limits are not
removed or relaxed. This implements the user's requested model extension, not
production admission or a replacement of the prior full-truncation MC schemes.

## Independent references

The retained [fixture](../../fixtures/rough-volatility/fourier.json) is generated
only with explicit `--write`; the normal checker recomputes and compares it.
It contains 24 complex transforms and 12 call prices. All use nonzero vol-of-vol,
nonzero mean reversion and unequal initial/long-run variance except explicit
boundary tests. Reference generation does not import the Rust/Python extension.

Ordinary Heston uses its closed-form transform and adaptive scalar Fourier
quadrature. The moderate-rate three-factor lifted model uses SciPy DOP853 with
relative tolerances 2e-12/2e-13 (terminal transform comparison), instead of
product integration or the MC recursion. Its price integral compares cutoffs
180/240. Rough short-time transforms use an 80-digit fractional power series,
comparing 100/120 terms. This check is confined to a short maturity/frequency
range where the two sums agree; it is not a global power-series solver.

Rough one-year prices use a separate explicit fractional Adams predictor/
corrector at 2048 and 4096 steps, vectorized over frequencies. This reference
integrates the interpolated R against the deterministic input curve g0 directly;
it does not use Rust's final trapezoidal integral of h or its implicit quadratic
root. Reference time-grid changes must remain below 1e-4. A separate Fourier
mesh/cutoff comparison, `(1024,160)` against `(2560,200)`, must change price by
less than 2e-7. These numerical comparisons are not rigorous reference-error
bounds or independent mathematical model derivations.

Production reference checks fix 1024 Riccati steps, 1024 Fourier intervals and
cutoff 160. The pre-run tolerances are 3e-6 in complex-transform norm and 0.003
in **price currency units**, not IV basis points. These budgets are common to
ordinary/lifted/rough cases. Per-row errors are printed in the retained log.
The time-resolution check compares 128/1024 steps at the fixed transform points.

## Contract and limit tests

Rust tests cover exact constant variance for H=0.01/0.1/0.3/0.5, absorbing zero,
zero maturity, characteristic conjugacy, unit transform at z=0/1, ordinary-Heston
boundary, splitting a common zero-rate factor, deterministic mean-reverting
variance, stiff finite-factor modes, parity and payment discounting, invalid
configurations, resource guards, immutable plan coordinates/fingerprints and
sequential refinement diagnostics. Complex numerical tests check arithmetic,
principal square roots, and division at very large/small scales. Kernel unit
tests independently check exact integrals of constant functions.

The Python suite checks public prices/transforms, exact constant variance,
zero initial variance with positive deterministic immigration, plan ownership,
refinement properties, rejection paths and explicit wheel-stub mutation guards.
The wheel contract now enumerates both new public classes, their signatures,
static factory and properties; checks are not disabled for the extension.

## CI and remaining scope

The dedicated read-only workflow executes the new fast tests in debug/minimal
configurations, and explicitly includes the otherwise ignored price panel in
release on Linux/macOS/Windows. Its reference job recomputes all retained
references and runs corruption guards; it does not regenerate fixtures.
Missing evidence upload fails the job. The usual wheel CI executes the new
Python runtime tests and example. The source-archive checker requires the code,
fixture, docs, example and workflow gates.

Defining or triggering these jobs is not evidence of their completion. Local
commands, toolchain, results and remote status are separately recorded in the
delivery report and logs. Initial design used v0*T as the Black control; before
final validation it was changed to the half-moment control to cover v0=0 with
positive variance immigration. No acceptance threshold was relaxed.

Neither this reference panel nor small refinement changes prove general
continuous-time accuracy, a martingale certificate or absence of time/cutoff
bias. This increment does not measure full-truncation MC bias against the new
continuous-time solver, prove joint factor/time limits, or add automatic
calibration, Greeks/AAD, VIX/SSR, stochastic rates/dividends, LSV, or path-dependent
Fourier payoffs. The other four model families retain their existing MC APIs.
