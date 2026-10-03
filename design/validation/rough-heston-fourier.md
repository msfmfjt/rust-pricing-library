# Rough and Lifted Heston Fourier validation

## Scope and reference model

This change adds an experimental continuous-time-model transform and a separate
positive-forward vanilla pricer, not a replacement for any MC path algorithm.
The [calculation specification](../../docs/models/rough-heston-fourier.md)
defines the parameter convention, affine exponent, numerical integrators and
inversion. In particular, the initial-variance exponent is `v0*integral R`,
not `v0*psi(T)` for a fractional kernel. No Greeks, calibration or composition
support is inferred from this new price-only method.

The branch is based on PR #121's source at
`8750d13aa9d302748ae0b23c124c9031d0d3982f` (tree
`c858296ba7e628fb8447113a5aae788e430edbd8`). No existing production path formula,
RNG layout, public MC API, default model parameter, schema or Cargo dependency
is changed. The new independent complex utility belongs to `pricing-numerics`;
financial Riccati/inversion code belongs to `pricing` and bindings to
`pricing-python`.

## Independent transform and price references

[`fourier.json`](../../fixtures/rough-volatility/fourier.json) retains inputs
and independently computed results. `scripts/check_heston_fourier.py` never
imports the pricing extension or production kernel helpers. Its default mode
only verifies retained values; the explicit `--generate` flag is for reviewed
regeneration, not CI repair.

The parameters are `v0=.04, kappa=.7, theta=.055, nu=.18, rho=-.65`.
Fractional log transforms use H=.1/.3, T=.25/1 and z=.5+.7i/2i/1+.7i:
12 complex values from a 70-digit fractional power series. The series coefficients
are formed recursively by matching powers `t^(n*alpha)` in the Riccati equation.
Orders 120 and 180 must differ by less than 1e-30 before a value is retained.
Rust's implicit product integrator at 2,048 steps is compared with these values
at absolute complex tolerance 5e-5. The series is used only within its checked
convergence region; no global convergence is claimed for arbitrary frequencies.

Eight Markov transform rows use T=.25/1 and z=.5+.7i/2i/1+.7i/.5+32i.
The ordinary Heston closed transform is cross-checked against an independently
implemented adaptive DOP853 ODE. A three-factor lift with weights [.2,.4,.5]
and rates [.1,1,8] is cross-checked with DOP853 and BDF integrations. Rust's
1,024-step result is compared with the retained values at tolerance 2e-5.

Twelve call prices use these two Markov models, the two maturities, F=100,
D=.97 and K=80/100/120. The independent generator uses P1/P2 inversion with
adaptive quadrature, not the production half-moment/Black-control/Simpson
formula. Cutoffs 200/300 must agree within 2e-7. The fixed Rust tolerance is
0.001 currency units; 1,024 time steps, 1,024 intervals and cutoff 192 are used.
The observed maximum absolute price gap was approximately 7.901e-8. These are
numerical reference values with stated cross-checks, not symbolic exact prices.

Additional controls cover H=1/2, zero and near-zero vol-of-vol, nonconstant
ordinary-Heston deterministic variance, zero maturity, moment-strip inputs,
conjugation, characteristic-function magnitude, put-call parity, zero-rate
and split factors, stiff lift rates and invalid inputs/work limits. Directly
returned transform identities at z=0/1 are not empirical martingale evidence.
The underlying complex root and exponential/power hat-weight mass checks are
separate unit tests. They do not certify all extreme-scale floating-point cases.

## Fourier numerical refinement

Four models are used: Rough Heston H=.1/.3, the explicit three-factor lift,
and the H=.1 20-factor lift with geometric ratio 2.5. At T=1 and K=80/100/120,
compare 1,024 to 2,048 Riccati steps while holding 512 frequency intervals and
cutoff 128 fixed. Then double cutoff to256 and intervals to 1,024, preserving
frequency spacing. The fixed comparison tolerance is 0.003 currency units.
The largest observed time-grid difference was about 1.382e-6; cutoff differences
were below the printed 1e-9 precision. Frequency-grid differences and retained
last-half envelope integrals are printed separately and required below 1e-5.

These checks do not prove continuous-time accuracy across all parameter
regimes. Neither reported diagnostic measures Riccati error or bounds the
omitted infinite-frequency tail. The high-precision fractional reference panel
checks selected low/moderate frequencies; no independent rough-Heston full
strike-surface price oracle is claimed.

## Public MC comparison and original failures

Use the unchanged public MC adapter for one-year ATM Europeans with F=100,
zero rates and no dividends, against the Fourier price at 2,048 steps and
cutoff 256. There are four models and seeds 91/1973: eight MC comparisons. The
fixed budgets, established before the first numerical run, are

```text
abs(MC price - Fourier price) + 4*MC standard_error <= 0.10
0 < MC standard_error <= 0.015
```

These are **price currency units, not IV basis points**. The standard error is
across 16 independent scramble means; antithetic pairs are averaged within a
scramble. It excludes path-discretization, full-truncation and model bias. No
paired SE is claimed between deterministic Fourier and MC estimates.

The initial configuration was 512 MC steps, 2,048 points per scramble,
16 scrambles and Brownian bridge with antithetic sampling: 65,536 evaluated
paths per row. Four of eight rows failed. The H=.1 seeds had combined values
0.212120833/0.161877693 and SEs 0.029323351 / 0.025647913. The 20-factor lift had
SEs 0.018080364/0.015457095, exceeding the SE cap even though its combined
values were below 0.10. The other four rows passed. Initial failure logs are
retained and are not presented as a successful acceptance run.

First rerun: increase **all** models to 16,384 points per scramble, keeping
16 scrambles, original point prefixes, seeds, grids, parameters, Fourier
settings and both budgets. This gives 524,288 evaluated paths per row. For
H=.1 at 512 steps, seed 91 retained gap 0.085264533 and SE 0.005479791:
combined 0.107183695 still exceeded 0.10. Seed 1973 gave gap 0.074114639,
SE 0.003818610 and combined 0.089389078. More samples did not eliminate the
coarse-grid discrepancy; it is not appropriate to treat the whole gap as
sampling error.

Second rerun: refine only H=.1 to 1,024 MC steps, preserving the increased sample
counts, seeds, all parameters and both error budgets. The other models retain
512 steps. The changed time grid changes the production random-coordinate
layout. Retaining the seed does not provide a common-Brownian coupling across
these separate grid runs, and no paired uncertainty between them is claimed.
The fixture records the final per-model grid settings explicitly. This is a
finite numerical comparison, not a guarantee of bias below 0.10 at all inputs.

## API, packaging and CI guards

The frozen Python plan/result classes have explicit stub-member, signature,
static-method and property contracts. The wheel example exercises both models;
ordinary Heston prices, fractional transforms, unsupported models and invalid
inputs are checked through the actual extension. Stub mutations removing either
class or changing the cutoff default must be rejected by the wheel checker.

Four independent-reference guard tests reject corrupted/nonfinite results,
missing rows and exact protocol/parameter mutations, and require all nine
workflow/evidence fragments plus the wheel example invocation. Thirteen new
source/reference/document members are explicitly required in the source archive.
The dedicated read-only workflow runs debug, minimal and release checks on
Linux/macOS/Windows, explicitly includes the ignored numerical panel, and
retains `heston-fourier.log`; missing artifacts are errors, not silent skips.

The local environment is Linux x86_64, pinned Rust/Cargo 1.98.1 and CPython 3.13.5.
The pinned compiler and locked vendor dependencies were restored from the
previously retained validation artifact and hash-verified. Cargo.lock and all
manifests remain unchanged. The optimized CPython 3.13 wheel was built offline;
all 169 Python tests and all wheel examples passed. The local offline smoke
environment inherits installed NumPy/system packages; remote CI still requires
normal clean wheel installation. Local success is not a claim of remote CI
success for a new commit.

Two preliminary compile mistakes (Option/Result mapping and a missing borrowed
float dereference in a test) were corrected before acceptance testing. No
lint suppression or tolerance change was used. Original failing build logs
are retained with the delivery alongside numerical failure logs.

## Executed fast regression results and outcome reporting

`cargo test --locked --workspace` completed with **665 passed, 0 failed,
50 ignored**. This does not claim execution of all pre-existing ignored
acceptance panels. The five new fast integration tests passed in debug and
no-default-features builds, and workspace/all-target/all-feature Clippy passed
with warnings denied. Formatting, schema and dependency-direction checks passed.
The committed source archive and Markdown links were checked; an initial
archive missed the newly written validation document, which was added without
weakening the archive guard. Four independent-reference guards also passed.

Individual release/MC run outcomes are recorded in the retained numerical
logs, not inferred from this protocol. The delivered report and PR description
identify the executed release configuration and its outcomes. Defining the
workflow does not assert that a remote workflow run has already succeeded.
The local wheel used inherited NumPy2.2.4; independent references used
NumPy2.3.5, SciPy1.17.0 and mpmath1.3.0 in the reference environment.

## Limits

No generalized-forward-variance initial curve, automatic error-controlled mesh,
Fourier calibration, Greeks/AAD, VIX/SSR, path-dependent Fourier payoff,
stochastic-rate measure change or cash-dividend payoff mapping is implemented.
Existing full-truncation MC bias remains a separate numerical issue; the new
method supplies a distinct comparison path rather than certifying that bias.
Resource limits and small diagnostic numbers do not constitute production
admission. Broad stressed parameter regions, short expiries/far-wing prices,
positive-leverage moment boundaries and overall performance remain further
validation work.
