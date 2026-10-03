# Heston Fourier parameter-risk validation

## Scope and frozen protocol

Parent: PR #125, `575956bda604ee597524713c3397708bb9fcafbe`.
Five first derivatives at fixed kernel/F/K/D/T; no Hurst or lift-factory risk,
calibration, implied-volatility Vega, VegaKT, parameter Hessian or reverse AAD.

Fixtures in `fixtures/rough-volatility/parameter-risk.json` record every input,
parameter order and acceptance budget. These were fixed before the numerical
Rust comparisons. `scripts/check_heston_parameter_risk.py` regenerates references
only with `--generate`; its default checks retained data without overwriting it.

All price cases use F=100, D=.97, v0=.04, kappa=.7, theta=.055, nu=.18, rho=-.65.
The explicit lift has weights [.2,.4,.5] and rates [.1,1,8]. Maturities are .25/1,
strikes 80/100/120. Ordinary Heston is the H=.5 boundary, not an unrelated model.

## Independent references

Twelve Markov price rows (60 scalar derivatives) use adaptive DOP853 integration
of the continuous-time finite-factor system and its variational equations.
P1/P2 inversion integrates on real exponents zero and one using Gauss-Legendre
quadrature. This is not production implicit product integration, half-moment
inversion, Black-control differentiation or Simpson quadrature. The reference
script never imports the pricing extension.

The independent references cross-check order/cutoff 160/192 against 240/256
(with absolute derivative tolerance 2e-7), and each of the five derivatives
against centered differences of independently evaluated prices at two bumps,
1e-5 and 5e-6 (tolerance 2e-5). The finite differences use only the primal ODE,
not its variational equations. They do retain finite-bump/integration error.

Twelve fractional transform rows (60 complex derivatives) use mpmath 60-digit
fractional power series at orders 100 and 140 (agreement better than 1e-25),
differentiated at high precision. H=.1/.3, T=.25/1, z=.5+.7i, 2i, 1+.7i.
This covers selected moderate frequencies, not a complete rough price surface.

Rust acceptance: absolute price-derivative gap <=2e-4 and complex
log-transform-derivative gap <=5e-4, in their natural per-parameter units.
These are not IV basis points. Independent Markov Greek references do not
constitute an independent full rough-Heston strike-surface Greek reference.

## Finite refinement and fast checks

Eighteen price cases / 90 derivatives compare Riccati512/1024 at cutoff128 and
512 frequency intervals, then cutoff128/256 at 1024 time steps with fixed
frequency spacing .25 (512/1024 intervals). Models are rough H=.1/.3 and the
explicit lift, with both maturities and all strikes. The separate time/cutoff
absolute budgets are 5e-4, fixed before the first numerical run.

Fast tests check two production bump sizes, both call/put, transform tangents,
exact affine exponent directions, zero maturity including ATM, zero-control
rejection, moment-strip inputs and identities, conjugacy, inward boundary
sensitivities, homogeneity, discount scaling, independently compiled coarse
quadrature, and same-driver factor splitting. A unit test enforces the tangent
work budget before allocating the frequency grid. Neither internally enforced
moment identities nor passing numerical panels is a martingale certificate.

The new read-only three-OS workflow explicitly includes both ignored numerical
tests in release, retains `heston-parameter-risk.log`, and errors on missing
artifacts. Guards reject corrupt references, changed inputs/thresholds/order,
missing workflow gates, missing example execution and absent archive files.
Python tests check frozen properties, bindings, independent prices, bump risk,
deterministic Black variance risk and required stub members.

## Interpretation

Tangent propagation differentiates the selected discrete implicit solution;
continuous-time sensitivity still has Riccati and Fourier numerical error.
Coarse/fine differences and retained tail envelopes do not bound total error.
The documented finite comparisons do not certify short-expiry/far-wing/stress
or moment-boundary regimes. Initial-zero-variance and kernel sensitivities
remain excluded. Do not treat these as recalibrated market risk or model
admission evidence beyond the specific tested cases.

Executed results and exact publication provenance are retained in the delivery
report and PR description; creating a CI workflow is not a claim that it passed.

## Initial precision failure and retained rerun

The original 256/512-step refinement exceeded the unchanged 5e-4 time-difference
budget at H=.1, T=1, ATM for the initial-variance derivative (7.552773660677e-4).
The first run stopped at that row. A full diagnostic rerun retained every row
and counted the failures without changing models, grids or numerical budgets.
The final protocol refines the time pair to 512/1024 **for all 18 cases**, not
only the failing scenario; both cutoff values and fixed frequency spacing,
parameters, strikes, maturities and all tolerances are unchanged. Reference
values were not regenerated or adjusted to match production. Both initial logs
and the initial fixture/test snapshot are retained in the delivery. A passed
finer comparison does not retroactively make the original panel a pass.
