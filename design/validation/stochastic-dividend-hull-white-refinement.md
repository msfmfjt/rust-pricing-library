# Stochastic-dividend Hull–White time-grid refinement

This protocol is fixed before candidate native execution. It extends the
fixed-grid price/AAD/Gamma checks without changing a production algorithm,
existing seed, acceptance budget or replay fixture. Parent: PR #98 at
`d8f464295a22617b0be017387dd5e2773368361e`.

## Independent discounted-moment reference

For maturity T, exponentially tilting by the integrated Gaussian short rate
changes the Brownian means by `-rho_jr eta(t) B_a(T-t) dt`. The normalized
discounted moments `m_f=E[D f]/P(0,T)`, `m_Y=E[D Y]/P(0,T)` therefore solve

```text
m_f' = -sigma rho_fr eta(t) B_a(T-t) m_f,
m_Y' = kappa alpha m_f + kappa(1-alpha)
       - (kappa + nu_D rho_Dr eta(t) B_a(T-t)) m_Y.
```

An independent fixed-step RK4 solver, split at every volatility knot, supplies
the continuous-time target. Compare 16,384 and 32,768 steps within `2e-12`.
At T=1, use kappa=0.7, alpha=0.6, nu_D=0.35, sigma=0.2,
rho_fD=-0.25, rho_fr=0.25, rho_Dr=-0.2, and a=0 / 0.4.
HW volatility knots are 0, 0.375, 0.75, 1.125 with values
0.04, 0.065, 0.025, 0.09.

Integrate the actual production Buehler one-step map under the tilted Gaussian
law using eight-point Gauss–Hermite quadrature in each independent f/Y normal.
Its affine dependence on incoming f/Y closes this first-moment recursion.
Use 16, 32, 64, 128, 256 steps. The last Y error must be below `5e-8`, with
successive error ratios above 3.5; the f error is below `2e-12` at every level.
The independently integrated continuous-time Y target must also match the
compiled collateral dividend forward divided by its Q mean within `2e-11`.
This is a deterministic weak-convergence check for these moments, without MC SE.

## Exact Gaussian coupling

Build four-coordinate step covariances by independent 64-panel Simpson
integration of the elementary Brownian/OU/integral kernels. A test-local
Cholesky maps standard normals into fine innovations. For a block of steps,
sum both Brownian increments and compose

```text
I <- I + B_a(h) x + epsilon_I,
x <- exp(-a h) x + epsilon_x.
```

Solve the coarse Cholesky system for its normal coordinates. Do not sum the
integrated-rate innovations alone: they also contain earlier rate states.
Check the independent covariance against the public transition within `2e-13`.
At shared nodes, f, x and I must agree within `3e-12`; at kappa=0, Y must also
agree within `3e-12`. Check positive/negative normal paths and a=0 / 0.4.

## Paired price, Delta and finite-bump Gamma panel

Use 4,096 antithetic MC units, seeds 421 and 1607, the five grids above, both
rate reversions, and cash means 4 at 0.5, 3 at 1, and 12 at 1.5. Compare a
post-cash European call at T=1 and an Asian call with post-cash observations
at 0.5 / 1 (weights 0.4 / 0.6), paid at 1.25. Both strikes are 100.
The knot at 1.125 and cash at 1.5 affect the reserve and delayed payment without
extending the simulated path beyond T=1.

From public physical paths evaluate the contractual payoff and its analytic
Spot derivative; evaluate central differences of that Delta at fixed Spot
widths 0.5, 1, 2. Every shifted scenario funds the same complete cash reserve.
All levels share the aggregated Gaussian innovations above. Average the two
antithetic signs before computing any means, paired differences or sample SE.

Emit JSON rows with absolute level estimates/SEs and each level's differences
against the 256-step reference. For the 128-minus-256 comparison require
`abs(paired_mean) + 4*paired_SE` below the fixed absolute budgets:

| Quantity | Budget |
| --- | ---: |
| Price | 0.02 |
| Delta | 0.002 |
| Each of Gamma(0.5), Gamma(1), Gamma(2) | 0.002 |

These are selected finite-grid consistency budgets. A noisy vanilla derivative
need not converge monotonically in one sample. The finest grid is not an exact
option oracle; no general convergence order, continuous-time Gamma accuracy,
zero-bump limit, smoothing-bias bound or parameter/correlation-risk acceptance
is claimed. Small paired SE or a zero crossing count does not bound bias.

## Execution

Run the two deterministic/coupling tests in ordinary discovery. Run the ignored
release panel with `cargo test --locked --release -p pricing --test
stochastic_dividend_hull_white_refinement -- --include-ignored --nocapture` in
the existing three-OS focused job, retaining its log as an artifact. Preserve
all inherited format/Clippy/workspace/wheel/source/docs/legacy gates. Record
candidate SHA and observed/pending results in the PR.

See the [model reference](../../docs/models/stochastic-dividends-hull-white.md).
