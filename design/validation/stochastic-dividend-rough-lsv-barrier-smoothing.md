# Rough residual-LSV Barrier smoothing width

The [Asian/Barrier grid panel](stochastic-dividend-rough-lsv-path-refinement.md)
holds smoothing half-width 8 fixed. This panel instead fixes both the
calibrated leverage surface and the 128-step valuation grid while changing
half-width through 8, 4, 2, 1 and 0.5. It separates smoothing effects from
time-grid and calibration changes.

## Contract, samples and estimands

Use the same two-date, up-and-in call: Spot 100, strike 80, barrier 105,
valuation 2026-09-04, monitoring 2027-03-05 and 2027-09-03, payment 2027-12-04.
Cash means 5/3/12 occur at `182/365`, `364/365` and 1.4. Monitor both sides
of the first two cash jumps and use post-cash terminal intrinsic value.
Discount to payment at `0.95^(456/365)`, with carry `0.98^t`.

H is 0.1/0.3, eta 0.6, dividend mean reversion 0.7, linkage 0.6, dividend
volatility 0.35, and correlations -0.25/-0.4/0.15. The nonflat residual
local-variance target and 16-step/512-particle calibration (seed 42,
bandwidth 0.35, minimum effective samples 5) are unchanged from the grid
panel. The 128-step path retains every calibration knot bit-for-bit.

Each H/valuation-seed combination uses 1,048,576 independent antithetic
units, with seeds 193 and 877. A unit contains two opposite-normal paths;
all widths, the hard payoff and the finite Spot bumps reuse those same
paths. The test computes two-pass sample variances on the unit-level
estimates or paired differences, not on individual correlated signs.

For every width report:

- Smoothed price and physical-Spot Delta, and their conditional SEs.
- Hard, unsmoothed price and its SE on the same finite grid and surface.
- Smoothed-minus-hard price and its paired SE.
- Except for the first width, Delta minus the previous wider-width Delta,
  with its paired SE. An individual Delta SE and a paired-difference SE
  describe different estimands and must not be substituted for each other.

The hard payoff is an unsmoothed **price** reference for this finite
simulation scheme. It is not a continuous-time price oracle. No hard
pathwise Delta is computed: differentiating only the vanilla payoff on
active paths would omit the barrier-crossing contribution.

## Independent payoff and finite-bump controls

Stock and its physical-Spot derivative are reconstructed from the primal
normalized f/Y states. As in the grid panel, normalized states and cash
jumps are invariant to physical Spot with the leverage surface re-anchored
to funded residual equity. Thus a finite physical-Spot shift b changes each
stock observation by exactly `b*(0.98/0.95)^t*f_t` for the finite algorithm.

The smoothed jump score is post-cash stock minus 105 plus the compact
positive part of the cash jump. Both the maximum and indicator use the
selected half-width. Centered polynomials implement these functions and
their derivatives independently of the production smoothing and payoff
helpers. The hard score uses the ordinary positive part; a hard hit at
either monitoring date activates the post-cash call.

Three fast controls check:

1. Price, Delta and antithetic SEs for all five widths against the public
   estimator on the 128-step grid, for both H values, within 2e-10. Leverage
   values must be identical across widths. The hard price and SE also match;
   requesting hard Spot risk is rejected without changing the price.
2. Smoothed derivatives against central differences on synthetic paths with
   positive, zero and negative cash jumps; exterior values agree exactly
   with the hard payoff. This exercises both branches of the jump maximum.
3. Hard prices and SEs under Spot shifts +/-0.5 and +/-0.25 against full
   production recalibration at the shifted Spot, within 2e-10.

The long panel also reports central differences of the hard **prices** at
Spot bumps 0.5 and 0.25, and the paired difference from the width-0.5 Delta.
These are finite-bump diagnostics, not unbiased hard Delta references.
Neither agreement nor a small reported SE removes finite-bump bias, so
these diagnostics do not act as a hard-Delta accuracy gate.

## Numerical gates and execution

The following budgets are set before the first numerical run. They apply
separately to both H values and both seeds:

- At half-width 0.5, `abs(smoothed price - hard price) + 4*paired SE < 0.02`,
  with paired SE below 0.004.
- For the Delta change from width 1 to width 0.5,
  `abs(Delta change) + 4*paired SE < 0.01`, with paired SE below 0.002.

Earlier widths are diagnostic. No monotonic bias/SE decrease or convergence
order is required. These are width-comparison budgets, distinct from the
grid panel's time-step budgets. They cannot be combined into a general
continuous-time accuracy claim. In particular, a small adjacent-width Delta
difference does not bound its remaining bias relative to hard Delta.

The test emits 20 width rows and eight finite-bump rows before reporting any
numerical-gate failures. Run all checks with:

```sh
cargo test --locked --release -p pricing --lib lsv_barrier_smoothing_tests -- --include-ignored --nocapture
```

The long panel is ignored in normal debug discovery. A separate three-OS
release CI job retains `stochastic-dividend-rough-lsv-barrier-smoothing.log`,
so this sample count does not extend the existing risk/refinement job's time
limit. Report actual outcomes against the tested revision.

## Initial local acceptance observation

The initial release panel passed all predeclared gates at the stated sample
count, with no changes to widths, seeds or budgets. The observed price
differences on the fixed grid/surface were:

| H | Seed | Width-8 price minus hard price | Width-0.5 price minus hard price | Width-0.5 paired SE |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 193 | 0.133221022 | 0.000574332 | 0.000400170 |
| 0.1 | 877 | 0.131465219 | -0.000036129 | 0.000396353 |
| 0.3 | 193 | 0.140386983 | 0.000844831 | 0.000405511 |
| 0.3 | 877 | 0.139090182 | 0.000244944 | 0.000401717 |

Maximum `abs(gap)+4*paired SE` was 0.002466874 for width-0.5 price versus
hard price and 0.007709262 for the width-1 to width-0.5 Delta change. The
respective maximum paired SEs were 0.000405511 and 0.001266176.
The narrowing-width Delta comparisons became noisier in these samples.
This confirms a material smoothing effect at width 8 despite the earlier
panel's small time-grid differences. It does not establish hard-Delta bias
or identify a universally suitable smoothing width.

This panel changes no production API, smoothing default, calibration or
reported uncertainty field. The SE remains conditional on the fixed particle
calibration and grid. General barrier directions/styles, continuous monitoring,
other risks, smoothing of other payoff types, RQMC and calibration uncertainty
are outside this panel.
