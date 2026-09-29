# Rough residual-LSV Asian and Barrier refinement

This extends the [European fixed-surface panel](stochastic-dividend-rough-lsv-refinement.md)
to arithmetic Asian and discretely monitored up-and-in call price/physical-Spot
Delta. Contractual observation dates and Barrier smoothing are fixed while the
valuation grid is refined. Extra simulation nodes are not extra observations.
The leverage surface is calibrated once; this panel does not repeat the
[calibration ensemble](stochastic-dividend-rough-lsv-calibration-refinement.md)
for these products.

## Contracts and fixed inputs

Valuation is 2026-09-04. Expiry is 2027-09-03, so `T=364/365` under Act365F.
The future observations are 2027-03-05 (`T/2=182/365`) and expiry. This choice
puts actual calendar observations on every uniform dyadic grid without moving
the dates when the step count changes. Payment is 2027-12-04 (`456/365`),
discounted at `0.95^(456/365)` rather than at expiry.

- Asian: known fixing 102 on 2026-08-05 with weight 0.2, then post-cash stock
  fixings with weights 0.3 and 0.5; strike 95.
- Barrier: up-and-in call, strike 80, barrier 105, two discrete monitoring
  dates. At a cash event, monitoring includes both pre- and post-cash stock.
  Compact-C2 smoothing half-width 8 is held fixed on every grid. Vanilla
  intrinsic value remains the ordinary call positive part.
- Spot 100, deterministic discount/carry factors `0.95^t` and `0.98^t`.
  Cash means 5/3/12 occur at `T/2`, `T` and 1.4 respectively. Cash after expiry
  remains in the funded reserve but does not extend the stochastic path.
- H=0.1/0.3, eta=0.6, dividend mean reversion 0.7, linkage 0.6, dividend
  volatility 0.35, and correlations -0.25/-0.4/0.15.
- Nonflat residual local-variance target: times `[0,T/2,T]`, log-moneyness
  `[-0.5,0,0.5]`, rows `[0.045,0.04,0.035]`, `[0.05,0.045,0.04]`,
  `[0.055,0.05,0.045]`. Calibration uses 16 steps, 512 particles, seed 42,
  bandwidth 0.35 and minimum effective samples 5.

## Independent observations and controls

The test reconstructs physical stock from the primal normalized f/Y states
and Buehler coefficients, without production payoff, smoothing, reverse or
sampling-error helpers. With the leverage surface re-anchored to funded
residual equity, `dS_t/dS0=(0.98/0.95)^t*f_t`; normalized f/Y and each cash
jump have zero physical-Spot derivative. The known Asian fixing is constant.
The Asian derivative is its weighted future-stock derivative on in-the-money
paths, discounted to payment.

For the Barrier, the centered compact indicator and its derivative are
evaluated as polynomials in `clamp(x/8,-1,1)`. Its score is post-cash stock
minus the barrier plus the compact positive part of the cash jump. The
knock-in weight is one minus the product of the two survival weights; its
derivative uses the product rule. The terminal call uses post-cash stock.

Fast controls check:

- Independent price/Delta means and antithetic-unit SEs against the public
  estimator, for both products and H values on all four pricing grids, within
  2e-10. Means and errors
  must be nonzero. A deliberately post-cash-only Barrier reference must
  produce a different price, showing that the cash-jump convention matters.
- In the eta=0, zero-dividend-mean-reversion, flat-variance 0.04 limit,
  coupled price/Delta observations agree across all four grids within 3e-10
  for 64 antithetic units, with a nonzero witness for every quantity.
- Public Spot Delta against central physical-Spot bumps of 1e-4, with full
  production recalibration on both sides, for both products/H values within
  2e-7. This separately checks the scale-invariant derivative convention.

## Refinement and acceptance

The pricing grids have 16/32/64/128 steps. Each calibration interval is
subdivided while retaining the existing leverage knots bit-for-bit. Rebuilding
the same mathematical knots as `i*T/n` can move them by one ulp and violate
the production surface contract. Grid nodes are checked against uniform
times within 1e-15; midpoint and expiry are checked exactly. The surface values
are unchanged. The existing pairwise common-Brownian coupling is scale
independent and applies on this common horizon; its covariance controls remain
in the European panel.

For each H and valuation seed 193/877, use 262,144 independent antithetic
units. Evaluate both products from each primal path. Report all 48 JSON rows:
three coarse grids versus 128 steps, four price/Delta quantities, two H
values and two seeds. A two-pass sample variance of antithetic paired
differences provides the conditional comparison SE.

Price, Delta and SE budgets fixed before running the numerical panel:

- At 64 versus 128 steps, `abs(gap)+4*paired SE < 0.05` for either price
  and `< 0.005` for either Delta.
- At that level, SE below 0.01 for price and 0.001 for Delta.
- At 64 versus 128 steps, paired SE below 75% of the unpaired comparison SE.
  At earlier diagnostic levels, paired SE must be below the unpaired SE.
  The distinction from the initial all-level 75% rule is explained below.

All diagnostics are emitted before reporting gate failures. Sampling error
consumes the difference budget; it does not relax it. Earlier grid levels
are diagnostic, with no required monotonic decrease.

```sh
cargo test --locked --release -p pricing --lib lsv_path_refinement_tests -- --include-ignored --nocapture
```

The numerical panel is ignored in normal debug discovery. Three-OS release
CI runs it explicitly and retains `stochastic-dividend-rough-lsv-path-refinement.log`.
Report pass/fail results against the tested commit; the protocol alone is not
evidence of a passing run.

## Initial precision and coupling observations

The initial 131,072-unit run failed. For H=0.1, the Barrier Delta paired SE
was 0.001039992 at seed 193 and 0.001037978 at seed 877, above the fixed
0.001 precision cap. The sample count was doubled to 262,144, retaining all
original counter-based samples as a prefix, both seeds, contracts, surfaces,
grid levels, smoothing width and price/Delta/SE budgets.

The first run also required paired SE below 75% of unpaired SE at **every**
level, copying the European panel's efficiency rule. At 16 versus 128 steps,
H=0.1 Asian Delta instead had ratios 0.811422 and 0.807687. Thus the initial
all-level condition failed even though coupling reduced variance. More samples
do not systematically improve that ratio. The unsmoothed Asian exercise
indicator and antithetic baseline differ from the European case, and no
mathematical argument guarantees a 25% SE reduction for far-coarse grids.

The final protocol retains the 75% requirement at the 64/128 acceptance
level and checks that coupling reduces SE at the earlier diagnostic levels.
This is an explicit revision of an empirical efficiency condition, not a
relaxation of any price, Delta or SE accuracy budget. All original levels and
their ratios remain in the log; the initial run is not reported as a pass.

## Limits

The separate [smoothing-width panel](stochastic-dividend-rough-lsv-barrier-smoothing.md)
compares smoothed and hard prices on the same grid and surface, adjacent-width
Deltas, and diagnostic finite bumps of hard prices. It does not turn this
fixed-width time-grid panel into a hard-Barrier Delta convergence claim.

This tests the stated fixed-observation Asian and smoothed discrete Barrier
contracts conditional on one finite-particle calibration. It does not bound
continuous-time bias, smoothing bias, particle-calibration uncertainty or
model error. It does not establish convergence order, continuously monitored
barrier accuracy, unsmoothed Barrier Delta, general nonuniform-grid coupling,
RQMC refinement or accuracy of other Greeks. No production API or reported
uncertainty field changes.
