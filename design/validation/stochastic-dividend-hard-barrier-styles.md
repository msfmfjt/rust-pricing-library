# Hard Barrier directions and option sides

The production [hard Barrier Spot-risk method](stochastic-dividend-hard-barrier-spot-risk.md)
supports discrete Up/Down and Call/Put for both knock-in and knock-out,
without rebates. This panel validates the added Up Put, Down Call and Down
Put combinations. Existing Up Call reference gates are retained unchanged.

## Cash ordering and conditional intervals

At each monitored node the symmetric dividend split gives `Y_next=u+link*f`.
With reconstruction coefficients `(a,b,c)` and event cash mean `cash`, stock is

```text
S_post = (a+b*link)*f + b*u + c,
S_pre  = (a+(b+cash)*link)*f + (b+cash)*u + c.
```

Positive cash makes pre-cash stock the maximum and post-cash stock the minimum.
Up survival therefore requires `S_pre < B`, giving an upper f cutoff; Down
survival requires `S_post > B`, giving a lower f cutoff. A nonpositive cutoff
means impossible Up survival and unrestricted Down survival. At unmonitored
cash dates neither direction introduces a Barrier observation.

For positive lower cut `b`, Down survival conditions the free equity normal
on `Z > (log(b)-mu)/s`. Reflecting `Z` changes this into upper truncation of
`-Z`. Both survival-probability and normal-transport tangents include this
sign change. The reflected input is used for antithetic samples as well.

At expiry, intersect the surviving f interval with the exercise interval.
Calls integrate above the exercise cutoff; Puts integrate below it. Empty
intervals contribute zero price and tangent. Calls use upper lognormal tail
moments, Puts use lower moments, avoiding subtraction of nearly complete
moments for an out-of-the-money Put. The calculation differentiates both
endpoints and the state-dependent conditional mean and scale.

The estimator label is now
`buehler-rough-residual-lsv-hard-barrier-survival-spot-v2`. Result fingerprints
include that label, so they also change for Up Calls. This identifies the
revised terminal interval calculation; the compiled price-only plan and its
ordinary valuation fingerprint are unchanged.

## Independent reference and controls

The [fixture](../../fixtures/stochastic-dividends/rough-barrier-styles-reference.json)
retains twelve contracts: H=0.1/0.3, three added direction/side pairs, and
knock-in/out. It links the retained eight-step nonflat surfaces and four
monitoring dates from the [survival reference](stochastic-dividend-survival-barrier-reference.md).
All have eta=0.6, kappa=0.7, dividend volatility=0.35, equity linkage=0.6,
correlations SD=-0.25, SV=-0.4, DV=0.15, Spot=100, cash means 5/3/12 and
payment at day 456. Up Puts use B=105, K=110; Down Calls/Puts use B=95, K=100.
Cash is paid on the second and fourth observations; the last funded cash
mean is beyond expiry.

The independent NumPy implementation extends the earlier survival reference
with direction, side and style arguments. Default Up Call behavior retains
the existing batch values within 1e-9. It does not call production transition,
payoff, derivative, RNG or reduction code. Calibration correctness remains
outside its scope: the surfaces are retained inputs.

The [Python controls](../../tests/python/test_rough_dividend_barrier_styles.py) check:

- All eight direction/side/style combinations in a deterministic-dividend,
  single-monitor limit against direct Gaussian payoff quadrature split at the
  Barrier and exercise boundaries. Price differences of this separate
  quadrature check Delta, within 4e-9. It uses no truncated-moment formulas.
- Nonzero-eta/kappa per-sample analytic Delta against two common-input price
  bumps for all twelve retained cases, within 2e-6.
- Retained batch means, aggregate SEs and reference precision caps.

Production controls extend the full-recalibration Spot-bump checks to all
eight combinations at H=0.1/0.3/0.5, with and without terminal monitoring.
Knock-in/out parity and notional scaling cover all direction/side pairs.
A further test distinguishes impossible Up survival from unrestricted Down
survival when the equity cutoff is nonpositive. Rebate, historical/initial
monitoring, smoothing and degenerate-conditioning rejection remain covered.

## Sampling and acceptance

Each independent reference uses PCG64 seed 20261003 and 32 batches of 32,768
antithetic pairs: 2,097,152 signed paths. Batch means, price, Delta and their
SEs are retained. Equal-dimension cases reuse the same random inputs, so their
errors are correlated. The SE for each case is calculated across its own
32 independent batch means.

Rust uses 16 independently randomized Sobol scrambles, 16,384 points each,
antithetics and Brownian bridge at seeds 193/877. Each scenario evaluates
524,288 signed samples. The public integration test recompiles and checks the
retained grid and leverage before comparison. For each of the 48 result rows:

```text
combined_SE = sqrt(reference_batch_SE^2 + production_scramble_SE^2),
abs(error) + 4*combined_SE < 0.05 for price, < 0.02 for Delta;
reference_SE < 0.0075 for price, < 0.0015 for Delta;
production_SE < 0.005 for price, < 0.003 for Delta.
```

These counts, cases, seeds and budgets were set before generating the new
references or running the Rust comparison. All 48 comparisons passed on the
first numerical run. Maximum price/Delta margins were 0.023707561/0.002092409;
maximum production SEs were 0.001745704/0.000384682. These are empirical
sampling-error gates, not certified errors across all parameter regimes.

```sh
python -m unittest discover -s tests/python -p test_rough_dividend_barrier_styles.py -v
python tests/python/rough_dividend_barrier_styles.py
cargo test --locked --release -p pricing --test stochastic_dividend_barrier_reference production_hard_barrier_directions_and_sides -- --include-ignored --nocapture
```

The [regeneration command](../../tests/python/rough_dividend_barrier_styles.py)
verifies every retained batch within 1e-9 and emits 24 aggregate price/Delta
rows. Linux CI retains them in `stochastic-dividend-barrier-styles-reference.log`.
The three-OS Barrier job's five numerical panels emit 148 rows, including the
48 new comparisons. Source archives require the implementation, controls,
fixture, documentation and CI regeneration command.

Rebates, continuous monitoring, initial/historical hit states, other Greeks,
Python/request-level dispatch, calibration uncertainty and continuous-time
accuracy remain separate work.
