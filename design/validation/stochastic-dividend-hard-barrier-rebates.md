# Fixed cash rebates for hard Barrier Spot risk

The [hard Barrier Spot-risk method](stochastic-dividend-hard-barrier-spot-risk.md)
accepts the existing contract's optional positive fixed cash rebate. It pays
when the vanilla branch is inactive, at the contractual payment date. The
rebate is independent of notional: notional=2 and rebate=7 pays 7 on the
inactive branch. An active out-of-the-money option pays zero, not a rebate.
There is no payment at first hit.

## Estimator and derivative

Let V be the coupled vanilla conditional payoff, Q the survival-weighted
vanilla payoff, P the survival probability, N notional, R the fixed rebate,
and D the deterministic payment-date discount. The estimators are

```text
knock-out = D * (N*Q + R*(1-P)),
knock-in  = D * (N*(V-Q) + R*P).
```

Each conditional path accumulates survival and its Spot derivative through
all observations. At expiry the digital probability is integrated separately
from the exercise interval. An empty exercise interval therefore zeros only
the vanilla payoff; it must not discard a rebate. With no terminal observation,
the accumulated survival probability still determines the rebate.

For each monitored step, the first-hit mass is the survival weight entering
that step multiplied by the complementary conditional probability. The code
sums these disjoint masses directly, including the terminal step. This avoids
losing a rare-hit rebate when the survival CDF rounds to one. The hit tangent
is the negative survival tangent. No probability floor or clipping is added.

Payoff and rebate are combined before MC antithetic-pair or RQMC scramble
reduction, preserving covariance in both price and Delta errors. In/out
parity includes exactly one discounted rebate, with cancelling rebate Deltas.
The method label remains `buehler-rough-residual-lsv-hard-barrier-survival-spot-v2`;
the compiled fingerprint already includes the contract rebate.

## Controls and independent references

Rust controls cover both rebate-free and rebate-bearing contracts in the
full-recalibration Spot-bump matrix: H=0.1/0.3/0.5, Up/Down, Call/Put,
knock-in/out, terminal/nonterminal observation and two bump widths. They also
check fixed cash versus notional scaling, in/out parity, empty exercise
intervals, rare-hit probabilities below machine epsilon, explicit MC errors,
antithetic sampling and worker replay.

The independent NumPy code uses the same retained discrete law but no
production paths, payoffs, derivatives or random generator. Direct Gaussian
payoff quadrature in a deterministic-dividend initial/final-monitor limit checks all
eight contract variants, with (N,R)=(1,0)/(2,7) and strikes 1/80/110/1000.
Richardson-extrapolated price bumps check Delta within 8e-9; the large strike
range includes empty active exercise regions. Common-input nonzero-eta/kappa
bumps additionally check analytic path tangents within 2e-6.

The [rebate fixture](../../fixtures/stochastic-dividends/rough-barrier-rebates-reference.json)
retains sixteen cases: H=0.1/0.3 times all eight direction/side/style variants,
N=2 and R=7. The eight-step grids, four monitoring dates, nonflat leverage,
market and payment at day 456 come from the existing
[survival reference](stochastic-dividend-survival-barrier-reference.md).
Up Calls use B=105, K=80; Up Puts use B=105, K=110; Down Calls/Puts use
B=95, K=100. Cash means are 5/3/12, including funded cash beyond expiry.

Before generating the new references or running production comparisons, fix
PCG64 seed 20261003, 32 independent batches of 32,768 antithetic pairs per
case, and Rust seeds 193/877 with 16 scrambles of 16,384 points, antithetics
and Brownian bridge. There are 64 production price/Delta comparisons:

```text
combined_SE = hypot(reference_batch_SE, production_scramble_SE),
abs(error) + 4*combined_SE < 0.10 for price, < 0.04 for Delta;
reference_SE < 0.015 for price, < 0.003 for Delta;
production_SE < 0.010 for price, < 0.006 for Delta.
```

These sampling budgets scale the earlier notional-one contract panel's
budgets by two. They do not bound time-grid bias or calibration uncertainty.
Existing rebate-free panels retain their original budgets and batch values.
All 64 new comparisons passed on the first numerical run. Maximum
abs(error)+4*combined_SE was 0.044258073 for price and 0.003920732 for Delta;
maximum production SEs were 0.003267850/0.000674087, and maximum reference
SEs were 0.009362144/0.000558680. No counts, seeds or budgets were changed.

```sh
python -m unittest discover -s tests/python -p test_rough_dividend_barrier_styles.py -v
python tests/python/rough_dividend_barrier_styles.py rough-barrier-rebates-reference.json
cargo test --locked --release -p pricing --test stochastic_dividend_barrier_reference production_hard_barrier_rebates -- --include-ignored --nocapture
```

Linux CI regenerates every retained batch within 1e-9 and stores 32 aggregate
rows in `stochastic-dividend-barrier-rebates-reference.log`. Its reference job
allows 30 minutes for the expanded panel. Three-OS Barrier integration emits
212 comparison rows across six numerical panels. Source archives require the
fixture, documentation and regeneration command. Historical hit states,
continuous monitoring, other hard-payoff Greeks and automatic request-level
risk dispatch remain outside this method. Python exposes the dedicated
method; see the [API contract](stochastic-dividend-hard-barrier-spot-risk.md#python-api).
