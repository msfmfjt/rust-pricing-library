# Fixed-surface hard Barrier time-grid refinement

This panel extends the [multi-step survival reference](stochastic-dividend-survival-barrier-reference.md)
to 16/32/64/128 evolution steps. For each H=0.1/0.3, it keeps the same
retained eight-step calibration and four contractual observation dates. Only
the evolution grid changes. Both price and analytic physical-Spot Delta use
the independent hard-payoff survival estimator, with no indicator smoothing.

The comparison is against the finite 128-step law. It does not establish a
continuous-time error bound or a convergence rate.

## Fixed inputs

The [configuration](../../fixtures/stochastic-dividends/rough-hard-barrier-refinement.json)
selects `h01_n8_four` and `h03_n8_four` from the retained survival fixture.
Each case has eta=0.6, dividend mean reversion kappa=0.7, equity linkage=0.6,
dividend volatility=0.35 and correlations SD=-0.25, SV=-0.4, DV=0.15.
The market and contract remain Spot=100, strike=80, up-and-in barrier=105,
cash means 5/3/12, four monitoring dates at days 91/182/273/364 and payment
at day 456. Cash-date monitoring uses pre-cash stock; terminal intrinsic
value uses post-cash stock. The last cash date remains beyond expiry.

Subdividing each original interval retains its endpoint exactly and repeats
its left-constant leverage row. Neither the surface's time dependence nor its
log-equity interpolation is recalibrated. The terminal row and observation
times are retained. The Python reference uses repeated rows; the Rust control
keeps the original surface object on the finer path grid and checks its knots
and values are unchanged.

## Raw Gaussian coupling and conditional paths

The [independent implementation](../../tests/python/rough_dividend_hard_refinement.py)
draws fine-grid independent normals in dividend, orthogonal-volatility,
newest-cell residual and free-equity order. Dividend, volatility and
free-equity Brownian increments aggregate by summing each block of b cells
and dividing by sqrt(b). A coarse newest-cell residual also needs a
covariance-preserving projection onto the fine coordinates; merely summing
fine residuals would give the wrong coarse/fine Volterra covariance.

In units where the fine cell has length one, let

```text
p = H + 1/2, a = sqrt(2H)/p, r = (1/2-H)/p,
C_H(k) = 2H * integral_0^1 x^(H-1/2)*(k+x)^(H-1/2) dx.
```

For a fine cell at distance d from the coarse endpoint, its covariance with
the coarse newest integral is `a*(d^p-(d-1)^p)` for the volatility Brownian
increment and `C_H(d-1)` for the fine newest integral. Projecting out the
coarse Brownian average determines the residual's fine-normal coefficients.
An additional independent normal supplies the remaining conditional variance.
The implementation rejects invalid residual variance rather than flooring it.
The H=1/2 limit has no newest-cell residual contribution to evolution.

The cross-kernel integral uses Gauss-Legendre quadrature after the substitution
`x=u^(1/H)`. This is independent of the earlier Rust coupling's Simpson rule
and change of variable. Nonuniform base grids and non-dyadic coupling ratios
are rejected.

This construction gives the common-Brownian joint Gaussian law for each
coarse/fine input pair. Survival conditioning subsequently transforms equity
inputs separately on each grid, so the conditioned paths themselves are not
one common Brownian path. Each coarse level uses a separate extra residual;
the joint law between different coarse levels is not used to fit convergence
rates or extrapolate. All reported differences are paired with the same
128-step reference.

## Controls

The [Python tests](../../tests/python/test_rough_dividend_hard_refinement.py) check:

- Cross-kernel integrals at two quadrature orders against retained independent
  constants, and the complete linear-map covariance for H=0.01/0.1/0.3/0.49/0.5
  and block sizes 1/2/4/8.
- Exact preservation of surface rows, original knots and monitoring times;
  128-step analytic per-sample Delta against common-input price differences.
- Rejection of nonuniform coupling; regeneration of all retained state paths.
- The flat-surface, zero-eta/kappa limit at 2/4/8 steps against the separate
  bivariate price/Delta reference, including sampling error.

The [Rust control](../../crates/pricing/src/engine/risk/stochastic_dividends/lsv_hard_refinement_tests.rs)
maps the dividend-first normals to production's equity-first Cholesky order
and compares the unconditioned f/Y states with the
[independent retained checkpoints](../../fixtures/stochastic-dividends/rough-hard-barrier-paths.json).
There are 24 paths (two H values, four grids, three deterministic input
patterns) and four observations per path: 192 state comparisons within 2e-11.
This checks the production transition law on every retained refinement grid;
it is not a production hard-Delta estimator.

## Sampling, acceptance and results

For each H and seed 193/877, NumPy PCG64 generates 32 independent batches of
8,192 antithetic pairs: 524,288 paths per grid. A pair uses opposite normals
and complementary equity uniforms. The paired SE is the sample standard
deviation of the 32 batch differences divided by sqrt(32). Unpaired SE is
also reported, with no efficiency-ratio acceptance requirement. Equal seeds
across H values reuse inputs; these scenarios are not independent replicas.

For every H/seed, the 64-versus-128 comparison must satisfy

```text
abs(price gap) + 4*paired SE < 0.05, paired SE < 0.01;
abs(Delta gap) + 4*paired SE < 0.01, paired SE < 0.002.
```

The 16/32-step comparisons are retained diagnostics. Four-SE margins are
empirical Monte Carlo acceptance budgets, not certified error bounds.

An initial feasibility run with 16 batches of 1,024 pairs had insufficient
precision: H=0.1, seed=193 gave price/Delta margins 0.107324/0.029962 at
64 versus 128, with SEs 0.019035/0.006677. It is not reported as a pass.
The final sampling counts were fixed before the main run; seeds, scenarios
and accuracy gates were unchanged. Changing the array dimensions changes
the random-number layout, so this is not described as retaining a sample
prefix. The earlier 4/8-step comparison gates are unchanged.

The completed panel on 2026-10-03 passed all eight acceptance rows:

| H | Seed | Quantity | 64 minus 128 | Paired SE | Absolute gap + 4 SE |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.1 | 193 | Price | 0.00840357 | 0.00620534 | 0.03322495 |
| 0.1 | 193 | Delta | 0.00006401 | 0.00133956 | 0.00542225 |
| 0.1 | 877 | Price | -0.00040975 | 0.00677520 | 0.02751056 |
| 0.1 | 877 | Delta | -0.00000383 | 0.00135639 | 0.00542940 |
| 0.3 | 193 | Price | 0.00105852 | 0.00298367 | 0.01299321 |
| 0.3 | 193 | Delta | -0.00012982 | 0.00101149 | 0.00417580 |
| 0.3 | 877 | Price | -0.00079786 | 0.00234385 | 0.01017326 |
| 0.3 | 877 | Delta | -0.00000105 | 0.00100904 | 0.00403722 |

The earlier-grid differences are not monotonic; no order is inferred from
them. The 64/128 paired-to-unpaired SE ratios were 0.153-0.413 for price and
0.437-0.691 for Delta. These are observed efficiencies for this panel.

```sh
python -m unittest discover -s tests/python -p test_rough_dividend_hard_refinement.py -v
python tests/python/rough_dividend_hard_refinement.py
cargo test --locked -p pricing --lib lsv_hard_refinement_tests
cargo test --locked -p pricing --no-default-features --lib lsv_hard_refinement_tests
```

The Python command emits 24 JSON rows, including the means, gaps and both SE
estimates. The Linux independent-reference CI job runs the panel and retains
`stochastic-dividend-hard-refinement.log` alongside the earlier survival log;
its allowance is 20 minutes. Ordinary Rust tests include the state control,
and Python discovery includes all six new controls. Source archives require
the implementation, tests, fixtures, this document and the CI command.

Calibration uncertainty, continuous-time bias, other barrier styles, general
nonuniform coupling and other Greeks remain outside this panel. The public
pathwise Spot-risk method still rejects unsmoothed Barriers. The dedicated
[hard Barrier Spot-risk method](stochastic-dividend-hard-barrier-spot-risk.md)
subsequently adds production survival conditioning for up/down knock-in/out calls and puts,
with its own contract, numerical-error policy and independent-reference checks.
