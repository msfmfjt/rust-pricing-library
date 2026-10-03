# Multi-step hard Barrier reference by survival conditioning

The [two-step conditional reference](stochastic-dividend-conditional-barrier-reference.md)
uses deterministic quadrature. This panel extends independent hard price and
Spot Delta checks to 4/8 evolution steps and 2/4 monitoring dates with
eta=0.6, kappa=0.7, H=0.1/0.3, and nonflat frozen leverage surfaces.

The new reference is a Monte Carlo estimate with its own sampling error.
It is not an exact number or a continuous-time price. Comparison gates include
both the reference's batch error and the Rust estimator's scramble error.

## Cases and retained data

The [fixture](../../fixtures/stochastic-dividends/rough-survival-barrier-reference.json)
links the market/contract and calibration inputs from the previous panels,
and retains each grid, monitoring dates/indices, complete calibrated surface,
32 reference batch means, aggregate price/Delta and their standard errors.
Cash means remain 5/3/12; the last cash date lies beyond expiry. Two-date
monitoring uses days 182/364; four-date monitoring adds days 91/273. Cash-date
up-barrier monitoring includes pre-cash stock, and the terminal intrinsic
value uses post-cash stock. Delayed payment remains at day 456.

| Case | H | Steps | Observations | Hard price (SE) | Hard Delta (SE) |
| --- | ---: | ---: | ---: | ---: | ---: |
| h01_n4_two | 0.1 | 4 | 2 | 12.07764836 (0.00345605) | 0.96064090 (0.00014744) |
| h01_n8_two | 0.1 | 8 | 2 | 12.54029638 (0.00450472) | 0.94919402 (0.00022027) |
| h01_n8_four | 0.1 | 8 | 4 | 13.20761103 (0.00404067) | 0.99739559 (0.00019383) |
| h03_n8_four | 0.3 | 8 | 4 | 13.25328933 (0.00404221) | 0.99714945 (0.00018489) |

Rust independently recompiles each case and checks the retained grid and
leverage values (2e-13 absolute tolerance for leverage). The two H=0.1
8-step cases have identical surfaces, so their difference isolates the
observation schedule. Each evolution grid has its own calibration: differences
between 4 and 8 steps include recalibration effects and are not evidence of
convergence or a time-error bound.

Quarter-date floating-point subtraction can cause `ceil(dt/max_step)` to add
an unintended extra subdivision. The test's maximum-step bound allows eight
machine epsilons of relative slack; the compiled grid is then checked against
the retained contractual times. No production grid policy is changed.

## Conditional evolution and the hard payoff

The [NumPy implementation](../../tests/python/rough_dividend_survival_reference.py)
uses independent dividend and volatility Brownian coordinates D and V, an
independent newest-cell residual, and uniforms for equity. Given D and V,
the equity normal has mean

```text
rho_SD*D + q*V,
q = (rho_SV - rho_SD*rho_DV) / sqrt(1-rho_DV^2),
```

and variance `1-rho_SD^2-q^2`. The volatility Brownian increment is
`sqrt(dt)*(rho_DV*D+sqrt(1-rho_DV^2)*V)`. This ordering allows the Volterra
history to stay fixed while the equity normal is conditioned on survival.

The history uses directly integrated cell-average weights

```text
w_ij = sqrt(2H) * [(t_i-t_j)^(H+1/2) - (t_i-t_(j+1))^(H+1/2)]
       / [(H+1/2)*dt_j].
```

Only the newest cell also uses its independent residual
`dt^H*(1/2-H)/(H+1/2)`. The variance centering is the sum of squared
Brownian loadings and that residual's variance on the actual grid; it is not
replaced by `t^(2H)`. Each equity step uses the previous node's history.

At a monitored node, the symmetric dividend update makes next Y affine in
next f, conditional on the dividend innovation. Pre-cash stock is therefore
affine in next f. A missed up-barrier imposes one upper cutoff b on the
remaining equity normal. Sample it as

```text
p = Phi(b), z = inverse_Phi(u*p), u uniform on (0,1),
W_next = W*p.
```

At unmonitored nodes, use the ordinary equity normal and leave W unchanged.
At the final step, integrate the lognormal equity analytically between the
exercise cutoff and, if monitored, the hard-barrier cutoff. The weighted
result is the knock-out call. Subtract it from a coupled, unconditioned
vanilla call (also integrating the last equity normal) to obtain the knock-in
price. The reference does not smooth a sampled Barrier indicator.

## Analytic Spot derivative

The physical-Spot convention reanchors the frozen surface to funded residual
equity. The unconditioned normalized law is invariant under Spot. The
survival-conditioned samples do move with Spot, because their cutoff changes.
Propagate f/Y tangents, the piecewise-linear leverage slope, and

```text
dp = phi(b)*db,
dz = u*dp/phi(z),
dW_next = dW*p + W*dp.
```

The final truncated moments are differentiated with respect to their mean,
variance, stock coefficients and both cutoffs. Differentiate the weighted
knock-out value, including dW, before subtracting from vanilla Delta. Thus
all earlier monitoring boundaries contribute through the probability and
transport tangents; the terminal boundary is included analytically.

Zero numerical survival is an absorbing zero-weight path. Positive
probabilities are never floored or clipped; inverse-CDF inputs outside the
open unit interval raise an error. The retained scenarios pass without
probability floors or clipping.

## Independence, controls and uncertainty

No production calibration, evolution, payoff, risk, normal generator or
reduction function is called by the Python valuation reference. Calibration
outputs are retained inputs, so calibration correctness is outside its scope.
The only shared Python helpers are the independent normal CDF and density
from the earlier test references.

Reference sampling uses NumPy PCG64 seed 20261003, 32 batches of 65,536
antithetic pairs: 4,194,304 paths per case. Each pair averages normals z/-z
and uniforms u/(1-u). Standard errors are the sample standard deviation of
the 32 batch means divided by sqrt(32). The same reference draws are reused
across equal-dimension cases; errors between cases are not independent.
The Rust valuation uses independently randomized Sobol scrambles.

The [controls](../../tests/python/test_rough_dividend_survival_reference.py) check:

- Normal inverse-CDF round trips and the first hybrid cell's variance, plus
  the Brownian H=1/2 limit on an irregular grid.
- Two-step survival estimates against the separate deterministic-quadrature
  references, with both price and Delta within four batch SEs plus 2e-7.
- Analytic per-sample Delta against common-input price differences at three
  Spots and two bump sizes for every 4/8-step case (2e-6 absolute tolerance).
- Zero knock-in value with no monitoring, date/index consistency, retained
  means/SEs, and the positive paired price effect of two added observations.

A dedicated Linux CI job regenerates every reference batch from scratch and
checks it within 1e-9; it never rewrites the fixture. These controls establish
finite-grid implementation checks, not a rigorous probabilistic error
certificate or continuous-time accuracy.

## Rust gates and execution

For each case and valuation seed 193/877, run hard price and width-0.5
price/Delta with 65,536 points per scramble and 32 independent scrambles,
antithetics and Brownian bridge. Set
`combined_SE=sqrt(reference_SE^2+valuation_SE^2)` and require
`abs(value-reference)+4*combined_SE < 0.03` for price and `< 0.015` for Delta.
Reference SE must be below 0.005/0.0005 for price/Delta, and valuation SE
below 0.003 for either quantity. The two-step quadrature gates are unchanged.
All 24 new comparison rows are retained, including both SE contributions.

The initial panel passed at these settings without changing its seeds,
counts or gates. The maximum comparison bounds were 0.022119615 for hard
price, 0.021955278 for width-0.5 price, and 0.009478830 for width-0.5 Delta.
These are observed finite-sample values, not bounds across all parameters.

```sh
python -m unittest discover -s tests/python -p test_rough_dividend_survival_reference.py -v
python tests/python/rough_dividend_survival_reference.py
cargo test --locked --release -p pricing --test stochastic_dividend_barrier_reference -- --include-ignored --nocapture
```

The three-OS Barrier job retains 72 rows across the three numerical panels.
Its allowance increases from 15 to 20 minutes for the additional work.
The independent Python regeneration job retains its eight price/Delta rows
in `stochastic-dividend-survival-reference.log`. Source archives require the
implementation, controls, fixture, documentation and CI regeneration step.

Public unsmoothed Barrier Spot risk remains rejected. Finer-grid convergence,
calibration uncertainty, different barrier styles and a production hard-risk
estimator remain separate work; this panel validates the specified frozen
finite-grid laws and the public smoothed Delta against their hard references.
