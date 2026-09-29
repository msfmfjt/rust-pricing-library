# Rough residual-LSV conditional price and Delta refinement

This panel extends [price and risk validation](stochastic-dividend-rough-lsv-correlation-risk.md)
to time-grid sensitivity with nonzero rough vol-of-vol and dividend mean
reversion. It conditions on one calibrated leverage surface. The separate
[recalibration and particle panel](stochastic-dividend-rough-lsv-calibration-refinement.md)
recompiles each grid/count and measures variability across calibration seeds.

## Common Brownian construction

For each coarse/fine pair, fine equity/dividend/volatility Brownian increments
and the fine newest-cell Volterra integrals are sampled first. Coarse Brownian
increments are sums of the fine increments. Coarse newest-cell integrals are
sampled conditionally on all these fine variables, with one additional normal
per coarse cell for the remaining variance. Copying or averaging the fine
residual normals would not preserve the required cross-grid covariance.

With `p=H+1/2`, the kernel is `sqrt(2H)*(T-s)^(H-1/2)`. On each overlapping
fine cell, the Brownian cross-covariance follows its elementary power integral.
The covariance of two newest-cell integrals uses
`2H * integral_0^1 (offset+x)^(H-1/2)*x^(H-1/2) dx`, in units of
`fine_dt^(2H)`. An endpoint-regularizing substitution and 8,192-panel Simpson
integration evaluate this integral; zero offset has exact value 1. Conditioning
on the fine Brownian increment and independent near-cell residual gives the
coarse normal coefficients. Negative remaining variance is rejected.

At H=1/2 the newest-cell integral is the Brownian increment and its residual
is unused. At equal grid sizes the mapping is the identity. Supported grids
are uniform with dyadic integer refinement ratios and common endpoints.

Three fast tests check:

- Kernel integrals against independent 512-point Gauss-Legendre constants,
  using the different substitution `x=u^(1/H)`. Orders 256/512 agree within
  2e-12; the Simpson comparison budget is 3e-11.
- The implemented linear map, reconstructed with basis vectors, has independent
  unit-variance coarse normals and the specified Brownian/near-cell marginal
  and cross-grid covariances within 3e-11. H is 0.01/0.1/0.3/0.49/0.5 and the
  grid ratio is 1/2/4/8.
- Direct terminal price and scale-invariant physical-Spot Delta agree with the
  public rough-LSV estimator, including their nonzero sampling errors, within
  2e-11. This checks the payoff/Delta observations used by the refinement panel.

Each coarse/fine pair has the joint law of integrals of common Brownian paths.
The separate pairs do not jointly specify the correlations between different
coarse levels. No covariance or combined standard error across those levels is
reported or used.

The independent kernel constants can be reproduced with NumPy (repeat at
orders 256 and 512):

```python
import numpy as np
z, w = np.polynomial.legendre.leggauss(512)
u = (z + 1) / 2
for h in (0.01, 0.1, 0.3, 0.49):
    for offset in (1, 3, 7):
        value = np.dot(w, u**(0.5/h) * (offset + u**(1/h))**(h - 0.5))
        print(h, offset, float(value))
```

## Fixed-surface acceptance panel

Run all four checks, including the slower panel, with:

```sh
cargo test --locked --release -p pricing --lib lsv_refinement_tests -- --include-ignored --nocapture
```

The particle calibration is performed once at 16 steps with 512 particles,
seed 42, log bandwidth 0.35 and minimum effective samples 5. The residual
local-variance target has times `[0,0.5,1]`, log-moneyness `[-0.5,0,0.5]` and
rows `[0.045,0.04,0.035]`, `[0.05,0.045,0.04]`, `[0.055,0.05,0.045]`.
All pricing grids retain the same leverage values and knots exactly. Only
the valuation grid changes: 16/32/64 steps versus a 128-step reference.

Parameters are H=0.1/0.3, eta=0.6, dividend mean reversion 0.7, linkage 0.6,
dividend volatility 0.35, and correlations `(rho_SD,rho_SV,rho_DV)` equal to
`(-0.25,-0.4,0.15)`. Spot/strike are 100, maturity 1, discount 0.95 and carry
factor 0.98. Cash means 5/3/12 occur at 0.5/1/1.4. Exercise observes post-cash
stock at expiry; the final cash mean remains in the funded reserve.

Two valuation seeds, 193/877, each use 131,072 independent antithetic units.
The terminal call is reconstructed directly. With leverage reanchored to funded
residual equity, normalized f/Y are Spot invariant and the path Delta is
`0.98*f_T*1{S_T>100}`. The two-pass variance is computed on antithetic coarse-minus-fine
differences. JSON log rows retain both means, the paired gap/SE and the
incorrectly unpaired SE for comparison.

Budgets fixed before running the acceptance panel:

- For 64 versus 128 steps, `abs(price gap) + 4*paired SE < 0.05` price units
  and `abs(Delta gap) + 4*paired SE < 0.005` Delta units.
- Paired SE must be below 0.01 for price and 0.001 for Delta at that level.
- At every level, paired SE must be below 75% of the unpaired SE, so the
  common-noise comparison demonstrably reduces sampling noise.

Sampling error consumes the gap budget; it does not expand it. Earlier levels
are diagnostic and need not have monotonically decreasing observed gaps.
CI runs the panel in release mode on Linux, macOS and Windows and retains
`stochastic-dividend-rough-lsv-refinement.log`.

### Initial precision observation

The initial local release run used 8,192 units. At H=0.1, seed 193, 64 versus
128 steps, the price gap was -0.0269095173 with paired SE 0.0159754938, which
failed the fixed 0.05 budget and 0.01 SE cap. That run is not a pass. The
valuation count was increased to 131,072 to resolve the gap more precisely;
the original counter-based samples are retained as a prefix. Both seeds,
all parameters, grids, references and acceptance thresholds remain unchanged.
The panel reports every condition before failing, to preserve all diagnostics.

## Interpretation and remaining work

The reference is a finer finite grid, not an exact continuous-time price or
Delta. A small last-level gap does not prove convergence order or bound the
remaining bias. Results concern the fixed, finite-particle calibrated surface
and exclude calibration, smoothing and model uncertainty. Recalibrating on
each grid changes that surface and must be studied separately before claiming
end-to-end calibrated-model convergence. The separate replication panel checks
specified finite grid/count differences without claiming general convergence.
Asian/Barrier refinement, nonuniform
grids and RQMC coupling are not covered by this first panel.
