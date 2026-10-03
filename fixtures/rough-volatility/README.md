# Independent rough-volatility reference values

`reference.json` contains 30 stationary fractional-OU covariance values, four
one-cell rough Heston conditional-Gaussian checks and a constant-volatility
Black call price. It is read by the new Rust and Python tests.

The covariance reference is an independent spectral integral:

```text
rho(z) = 2 sin(pi H)/pi integral_0^infinity
         cos(z omega) omega^(1-2H)/(1+omega^2) d omega.
```

The generator uses SciPy's oscillatory quadrature and checks a separate
nonoscillatory second-difference integral. It also checks positive definiteness
on five irregular grids. The production Rust implementation does not call
SciPy; it uses globally adaptive Simpson quadrature of the nonoscillatory form.

The one-cell Heston values use analytic mean, variance and covariance of the
singular integral and the Brownian increment, not the production kernel code.
The Black value uses the scalar closed formula, not a production pricer.

Run from the repository root with NumPy and SciPy installed:

```shell
python scripts/check_rough_volatility_reference.py
```

Regeneration is explicit and must be reviewed, not used to make a failing
implementation appear correct:

```shell
python scripts/check_rough_volatility_reference.py --write
```

The script **does not execute Rust or the compiled Python extension**. A passing
reference check is not a claim that the implementation passes its tests. These
limiting-law and Gaussian checks do not bound general pricing bias. See the
[model contract](../../docs/models/rough-volatility-families.md) and
[validation record](../../design/validation/rough-volatility-families.md).


## Nondegenerate two-step prices

`two-step-prices.json` retains 18 call prices: the six families, each at strikes
90/100/110, spot 100, times `[0, 0.5, 1]`, zero rates and no dividends. Every model
has nonzero volatility randomness. These are prices of the specified **finite
log-Euler scheme**, not exact continuous-time rough-model prices.

For Heston, lifted Heston, quadratic Heston, mixed Bergomi and beta-one SABR,
the generator conditions on the first volatility innovation, integrates out
the two asset shocks analytically and integrates the resulting conditional
Black price over one Gaussian variable. Heston truncation kinks are explicitly
split. RFSV uses a separate two-dimensional stationary-fOU Gaussian quadrature
with the independently integrated spectral covariance. Two tolerances / orders
are checked before retaining a value. No production paths, RNG or pricing
helpers are used to generate these prices.

```shell
python scripts/check_rough_volatility_prices.py
# Explicit reviewed fixture regeneration only:
python scripts/check_rough_volatility_prices.py --write
```

The Rust release panel uses two fixed seeds, 16 scrambles and 4,096 antithetic
units per scramble for all 36 comparisons. Numerical budgets and scope are in
the [validation record](../../design/validation/rough-volatility-families.md).
A separate public-primal reconstruction checks sampling-error aggregation; it
shares the path simulator and is not an independent model-price reference.

## Pairwise refinement references

`refinement.json` stores six newest-cell Gaussian coupling cases, cross-checked
by endpoint-weighted and Gauss-Jacobi integration, and 12 geometric lift kernels
on five strictly positive lags. `scripts/check_rough_volatility_refinement.py`
checks retained values without regeneration; only `--write` replaces them.
The [protocol](../../design/validation/rough-volatility-refinement.md) describes
the original failed ratio sequence and the revised shrinking-bin sequence.
These references are not continuous-time option prices or price-bias bounds.

## Fixed-grid Lifted Heston factor-price inputs

[`lifted-factor-prices.json`](lifted-factor-prices.json) retains the full fixed
price experiment and rational-kernel references for the semi-implicit scheme.
The infinite-factor values are independently integrated over the entire Laplace
measure, not obtained from a larger finite-factor production run. The finite
factory kernels and the beta-integral identity are checked independently.

```shell
python scripts/check_lifted_heston_factors.py
python -m unittest discover -s scripts -p 'test_lifted_heston_factors.py'
```

The [protocol](../../design/validation/lifted-heston-factor-prices.md) distinguishes
factor-price comparisons from continuous-time accuracy. Regeneration uses the
explicit `--write` option and requires review; CI only verifies retained values.
