# Rough residual-LSV correlation risk validation

The public `evaluate_lsv_rough_bergomi_correlation_risk` contract has two
different calibration dependencies: equity/volatility correlation requires
particle recalibration; dividend/volatility correlation keeps the leverage
surface fixed. Both use central differences with common valuation noise.

## Executable checks

Run the Rust public-API checks with:

```sh
cargo test --locked -p pricing --test stochastic_dividend_rough_lsv
```

The panel uses 64 calibration particles with seed 42 and a fixed local-variance
target. Valuation uses seed 91, either 128 pseudo-MC sampling units or four RQMC
scrambles of 32 points. Antithetic sampling and Brownian bridging are tested
both off and on. Cash at 0.5 and after expiry at 1.4 remain active. Calibration
reverse traces are disabled throughout.

- Both reported correlation derivatives agree with independent full-recompile
  central price differences at bump 0.02, within 2e-10 absolute. Up/down equity
  correlation scenarios change leverage; dividend correlation scenarios retain
  it exactly.
- Baseline price, price SE and sampling counts match ordinary pricing exactly.
  Derivatives and SEs replay exactly with one and three workers and a fixed
  reduction block size of 16.
- At zero vol-of-vol, both derivatives and their sampling errors vanish within
  1e-11. Neither volatility-driver correlation enters equity/dividend dynamics
  in this limit, giving a check independent of the finite-difference oracle.
- Zero, negative and non-finite bumps, out-of-range scalar correlations and a
  scalar-valid but non-PSD joint correlation matrix are rejected.

The installed-wheel tests in
[`test_stochastic_dividends.py`](../../tests/python/test_stochastic_dividends.py)
also cover rough parameter risk, Local-variance reverse, VegaKT, selective
correlation recalibration, and API rejection/worker replay. The wheel smoke
check verifies runtime exports against the explicit type-stub contract,
including the rough LSV factory and both rough risk result classes. CI checks
the static stub contract before Rust compilation, so missing method/property
registrations are reported without waiting for wheel builds.

## Independent sampling-error reconstruction

Run the paired-uncertainty checks with:

```sh
cargo test --locked -p pricing --lib lsv_uncertainty_tests -- --nocapture
```

Six sampling-error tests cover 48 panels: three contracts, pseudo-MC/RQMC,
seeds 91/1973, and every combination of antithetic sampling and Brownian
bridging on/off. Each panel
fully recompiles the base and eight bumped plans for H, eta, equity/volatility
correlation and dividend/volatility correlation. The H bump is 0.01; the other
three bumps are 0.02. All plans use 64 calibration particles, calibration seed
42, and no retained reverse trace.

The reference evolves primal paths using the original random coordinates and
reconstructs physical stock from its affine coefficients. It does not call the
production payoff, smoothing, bumped-risk sampler, statistics reducer or
standard-error helpers. All contracts have unit notional and last fixing at
one year; simulation stops there even when payment and cash reserves extend
beyond that date.

| Contract | Independently reconstructed payoff |
| --- | --- |
| European | Strike-100 call, paid at expiry with discount factor 0.95. Cash at 0.5, at expiry 1.0 and after expiry 1.4 remains active. |
| Arithmetic Asian | Strike-95 call on a weighted average: 20% historical fixing 102, 30% post-cash stock at 182/365, 50% post-cash stock at expiry. Payment is at 456/365. |
| Up-and-in discrete Barrier | Strike-80 call, barrier 105, monitoring at 182/365 and expiry. The hit state includes both sides of cash jumps with compact-C2 width 8. Payment is at 456/365. |

For Asian/Barrier, fixed-cash means 6 and 2 coincide with the two future fixing
dates, and cash mean 3 after expiry at 1.4 remains in the funded reserve.
Delayed payment uses the independently extrapolated log-linear discount
`0.95 ** (456/365)`. Barrier smoothing evaluates the centered quintic and its
integral directly, instead of the production shifted-coordinate expression.

For pseudo-MC, the 64 independent units average antithetic partners before
forming up/down payoff differences. For RQMC, the reference first averages
16 points within each of four scrambles. In each case, it applies the ordinary
two-pass sample-variance formula to the resulting independent paired values.
The reference price, all four risk means and their five SEs must match the
public APIs within `2e-10 + 1e-11 * abs(reference)`.

Every risk SE must exceed 1e-8, and must differ from an incorrectly unpaired
up/down scenario SE by more than 1e-6. These guards ensure that a zero-error
case cannot pass vacuously and that the panel detects discarded common-noise
covariance. Baseline price/SE identity and sampling counts are also checked.
Every delayed-payment panel must differ by more than 1e-6 from incorrectly
discounting at the last fixing; every Barrier panel must differ by more than
1e-6 from monitoring only post-cash stock. A seventh test verifies that exact,
unsmoothed Barrier parameter/correlation risk is rejected while repeated price
evaluation remains unchanged.
The focused three-OS CI job runs these tests in release mode alongside the
public correlation tests and retains `stochastic-dividend-rough-lsv.log`.

## Interpretation and remaining validation

These are finite-particle, fixed-grid implementation checks. The paired
sampling errors condition on calibration and exclude calibration, time-grid,
smoothing and model errors. The independent reconstruction verifies the paired
SE aggregation for price and the four rough parameter/correlation risks at the
stated inputs and three contracts. It shares the underlying primal path scheme
and random-number generators; it does not certify continuous-time Greek
convergence, pricing-scheme accuracy or SEs for other risk APIs and products.
Broad rough-dividend exotic accuracy remains outside this fixed-grid panel.

Stochastic rates remain unsupported for rough residual-LSV dividends. The
constant-residual-volatility Hull-White conditional cash-claim formula cannot
be reused for state-dependent rough/LSV volatility; that extension requires a
separate conditional-claim valuation design and validation.
