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

## Interpretation and remaining validation

These are finite-particle, fixed-grid implementation checks. The paired
sampling errors condition on calibration and exclude calibration, time-grid,
smoothing and model errors. The panel does not independently certify the
numerical value of every reported nonzero SE or continuous-time Greek
convergence; full-recompile differences share the same underlying path scheme.
Broad path-dependent rough-dividend accuracy remains outside this panel.

Stochastic rates remain unsupported for rough residual-LSV dividends. The
constant-residual-volatility Hull-White conditional cash-claim formula cannot
be reused for state-dependent rough/LSV volatility; that extension requires a
separate conditional-claim valuation design and validation.
