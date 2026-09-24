# Bergomi / stochastic cash-dividend validation

## Source and scope

Parent: merged PR #84, `25c1b025c2945dea3c665040bfad37f8f6e58a39`,
tree `baba26042ba98ca061e18667454a4a53bdd412fe`, on `feat/bergomi-pure-sv`.
That parent is not yet propagated into main; no other branch/PR is changed here.

This is single-asset deterministic-rate pure 1F/2F Bergomi, price only.
[ADR 0014](../adr/0014-bergomi-stochastic-dividends.md) records the boundary.

## Independent two-step reference

`tests/python/bergomi_dividend_reference.py` uses a separately constructed direct
covariance integral, NumPy Cholesky, tensor Gaussian quadrature for the first
step and final dividend innovation, and an analytic conditional Black price
for the final equity innovation. It calls no production pricing, transition,
correlation, reserve or random-number helpers. The dividend mean reversion is
nonzero, and the dividend is after option expiry, so both stochastic factors
remain exposed at the terminal payoff.

| Gaussian order | 1F | 2F |
| --- | ---: | ---: |
| 8 | 7.7722066487075425 | 7.775688587829926 |
| 12 | 7.772107935821505 | 7.775598328664534 |
| 16 | 7.772110840278060 | 7.775600401255288 |
| 20 | 7.772111247866506 | 7.775600673296135 |

These values were computed locally. Order-16/20 differences are below 4.1e-7.
This is an independent **finite-step scheme** reference, not continuous-time
price acceptance. No production MC price is used as the oracle. Public tests
use an absolute 0.002 budget plus six reported scramble standard errors.

## Added tests

Two private Rust tests check integrated covariance against direct Simpson
integration and weighted-OU centering against its closed-form variance.
Five Rust integration tests cover zero-vol-of-vol projection onto unchanged BS
paths, 2F-to-1F reduction without dropping unused coordinates, invalid PSD even
with zero loading, singular positive-semidefinite inputs, independent price
references, nonfinite inputs, worker replay and fingerprint sensitivity.
Four Python tests check both factories, price references, immutable metadata,
fixed-cash/zero-stock-volatility limits, worker replay, validation and risk rejection.

## Execution status

Local Rust is unavailable in this editing environment. Native validation and
full platform CI must be observed on the candidate source; no parent result
certifies this extension. Native results will be recorded on the PR together
with the measured source tree. The local reference above is not a claim that
Rust/Python extension tests have passed.

## Remaining acceptance

Broad continuous-time price/refinement panels, stochastic-dividend exotic
accuracy, full legacy release gates and platform replay remain distinct checks.
AAD/VegaKT, LSV recalibration, HW, rough and multi-asset coupling are not included.
The existing 4 bp ensemble-SE gate on this parent is untouched; PR #85 changes
that requirement separately. Never relax it implicitly to obtain green CI.
