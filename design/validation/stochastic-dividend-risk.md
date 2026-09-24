# Stochastic-dividend risk and timestep refinement

## Source and evidence

Parent: PR #86, commit `5998b24de8c8bc520adcee3e9df52a2ef2f49223`, tree
`1332a2aa5d5e9e2738c1637ab03261d575358b8c`. This is a separate child change;
no parent or main branch is modified. The parent still uses its 4 bp legacy
ensemble-SE gate; the separately proposed 5 bp requirement is not included.

Native results are not yet recorded for this candidate. Read the PR's observed
source-pinned results; do not interpret included tests as passing evidence.

## First-order derivative panel

`stochastic_dividend_risk.rs` uses full recompilation for central differences at
h=1e-5 and h=1e-6. The fixed error budget at **each** scale is
`3e-5 + 2e-5 * max(abs(AAD), abs(FD))`. Sampling errors never enlarge it.
The base panel covers BS/1F/2F, pseudo-MC and bridged RQMC, two seeds, three cash
means including at expiry and after expiry, and nonuniform discount/repo curves
with extrapolation. Every non-anchor parameter is checked; time-zero anchors
must report exactly zero. There are additional Asian/delayed-payment and
smoothed pre/post-cash barrier cases. Zero-parameter derivatives use an inward
h=1e-7 difference and a 3e-4 absolute budget.

Checks include equal price and price standard error from `evaluate` and
`evaluate_aad`, repeatability with one/three workers, risk vector/label ordering,
vol-point and zero-rate scaling, and explicit rejection of unsmoothed barriers.
These are finite-algorithm derivative tests. They do not establish price accuracy
or risk convergence for a continuous-time process or a market-calibrated model.

Python tests separately exercise immutable/copying results, sigma0/cash full
recompilation, limits, worker replay, methods and uncertainty metadata.

## Continuous-time moment reference

For constant sigma, use the Ito moment system for `(E[f²], E[fY], E[Y²])`, with
`E[f]=E[Y]=1`, and independently integrate it by fourth-order Runge--Kutta with
32768 steps. Compare exact Gaussian moments of the positive split on
16/32/64/128 steps. The final maximum normalized moment error must be below 1e-5;
each successive error ratio must exceed 3.5. This is a deterministic second-order
moment check, not a vanilla-price reference for stochastic volatility.

An independent local Python ODE check gave maximum errors
2.0897493238747344e-5, 5.225208012271665e-6, 1.3063541692837077e-6 and
3.2659176119054223e-7. These are Python derivation checks, **not native Rust results**.

## Common-noise price refinement

The ignored release test `coupled_bs_one_and_two_factor_price_refinement` covers
BS/1F/2F, two seeds, 4096 independent antithetic units each, and
16/32/64/128 steps versus 256. It builds Brownian/OU integral covariance directly
from exponential kernels and uses its own triangular factorization. Coarse
Brownian increments are sums of fine increments; coarse OU integrals are
exponentially weighted sums. All time resolutions therefore share the same
joint Gaussian innovations, rather than reusing a seed with unrelated paths.

The terminal call is evaluated from the public path plan with cash at 0.5, 1.0
and 1.4 years. Report all 24 paired differences and paired sampling errors.
At 128 versus 256 steps require `abs(mean difference) + 4 * paired SE < 0.02`
in **price units** for each model/seed. This new test budget was specified before
native observation. It is not an IV budget and changes no existing budget.

This bounds selected finite-grid differences, not absolute continuous-time
error, broad stress coverage or exotic accuracy. The finest grid has no claim
to be exact. No extrapolated price or risk is substituted into production.

## Commands

```shell
cargo test --locked -p pricing --test stochastic_dividend_risk -- --nocapture
cargo test --locked -p pricing --test stochastic_dividend_refinement
cargo test --locked --release -p pricing --test stochastic_dividend_refinement -- --include-ignored --nocapture
python -m unittest discover -s tests/python -p test_stochastic_dividend_risk.py -v
python examples/python/stochastic_dividend_risk.py
```

Normal checks include all/minimal Rust suites, Python wheel/stub/source contracts,
existing price regressions, formatting/Clippy, docs and supported-platform replay.
See [ADR 0015](../adr/0015-stochastic-dividend-risk.md).
