# Rough-family market-IV adjoint validation

## Frozen scope and numerical protocol

Base PR #135: `77da3ead5e96432e8499c49ae778d9cf7adccc31`, tree
`3252c457b093d5a3071aa66b55e12327b6dd1431`. Add the discrete market-IV transpose;
do not change path, particle, interpolation, model, RNG or payoff formulas.
Parent functions retain the identity projection in their aggregation path.

Four fast Rust tests cover the strict source contract, two-size Dupire transpose
bumps, six nondegenerate families with both pseudo-MC and RQMC, and an independent
RQMC gradient/standard-error reconstruction. A separate ignored test is executed
explicitly in release CI, not silently omitted from acceptance evidence.

Quotes: times [.2,.6,1.2], x [-.9,-.45,.05,.5,.9], IV at row i:
`.2+.002*i-.003*x+.002*x*x`. Target times [0,.13,.37,.7,1], x
[-.55,-.13,.18,.6]. Price F/Spot100, strike100; cash3 at .25 and cash4 at1.5
for the main chain-rule panel. Preserve nonflat carry and the full future cash
reserve. Calibration128 particles, seed429, bandwidth.5, ESS3. Valuation seed819;
MC128 antithetic units, RQMC4x64 antithetic/Brownian bridge. All six native random
layouts are covered, with model parameters retained in the test source.

Direction `cos(.7*j)` at all15 quotes; bumps1e-6 and5e-7. Rebuild the surface,
Dupire target and particle calibration at both bump endpoints. Compare against
the analytic quote-vector dot product with fixed relative budget
`3e-6*(1+abs(reference))`. The separate Dupire scalar-dot test uses2e-7. These
checks measure discrete derivatives, not independent continuous-time prices.

The Python suite additionally rebuilds the natural-cubic/time/Dupire nodes in
pure Python via tridiagonal equations (no extension surface or grid construction
for this independent target), then bumps that target before production particle
calibration. It tests individual quote0/7/14 and parallel shifts for Asian calls
with delayed payment, exact-source rejection, absent reverse traces, smoothed
and unsmoothed Digital behavior, immutable/copy-returning properties and workers.

For RQMC reconstruction, obtain native shocks and recorded paths, independently
seed an un-discounted no-dividend European payoff, aggregate leverage adjoints,
reverse particles and target interpolation, and apply the quote transpose. The
four scramble vectors are independently reduced with a two-pass sample-variance
formula. Check each marginal and the sum; require a material (>1e-4) difference
from a deliberately covariance-ignoring root-sum-square estimate. This tests
statistics/orchestration but shares the production path/adjoints, not an
independent path generator.

## Analytic Black limit

Six deterministic-variance limits, variance.04, one-year ATM F=K100, D1, no cash.
Source2x3 quotes all.2; target9x5 over [0,1] with steps1/8. Two valuation seeds91
and1973, eight scrambles x2048points with antithetic/Brownian bridge,32,768 paths
per row. Calibration settings unchanged. Independently evaluated constants:
price `7.965567455405804`, parallel Black Vega `39.69525474770118` per absolute IV.

Gates fixed before first execution:

```text
abs(price - reference) <= 5*price_SE + 2e-4
abs(parallel_vega - reference) <= 5*parallel_SE + 0.02
0 < parallel_SE <= 0.1
```

Report all12 rows including unfavorable outcomes; no seed/count/budget edits
based on results. This is a constant-variance-limit check, not general LSV price
accuracy or a statistical simultaneous-coverage statement.

## Regression and evidence

The continuation report records actually completed local commands and current
remote status separately. Check default/minimal/release targets, full debug
workspace, Clippy, formatting, Python wheel/package contracts, required source
files, and both parent risk targets. Preserve failed/incomplete command logs.
Extend the existing three-OS rough-family AAD workflow with separate market-IV
commands and retained `rough-family-market-iv.log`; do not remove older gates.

Sampling errors condition on one calibration. Pseudo-MC quote SE is None.
No full error bound, unrestricted payoff smoothness, stochastic model-parameter
calibration derivative, arbitrary-strike Spot convention, Gamma or paper-style
VegaKT projection is claimed. See [calculation specifications](../../docs/models/rough-family-market-iv.md).
