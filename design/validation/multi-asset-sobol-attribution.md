# Multi-asset Sobol sampling attribution

The [refinement/stress panel](extended-model-refinement-stress.md) initially
failed for the three-year 2F-LSV/BS basket on all three CI platforms. The
maximum IV error was within its 15 bp budget, but ensemble SE reached
8.9042 bp (limit 4) and conditional pricing SE 4.1290 bp (limit 3). The
same-parameter LSV constituent passed. The investigation below identifies
poor Sobol coordinate placement as a major source of the sampling error,
and corrects the placement to the existing simulation specification.

## Controlled experiment

Use the unchanged nu=0.6, 131,072-particle calibration, bandwidth 0.035,
requested 192 time steps and three calibration seeds 41709/42903/44001.
Compile each calibration once. Use the same eight pricing scrambles for all
three calibrations, with master seed `15111065706836414446`, and compare
2,048-point and 8,192-point prefixes. The second sampling layout changes only
the mapping from Sobol dimensions to independent bridge normals; calibration,
Gaussian transition loadings, target curves and payoff functions stay fixed.

The [manual attribution test](../../crates/pricing/src/engine/multi_asset/local_correlation/stress_diagnostics.rs)
records all three strikes, both constituent prices, forward means and spatial
boundary visits on the same paths. The legacy baseline's price and SE were
checked against the public compiled plan before the correction. The retained
test checks the corrected layout against that public plan. The
[raw observations](extended-model-sampling-attribution.json) preserve the
pre-correction experiment, including all scramble means and the public replay.

The right-wing quote, x=0.36, gives the following 8,192-point results. Each SE
is for one fixed calibration using eight scrambles; it is not the acceptance
panel's mean over three independent calibration/pricing seed pairs.

| Calibration seed | Legacy IV error, bp | Corrected-order IV error, bp | Legacy pricing SE, bp | Corrected-order pricing SE, bp |
| --- | ---: | ---: | ---: | ---: |
| 41709 | 13.0290 | 0.8941 | 6.5747 | 3.1751 |
| 42903 | 10.7169 | -1.1465 | 6.6388 | 2.9219 |
| 44001 | 11.3457 | -1.6921 | 6.9650 | 2.5705 |

Calibration variation is measurable and remains relevant. With common pricing
scrambles and the corrected order, right-wing price differences relative to
seed 41709 are -2.04 bp (paired SE 0.78) and -2.59 bp (paired SE 0.73), using
target-vega conversion. The largest ATM contrast is -2.96 bp (paired SE 0.61).
These results distinguish a few-bp calibration effect from the much larger
legacy pricing uncertainty. Common-random-number estimates are correlated;
their pricing SEs must not be combined as though they were independent.

## Why the old layout is problematic here

The compiled grid has 309 intervals. Exact target dates plus the maximum-step
subdivision produce more intervals than the requested 192; the investigation
does not alter this grid. Two spots and two OU factors at each of the two
correlation endpoints give eight independent factors and 2,472 Sobol dimensions.

The old sampler used `factor * steps + bridge_rank`. Its terminal bridge
normals therefore occupied dimensions 0, 309, 618, ..., 2163. Dimensions 309
and 1236 directly influence different spot components of the basket. They
have a poor joint projection in the bundled Joe–Kuo direction set.

The [direction-number probe](../../scripts/diagnose_stress_sobol.py) uses exact
GF(2) matrix ranks to inspect that projection. On a dyadic grid with 64 bins
in the first coordinate and two in the second, only 64 of the 128 cells are
occupied at 2,048, 8,192 and 32,768 points. At 32,768 points those cells each
contain 512 points and the other half are empty. The production triangular
linear scramble preserves the leading-row rank; digital shifts change which
cells are occupied. Increasing the prefix within those tested powers of two
therefore does not remove this particular coarse projection defect.

These occupancy counts describe the base Sobol points before antithetic
payoff averaging. The pricing experiment includes production antithetics;
its observed SE reduction is the direct evidence for the pricing impact.

| Projection at 32,768 points | Zero-based dimensions | 2D t-value | Occupied cells in the 64 × 2 grid |
| --- | --- | ---: | ---: |
| Legacy joint 2F stress | 309, 1236 | 9 | 64 / 128 |
| Corrected joint 2F bridge order | 1, 4 | 2 | 128 / 128 |
| BS/BS stress with its four factors | 309, 618 | 3 | 128 / 128 |
| One-year 2F case with 128 intervals | 128, 512 | 3 | 128 / 128 |

The legacy problematic pair fills this 64 × 2 grid at 65,536 points. That
geometric fact alone does not certify a price budget at that resolution.
The first LSV constituent does not directly use the same problematic pair;
the basket combines both spots. This explains why reducing vol-of-vol or
increasing calibration particles did not address the principal sampling issue.
It does not imply that every error in the panel comes from one projection.

## Correction and compatibility

[Requirements section 5.1](../requirements-v1.0.md#51-path-generation) and
[architecture section 9](../architecture-v0.1.md#9-random-source-and-simulationplan) specify bridge-rank-major,
factor-minor Sobol coordinates: `bridge_rank * factor_count + factor`. The
generic `BrownianBridgePlan` already provides that order. The multi-asset
sampler now follows it when RQMC and Brownian bridge are both enabled.

The mathematical dynamics, calibrations, correlation endpoints, antithetics,
direction numbers, scramble implementation and error budgets do not change.
Pseudo-MC and unbridged RQMC retain their previous coordinates. The affected
RQMC bridge plans use fingerprint domain
`multi-asset-bs-lv-affine-v2-bridge-rank-major-before-correlation`; finite
sample paths, prices and risks differ from version 1. API and wire inputs
remain compatible. Historical measured data and replay observations are
preserved rather than relabelled as corrected results.

Fast sampling-contract tests reconstruct each factor's terminal Brownian value
from generated increments and require it to use a leading Sobol coordinate.
They also check the retained pseudo-MC and unbridged-QMC layouts. Existing
multi-asset tests cover analytic prices, worker replay and risk identities.
Two pathwise zero-vol-of-vol/rough-limit fixtures compare models with different
factor counts. They now use unbridged QMC to retain common stock coordinates;
their exact price/risk identity tolerances are unchanged. A bridge-rank-major
mapping depends on the full factor count, so those different models no longer
have identical finite-sample stock paths under a shared bridged-QMC seed.

## Corrected validation

The original strict 2F basket stress gate **passes** with the corrected public
sampler. The [corrected observations](multi-asset-sobol-corrected-panel.json)
retain all nine price runs and calibration diagnostics, plus hashes of the
sampler and acceptance sources. The target IVs, 131,072 particles, bandwidth
0.035, requested 192 steps, 32,768 points per scramble, eight scrambles, three
independent calibration/pricing seed pairs and every budget match the legacy
nu=0.6 basket observation exactly. All figures below are IV basis points.

| Log strike | Legacy IV error | Corrected IV error | Legacy ensemble SE | Corrected ensemble SE | Legacy pricing SE | Corrected pricing SE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.36 | -0.2745 | 0.3924 | 1.5545 | 0.5967 | 1.1894 | 0.2354 |
| 0.00 | 2.5855 | -0.5377 | 7.2514 | 0.3674 | 3.0964 | 0.3674 |
| 0.36 | 6.6614 | -0.5003 | 8.9042 | 0.7274 | 4.1290 | 0.7274 |

The corrected maximum absolute IV error is 0.5377 bp (limit 15), RMSE 0.4808 bp
(limit 8), worst individual seed error 1.5131 bp (limit 30), maximum ensemble
SE 0.7274 bp (limit 4), and maximum conditional pricing SE 0.7274 bp (limit 3).
This is the unchanged fixed regression panel, not a confidence-interval coverage
claim. Its public test completed in 1,462.30 seconds on the local runner.

Local regression checks with pinned Rust 1.98.1 pass: 515 workspace/all-feature
tests, 501 no-default-feature tests, and all 89 Python API tests against the
current CPython 3.12 release extension. The Python check imports the built
shared library directly; wheel installation is covered by CI. Clippy passes
for all workspace targets and features with warnings denied. These counts
exclude ignored studies; the retained attribution study is a manual diagnostic.
Formatting, reference-fixture checks (67/114/61 cases), schemas and Markdown
links also pass. The source-archive check includes the new diagnostics and
both sampling evidence files.

Reproduce the original public acceptance gate, the controlled attribution
study, and the exact projection probe separately:

```sh
cargo test --locked --release -p pricing --test extended_model_acceptance multi_asset::two_factor_basket_stress_repricing -- --ignored --exact --nocapture
cargo test --locked --release -p pricing --lib engine::multi_asset::local_correlation::stress_diagnostics::two_factor_stress_sampling_attribution -- --ignored --exact --nocapture
python3 scripts/diagnose_stress_sobol.py
```

Local numerical measurements use the same documented build workaround as the
[original stress study](extended-model-refinement-stress.md#pre-correction-measurements).
The full 74-panel native run on Ubuntu, macOS and Windows must be checked at
the corrected PR head; the pre-correction 73 passing panels do not establish
their result under the new layout.

The original nu=0.8 measurements and fixed-cash domain failures remain historical
evidence. This correction does not certify those unrerun stress settings,
arbitrary extreme parameters or whole-grid feasibility. Spatial extrapolation
and projection diagnostics remain material despite the sampling improvement.
