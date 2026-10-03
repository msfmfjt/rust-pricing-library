# Rough families: MC Spot Delta and particle LSV AAD

[Model families](rough-volatility-families.md) · [Example](../../examples/python/rough_family_aad_lsv.py) ·
[Validation protocol](../../design/validation/rough-family-aad-lsv.md)

## Scope and coordinate contracts

Experimental additive APIs for the six rough model families. Existing pure-price
requests, random coordinates and model parameters are retained. The new methods
must be requested explicitly; a generic rough request with risk flags is still
rejected. Stochastic rates/dividends, multi-asset composition, Gamma, reverse-mode
model-parameter risk and market-IV VegaKT are not implemented by this extension.

| Entry point | Output / held fixed |
| --- | --- |
| `RoughVolatilityPlan.evaluate_delta()` | Physical Spot Delta; model parameters, rate/repo curves and cash/proportional dividends fixed |
| `RoughVolatilityPathPlan.reverse_initial_forward(...)` | Initial path-coordinate VJP, all variance histories and shocks fixed |
| `RoughFamilyLsvPlan.evaluate()` (Python) | Particle-calibrated LSV price |
| `RoughFamilyLsvPlan.evaluate_local_variance_risk()` (Python) | Full discrete recalibrated local-Dupire-variance-node VJP, model parameters and particle randomness fixed |

Rust's high-level LSV type is `RoughFamilyLsvPricingPlan`; Rust additionally exposes
`RoughFamilyLsvPlan`, its recorded path and `CalibratedRoughFamilyLsv`.
The low-level LSV path's initial-forward adjoint holds the leverage grid anchor
and axes fixed. It is **not** the high-level physical Spot Delta, and is not a
recalibrated Spot derivative. No high-level LSV Spot Delta is exposed here.

## Pure MC physical Spot Delta

A recorded path stores the existing primal path and one initial-state derivative
per step. With fixed volatility histories and fixed shocks, log Euler gives
`dF_next/dF = F_next/F`. Normal SABR (`beta=0`) gives one. For `0<beta<1`, the
positive absorbing-Euler proposal has derivative
`1 + beta * F^(beta-1) * sqrt(V*dt) * z`; the absorbed branch has derivative zero.
An exactly zero proposal, or zero initial state in this absorbing case, is
rejected for the derivative rather than silently inventing a boundary value.
These are derivatives of the finite discrete path, not an exact CEV transition.

For arbitrary observation seeds `g_j`, reverse time by
`bar_F_j = g_j + bar_F_(j+1) * dF_(j+1)/dF_j`. There is no price bump, shock
adjoint or hidden derivative of volatility parameters. In all six base models
the variance history is independent of the initial price coordinate at fixed
parameters and shocks. A general model in which volatility inputs depend on
Spot would require additional chain-rule terms and is not inferred here.

The high-level method first reverses the existing payoff graph, including its
payment-date discount, then the affine escrow observations. A physical
post-dividend observation is `S_j = scale_j * F_j + reserve_j`; the pre-dividend
observation includes the contractual affine jump. The existing dividend-node
reverse supplies both state seeds and coefficient seeds. The initial-forward
adjoint and the dividend plan's direct Spot adjoint are added. Cash amounts,
proportional fractions, discount and repo curves are held fixed. Dividends beyond
option expiry remain in the reserve; simulating their later dates is unnecessary.
Discounting is performed only by the payoff graph, not a second time by the VJP.

Unsmoothed discontinuous products, including digital and barrier payoffs that
lack a supported pathwise contract, are rejected. Explicit supported payoff
smoothing retains its existing meaning: the derivative is of that smoothed
payoff, not a distributional derivative of an unsmoothed contract. At isolated
vanilla/maximum kinks the shared payoff tape retains its branch convention;
this does not assert an ordinary derivative at a deterministic atom on a kink.

`RoughVolatilityDelta.price` is the existing price/SE result. Delta's standard
error is computed from independent MC units (one antithetic pair is one unit),
or from independent scramble means for RQMC. It excludes discretization bias,
model error and calibration uncertainty. Python result fields are read-only.

## LSV with an unchanged variance driver

For a positive normalized funded-forward state, introduce a squared leverage
surface `ell=L^2` and evolve

```text
dF/F = sqrt(ell(t,F) * V_t) dB_t
ell(t,x) = local_variance(t,x) / E[V_t | F_t=x].
```

`V_t` is the actual base-model variance, **not a unit-mean variance multiplier**.
The given rough model, its initial variance curve and all stochastic parameters
stay fixed during particle calibration and during its reverse. Price and
variance shocks retain the native correlation and hybrid-kernel layout.
RFSV retains its independent price Brownian motion and stationary or conditioned
initial Gaussian law. Rough SABR is accepted here only at `beta=1`; applying
relative Dupire calibration to a non-lognormal coordinate is not silently done.

For Quadratic rough Heston this defines a **fixed-driver LSV extension**: its
latent state keeps the original `sqrt(V) dB` feedback while the price acquires
`L sqrt(V) dB`. It does not redefine that latent state as a convolution of the
new leveraged returns. Such a different feedback model needs a different joint
calibration and reverse and is not supplied under this name.

Calibration uses the existing quartic kernel estimator, minimum effective
sample count, nearest supported donor rule and interpolation from
[LSV calculation specifications](lsv-calculation-specifications.md). Each
particle's base variance history is precomputed with Philox and the separate
`LsvCalibration` random domain. No unused unlevered price path is generated.
The first-row moments are empirical moments of `sqrt(V_0)`; they are not assumed
to be one. In particular stationary RFSV may have different `V_0` per particle.
A nonpositive first-row conditional second moment is rejected, not floored.

Subsequent zero base-variance nodes caused by full truncation produce an identity
asset step and zero leverage adjoint. The original raw-negative diagnostics
remain available through the base path API. If the conditional denominator
itself is zero or invalid, calibration fails explicitly. This handling neither
proves positivity of the raw Volterra discretization nor removes its bias.

## Discrete reverse through calibration

The pricing tape reverses the log-Euler step and the spatial interpolation of
`ell`, returning an adjoint per **squared leverage node**. For nonzero `V_j`,

```text
bar_ell_j += bar_F_next * F_next
            * (-dt/2 + sqrt(dt)*z/(2*sqrt(ell_j*V_j))) * V_j.
```

For `V_j=0`, use exactly zero rather than evaluating `0/sqrt(0)`. The state
adjoint also includes the derivative of the interpolated leverage with respect
to log-moneyness. Price-only and recorded paths use the same primal recurrence.

The existing particle reverse then propagates these leverage-node adjoints
through the conditional moment ratios, kernel weights and **earlier particle
states**. It is not merely dividing by a frozen conditional expectation. It
returns sensitivities to the original local-variance target nodes, including
the existing market-coordinate target pullback. This is a hand-written
discrete adjoint, not finite differences and not a generic scalar AD tape.

The derivative is conditional on the realized particle sample, fixed bandwidth,
donor/support active set and interpolation branch. Support switches and knot
crossings can be nonsmooth. `retain_reverse_trace=False` permits pricing but
makes recalibrated risk an explicit error. The standard error of node risks in
RQMC is across pricing scramble means conditional on the one calibration;
it does not include variation of the calibration seed. The inherited pseudo-MC
node-risk result has no per-node SE (`None`), and this absence is not reported as
zero uncertainty.

**Local variance nodes are not market implied-volatility quotes.** The output
coordinate is `relative_dupire_variance_nodes_in_f`. To obtain market-IV VegaKT
one must additionally differentiate the chosen arbitrage-free surface fitting,
Dupire construction and their conventions. That chain is outside this API.

## Time grids, resource bounds and reproducibility

Price and calibration grids can differ as in the existing LSV engine. Only
Brownian increment blocks receive the Brownian-bridge transform: two for rough
Heston, lifted Heston, mixed rough Bergomi and lognormal rough SABR; one for
quadratic rough Heston and RFSV. Hybrid near-cell residuals and RFSV Gaussian
levels are not Brownian increments. Full native random dimensions are retained,
including RFSV's `2*n+1`. MC and calibration have separate random domains.

The local-volatility compiler includes dividend dates up to expiry in the time
grid, while retaining the complete dividend schedule for the physical mapping.
The public strict dividend-checkpoint constructor is unchanged; the pricing
compiler uses a horizon-limited internal constructor. This also corrects the
previous rejection of otherwise valid expiry-bounded LV targets with later cash
reserves. Existing requests without later dividends preserve their old grids.

Before allocating the new calibration table, checked arithmetic bounds particle
rows at 8,000,000, target nodes at 1,000,000 and estimated history/kernel-regression
work at 2 billion operations each. Native path step caps and existing pricing
limits also apply. These guards are not latency or accuracy guarantees. There
is no shared mutable calibration state across worker threads. Fingerprints
include the complete model, target, configuration and execution policy.

## Example and status

The [complete Python example](../../examples/python/rough_family_aad_lsv.py)
shows physical Spot Delta and LSV local-variance-node risk with pre- and
post-expiry cash dividends. It intentionally uses a small workload and does not
claim five-IV-bp calibration accuracy. Supplying a local-volatility target built
from SSVI uses the existing surface/Dupire construction; it does not make the
returned local-node adjoints market-IV sensitivities automatically.

This extension does not address the existing small-H full-truncation/time-grid
bias. A derivative may closely match the derivative of a biased discrete price.
See the [validation record](../../design/validation/rough-family-aad-lsv.md) for
the finite comparisons actually executed and remaining acceptance work.
