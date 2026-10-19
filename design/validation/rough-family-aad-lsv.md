# Six-family MC Delta and LSV discrete AAD validation

## Fixed scope

Based on main `267da1aa0104116317d7da3ff427d069db4c3a25`, source tree
`13976ef1330e1505ae8a982a3c009c3bae67eebb`. Add explicit fixed-model MC Spot Delta
and a fixed-driver LSV price/local-variance-adjoint adapter. No Gamma, market-IV
VegaKT, stochastic-model-parameter reverse, stochastic rates/dividends or
multi-asset support is claimed. The detailed
[calculation specification](../../docs/models/rough-family-aad-lsv.md) is normative.

Production and test responsibilities remain in three crates. The existing LSV
particle/conditional estimator and its reverse are reused; no new dependency,
manifest/lockfile, JSON model tag or volatility-parameter convention is added.
Volatility-history extraction must retain original pure paths. The one deliberate
shared behavior correction stops the LV pricing grid at expiry while retaining
later cash dividends in the physical escrow mapping.

## Nondegenerate derivative contracts

`crates/pricing/tests/rough_family_aad_lsv.rs` fixes six nondegenerate models,
nonuniform grids, seeds, two bump sizes and thresholds in source. Finite
differences are independent derivative checks of the discrete primal, not a
production calculation and not independent continuous-time model references.

- Initial-state VJP: all six families, 16 paths each, five observation seeds;
  retained recorded/primal paths equal; bumps 1e-3/5e-4, tolerance
  `2e-9*(1+abs(reference))`. Bad seed shapes/nonfinite seeds are rejected.
- Frozen-leverage path VJP: all six families, nonuniform 0/.15/.4/.7/1-year grid,
  initial state 99 with anchor 100; initial and leverage-direction bumps
  1e-5/5e-6, tolerance `2e-8*(1+abs(reference))`. Explicit truncated-zero-variance
  step checks identity evolution and zero leverage adjoint.
- Full calibration pullback: 128 particles, seed429, bandwidth.5, minimum ESS3,
  5x5 positive sloping local-variance grid. Recalibrated bumps 1e-7/5e-8,
  tolerance `2e-6*(1+abs(reference))`. Check unchanged donor topology, nonunit
  initial second moments, first-row leverage equation and worker1/3 equality.
- Public pure physical Spot Delta: all six models, both MC and RQMC; antithetic
  and Brownian bridge; fixed cash3 at t=.25 and cash4 at t=1.5, one-year expiry;
  Spot bumps 1e-4/5e-5, tolerance `2e-8*(1+abs(reference))`. Check price and price
  SE equality with price-only evaluation, Delta/SE equality across worker1/3.
- Public LSV risk: all six models, both engines, full target recompile and
  recalibration for each local-variance-direction bump 1e-7/5e-8;
  tolerance `3e-6*(1+abs(reference))`. Check price/worker equality and the
  conditional node-SE contract; missing traces fail explicitly.
- Independently reconstruct four scramble means and two-pass SEs from native
  uniforms, inverse normals, Brownian blocks and terminal payoffs. All six
  native layouts are used, including RFSV's extra initial Gaussian coordinate.
  This shares the production primal path and RNG, but independently tests
  orchestration, grouping and payoff reduction, not the RNG distribution.
- Future-dividend regression: expiry-bounded target is accepted, target ends at
  expiry, and deleting the future cash dividend changes the price. Later cash
  is retained in reserve, not silently discarded to make compilation succeed.
- Reject nonlognormal SABR LSV and resource overflow; pure SABR beta0/.5
  initial-state reverse remains available and checked independently by bumps.

These fixed finite panels do not establish unbiased Greek estimates, correct
behavior at every support switch, or broad path-dependent product admission.

## Independent Black-limit numerical panel

The ignored release test `six_family_black_limit_price_delta_and_lsv_node_risk`
is explicitly executed by dedicated CI. Six constant-variance model limits,
two original seeds91/1973, one-year ATM call, F100, unit discount, no dividends.
Zero vol-of-vol and variance.04 are used; RFSV mean log vol is log(.2).
Eight steps; 128 calibration particles, bandwidth.5/ESS3; 8 independent RQMC
scrambles, 2,048 points per scramble, antithetic pairs and Brownian bridge.

Independent analytic Black references are price7.965567455405804,
Delta.539827837277029 and parallel local-variance derivative99.23813686925295.
The last reference differentiates C(sigma) through variance=sigma^2; it is not
market-IV Vega. Sum all target-node adjoints for this additive variance direction.

Fixed gates before first numerical run:

```text
abs(pure price - Black) <= 5*pure price SE + 2e-4
abs(pure Delta - Black Delta) <= 5*Delta SE + 2e-4
abs(LSV price - Black) <= 5*LSV price SE + 2e-4
abs(sum node adjoints - Black variance derivative) <= 5*sum node SEs + .02
```

The sum of marginal node SEs is conservative, not the SE of their correlated sum.
Per-row finite statistical gates do not make simultaneous coverage guarantees.
Record all twelve `AAD_LSV_BLACK` rows. This validates independent deterministic
volatility limits; it is not nondegenerate LSV smile/calibration accuracy evidence.

## Failure records and disclosure

Initial test scaffolding used an outdated JSON engine field, then exposed the
pre-existing post-expiry dividend grid rejection. Extending the target beyond
expiry was also rejected by the existing target contract. Keep the expiry-bound
fixture and full cash schedule; fix the compiler's checkpoint horizon instead.
A Python Asian fixture initially supplied three unit weights rather than weights
summing to one; replace each with 1/3 to satisfy the existing product contract.
The corrected Asian test covers all six models, delayed payment, fixed-cash Spot
Delta, and a direct SSVI-to-Dupire target with finite LSV node risk. This is an
integration check, not an independent nondegenerate SSVI accuracy guarantee.
Two new test API-name/bridge-arity mistakes and Clippy range-loop warnings were
corrected without relaxing numerical conditions. Preserve initial logs separately
from successful reruns. Whole-workspace and wheel results are recorded in the
accompanying delivery report, not inferred merely from this protocol.

## API, replay and evidence gates

Python tests exercise all six new Delta/LSV routes, full recalibration bumps,
worker replay, owned arrays/frozen result properties, unsupported nonlognormal
LSV, unsmoothed digital rejection, and mutation of the checked-in stub contracts.
Wheel smoke tests execute the complete new example in addition to existing ones.
The source-archive gate requires all new modules, tests, docs, example and CI.

The dedicated read-only Linux/macOS/Windows workflow runs debug, minimal and
release tests, explicitly includes the ignored numerical panel, and retains
`rough-family-aad-lsv.log`; absent evidence is an upload error. Other existing
rough/LSV numerical CI checks remain active. Local and remote completion are
reported separately. A newly defined workflow does not imply successful CI.

## Remaining work

Full stochastic-parameter AAD (including fractional kernels and RFSV covariance),
MC Gamma, high-level frozen/recalibrated LSV Spot Delta, market-IV VegaKT,
broader smoothed/discontinuous/path-dependent payoff validation, conditional
support-switch analysis and calibration-noise uncertainty remain separate tasks.
Rough SABR LSV beta<1 and quadratic feedback driven by the *leveraged* return
rather than the unchanged variance driver are not covered. Finite derivative
agreement does not remove inherited discretization or calibration bias.
