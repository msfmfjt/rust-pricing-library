# Heston IV grid validation: protocol and evidence

## Scope and fixed comparisons

Additive to PR #128, source tree `4938d14b174b93328c0d8a8e6463d547dafe1b83`.
Production pricing, Riccati tangents, Black inversion, least-squares optimizer,
quote conventions, schemas and dependency manifests are unchanged. The new layer
orchestrates existing calls and provides explicit diagnostics and bounded stages.
The [specification](../../docs/models/heston-iv-refinement.md) defines acceptance.

The retained [protocol](../../fixtures/rough-volatility/iv-refinement.json) hashes
the parent independent IV fixture. Four ordinary-Heston/explicit-lift fits use
those independently computed targets, not targets regenerated from this change.
Two SSVI fits use the parent's nine target coordinates and two starting points.
Their numerical prices still use production Fourier; this is not a new independent
rough-price oracle.

The original six-scenario panel is fixed at `N=32, M=256, U=64`, at most two stages,
60 iterations and 100 evaluations per stage, scaled solver tolerance `2e-6`, raw
IV fit tolerance `5e-4`, and grid tolerance `1e-4`. All six passed on their first
numerical run, at the first stage. Parent bounds and targets were retained.

A subsequent pair of SSVI scenarios specifically exercises a successful warm-start
refinement. Its grid tolerance is `1e-5` (0.1 IV bp), with the same fit tolerance,
objective, solver options, bounds, starting vectors and quote inputs. The final
pair starts at 64 time steps with 256 intervals and cutoff 64. Validation after
its second calibration stage therefore uses joint grid `(256,4096,256)`.

## Initial failure and precision change

The added tight pair initially used 32 time steps like the original six. Its first
case reached the second stage and failed the unchanged `1e-5` grid criterion:
maximum IV grid change `1.173437067337e-5` (approximately 0.11734 bp). That test run
stopped at the first added case; it is not evidence for the other added start.
The original six conditions had all passed again before this failure.

Only the tight pair's initial time resolution was doubled to 64. Both starts use
the same revised grid; no threshold, target, parameter bound, frequency grid or
optimizer tolerance was relaxed. The original six retain the original grids.
Logs and both pre-change protocol/test snapshots are retained in the delivery.
The initial test build also used a nonexistent `SurfaceValidationTolerance::default`
and was fixed to the repository's existing `local_vol_vegakt_v1` constructor; no
numerical acceptance was run or claimed for that failed build.

## Contracts and checks

The six fast Rust integration tests cover independently known constant-volatility
IV; conflicting identical quotes whose analytic IV least-squares optimum is 0.25;
separate public repricing at every probe and non-initial point for both families;
exact warm starts and stage grids; stopping at acceptance; preserving original
calibration outputs and input problems; raw rather than scaled tolerance units;
duplicate/order preservation; rejecting impossible future grids and excessive
aggregate budgets; and rejecting a coarse fit that fails the grid-change criterion.

Python tests cover the same high-level reports, independent constant-IV values,
frozen classes and copying of returned lists. The existing stub mutation suite
also exercises the new HestonIv classes and every required member. New protocol
mutation tests reject changed thresholds, parent references, bounds, starts,
probe order and missing CI/evidence requirements. Source archive membership and
the registered example are required, not optional.

The dedicated three-OS workflow runs debug/minimal tests and explicitly includes
the ignored numerical panel in release; it retains `heston-iv-refinement.log`
with missing artifacts treated as failures. Full workspace, Clippy, formatting,
wheel and source checks are recorded separately. Defining a workflow is not a
claim that its remote run passed.

## Interpretation

Small directional or joint IV differences do not bound continuous-time error or
Fourier truncation error. All checks retain the same calibration quote set; there
is no out-of-sample or market-data test. Boundary stationarity is not a global
optimality or parameter-identification certificate. Raw price/IV sampling errors
and the accuracy of the Monte Carlo scheme are separate questions. All new APIs
remain experimental.

## Final numerical observations

The frozen protocol's eight scenarios passed their respective policy checks. The
original six report lines reproduce their original printed values exactly. Both
tight SSVI cases used two stages: initial maximum grid changes were approximately
`1.17355e-5`, and second-stage maximum grid changes were `4.110060217499e-6`
(0.0411006 IV bp). Their final maximum target residuals across all five grids
were at most `8.361641129029e-5` (0.836164 IV bp). The stricter optimizer residual
criterion `2e-6` remained unmet for SSVI, so its calibration `fit_achieved` remained
false even though the distinct finite-grid policy passed. No flag was overwritten.

Early Python tests incorrectly called the pre-existing keyword-only model factory
positionally; the test and example calls were corrected without changing that API.
The next Python run identified that new refinement input errors wrapped as
`FourierError::InvalidInput` were surfaced as `PricingError`. A mapper limited to
the new refinement methods now returns `ValidationError` for these invalid inputs;
the original APIs and all other numerical-failure mappings are unchanged. The
expected exception was not broadened. Earlier failure logs remain retained.
