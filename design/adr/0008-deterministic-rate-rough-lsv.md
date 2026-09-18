# ADR-0008: Deterministic-rate rough Bergomi LSV

Status: Accepted, experimental model contract
Date: 2026-09-18
Owners: pricing library maintainers
Baseline: main `935cdd8a339ed4437b0cd6c7ba77edf1c380f633`, with the parallel
calibration and price-only paths of the `perf/bergomi-quality-speed` branch.

## Context

[ADR 0005](0005-rough-bergomi.md) placed rough Bergomi inside the equity/Hull–White
hybrid engine. Deterministic rates were available only as a Hull–White model with
zero rate volatility, which still requires HW rates, a three-driver correlation,
an HW calibration target and five Gaussian blocks per step. One- and two-factor
Bergomi LSV, by contrast, have a deterministic-rate plan that takes the standard
`PricingRequest` with a Local Volatility target. The user asked to fill that gap
for rough Bergomi.

The existing deterministic-rate plan is generic over `BergomiDynamics`, whose
factor state evolves from its current value alone. The rough driver is a
Volterra convolution of the whole history, so it cannot implement that trait.

## Decision

Add a deterministic-rate rough-LSV entry point beside the Bergomi one, sharing
everything that does not depend on the variance driver.

- The particle calibration core is generic over a private particle driver: the
  existing Markov factors keep their per-particle state and random coordinates,
  and rough Bergomi reads a precomputed per-particle multiplier history. The
  calibration reverse is shared unchanged.
- `BergomiLsvPricingPlan` and the new `RoughBergomiLsvPricingPlan` are thin
  public wrappers over one private pricing core: target refinement, calibration,
  fingerprinting, the MC/RQMC loop and the calibrated local-variance reverse.
  The Bergomi plan's public API, fingerprint and results are unchanged.
- The rough path plan uses the ADR 0005 kappa=1 hybrid scheme, Volterra weights
  and discrete-variance centring on the execution grid, with three factor-major
  Gaussian blocks per step: spot, orthogonal variance and near-cell residual.
  The near-cell integral is its exact regression on the variance increment plus
  an independent residual. Its first two blocks use the same coordinates as
  one-factor Bergomi LSV.
- `eta` remains the log-variance coefficient. There is no uncalibrated pure rough
  plan and no multi-asset form at deterministic rates in this change; the
  Hull–White plans remain the entry points for those.

## Consequences

- New public Rust items: `RoughBergomiLsvPricingPlan`, `CalibratedRoughBergomiLsv`,
  `calibrate_rough_bergomi_lsv(_parallel)`, `RoughBergomiLsvPlan`,
  `RoughBergomiLsvPath` and `ROUGH_BERGOMI_LSV_SCHEME`. New Python class
  `RoughBergomiLsvPlan`. No wire-format, schema or JSON changes.
- Price scheme `rough-bergomi-lsv-hybrid-kappa1-log-euler-v1`, fingerprint tag
  `pricing/rough-bergomi-lsv-plan/v1`, RQMC dimension `3*N`.
- Calibration memory grows by `particles * time nodes` multipliers and costs
  O(particles * N^2) once; each pricing path costs O(N^2).
- Existing one- and two-factor calibrations are bit-identical before and after
  the refactor.

## Alternatives considered

- **Adapter over the Hull–White engine with zero rate volatility.** Smallest
  change, but it keeps five Gaussian blocks and the HW covariance work per step,
  a different random layout from the Bergomi plans, and an HW target type in a
  deterministic-rate API.
- **Implementing `BergomiDynamics` for rough Bergomi.** Not possible without an
  approximate Markovian lift, which would change the numerical scheme.
- **A separate copy of the calibration and pricing loops.** Rejected to keep a
  single implementation of the particle estimator, its reverse and the pricing
  loop.

## Validation

- The three-block law equals the HW hybrid covariance restricted to
  `(dW_S, dW_v, J)` at zero rate volatility, for several H and correlations.
- At H=0.5 with `eta=2*nu`, calibrated leverage, moments, ESS, the calibration
  VJP, pricing paths and pseudo-MC plan prices match zero-mean-reversion Bergomi
  LSV up to roundoff and the absorbed centring factor.
- The pathwise reverse matches finite differences for squared leverage, initial
  f and all three shock blocks on a nonuniform grid; the calibration VJP matches
  recalibration by finite differences and includes calibration feedback.
- Price-only paths equal recorded paths bit for bit; parallel calibration equals
  sequential calibration for 1, 3 and 8 workers; the plan replays across worker
  counts and its local-variance risk matches bump-and-reprice.
- Python tests cover the public plan, validation errors and a flat-target
  repricing check against the Bergomi LSV plan.
