# Fixed-kernel Heston MC parameter reverse: validation protocol

This increment differentiates the existing discretized Pure-SV variance/asset
path. It does not validate a new model or remove full-truncation bias.

## Fixed tests

1. Both Heston families, H=.1/.3/.5 (finite lift fixed), nu=.15/.7, 16 paths each,
   seed91, nonuniform times0/.07/.21/.63/1. Seed all five forward and diffusion-
   variance observations. Compare all five scalar directions at bumps2e-7/1e-7,
   tolerance2e-5*(1+abs(reference)); require unchanged truncation counts, identical
   primal paths and more than50 observed negative nodes. Initial-forward checks
   use bump1e-3 and tolerance2e-9*(1+abs(reference)). No failed path is discarded.
2. High-level price reverse: both models, MC128 antithetic units and RQMC4x64,
   bridge, seed819, maximum step.2, cash3 at.25 and4 at1.5. All five parameters,
   bumps1e-6/5e-7, tolerance3e-5*(1+abs(reference)); all forty comparisons retained.
   Workers1/3 must agree numerically; execution-dependent fingerprints differ.
3. H=.5 rough versus equivalent single-factor lift, twelve paths; direct analytic
   one-step terminal-variance derivatives; zero seeds; invalid seeds/normals,
   unsupported family, v0=0, rho endpoints and exact-zero raw-variance rejection.
4. Release Black constant-variance panel: both models, kappa=nu=0, v0=theta=.04,
   H=.2, T1, S=K100, unit discount/repo, no cash, four steps, RQMC8x4096 with
   antithetic/bridge, seeds91/1973. Independent analytic dPrice/dv0 reference
   99.23813686925295. Fixed bound abs(error)<=5*SE+.02;0<SE<=.1. Kappa, theta and
   rho sensitivities must be zero. Nu is an inward finite-path derivative and
   is not asserted zero. This is a degenerate limit, not nondegenerate pricing
   or IV5bp admission.

A separate Rust control reconstructs MC/RQMC sampling means and errors from
low-level path adjoints, sharing the primal path/RNG and reverse but not the
high-level estimator. Python tests check full parameter-recompiled price bumps
and additionally exercise Asian observations,
nonflat carry, delayed payment, future/proportional dividends, output ownership,
unsupported families, boundary errors, and hard discontinuity rejection.

## Evidence interpretation

Finite-difference agreement checks the discrete derivative, not continuous-time
bias, model calibration or moment stability across the parameter domain.
New low-level variance observation seeds also verify adjoints beyond those
needed by one particular European payoff. The final public-call price must equal
the old price API. Marginal SEs cannot be combined under independence.

The focused read-only CI runs default/minimal/release, explicitly executes the
ignored Black panel and retains the log. Full repository CI remains unchanged.
Record actual completion in the PR/report, not by assuming requested runs passed.
