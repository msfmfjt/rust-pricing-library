# Heston LSV fixed-target parameter reverse validation

Baseline: PR140 `fb2597622a61c9b81d3a3d247d8ef3fb3705196d`.
This increment changes no primal path, model, calibration or payoff formula.

## Fixed tests

The numerical limits below were written into the tests before their numerical
execution. Finite differences always rebuild the model and redo particle
calibration, fixing the same target, grid, sample seeds and bandwidth. They are
checks of discrete derivatives, not independent continuous-time price oracles.

1. Calibration-only Leverage VJP: power kernels H=.1/.3/.5 and an explicit
   three-factor lift with weights .2/.8/1.3 and rates0/.7/12. Nonuniform five-time,
   five-space target,128 particles, seed429, bandwidth.5, ESS3. Seeds on every
   Leverage node; all five scalar parameters and H for power kernels, at
   bumps2e-7/1e-7 (second-order backward at H=.5). Donor/extrapolation topology
   must agree for all bumps; extrapolated nodes must actually be present.
   Bound:3e-5*(1+abs(FD)). **46 comparisons**.
2. High-level recompiled European prices: same four model settings, MC128
   antithetic units and RQMC4x64 with bridge/antithetic, valuation seed819,
   pre-expiry cash3 and post-expiry cash4, nonflat curves. Every scalar/H direction,
   two bumps1e-6/5e-7, same bound. **92 comparisons**. Check unchanged prices and
   old node adjoints, numeric worker1/3 equality, distinct worker fingerprints,
   fixed-scalar versus optional-H equality, and direct+calibration decomposition.
3. Deterministic time-dependent variance identity: nu=0, otherwise nonzero mean
   reversion and v0 != theta. Every particle has the same V_r; L_r^2=a_r/V_r
   cancels all deterministic variance-curve changes at fixed target when grids
   coincide. Direct v0 risk must be nontrivial, total deterministic-parameter
   risks and SEs <=2e-10. Do not impose zero nu sensitivity at its boundary.
4. Independent two-pass scramble statistics: explicitly rebuild all eight
   scrambles and antithetic paths, reverse direct and calibration contributions,
   then recompute their summed mean/SE without the production reducer. Agreement
   <=1e-10. The wrong independent-SE formula must differ by more than1e-3.
5. Domain/trace rejection: no trace, finite-lift H request, unsupported variance
   family, seed shape/nonfinite values. Python adds immutable/copy fields,
   delayed-payment Asian, mixed proportional/future cash, hard Digital rejection
   and smoothed Digital directional bumps. All target bumps recompile/recalibrate.
6. Release Black panel: **eight rows**, four model settings x seeds91/1973;
   S=K100,T1,unit curves,no dividends, target variance.04, four intervals,
   RQMC8x2048 antithetic/bridge and128 particles. Independent Black price
   7.965567455405804; fixed gate |price-reference|<=5*SE+2e-4.
   All deterministic-parameter total sensitivities must be <=1e-8 by the exact
   finite cancellation identity. Nu is excluded from that zero-risk assertion.
   The price is a stochastic estimate; the cancellation check is not a
   confidence interval.

## Initial results and limitations

The first completed fast numerical run passed all four integration tests:
46 calibration comparisons, max absolute gap1.139972312103055e-7;
92 high-level comparisons, max1.026049045727007e-7. The separate covariance-control
unit test passed. A variance-only reverse control also passes while an unused
unlevered asset path overflows: unrelated Pure-SV asset generation must not be
required for an LSV variance pullback. The first release Black run passed all
eight rows, with maximum deterministic-parameter residual
7.105427357601002e-15; no calibration/pricing error budget was inferred from
that cancellation. No numerical threshold, model, seed or bump was changed.

An initial test compilation used the wrong `ExecutionPolicy::new` arity/type;
corrected to u32 worker count and explicit optional reduction block. The initial
compiler log is retained separately. The release Black fixture was checked
against the v1 golden schema (its carry key is `dividend_curve`) before numerical
execution; this is not a parameter fitted to a price outcome.

Final execution counts and configuration-specific outcomes are supplied in the
PR/delivery report, rather than inferred from parent-head tests. Debug tests may
omit local debug symbols for container memory. Hosted three-OS numerical tests
use the unchanged standard release profile and retain logs as required artifacts.

Derivative consistency does not establish finite-particle, continuous-time,
calibration-noise or IV5bp accuracy. The fixed donor/support/interpolation branch
contract excludes switching sensitivities. Negative-raw-variance and singular
boundary behavior are inherited from the already independently tested Heston
reverse. The deterministic zero-risk identity tests nontrivial cancellation,
not the population derivative of arbitrary nondegenerate LSV dynamics.
