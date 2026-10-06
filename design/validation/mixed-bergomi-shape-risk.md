# Mixed rough Bergomi shape-risk validation

Base: PR143, f258e624ddebb470fc5dc813a922e7c42d383a12.
No changes to primal pricing, random layout, curves, calibration or existing risk
arithmetic. No new dependencies, schema or Cargo.lock changes.

Tests use m=3, weights (.2,.35,.45), eta (.3,.8,1.1), rho=-.6.
Curve choices: original knots (0,.4,.9,1.4) with values (.04,.05,.045,.055),
or exponential initial .04/growth .2. All numerical inputs/budgets are encoded in
the tests, not selected by observed output.

- Path transpose: H=.03/.1/.5, 16 paths/seed91, times 0/.07/.21/.63/1/1.6,
  fixed forward and variance seeds at every observation, widths 2e-7/1e-7.
  All 960 scalar comparisons use 3e-5*(1+abs(reference)). Includes original
  xi interpolation, flat extrapolation and exponential coordinates.
- Calibration transpose: H=.1/.3/.5, nonuniform 5x5 target, 128 particles/seed429,
  bandwidth .5, ESS3, both curve types/all coordinates/two widths; 60 comparisons.
  Require donor topology unchanged and actual extrapolated nodes. Same bound.
- Fully rebuilt prices: H=.1/.5, MC128 or RQMC4x64, anti/bridge, seed819,
  cash3 at .25 and cash4 at1.5, nonflat carry. Pure and LSV, both curves,
  all directions at 1e-6/5e-7: 160 comparisons with the same bound. Includes worker
  reproducibility, old-result identity, separate risk fingerprints, and LSV xi
  cancellation <1e-8 with a nontrivial direct contribution.
- Separate eight-scramble two-pass covariance reconstruction checks that total
  errors are formed AFTER adding direct/calibration contributions. Incorrect
  independent-SE addition must differ detectably; xi cancellation is not bypassed.
- Domains: wrong family, zero weights/xi, missing trace, invalid seeds, excessive
  dimension, fixed rho endpoints, one-component/constant curves, identical etas.
- Independent constant-variance Black: two seeds91/1973, RQMC8x4096 anti/bridge,
  S=K100,T1, unit carry, xi=.04, eta0. dP/dxi=99.23813686925295;
  abs(error)<=5SE+.02 with 0<SE<=.1. Weight transfer is zero.
- Independent NONDEGENERATE two-step law: H=.1/.3/.5, weights .35/.65,
  eta .3/.8, rho-.6, xi at0/.5/1 =.04/.05/.045. Three Gaussian coordinates
  integrated with Gauss-Hermite, last asset normal analytically via conditional
  Black. Differentiate the first-step asset level for xi0 and second-step variance
  for xi(.5) and the mixture transfer. Terminal xi(1) risk is exactly zero.
  Retain orders48/64 (gap<1e-7), plus independent price stencils5e-6/2.5e-6
  (difference<1e-6*(1+abs(reference))). Production RQMC8x4096, two seeds:
  price and risk gaps<=5SE+.003, nonzero-coordinate 0<SE<=.2.
- Python adds delayed-payment Asian, fixed/proportional/future cash, original
  curve knot beyond final observation, copied/read-only results, workers,
  hard Digital rejection and smoothed Digital bumps.

Discrete bump agreement and two-step expectations do not certify continuous-time
accuracy or 5 IV bp pricing. LSV xi cancellation is a structural identity, not a
confidence interval or real-market calibration result. Risk SE excludes calibration
randomness, time-grid/support/donor switching and model bias.

Strict Rustdoc is checked without warning suppression. Existing bracket formulas
are wrapped as inline code to avoid five false intra-doc links; this is doc-only.

## Execution issues retained separately

The initial usage example named a nonexistent `RiskRequest.price_only` helper and
omitted required `worker_threads`; it was corrected to the existing API. No
pricing input or numerical gate was changed. The first full-smoke attempt stopped
at that example; its failed log is retained separately from the completed retry.
A concurrent debug/release compilation was killed with SIGKILL under the
container memory limit; the release execution is repeated without a concurrent
large compile. Standard release flags and numerical acceptance limits are kept.
