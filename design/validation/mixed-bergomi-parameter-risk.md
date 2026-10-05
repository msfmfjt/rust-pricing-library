# Mixed rough Bergomi eta/rho reverse validation

Base: PR #141, commit `f4e1e7bf17e0cd7df8c3823537d47c359a2848ad`, source tree
`bbd2b06c2af0ae04b05140f7c049b534465b429a`. No parent/main mutation.

## Predeclared checks

The following inputs/budgets are declared before executing the new Rust tests.
They must not be weakened after observing a failure.

* Path VJP: H=.03/.1/.3/.5, eta=.3/.8, rho=-.6, weights=.35/.65;
  xi knots (0,.04),(.4,.05),(1,.045); times 0/.07/.21/.63/1;
  seed91,16 paths, seeds on all forward/variance nodes. All3 parameters,
  bumps2e-7/1e-7, bound2e-5*(1+abs(FD)). 384 comparisons.
* Calibration: H=.1/.3/.5,128 particles,seed429,bandwidth.5,ESS3,5x5 nonuniform
  target; all3 directions,two widths2e-7/1e-7. Fixed donor topology.
  Bound3e-5*(1+abs(FD)),18 comparisons.
* Price: same3 H settings, MC128 and RQMC4x64, antithetic/bridge, cash before
  and after expiry, nonflat curves. Pure SV and fully recalibrated LSV,
  all3 directions,widths1e-6/5e-7. Bound3e-5*(1+abs(FD)),72 comparisons.
* Independent two-step expectation: constant first-step variance .04, xi(.5)=.05,
  dt=.5, S=K100, unit curves, no cash, eta=.3/.8,rho=-.6,weights=.35/.65,
  H=.1/.3/.5. Direct three-dimensional Gaussian integration of conditional Black,
  independent analytic eta/rho partials, orders32/48/64;48/64 gap<=1e-7.
  Rust uses RQMC8x4096, seeds91/1973. Price/each parameter gap<=5SE+.002;
  each parameter has 0<SE<=.05. These are expectations of the TWO-STEP law,
  not continuous-time rough prices or calibrated-LSV oracles.

Additional tests reconstruct RQMC direct/calibration means and total SEs independently
of the high-level reducer, require a detectable covariance contribution, preserve
old price results and worker1/3 numeric bits, check identical-component weight
splitting, zero-weight components, zero xi, eta=0, rho boundaries, invalid seeds,
non-Mixed inputs, missing trace and explicit smoothing. Python adds delayed-payment
Asian and proportional/future cash. H and weight/xi sensitivities are out of scope.

The shared particle-variance transpose is left unchanged except for sibling
visibility. Existing Heston scalar/Hurst/LSV tests remain enabled. No new dependency,
manifest, lockfile, JSON tag or model definition is introduced.

## Evidence policy

The authoring container has Python3.13 but no Rust compiler or direct external DNS.
Local Python mathematical/static checks are not represented as Rust compilation.
Native Rust, binding and package results are to be recorded from actual GitHub
Actions executions. Retain failed build/test logs separately. Run formatting before
native validation; any formatter-only source changes must be captured in the final
source/patch. Experimental status remains unchanged.

[Model contract](../../docs/models/mixed-bergomi-parameter-risk.md).
