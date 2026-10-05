# Mixed rough Bergomi Hurst reverse: fixed validation protocol

This increment appends Hurst after the existing component eta/shared rho risks,
for Pure SV and fixed-target particle LSV. It differentiates the normalized
Gaussian hybrid kernel AND the finite-grid centering variance. Existing primal,
eta/rho, target, Spot and Gamma methods must be preserved.

Inputs and tolerances below were declared before numerical test execution.
Keep initial failures; do not tune sampling seeds or numerical tolerances.

- Low-level path cotangents at every forward and diffusion-variance node:
  H=.001/.03/.1/.3/.5, 16 paths, seed91, times0/.07/.21/.63/1,
  eta=.3/.8,rho=-.6,weights=.35/.65,xi at0/.4/1=.04/.05/.045.
  H bumps2e-7/1e-7, second-order backward at H=.5. Bound3e-5*(1+abs(FD)).
- Kernel: 70-digit mpmath derivatives of direct power-integral expressions,
  all near/residual/older loadings and finite-grid variances for the same five H
  values on the above grid and0/1e-8/10.00000001. Bound2e-12*(1+abs(reference)).
- Nonzero residual derivative at H=.5: one-step terminal variance seed,
  dt1, one eta.7,xi.04, only residual normal1. Exact expectation -.7*.04*exp(-.7^2/2).
- Calibration-only: H=.1/.3/.5, nonuniform5x5 target,128 particles/seed429,
  bandwidth.5/ESS3, all Leverage nodes seeded, matching donor topology across
  H bumps2e-7/1e-7. Require donor nodes actually used. Bound3e-5*(1+abs(FD)).
- Full Pure/LSV request/model rebuild, H=.1/.3/.5, MC128 and RQMC4x64,
  antithetic/bridge, pre-expiry cash3/post-expiry cash4 and nonflat carry.
  H bumps1e-6/5e-7, backward at H=.5. Bound3e-5*(1+abs(FD)).
- Exact eta/rho prefix, price and old-fingerprint regression, workers1/3;
  zero eta/zero fixed xi, unsupported family/interior-rho domain, seed validation.
- Independent two-step conditional Black/Gauss-Hermite, H=.03/.1/.3/.5,
  dt.5,xi0.04/xi(.5).05,eta.3/.8,rho-.6,weights.35/.65,ATM100,unit curves.
  Quadrature48/64 difference<=1e-7, reference-price H differences2e-5/1e-5
  agree within2e-7*(1+abs(reference)). Retain 64-point references; no pricing
  extension is imported. Production RQMC8x4096, seeds91/1973, antithetic/bridge:
  price/Hurst gap<=5SE+.002, 0<HurstSE<=.1. These are two-step finite-law checks,
  not continuous-time or nondegenerate LSV accuracy certificates.
- Separate scramble reconstruction includes direct/calibration covariance and
  Hurst as an additional channel; dropping covariance must be detectable.
- Python: delayed-payment Asian, proportional/future cash, smoothed Digital,
  hard-risk rejection, copy/frozen getters and exact old prefixes.

Sampling errors exclude calibration-seed variation, active-set switches,
time-grid/smoothing/model bias. Pseudo-MC LSV errors are None. Weights and the
full xi curve are held fixed; their risks are not added by this increment.
