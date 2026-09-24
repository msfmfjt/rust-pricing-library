# Rough stochastic-dividend correlation risk: validation protocol

Specified before native execution against PR92 head
`509ccdfe65e4dcb09b0c6430b4cacb6475a917f9` (source tree
`781e195a293aba9201330c3cf73d863691f4c747`). Keep previous source/price fixtures,
seeds and tolerances unchanged. Parent test success is not child acceptance.

## Coefficient and history checks

Two private Rust tests:

1. All three symmetric correlation-entry tangents of the actual factor row
   versus independently recompiled factors: `3e-8` absolute. Cover H values
   0.01/0.1/0.49/0.5, zero correlations and two mixed correlation matrices.
   Independently differentiate the complete newest-cell covariance using
   elementary power integrals (`2e-12` absolute): f/dividend/vol cross terms,
   f/near and dividend/near terms, and zero variance derivatives for the vol
   and near-cell marginals. The grid variances must be bitwise unchanged when
   only correlations are recompiled.
2. Full-history pullback against two-scale, independently recompiled loading
   dot-products: h=1e-5 and h=1e-6, `3e-8` absolute, same H range, eta=0/0.6,
   zero/mixed correlations and an irregular grid. Eta=0 gives exact zeros.
   Perturbing final-step normals cannot affect any earlier left-endpoint
   loading or its correlation derivative. Wrong lengths/nonfinite seeds and
   normals must reject.

## Public Rust checks

Four integration tests:

- Three-entry full-recompile price comparisons at H=0.1/0.49/0.5, MC/RQMC,
  seeds 791/433, cash at 0.5/1.0/1.4 including post-expiry funding. Exact
  `evaluate_rough_aad()` prefix, price/SE, cash/curve accessors and repeated
  evaluations. One/three-worker numerical results must match bitwise;
  execution-policy fingerprints must retain their distinct identities.
- Irregular Asian observations with delayed payment, and a fixed-width smoothed
  barrier with pre/post dividend observations. Unsmooth discontinuous risk
  rejects and leaves price evaluation unchanged.
- Zero correlations, eta=0 and sigma0=0, including H=1/2. The last two
  volatility-correlation derivatives/SEs are exactly zero at eta=0 or sigma0=0;
  the direct dividend/equity partial remains active.
- Perfect and nearly singular 3x3 matrices reject only the new correlation
  scope, including eta=0 and H=1/2. Price, basic/H-eta AAD and Gamma retain
  support and their numerical results after the failure. Include a matrix
  with a nonsingular leading 2x2 block but singular final pivot.

Both h=1e-5 and h=1e-6 must satisfy
`abs(AAD-FD) <= 3e-5 + 2e-5*max(abs(AAD),abs(FD))`.
Sampling SE never widens this deterministic budget.

For unsmoothed terminal calls, **predeclare** the independent finite-hinge
correction established by the PR92 diagnostic. Public primal paths and original
random coordinates reconstruct discounted intrinsic values x0/x+/x-, with no
AAD or payoff-reverse helper. All three reconstructed prices must agree with
the corresponding full public price within `2e-12`. Define

`R_h = mean[(x_plus^+ - I0*x_plus - x_minus^+ + I0*x_minus)/(2h)]`,
`I0 = 1{x_base > 0}`.

Apply the derivative budget to raw FD minus this independently measured
remainder. The remainder is exactly zero without a branch change, is never
inferred from AAD-FD error, and any crossing/raw FD/remainder is printed.
Nonzero remainders are not relabelled direct-FD passes. Asian/smoothed-barrier
cases retain direct price-FD comparisons. This isolates the derivative of the
finite algorithm from a finite stencil that crosses the call's kink.

## Python and regression checks

Three new Python tests: all three raw-entry price FDs at both scales and
H=0.1/0.49/0.5; exact H/eta/basic prefix and cash/curve accessors; immutable
copies; worker replay and fingerprint identity; zero-loading and singular-domain
behavior without breaking older scopes. The new executable example prints the
three raw derivatives, sampling SEs and linearized +1 percentage-point changes.
Python FD budgets match Rust; these selected Python cases use direct FDs.

Run prior rough/stochastic-dividend integration tests (42), all ordinary release
Rust suites (workspace all features, bindings, minimal), full Python discovery,
wheel build/install, explicit static stub shape, source archive, 242 existing
reference cases, schemas, Markdown links, Rust docs and dependency direction.
Normal PR platform CI and full legacy extended acceptance remain separate.
Do not mix the separate PR85 5 bp requirement change into this feature.

## Interpretation and evidence

No finite-difference agreement establishes continuous-time rough-risk accuracy.
The standard errors come from the existing paired MC-unit/RQMC-scramble risk
reduction; no change to that estimator is proposed. H=1/2 is supported for
correlation differentiation despite the zero newest-cell residual. Singularity
of the augmented near-cell covariance is not itself a Brownian-domain failure.

Record exact input/formatted source trees, native run IDs, observed failures,
passed gates and outstanding platform/legacy gates in the pull request. Keep
draft until the intended acceptance scope is reviewed.
