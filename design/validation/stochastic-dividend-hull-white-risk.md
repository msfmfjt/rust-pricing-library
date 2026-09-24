# Stochastic-dividend Hull–White basic AAD validation

Scope: constant residual volatility, Buehler dividends, stochastic HW rates;
HW parameters/correlations fixed. Parent price implementation: PR #94, commit
`53a0de3ab593d431c0c82c511ea91368d63c85c6`.
This protocol is recorded before native execution. Results must be tied to the
actual candidate source; no parent or unobserved CI result is a child pass.

## Prespecified checks

Two new private Rust tests:

- Conditional A/B/C derivatives versus full recompilation, rate vol zero/nonzero,
  future volatility knots, sigma/kappa/nu zero and alpha endpoints. Central h=1e-5
  and 1e-6: 3e-7 absolute. Valid-side differences at boundaries: 2e-5 absolute.
- Independent elementary kernel integral at rate a=0 and dividend kappa=0:
  nonzero kappa tangent in the zero-valued A/C coefficients versus 20,000-panel
  midpoint integration, 2e-11 absolute; B and its tangents 2e-14.

Five new public Rust tests:

- All active basic labels, MC/RQMC and seeds 1973/1051, rate mean reversion zero
  and positive, piecewise vol including a post-expiry knot, and deterministic
  rate limit. Full-recompile h=1e-5 and 1e-6 differences use
  `3e-5 + 2e-5*max(abs(AAD),abs(FD))`, without sampling-SE widening.
- For terminal calls, use the already established independent hinge protocol:
  reconstruct base/up/down discounted intrinsics from public primal paths,
  verify each reconstructed call mean versus its full public price within 2e-12,
  and subtract the independently measured finite-stencil hinge remainder. Report
  crossings/raw mismatch; do not infer the remainder from AAD-FD error. This is
  predeclared, not a post-failure tolerance change.
- Irregular Asian observations with delayed payment, and smoothed barrier seeds
  before/after dividend settlement, use direct full-recompile FDs. Asian and
  barrier strike 20 separates state/discount reverse from the vanilla hinge.
  Unsmoothed barrier errors must not change subsequent price evaluation.
- Zero cash, zero sigma/kappa/nu and alpha endpoints use inward h=1e-7 (cash 1e-6),
  3e-4 absolute (cash 3e-5). A fixed singular correlation still supports AAD.
- One-step zero-rate case versus previous fixed-rate AAD under the identical
  first two normal coordinates: price 2e-12, all derivatives 2e-11. Baseline
  price/SE and numerical one/three-worker replay are exact within the HW adapter.
  All labels, cash/curve metadata, raw/scaled accessors and finite SEs are checked.

Four new Python tests:

- Direct two-width full recompilation of Spot, sigma, dividend parameters,
  post-expiry cash, discount and repo pillars with the same gradient budget.
- Immutable result/accessors, price and SE identity, RQMC unit count and exact
  numerical worker replay. Unsupported HW Gamma/model/correlation methods remain
  absent. API constructors still require price-only request flags.
- Zero cash and zero model/diffusion boundaries, inward h=1e-7 and 3e-4 absolute.
- No-cash delayed one-fixing Asian: Delta and initial-volatility Vega versus
  independent payment-measure Gaussian Black formula, `6*SE + 2e-5`. The path
  still ends at expiry, not payment. This is a special-case analytic risk oracle.

## Regression and interpretation

Run the prior HW price tests and 46 rough/dividend integration tests without
changing their prices, seeds or tolerances. Update only the former Python
`evaluate_aad` absence assertion for the now-supported method. Run Clippy,
all-feature/minimal/binding Rust suites, full Python discovery, wheel/example,
242 reference equations, schemas, archive/stub contracts, docs and dependency gates.

Sampling SE covers independent MC (antithetic) units or RQMC scramble means only.
No certification of continuous-time risk, adaptive quadrature error, smoothing,
calibration or model accuracy is implied. Full platform and extended legacy
acceptance remain separate from the focused tests above.

See [decision](../adr/0023-stochastic-dividend-hull-white-risk.md) and
[model reference](../../docs/models/stochastic-dividends-hull-white.md).
