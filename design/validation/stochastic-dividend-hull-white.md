# Stochastic-dividend Hull–White validation protocol

This protocol is recorded before native test execution. Existing acceptance
thresholds, random streams and all prior stochastic-dividend price/risk references
are unchanged. Evidence is tied to the source tree/commit that was executed.

## New kernel checks

1. Discounted conditional-cash coefficients satisfy the independent backward
   pricing equation, including rate/dividend and rate/equity mixed derivatives.
   Check a=0 and a=0.4, nonzero kappa, at a nontrivial f/Y/x/time state. Time
   derivative uses central h=1e-5; absolute PDE residual below 2e-8.
2. Joint four-coordinate covariance and actual factor reconstruction versus
   independent elementary-kernel Simpson integration across an HW volatility
   breakpoint: each entry below 2e-13 absolute.
3. Exact zero-rate coefficient formula, deterministic-cash limit within 2e-12,
   and sensitivity to post-option-expiry rate-volatility knots.

## Public Rust checks

- At zero rate volatility, first-two-coordinate projection reproduces f/Y states
  exactly and prior physical spots within 2e-12. Check pre/post ex-date values,
  payment settlement and post-expiry funding, with nonzero dividend kappa.
- One-step European prices with kappa=0 and 0.7, nonzero all three correlations,
  cash at expiry and beyond, against independent conditional Gaussian/Black
  integration. References from orders 32 and 40 are respectively
  7.029077680799455 / 7.029077680799348 and
  7.266840212859320 / 7.266840212859318. Each native price must be within
  `6 * sampling_SE + 0.002` in price units. These targets validate a finite split
  with continuous-time reserves, not the exact continuous-time option price.
- At kappa=0, collateral forwards equal `mean*exp(-nu_D*rho_Dr*L(0,T))`,
  independently evaluated for a constant HW volatility. Fixed cash recovers
  `Pi(0)=mean*P0(T)`. Metadata distinguishes cash Q means and forward amounts.
- At 32 steps/year, expected discounted/carry-adjusted stock plus paid cash
  equals initial Spot within `6*SE + 0.02` in price units; expected discount
  equals P0(T) within `6*SE + 1e-12`. This is a selected martingale diagnostic,
  not a grid-convergence certificate.
- One/three-worker numerical price and SE replay, distinct execution-policy
  identity, every new correlation/rate input in fingerprints, full-PSD and
  nonfinite-input errors, and singular but admissible matrices retain pricing.

## Python and packaging

Four new Python tests cover independent one-step prices, cash PV/forward metadata,
immutable copies, worker replay, future HW knots, deterministic fixed-cash limit,
explicit invalid inputs/unsupported risk requests, and one-fixing Asian delayed
payment versus an independent Gaussian Black law. Payment-lag price tolerance is
`6*SE + 0.002`; the grid must end at fixing rather than payment.

The Python oracle imports only NumPy/math, not production model/path/payoff/rate
helpers. Existing full Rust/Python discovery, schemas, 242 equation-reference
cases, links, stubs, dependency direction, docs and source archive are required.
Run new tests in release mode as well as normal platform discovery. A focused
native pass is not a pass for all platform/wheel/replay/legacy acceptance gates.

## Boundaries

No stochastic-dividend HW AAD/Gamma, Bergomi/HW coupling, market-IV calibration,
LSV/VegaKT, multiple assets, early exercise or continuous barriers. No relaxed
4-to-5 bp legacy acceptance threshold is folded into this slice. Sampling SE
excludes quadrature, time-step, smoothing, model and calibration errors. Adaptive
quadrature convergence estimates are not rigorous bounds; domain or convergence
errors must be reported rather than hidden.
