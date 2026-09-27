# Independent cash-dividend Hull–White risk oracle

This protocol is fixed before candidate native execution. Parent: PR #99 at
`ebf17dd7098edeca7933d9044688bac5012f6731`. Production price/AAD implementations,
existing seeds, budgets and replay fixtures are unchanged.

## Independent construction

The reference imports only Python standard-library modules and NumPy. It does
not call production pricing, model, covariance, payoff or adjoint code.

- Integrate the four Gaussian Brownian/OU/integral kernels with 32-point
  Gauss–Legendre quadrature, split at each HW volatility knot. This avoids
  subtracting nearly equal mean-reversion formulas near a=0.
- Compute conditional discounted dividend coefficients with fixed-step RK4
  under the maturity-forward Gaussian tilt. Starting at time t, set
  `(m_f,A,B,C)=(1,0,1,0)` and integrate to the cash maturity V:

  ```text
  l(s) = eta(s) B_a(V-s)
  m_f' = -sigma rho_fr l(s) m_f
  A'   = kappa alpha m_f - (kappa + nu_D rho_Dr l(s)) A
  B'   = -(kappa + nu_D rho_Dr l(s)) B
  C'   = kappa(1-alpha) - (kappa + nu_D rho_Dr l(s)) C
  ```

  Then the cash claim is `mean * P_HW(t,V,x) * (A f + B Y + C)`.
  Use 256 RK4 steps/year, splitting at volatility knots; 512 is an independent
  resolution control. This does not reuse the production adaptive cash integrals.
- Condition on `(W_D,x_T,I_T)`. After one actual Buehler split, physical Spot is
  affine in the remaining lognormal f_T. Integrate the call over f_T with Black,
  then integrate the three remaining Gaussian coordinates with tensor
  Gauss–Hermite quadrature of orders 32 and 40.
- Fund cash at and after expiry in initial risky Spot. Remove cash at expiry
  before payoff observation. Discount a delayed payment with
  `D(0,T) P_HW(T,U,x_T)`. Never extend the one-step path to the payment date.

## Controls and fixed budgets

The reference must reproduce the earlier constant-HW one-step price targets
7.029077680799348 (kappa=0) and 7.266840212859318 (kappa=0.7) within `2e-9`.
Its no-cash delayed-payment price must agree with the independent Gaussian Black
law within `2e-9`, for a=0 / 0.4, including a rate-volatility knot after expiry.

The cash-risk panel has Spot/strike 100, sigma=0.2, kappa=0.7, alpha=0.6,
nu_D=0.35, correlations (rho_fD,rho_fr,rho_Dr)=(-0.25,0.25,-0.2), and a=0 / 0.4.
Cash Q means are 3 at T=1 and 8 at 1.4. HW volatility is 0.04 before 1.15,
0.09 thereafter. Input curves are P0(t)=0.95^t and Q0(t)=0.98^t. A one-fixing
Asian observes at T=1 and pays on 2027-12-04 (456/365 from valuation 2026-09-04).

Reference prices at both Gaussian orders and both ODE resolutions must agree
within `2e-9`. Differentiate only these independent prices, using central
stencils at h and h/2; at a=0 use the second-order inward stencil
`[-3P(0)+4P(h)-P(2h)]/(2h)`. Spot and cash h=1e-3; HW volatility h=1e-6;
other parameter/log-DF/correlation h=1e-5. Reference partials at both Gaussian
orders and both widths must agree within `3e-6` absolute. Repeat the selected
order-40, half-width partials with 512 ODE steps/year under the same budget.
The native AAD target
is the order-40, half-width result; no reference is fitted to native output.

Check all 15 non-fixed partials: Spot, residual sigma, kappa/alpha/nu_D, both
cash Q means, nonzero discount/repo log-DF pillars, HW a and both volatility
knots, and all three Brownian correlations. The two time-zero curve partials
must be exactly zero. Preserve result-label order and price/SE identity.

Use antithetic, bridge RQMC with 8192 points, 8 scrambles and fixed seed 2909.
Native prices use `6*sampling_SE + 0.002`; each native partial uses
`6*sampling_SE + 2e-5`. Require finite nonnegative SE and record each reference,
estimate, sampling SE and difference as JSON. The full panel is run at a=0 and
a=0.4. No failed seed or threshold is replaced.

## Execution and limits

Run the independent reference controls locally without a Rust module. Full
native comparisons run in existing macOS/Windows clean-wheel discovery and
retain their output in job logs. Require source-archive membership, Markdown
links, Python syntax, and inherited platform gates. Tie evidence to candidate
SHA; report pending legacy CI separately.

These are independent expectations of a **single finite split** with
continuous-time reserves. They complement common-coordinate implementation
derivatives and the price/Delta/Gamma grid panel, but do not establish
continuous-time parameter-risk convergence. Gaussian/ODE/stencil refinements
are numerical consistency checks rather than rigorous error bounds. Sampling SE
does not include grid, quadrature, smoothing, calibration or model error.

See the [model reference](../../docs/models/stochastic-dividends-hull-white.md),
[rate-risk protocol](stochastic-dividend-hull-white-parameter-risk.md), and
[grid-refinement protocol](stochastic-dividend-hull-white-refinement.md).
