# Stochastic-dividend Hull–White correlation AAD validation

Scope: three raw Brownian-correlation partials for the dedicated BS/Buehler/HW
plan, including initial funding, future cash claims and delayed payment. Source
and results must refer to the candidate commit; parent passes do not certify it.

## Prespecified checks

- Private Cholesky and cash-coefficient checks use full-model central bumps at
  `h=1e-5` and `1e-6`. Include nonzero and all-zero correlations, `a=0/0.4`,
  piecewise rate volatility with a future knot, `kappa=0`, `alpha=0/1`, and
  `nu_D=0`. Absolute budget: `3e-8` per coefficient, without sampling-SE terms.
- A separate private test constructs a full-rank integrated covariance from a
  piecewise volatility kernel on a coarse grid at an instantaneous PSD boundary.
  The correlation context must still reject the driver law.
- Public Rust full-price recompiles use identical random coordinates for each
  central bump (`h=1e-5`, `1e-6`). Test delayed Asian MC and bridge/antithetic
  RQMC, and a smoothed barrier with pre/post cash seeds. Include a cash maturity
  and rate-volatility knot beyond expiry. Budget:
  `2e-5 + 2e-5*max(abs(AAD),abs(FD))`, without sampling-SE widening.
- Repeat a public bump panel at all-zero correlations and the Ho–Lee limit,
  then with zero equity/dividend diffusion, zero dividend mean reversion and a
  zero cash mean. Unsmoothed discontinuities, instantaneous PSD boundaries and
  zero simulated rate variance must fail explicitly in the new method.
- Verify price and price-SE identity; exact label/derivative/SE prefix equality
  with rate-parameter AAD; correct cash/curve metadata; finite nonnegative SE;
  exact one/three-worker derivative and SE replay.
- Python covers the appended labels/method, preserved public helper slices,
  copy immutability, worker replay, both full-recompile bump widths, and domain
  errors. API members/signatures, type stubs and source-archive membership are
  checked by the existing smoke/source scripts.

## Independent model check

The existing Python no-cash, delayed single-observation Gaussian test also
checks correlation risk. For expiry T, payment U, residual volatility sigma,
constant HW volatility eta and rho=rho_Fr, define

`V = Var(I_T) + sigma^2*T + 2*sigma*rho*eta*J(a,T)`

and the payment-measure forward F from the independently implemented Gaussian
law, with `d log F/d rho = -B(a,U-T)*sigma*eta*B(a,T)`.
Then the call's raw rho derivative is

`P(0,U)*F*(-B(a,U-T)*sigma*eta*B(a,T)*N(d1) + phi(d1)*sigma*eta*J(a,T)/sqrt(V))`.

The two unused dividend-correlation expectations are zero. Compare all three
AAD estimates to these analytic values within `6*sampling_SE + 2e-5`, using the
existing 8192 points, 8 scrambles and fixed seed. This stochastic analytic check
is separate from the deterministic same-coordinate derivative budgets above.

## Execution and interpretation

The focused three-OS risk CI job runs private correlation tests plus existing HW
risk and new HW correlation integration tests in release mode, retaining logs.
Also run normal formatting, Clippy, workspace regressions, Python wheel tests,
Markdown links and source archive checks. Preserve all inherited CI gates,
acceptance budgets and seeds. Report pending long-running acceptance separately.

Sampling SE measures MC unit or RQMC scramble uncertainty only. These tests do
not certify continuous-time risk convergence, quadrature/grid error, smoothing
bias, calibration effects or model adequacy.

See the [decision](../adr/0025-stochastic-dividend-hull-white-correlation-risk.md)
and [model reference](../../docs/models/stochastic-dividends-hull-white.md).
