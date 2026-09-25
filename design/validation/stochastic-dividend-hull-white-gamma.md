# Stochastic-dividend Hull–White Gamma validation

Candidate builds on PR #97 head `98409dcab71bfb3f95576eddc964730410b28297`.
No existing acceptance budget, seed, schema or replay fixture changes. Passes
must be associated with the new source commit, not inherited from the parent.

## Prespecified numerical checks

- Full-recompile AAD Delta bumps: European, delayed Asian and smoothed pre/post
  cash barrier; MC and antithetic/bridge RQMC; two seeds; piecewise HW with
  `a=0.4`, `a=0`, and zero-rate volatility. Include a rate knot and cash maturity
  beyond expiry. Each half/base/double Gamma agrees within `2e-12` absolute.
- Exact base price/price-SE/Delta/Delta-SE identity with existing pricing/basic
  AAD. Exact numerical one/three-worker replay after excluding execution-policy
  fingerprints; absolute/relative equivalent-width equality with distinct risk
  fingerprints; seven payoff evaluations per simulated path.
- Independent one-step Ho–Lee construction with `kappa=0` and zero equity/rate
  and dividend/rate correlations. Construct `x=eta*z2` and
  `I=eta*(z2/2+z3/sqrt(12))` for `T=1`, explicit lognormal f/Y, future discounted
  cash and delayed-payment discount. Compute Delta and all paired Gamma/gap
  samples without production path/payoff/reverse/covariance helpers. Compare
  means and sample SE within `2e-12` for MC/RQMC, antithetics on/off. RQMC units
  are scramble means.
- Zero-rate fixed-cash limit: compare each finite-bump target against independent
  Black Delta differences, with cash at expiry and after expiry. Budget:
  `6*sampling_SE + 3e-5` using 32768 points and 8 scrambles.
- Reject invalid/indistinguishable bumps, unfunded downward scenarios and
  unsmoothed discontinuities without mutating basic risk. Zero-diffusion deep-ITM
  Gamma and SE are below `1e-12`; perfect correlations retain Gamma support.

## Python and independent stochastic-rate reference

Three Python tests cover public methods, recompiled Delta bumps, paired gap
metadata, immutable copies, worker replay, absolute/relative identity, invalid
conventions and funding, and singular-law support. Type stubs, runtime signature
and public member sets, and source-archive membership must match.

The third test uses the independent no-cash delayed-payment Gaussian law at
nonzero HW volatility and equity/rate correlation. With
`V=Var(I_T)+sigma^2*T+2*sigma*rho*eta*J(a,T)` and payment-measure forward
`F(S)=S*Q(T)/P(T)*exp(-B(a,U-T)*(eta^2*B(a,T)^2/2+sigma*rho*eta*B(a,T)))`,
the call Delta is `P(U)*(F(S)/S)*N(log(F(S)/K)/sqrt(V)+sqrt(V)/2)`.
Compare the three finite Delta-difference targets within
`6*sampling_SE + 3e-5`, at 8192 points, 8 scrambles, fixed seed 1801. This is a
known finite-bump target rather than an analytic zero-bump Gamma assertion.

## Execution and interpretation

Run the new Rust test target in the existing three-OS focused HW risk CI job,
alongside basic/rate/correlation risk tests. Run formatting, Clippy, workspace
regressions, wheel/Python/source checks and legacy acceptance separately. Record
observed results and pending gates in the PR, tied to the candidate SHA.

Neither small paired SE nor small ladder gaps certify general nonsmooth Gamma
accuracy or continuous-time convergence. SE measures only fixed-bump MC units or
RQMC scramble uncertainty. It excludes time-grid, quadrature, smoothing,
calibration and model error.

See the [decision](../adr/0026-stochastic-dividend-hull-white-gamma.md) and
[model reference](../../docs/models/stochastic-dividends-hull-white.md).
