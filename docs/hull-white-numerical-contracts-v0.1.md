# Equity/Hull–White numerical contracts

Status: experimental, price only. Date: 2026-09-12.
Decision: [ADR 0001](adr/0001-hull-white-equity-hybrid.md).

## Model and deterministic curve shift

Let P0(t) be the supplied discount curve, f0(t) its instantaneous forward rate,
and a >= 0 constant. The model is

```text
dx(t) = -a*x(t) dt + sigma_r(t) dW_r(t),   x(0) = 0,
I(t)  = integral_0^t x(s) ds,
V(t)  = Var[I(t)],
r(t)  = f0(t) + x(t) + 0.5*V'(t).
```

Rate volatility is left constant on `[time_i,time_{i+1})`, begins at zero,
and extrapolates flat after the last node. Initial curve rates may be negative.
The a=0 Ho–Lee limit and sigma_r=0 limit are supported. The shift equals
`Cov[x(t),I(t)]`; its time integral over `[s,t]` is `(V(t)-V(s))/2`.
No numerical differentiation of the input curve is needed. In particular,

```text
Dbar(t) = D(t)/P0(t) = exp(-I(t) - V(t)/2),
E[Dbar(t)] = 1.
```

`HullWhiteModel.bond_price(curve,t,T,x)` takes the shifted OU state x, not the
short rate. With `B(a,h) = (1-exp(-a*h))/a` and its continuous a=0 limit,

```text
P(t,T)/(P0(T)/P0(t))
  = exp(-B(a,T-t)*x - (V(T)-V(t))/2 + Var[J(t,T)]/2),
J(t,T) = integral_t^T sigma_r(u)*B(a,T-u) dW_r(u).
```

The bond option is the usual forward Black formula with discount P0(expiry),
forward P0(maturity)/P0(expiry), and total log variance
`B(a,maturity-expiry)^2 * Var[x(expiry)]`. These are initial-curve formulas,
not a calibration to rate option quotes.

## Exact joint innovations and equity coordinate

The Brownian drivers are ordered equity, Bergomi volatility, short rate. All
three pairwise correlations are explicit and the entire 3x3 matrix must be
positive semidefinite. Per step `[s,t]`, sample the joint Gaussian vector

```text
[ integral_s^t dW_S(u),
  integral_s^t exp(-k*(t-u)) dW_v(u),
  integral_s^t sigma_r(u)*exp(-a*(t-u)) dW_r(u),
  integral_s^t sigma_r(u)*B(a,t-u) dW_r(u) ].
```

It requires four independent normal coordinates, including the extra rate
integral innovation. Covariances integrate across all rate-volatility knots,
even if a knot lies inside an equity step. Stable series handle small a*dt.
The loading uses Cholesky on covariance normalized to correlation scale;
negative residuals within 2e-12 are treated as roundoff, and larger violations
or inconsistent singular rows are errors. Singular Brownian correlations and
zero volatility are supported.

Let F0(t) be the deterministic canonical forward before discrete dividends.
The simulated state is `f(t)=S0*S_canonical(t)/F0(t)`. Its dynamics and the
one-factor Bergomi volatility multiplier are

```text
df/f = (r-f0)dt + L(t,log(f/S0))*exp(nu*X_v) dW_S,
dX_v = -k*X_v dt + dW_v,   X_v(0)=0.
```

For BS+HW the diffusion is constant. For LSV+HW, freeze squared leverage and
the Bergomi multiplier at the step's left endpoint, while integrating the
random rate exactly. Equity uses positive log-Euler. Thus BS+HW terminal
sampling is exact; LSV equity discretization still has time-step bias.

At observations, convert f to the canonical state with F0(t)/S0, then apply
the existing affine payoff coordinates. This restores deterministic carry
as well as stochastic-rate drift. Proportional dividends reuse the existing
pre/post event order, including discrete barriers. Fixed-cash dividends are
rejected because a correct hybrid coordinate must also evolve the associated
stochastic bonds. Existing deterministic LV/LSV kernels are unchanged.

Payoffs known by expiry T but paid at U >= T use conditional discounting:
`P0(U) * Dbar(T) * [P(T,U)/(P0(U)/P0(T))]`. No unnecessary post-expiry path
is simulated. The Gaussian BS reference includes the payment-measure shift
of the terminal equity distribution when U > T.

## Paired smile target and discounted calibration

`HullWhiteLsvTarget` holds a Dupire local-variance grid and matching samples
of the T-forward log-density:

```text
p_log(t,k) = K*C_KK(t,K)/P0(t) = f*p_f^T(t,f),
k = log(K/F0_contract(t)) = log(f/S0).
```

`flat` constructs both inputs from a flat market implied volatility.
`from_surface` / Python `from_essvi` computes both from the same analytical
smile derivatives. `from_grid` accepts explicit density samples; the caller
must establish their economic consistency with the supplied local variance.
Density samples are finite and nonnegative, in time-major row order. The t=0
row is unused (constructors use zero for the singular initial distribution).
Targets with floor/cap repairs are rejected to preserve paired consistency.

The target must begin at zero, end at expiry and contain every contractual
observation time exactly. The pricing scheme retains the target's calibrated time
knots. A target/request grid match permits only relative differences up to
8 machine epsilons for JSON decimal roundtrips; no market discrepancy or grid
resampling is allowed. Fingerprints retain the actual bit patterns of both.

Writing y=r-f0 and a_v=exp(nu*X_v), the calibration identity is

```text
m2_D(t,k) = E[Dbar*a_v^2 | f] / E[Dbar | f],
L^2(t,k)  = [sigma_Dupire^2(t,k)
              - 2*E[Dbar*y*1(f>K_f)]/p_log(t,k)] / m2_D(t,k),
K_f = S0*exp(k).
```

For justification, apply Ito/Tanaka to `Dbar*(f-K_f)+`. Its rate drift is
`K_f*Dbar*y*1(f>K_f)` and its diffusion term is one half the discounted
conditional variance times `K_f^2*p_f`. Equating its expected time derivative
to that of the target normalized call gives the formula above. This is our
equity/Bergomi/HW specialization of discounted stochastic-rate calibration;
it is not the FX/Heston/CIR model used in the cited hybrid paper.

The finite algorithm uses quartic log-coordinate kernels `(1-u^2)^2` for
`abs(u)<1`, multiplied by Dbar. ESS is `(sum w)^2/sum(w^2)`. Independent
calibration particles estimate the conditional second moment and the digital
rate expectation. To reduce noise, use the identity `E[Dbar*y]=0` and the
centered estimator

```text
q_hat = [sum(Dbar*y*indicator)
          - sum(Dbar*indicator)/sum(Dbar) * sum(Dbar*y)] / particle_count.
```

The empirical ratio introduces finite-particle bias. It is part of the
versioned scheme, not an unbiased control-variate claim. The variance correction
is `2*q_hat/p_log`. A supported node needs sufficient ESS and, for stochastic
rates, strictly positive target density. Unsupported nodes copy both the
conditional moment and rate correction from the nearest supported log node
(lower-index tie), retaining their own target local variance. No supported
node, non-finite estimates or nonpositive corrected leverage variance is an
error. The correction is never silently clipped. Tail density underflow may
cause explicit fallback. At t=0, and when both rate volatility and nu vanish,
the analytical conditional moment is one and correction zero.

Diagnostics report row-wise minimum supported ESS, fallback count, mean Dbar,
mean Dbar*f (target S0) and maximum absolute rate correction. Rust also exposes
the full conditional second-moment and correction arrays. Broad refinement
across particle count, bandwidth, seeds, smiles and maturities is still required.

## Randomness, uncertainty and API boundary

- Scheme: `equity-hw1f-joint-gaussian-log-euler-v1`.
- Calibration: `lsv-hw-discounted-quartic-centered-rate-v1`.
- Four factor-major normal blocks. Antithetics reverse all four; Brownian
  bridge transforms each independent block before joint covariance loading.
- Calibration uses the dedicated `LsvCalibration` Philox domain and explicit
  seed. Pricing uses the independent valuation domain or randomized Sobol.
- Fixed reduction blocks give worker-count-independent numerical results on
  the same platform/build. The fingerprint also records the execution policy,
  so plans with different worker counts need not have identical fingerprints.
- The fingerprint includes the request/base plan, rate schedule, all three
  correlations, factor parameters, paired target, particle settings, actual
  time grid and realized leverage. No global RNG or mutable market state.
- Pseudo-MC standard errors use independent paths/pairs. RQMC errors use
  independent scramble means. LSV errors are conditional on one calibration;
  they exclude calibration randomness, kernel bias and time discretization.
- No hybrid AAD, Delta/Gamma, volatility Vega, market-IV VegaKT or DV01 is
  returned. Risk requests and retained calibration reverse traces are rejected.
- Stable JSON is unchanged and does not serialize the hybrid configuration.
  Persist explicit compile inputs as well as the request to reproduce a plan.

## References

- Fries, [A Short Note on the Exact Stochastic Simulation Scheme of the
  Hull-White Model and Its Implementation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2737091),
  exact joint short-rate and numeraire simulation.
- Cozma, Mariapragassam and Reisinger, [Calibration of a Hybrid Local-Stochastic
  Volatility Stochastic Rates Model with a Control Variate Particle Method](https://arxiv.org/abs/1701.06001),
  discounted hybrid calibration and particle variance reduction.

The implemented specialization and empirical centering policy are specified
above; the references do not establish acceptance of this implementation.
