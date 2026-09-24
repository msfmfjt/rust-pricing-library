# ADR 0015: explicit stochastic-dividend first-order reverse and refinement

Status: implemented candidate; native validation required before acceptance.

## Decision

Extend the dedicated BS/1F/2F Bergomi Buehler plan with `evaluate_aad`, returning
`StochasticDividendAadRisk`. Constructors continue to require price-only requests;
this avoids accidentally routing request Greeks through fixed-cash engines.
The standalone `evaluate` path, its arithmetic, sampler, fingerprint and scheme
are unchanged. There is no new dependency, wire variant or crate.

The reverse differentiates the compiled payoff and the positive split, with raw
parameters in this order: Spot; residual initial volatility sigma0; dividend
mean reversion kappa; equity linkage alpha; dividend volatility nu; every cash
mean in supplied schedule order; discount-curve log-DF pillars; repo-spread-curve
log-DF pillars. Curve labels include the fixed time-zero anchors with zero risk.
Cash event labels retain event IDs and ex-dates, including dates beyond expiry.

Bergomi factor parameters, all correlations, ex-dates, grid topology, notional,
strike and smoothing width are fixed. There is no covariance/Cholesky derivative,
market-IV recalibration, VegaKT, Gamma, rough/HW/multi-asset extension or
calibration to dividend options in this decision.

## Reverse specification

Use the actual primal states and recompute the same OU sequence for sigma0
loadings. The OU process is independent of the active parameters in this scope.
Do not divide by sigma0: the derivative at zero is obtained from its exponential
loading. Differentiate both drift halves and the multiplicative dividend noise.
The convex blend's `current == target` shortcut is a roundoff-preserving identity,
not a branch that discards state/parameter derivatives. At parameter boundaries
these are inward one-sided derivatives. Payoff tie rules remain those of the
compiled payoff graph.

At each observation reverse both post-dividend Spot and, when present, its
pre-dividend value. Differentiate `S=a f+b Y+c` and the realized event cash.
Differentiate initial funding from **all** supplied means, not just in-expiry
payments. Transpose carry `log G = log D_s - log D_r` and payment discounting
through the existing log-linear curve interpolation and extrapolation, with a
fixed log(1)=0 anchor. These curve risks hold the other curve and cash means fixed.

The reserve-coefficient Jacobian is compiled once per risk evaluation, not per
path. Every path uses a manual simulation reverse plus the existing payoff tape;
no finite differences or repeated pricing are used to produce the AAD result.
Derivative tests use separately recompiled perturbed plans.

Unsmooth discontinuous products reject `evaluate_aad` before sampling. Explicit
compact-C2 smoothing enables the shared barrier/digital adjoints; derivatives
are of that smoothed price, not a claim of unbiased unsmoothed discontinuous risk.
American exercise and continuous barriers remain unsupported by this plan.

## Reporting and compatibility

Raw initial-volatility and dividend-volatility derivatives scale by 0.01 for a
+1 vol-point report. Zero-rate pillar derivatives scale by `-t * 1e-4` for signed
DV01. These are first-order changes, not repriced finite shocks. Initial-volatility
Vega is not physical-stock market-IV Vega.

MC errors use independent antithetic pair means when enabled. RQMC errors use
independent scramble means only. All are conditional on parameters and grid and
exclude discretization/model uncertainty. Risk evaluation retains price-only
random coordinates, antithetic ordering and deterministic block reductions.
Python result objects are immutable and vector getters return copies.

## Validation

See [validation record](../validation/stochastic-dividend-risk.md) and the
[model reference](../../docs/models/stochastic-dividends.md#first-order-risk).
The new release refinement job is separate from the existing price/risk gates.
No existing tolerance, seed, platform contract or frozen replay is changed.
