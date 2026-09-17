# Equity/Hull–White AAD calculation specifications

Date: 2026-09-13. Status: experimental first-order implementation of H5.
Quote-backed VegaKT is described in the [VegaKT contracts](hull-white-vegakt.md).
These contracts extend the [hybrid](hull-white-calculation-specifications.md) and
[cash-dividend](hull-white-cash-dividends.md) price contracts.
The [rough extension](rough-bergomi.md) reuses this VJP with a fixed
Volterra driver and adds the pure model's `initial_volatility` sensitivity.

## API and risk coordinates

Use `HullWhiteEquityPricingPlan::evaluate_aad()` in Rust or `plan.evaluate_aad()`
in Python. BS needs no calibration trace. LSV compilation must opt into
`LsvParticleConfig::retain_reverse_trace` / Python `retain_reverse_trace=True`.
The default is false; requesting LSV AAD without the trace is an error.
See the [runnable example](../../examples/python/hull_white_lsv.py).

The immutable Python `HullWhiteAadRisk` contains a price, `parameter_labels`,
`derivatives` and optional `standard_errors` in the same order:

| Order / label | Coordinate / units |
| --- | --- |
| `spot` | Currency per unit input pre-event S0; initial residual equity, normalized log-coordinate lookup and particle recalibration all move |
| `bs_volatility` (BS only) | Currency per unit absolute residual-equity volatility; `vega` is `None` for LSV targets without a retained market-IV source |
| `discount_log_df[i]` | dPrice / d log P0(t_i), including the HW initial-curve fit, reserve and payment discount |
| `dividend_log_df[i]` | dPrice / d log Q0(t_i), including continuous carry and the reserve |
| `local_variance[i]` (LSV) | dPrice / d effective relative local-variance sample; paired density held fixed |
| `forward_log_density[i]` (LSV) | dPrice / d p_log, with p_log=K*p_F^T(K); this is **not** the logarithm of a density |

Quote-backed targets append row-major `market_iv[i]` and `parallel_market_iv`
derivatives. Their axes, units and interpolation are specified in the
[VegaKT contracts](hull-white-vegakt.md).

Both target arrays use row-major `(time_nodes, log_moneyness_nodes)` order.
Convenience properties expose `delta`, `vega`, the four node-adjoint arrays,
the curve times, and target axes. Curve arrays include the zero-time anchor,
whose value is fixed at one and whose adjoint is zero. Log-linear interpolation
and terminal-segment extrapolation are transposed onto the original pillars.

`discount_node_dv01[i] = -1e-4 * t_i * discount_log_df_adjoints[i]` is the signed
first-order price change for a **+1 bp** bump to that continuously compounded
zero-rate pillar. `parallel_discount_dv01` is their sum. It is not an unsigned
loss convention. Its SE cannot be inferred by adding the individual node SEs.

HW/Bergomi parameters, correlations, dividend quotes, dates, RNG draws, bandwidth,
ESS threshold and grid axes are fixed. LSV target samples are fixed at their
relative log coordinates `log(F/S0)` during Spot and curve risk. Physical market
quotes must first be converted to the cash model's escrow coordinate; changing
that conversion or refitting a smile requires an additional caller-supplied
chain rule. The stable request still remains price-only, including its risk
flags; the explicit AAD method defines these coordinates independently of those
flags. Gamma, model-parameter risk and dividend-amount risk are not returned.
Market-IV VegaKT is available only for the explicitly retained quote source.

## Path, payoff and curve reverse

The compiled payoff supplies adjoints at every observed post/pre-dividend Spot.
For fixed cash D and beta, reverse `S_pre=(S_post+D)/(1-beta)` by adding
`S_pre_bar/(1-beta)` to the post-event seed. Then reverse
`S_post=c*R+A(t,x)` into R, c and the individual HW bond reserve coefficients.
This includes valuation-date and expiry-date dividend collisions.

At fixed centered HW states and fixed factor parameters, one equity step is

```text
R_next = R * exp(integrated_x + integrated_shift - v*dt/2 + sqrt(v)*dW_S),
v = L2 * exp(2*nu*X_v).

exponent_bar = R_next_bar * R_next,
v_bar = exponent_bar * (-dt/2 + dW_S/(2*sqrt(v))),
L2_bar = v_bar * exp(2*nu*X_v).
```

Transpose the squared-leverage interpolation weights and its log-coordinate
slope. The latter includes both `dF/F` and `-dS0/S0`, as well as the stochastic
reserve contribution to `F=R+(A-A0)/c`. With BS volatility sigma, use
`sigma_bar += exponent_bar*(-sigma*dt+dW_S)` directly; this has a finite right
derivative at sigma=0 and avoids a variance-to-volatility 0/0.

Re-fitting HW to a changed initial curve leaves its centered OU states, relative
discount `Dbar=exp(-I-V/2)`, relative conditional payment bond and `r-f0` unchanged
when a/sigma_r are fixed. Initial-curve dependence remains in the physical Spot
map, cash reserve and the `P0(payment)` prefactor. Their transposes therefore
include both the equity drift and stochastic-discount fit. A payment log-DF
seed equals the discounted payoff. Reserve seeds also reach dividends after
the option expires; their cash amounts remain fixed.

Continuous payoffs use the existing graph's branch/tie conventions. Digital and
Barrier AAD require explicit compact-C2 smoothing. An unsmoothed discontinuous
payoff is rejected instead of returning a zero pathwise indicator derivative.
No smoothing is silently added to the price or to the calibration estimator.

## Discounted particle-calibration VJP

Store the particle states, supported-node kernel weight sums and donor identities
at each time row. Reverse rows from last to first. Each row first receives seeds
through the evolution of all subsequent particles, then through its regression.
Normal draws are regenerated from their original four-block coordinates.
Memory for checkpoints is O(particles * time nodes), not a scalar tape of every
particle/kernel pair. Price compilation without trace retains no checkpoints.

For the cash quadratic, write `a=M2`, `b=rho_Sr*M1`, `c=M0` and
`a*L^2+2*b*L+c=target_var-correction`. For an outgoing squared-leverage seed g:

```text
v_bar = g*L/(a*L+b),
target_var_bar += v_bar,       correction_bar -= v_bar,
M2_bar -= v_bar*L^2,
M1_bar -= v_bar*2*rho_Sr*L,   M0_bar -= v_bar.
```

The upper-root denominator must be positive and finite. A double-root singularity
is a reverse error even if the forward price can be computed. The zero-bond
limit reduces to the original variance quotient. At time zero the analytic
residual ratio is one and the bond loading/strike ratio remains active. At zero
rate volatility and zero vol-of-vol, L2 equals the target and density adjoints
vanish.

For each moment `M=sum(w_i*g_i)/sum(w_i)`, propagate
`g_i_bar += M_bar*w_i/sum(w)` and
`w_i_bar += M_bar*(g_i-M)/sum(w)` through the quartic log-coordinate kernel and
the particle's `eta=a_v*R/F` and `zeta/F`. Fallback-cell moment and correction
seeds reach the same recorded donor; their target variance seeds remain at the
destination cell. The rate correction's density denominator and `1+A0/(c*K)`
factor are also active.

The empirical digital rate estimator uses hard indicators. For these risk
coordinates, discounted weights and centered rate values are constant, while
indicator derivatives are zero between crossings. Sorting, support/ESS selection,
fallback donors and interpolation branches are likewise held fixed. The result
is the almost-everywhere derivative of the realized finite algorithm; it is not
an unbiased continuum sensitivity. Recalibrated finite differences can jump when
a digital or support branch changes. Calibration-seed, particle/bandwidth/time
refinement and branch-stability studies remain necessary for H5 acceptance.

Both effective variance and forward-density adjoints are returned. For a smile
parameter theta, contract `var_bar*dvar/dtheta + density_bar*dp_log/dtheta`.
Multiplying only variance adjoints by `2*sigma` is insufficient when density
also changes. Tests check this with an actual flat-smile volatility bump.

The low-level Rust calibration retains a fingerprint of its public primal arrays.
Changing a surface or moment array after retaining the trace causes reverse to
fail explicitly. A recorded pricing path borrows its immutable source plan.

## Reduction and uncertainty

Method: `equity-hw-discrete-particle-vjp-v1`. Existing price defaults and RNG
coordinates are unchanged. Trace-enabled plans have a distinct fingerprint;
the retained checkpoint choice does not change their prices.

Reverse pricing paths into deterministic coefficient/leverage seed vectors and
reduce them with fixed blocks. Apply the calibration VJP once to a pseudo-MC
mean, or once per independent RQMC scramble. No calibration is rerun per pricing
path. RQMC risk SEs use scramble derivatives; pseudo-MC risk SEs are currently
`None`. Price SEs remain available for both engines. All SEs exclude calibration
randomness, kernel/time bias and branch-selection uncertainty.


## References

- [Giles and Glasserman, *Smoking Adjoints: Fast Evaluation of Greeks in Monte Carlo Calculations*](https://people.maths.ox.ac.uk/gilesm/files/NA-05-15.pdf), for reverse pathwise AAD and the reuse of one reverse pass for many sensitivities.
- [Capriotti, *Algorithmic Differentiation: Adjoint Greeks Made Easy*](https://www.luca-capriotti.net/pdfs/Finance/GD11LucaCapriotti.pdf), for practical AAD implementation patterns in financial Monte Carlo.
- [Hamdouche and Henry-Labordère, *Vega KT for LSV Models: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4304114), for the LSV calibration VJP and the variance/density target transpose.
- [Hull and White, *Pricing Interest-Rate-Derivative Securities*](https://doi.org/10.1093/rfs/3.4.573), for the stochastic-rate state that is differentiated through the physical-coordinate and discounting maps.

The retained trace, fixed branch set and finite-program derivative are
implementation contracts; the references do not imply a continuum or
re-optimized-exercise sensitivity.
