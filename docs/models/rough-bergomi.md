# Rough Bergomi and rough-LSV contracts

Date: 2026-09-13. Status: experimental implementation.

[Multi-asset rough-LSV with common HW](multi-asset-rough-bergomi.md) extends
this nonuniform hybrid scheme to jointly driven rough and Markovian assets.
The single-equity entry points and conventions below remain available.
[Deterministic-rate rough-LSV](#deterministic-rate-rough-lsv-without-hullwhite)
has its own entry point without Hull–White inputs.

## Model and input conventions

The Riemann–Liouville driver and unit-mean variance multiplier are

$$
X_H(t)=\sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dW_v(s),\qquad
a_t^2=\exp\left(\eta X_H(t)-\tfrac12\eta^2 t^{2H}\right).
$$

Pure rough Bergomi uses instantaneous variance `v_t=sigma0^2*a_t^2`.
The initial forward variance is flat: `xi0(t)=sigma0^2`, the unconditional
variance expectation under the simulation's risk-neutral measure. With
stochastic rates this is not a T-forward-measure expectation or a Black IV
term structure. This first API takes sigma0 from a Black–Scholes request;
it does not accept a forward-variance curve or fit H/eta to options.

Rough-LSV uses `v_t=L(t,F_t)^2*a_t^2`, where L is determined by the existing
discounted particle calibration. Without fixed cash, between proportional
dividend events, `dS/S=(r-q)dt+sqrt(v_t)dW_S`. With escrowed cash, volatility
applies to the residual risky equity, with the same stochastic reserve and
target F coordinate as the [cash model](hull-white-cash-dividends.md).
Hull–White rates, continuous carry and payment discounting retain the
[hybrid contract](hull-white-calculation-specifications.md).

Require finite `0 < H <= 0.5`, finite `eta >= 0`, and a positive-semidefinite
three-driver correlation matrix for `(W_S,W_v,W_r)`. H=0.5 is the Brownian
boundary, not a rough input. The rough model's equity/volatility correlation
must equal the full hybrid correlation's equity/volatility entry. Zero HW
rate volatility gives deterministic rates. Singular valid correlations and
zero rate mean reversion are supported.

`RoughBergomiModel.vol_of_vol` is **eta, the log-variance coefficient**.
The older `Bergomi1Factor` / `compile_lsv` vol-of-vol is nu, the log-volatility
coefficient. At H=0.5, compare rough-LSV to mean-reversion-zero Bergomi LSV
using `eta=2*nu`. Deterministic time centering is absorbed by the recalibrated
leverage surface in that comparison.

## Finite-grid Volterra scheme

The implementation uses the near-cell/older-cell construction of the
[Bennedsen–Lunde–Pakkanen hybrid scheme](https://arxiv.org/abs/1507.03004).
Our implementation extends the cell-average construction to a nonuniform
event grid and couples the near integral to HW rate noise as specified below.
For `0=t_0<...<t_N`, define

$$
\begin{aligned}
\widehat X_i &= J_{i-1}+\sum_{j=0}^{i-2}w_{ij}\Delta W_{v,j},\\
J_{i-1}&=\sqrt{2H}\int_{t_{i-1}}^{t_i}(t_i-s)^{H-1/2}\,dW_v(s),\\
w_{ij}&=\frac{\sqrt{2H}\left[(t_i-t_j)^{H+1/2}-(t_i-t_{j+1})^{H+1/2}\right]}
{(H+1/2)(t_{j+1}-t_j)}.
\end{aligned}
$$

Use the **discrete** variance

$$
V_i=(t_i-t_{i-1})^{2H}+\sum_{j=0}^{i-2}w_{ij}^2(t_{j+1}-t_j),\qquad
\widehat a_i^2=\exp(\eta\widehat X_i-\eta^2 V_i/2).
$$

Thus `E[a_i^2]=1` on each finite grid. V_i is generally smaller than the
continuous variance `t_i^(2H)`; it converges to that variance under refinement.
Centering does not remove the remaining covariance or equity time-step bias.
H=0.5 gives ordinary Brownian increments and V_i=t_i. The public low-level
`RoughBergomiDriverPlan` exposes times, discrete variances and raw Gaussian
driver evolution for verification. Internally the hybrid state stores
`Y_i=Xhat_i-eta*V_i/2`, so the shared engine uses `exp(eta*Y_i)`.

Each time interval jointly samples `(dW_S,dW_v,U_r,I_r,J_near)`, where U_r is
the OU rate noise and I_r the integrated OU noise. The first four variables
use the existing exact HW covariance with zero volatility-factor mean
reversion. For an interval `[s,t]`, write h=t-s and p=H+1/2. The added entries
are

$$
\begin{aligned}
\operatorname{Var}(J)&=h^{2H},&
\operatorname{Cov}(J,\Delta W_v)&=\sqrt{2H}\,h^p/p,\\
\operatorname{Cov}(J,\Delta W_S)&=\rho_{Sv}\sqrt{2H}\,h^p/p,\\
\operatorname{Cov}(J,U_r)&=\rho_{vr}\sqrt{2H}\int_s^t
\sigma_r(u)(t-u)^{p-1}e^{-a(t-u)}\,du,\\
\operatorname{Cov}(J,I_r)&=\rho_{vr}\sqrt{2H}\int_s^t
\sigma_r(u)(t-u)^{p-1}B(a,t-u)\,du.
\end{aligned}
$$

All piecewise rate-volatility knots inside the interval are included. A
cancellation-safe series handles small a*h; a power change of variables
removes the endpoint singularity before adaptive quadrature for large a*h.
Unresolved quadrature and inconsistent covariance are explicit errors.
The normalized PSD Cholesky clips only roundoff-sized negative residuals.
At H=0.5 the near integral reuses the volatility-increment loading exactly.

Five factor-major Gaussian blocks are required even when a loading is zero.
Pseudo-MC keeps the existing first four coordinate blocks and appends the
fifth. RQMC uses dimension `5*N`, the existing direction-table limit, and the
same independent-scramble policy. Brownian bridge acts separately on each
block; antithetic paths negate all five blocks. The stock step uses variance
at the left node, hence no future rough-driver noise enters its volatility.
Price and AAD share the path generation. Existing BS/Bergomi plans keep their
four-block random layout and scheme identifiers.

The direct history convolution costs O(N^2) per path, with O(N^2) compiled
weights and O(N) path scratch. Rough calibration precomputes the fixed driver
history in O(particles*N) memory, then runs the existing particle algorithm.
Retaining a reverse trace additionally stores the existing particle checkpoints.
This implementation does not use an FFT, Markovian lift or persistent scalar
tape for the history convolution. Very fine grids and large particle sets can
be expensive, especially for small H where refinement is important.

## Rust and Python API

Rust exports `pricing::models::RoughBergomi` and
`pricing::hull_white::HullWhiteEquityPricingPlan` constructors:

| Constructor | Input volatility / calibration |
| --- | --- |
| `compile_rough_bergomi` | Black–Scholes request sigma0; flat initial variance; maximum step |
| `compile_rough_bergomi_with_cash_dividends` | Same, with explicit escrowed cash |
| `compile_rough_lsv` | Matching Local Volatility request/target and particle settings |
| `compile_rough_lsv_with_cash_dividends` | Same, with explicit escrowed cash |

Python adds the immutable `RoughBergomiModel` and two static constructors on
`HullWhiteEquityPlan`. Both accept `cash_dividend_model="escrowed"` when needed:

```python
rough = rp.RoughBergomiModel(0.1, 0.8, equity_vol_correlation=-0.5)
# request contains Model.black_scholes(0.2) and a price-only RiskRequest.
pure = rp.HullWhiteEquityPlan.compile_rough_bergomi(
    request, rough, rates,
    equity_rate_correlation=0.25, vol_rate_correlation=-0.1,
    maximum_step=1/32, worker_threads=2,
)
price = pure.evaluate()
risk = pure.evaluate_aad()
```

For rough-LSV use `compile_rough_lsv(request, target, rough, rates, ...)`,
with the existing particle options and `retain_reverse_trace=True` for AAD.
Use `target.model` in that request. Calibration times must include the product
events and end exactly at expiry; pure rough constructs event substeps from
`maximum_step`. Rate-volatility knots are integrated inside a step and need
not be equity nodes. Both APIs release the Python GIL during compilation and
evaluation. See the [complete runnable example](../../examples/python/rough_bergomi.py).

Existing positive-horizon European, Asian, Lookback, Digital and discrete
Barrier payoff integration, payment lags and dividend event order are shared.
Digital/Barrier AAD still requires explicit payoff smoothing. Stable JSON
continues to describe the base request; the rough-model selection lives in
this explicit Rust/Python plan API and has no new wire tag or risk flags.

Price scheme: `rough-bergomi-hw-hybrid-kappa1-log-euler-v1`.
Calibration labels: `rough-lsv-hw-discounted-quartic-v1` and
`rough-lsv-hw-escrowed-quadratic-v1`. Pure rough has no calibration label/seed.
H, eta, correlations, the base request, time grid and calibration configuration
contribute to the plan fingerprint. `random_factor_count` reports five.

## Deterministic-rate rough-LSV without Hull–White

`RoughBergomiLsvPricingPlan` is the rough counterpart of `BergomiLsvPricingPlan`
([ADR 0008](../../design/adr/0008-deterministic-rate-rough-lsv.md)). It takes the
same Price-only `PricingRequest` with a Local Volatility target, calibrates
`L(t,F)^2 = sigma_LV^2 / E[a_t^2 | F_t]` with the existing particle algorithm and
evaluates the request's product with independent MC/RQMC paths. Discounting,
carry and discrete dividends follow the request's deterministic market, exactly
as for the Bergomi plans. No Hull–White model, rate correlation or HW target is
required.

The driver uses the same nonuniform kappa=1 hybrid scheme, Volterra weights and
discrete-variance centring as above. Without rates, each step needs three
factor-major Gaussian blocks instead of five: spot normals, variance normals
orthogonal to spot, and near-cell residual normals. With `h` the step,
`p=H+1/2` and `rho` the equity/volatility correlation,

$$
\begin{aligned}
\Delta W_v&=\sqrt h\left(\rho z_S+\sqrt{1-\rho^2}\,z_v\right),\\
J&=\beta\,\Delta W_v+\sigma_J z_J,\qquad
\beta=\frac{\sqrt{2H}\,h^{H-1/2}}{p},\qquad
\sigma_J^2=h^{2H}\left(1-\frac{2H}{p^2}\right).
\end{aligned}
$$

This reproduces `Var(J)=h^(2H)`, `Cov(J,dW_v)=sqrt(2H)h^p/p` and
`Cov(J,dW_S)=rho*sqrt(2H)h^p/p` exactly. The residual variance is nonnegative
because `2H <= p^2`, and it is exactly zero at H=0.5, where `J` equals the
Brownian increment. The law of `(dW_S, dW_v, J)` equals the HW engine's at zero
rate volatility; the random coordinates differ, so paths are not pathwise equal.

At H=0.5 with `eta=2*nu`, the driver equals zero-mean-reversion one-factor
Bergomi up to the deterministic centring `exp(-2 nu^2 t)`, which the calibrated
leverage absorbs. On the calibration grid the particles, leverage-times-multiplier
products and pseudo-MC prices then agree with `BergomiLsvPricingPlan` up to
roundoff, because the first two Gaussian blocks use the same coordinates. On a
finer execution grid they differ between leverage knots, where the rough centring
moves with the node time while the absorbed factor is constant per leverage row.

Calibration first generates each particle's full driver history from its own
calibration normals (spot, orthogonal and near-cell blocks at dimensions `r`,
`N+r` and `2N+r`), then runs the particle algorithm. This holds
`particles * time nodes` multipliers in memory, in addition to any reverse trace,
and costs O(particles * N^2) once. `calibrate_rough_bergomi_lsv_parallel` and
the plan spread both stages over the execution policy's workers with results
bit-identical for any worker count.

| Item | Value |
| --- | --- |
| Rust plan | `pricing::lsv::RoughBergomiLsvPricingPlan::compile(request, RoughBergomi, LsvParticleConfig, ExecutionPolicy)` |
| Rust calibration | `calibrate_rough_bergomi_lsv`, `calibrate_rough_bergomi_lsv_parallel`, `CalibratedRoughBergomiLsv` |
| Rust paths | `RoughBergomiLsvPlan::{pseudo_shocks, evolve_path, evolve_states}`, `RoughBergomiLsvPath::reverse` |
| Python | `rp.RoughBergomiLsvPlan.compile(request, hurst=..., vol_of_vol=eta, correlation=..., ...)` with the `BergomiLsvPlan` particle options |
| Price scheme | `rough-bergomi-lsv-hybrid-kappa1-log-euler-v1` |
| Random layout | three factor-major blocks per step; RQMC dimension `3*N` |
| Risk | Price and calibrated local-variance risk through `evaluate_local_variance_risk`, as for Bergomi LSV |

The pathwise reverse returns squared-leverage, initial-f and all three shock
adjoints; `orthogonal_shocks` holds the orthogonal-variance block followed by the
near-cell block. H, eta and the correlation contribute to the plan fingerprint
under the tag `pricing/rough-bergomi-lsv-plan/v1`.

This entry point has no uncalibrated pure rough model and no multi-asset form;
use the Hull–White plans for those. American/Bermudan exercise and continuous
Barrier monitoring are rejected as for the other LSV adapters.

## AAD and VegaKT

The existing [AAD](hull-white-aad.md) and
[quote-node VegaKT](hull-white-vegakt.md) contracts apply, with H/eta fixed.
There is no derivative of the Volterra weights or Cholesky loadings: the driver
is independent of the active Spot, initial-curve and target inputs. Reversing
the particle leverage calibration still includes its effect on all subsequent
particle paths. No bump-and-revalue is used in the production risk calculation.

| Plan | `vega` / available risk |
| --- | --- |
| Pure rough | dPrice/dsigma0, labelled `initial_volatility`; Spot Delta and initial-curve adjoints/DV01 |
| Rough-LSV from a grid/flat/eSSVI target | Paired local-variance and forward-density adjoints, Spot Delta and curve risk; no implied-IV Vega |
| Rough-LSV from `from_market_iv` | The above plus row-major IV VegaKT and parallel IV Vega through both target arrays and recalibration |

For pure rough, the sigma0 transpose is
`exponent_bar*(-sigma0*a_i^2*dt+a_i*dW_S)`, including the finite right derivative
at sigma0=0. The low-level Rust path-adjoint field retains its legacy name
`bs_volatility`; the facade labels it `initial_volatility` for rough.
VegaKT reports currency per unit input IV; market scaling multiplies by 0.01.
Pure rough has no quote-node VegaKT because no quote source is supplied.

H, eta, rate parameters, correlations, dividend amounts, dates, grids and random
draws remain fixed. Cash-mode IV quotes must already use the converted escrow
F coordinate. Physical-quote conversion, H/eta calibration, Gamma and parameter
Greeks are not differentiated. The existing hard digital/support/donor branch
conventions remain. Risk SEs for RQMC are conditional on the calibration and
exclude particle noise, discretization bias and branch-selection uncertainty.

The cash calibration rejects a negative quadratic discriminant or a nonpositive
leverage root. A positive IV surface alone does not guarantee a feasible target
for the chosen rate/dividend model and finite particle estimator. Diagnose the
rate correction, conditional moments, support and particle/grid refinement
before interpreting or changing such a target; no variance clipping is applied
to force a calibration.

## Scope and limitations

The finite-grid implementation remains subject to refinement across grids,
particle counts, bandwidths, seeds, adverse smiles and maturities. Nonflat
initial forward variance, rough Heston, fast convolution/lifts and H/eta
calibration/risk remain future extensions.

## References

- [Bayer, Friz and Gatheral, *Pricing under rough volatility*](https://doi.org/10.1080/14697688.2015.1099717), for the rough Bergomi model and its Riemann–Liouville volatility driver.
- [Bennedsen, Lunde and Pakkanen, *Hybrid scheme for Brownian semistationary processes*](https://arxiv.org/abs/1507.03004), for the near-cell/older-cell Volterra discretization.
- [Bergomi, *Smile Dynamics II*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1493302), for the Markovian Bergomi factor used in the H=0.5 comparison.
- [Guyon and Henry-Labordère, *The Smile Calibration Problem Solved*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032), for the particle-calibrated rough-LSV leverage construction.
- [Hamdouche and Henry-Labordère, *Vega KT for LSV Models: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4304114), for the retained LSV target-risk convention.
- [Hull and White, *Pricing Interest-Rate-Derivative Securities*](https://doi.org/10.1093/rfs/3.4.573), for the common one-factor stochastic-rate component.
