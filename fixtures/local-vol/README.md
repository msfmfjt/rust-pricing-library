# Local Volatility and VegaKT equation fixtures

`reference-cases-v0.1.json` is the Gate L0 equation-level reference set for
policy `local_vol_vegakt_v1`. It is intentionally independent of any production
Rust implementation.

The SSVI and eSSVI values were evaluated with 70-digit Python `Decimal`
arithmetic. Transition-cell values use SciPy's binary64 normal CDF only when the
fixture was authored; the committed checker uses the Python standard-library
`erfc` formulation and a documented binary64 tolerance. Algebraic cases use
decimal arithmetic or exact rational identities.

Sources and equation provenance are recorded in every case group. The VegaKT
paper's published figures are not used as numerical truth because their full
market surface, grids, and raw values are unavailable. Updates require review of
the inputs, policy version, provenance, and all expected values.

Run:

```shell
python scripts/check_local_vol_reference_fixture.py
```

`gamma-numerics-v0.1.json` adds eight independent numerical regression cases for
Gamma transition tails, short steps, and narrow strikes. The reference uses
120-digit Decimal power-series integration with exact binary64 inputs. Rust
tests use relative tolerance `1e-10` with no absolute-error floor. See the
[validation record](../../design/validation/local-vol-gamma-numerics-v0.1.md) for
the corrections and the still-failing end-to-end VegaKT gate.

```shell
python scripts/check_local_vol_gamma_numerics.py
cargo test --locked -p pricing --test local_vol_gamma_numerics
```

`gamma-recovery-v0.1.json` contains two independent conditional vanilla-price
bump oracles on a non-uniform grid. Only one time interval's local volatility
is bumped; the remaining conditional variance is held fixed. These validate
the units and normalization of the positive-time Gamma recovery bridge, not
the missing vanilla-weight/IV decomposition.

```shell
python scripts/check_local_vol_gamma_recovery.py
cargo test --locked -p pricing --test local_vol_gamma_recovery -- --nocapture
```

`vegakt-paper-v0.1.json` adds eight total-spot-Gamma price-difference cases and
four quoted-IV price-bump cases for SSRN 4107770 equations (5) and (9). The
120-digit reference constructs and prices a piecewise cubic payoff; it does not
reuse the native total-derivative kernel. Both spot and quoted-IV stencils are
checked at two bump sizes. Relative tolerance is `1e-9`, with no absolute floor;
the prior fixture and production thresholds are unchanged. See the
[paper operator record](../../design/validation/local-vol-vegakt-paper-v0.1.md).

```shell
python scripts/check_local_vol_vegakt_paper.py
cargo test --locked -p pricing --test local_vol_vegakt_paper -- --nocapture
```

`step-adjoints-v0.1.json` contains 136 independent derivatives across four
path-price cases: nonflat path dependence, flat quadratic payoff, and both
spatial clamps. The original three-row grid is deliberately different from the
five simulation intervals. A 100-digit Decimal forward replay bumps individual
step variances, effective step volatilities, spatial hats, and original grid
nodes at two bump sizes. It never calls the native reverse. See the
[step-adjoint record](../../design/validation/local-vol-step-adjoints-v0.1.md).

```shell
python scripts/check_local_vol_step_adjoints.py
cargo test --locked -p pricing --test local_vol_step_adjoints -- --nocapture
```

`step-recovery-v0.1.json` adds means, sample variances and standard errors from
five independent paths and five antithetic pair means. The 100-digit Decimal
reference uses forward price differences and two-pass moments, including row,
retained/excluded and all-step sums with their covariance. Analytic flat density
and full-grid hat areas independently normalize positive-time Gamma moments.
The initial sensitivity is retained without a Gamma estimate. See the
[step-recovery record](../../design/validation/local-vol-step-recovery-v0.1.md).

```shell
python scripts/check_local_vol_step_recovery.py
cargo test --locked -p pricing --test local_vol_step_recovery -- --nocapture
```

`direct-variance-v0.1.json` contains 215 path derivatives for up/down barriers,
knock-in/out payoffs with rebates, nonflat/flat variance, shortened monitoring
and an already-touched endpoint. Another 80 price derivatives form independent
path/pair moment references. The 100-digit Decimal oracle prices a product of
bridge survival factors with trapezoidal endpoint variance; it does not call
native reverse or statistics. Direct terminal risk is retained without claiming
Gamma recovery. See the [validation record](../../design/validation/local-vol-direct-variance-v0.1.md).

```shell
python scripts/check_local_vol_direct_variance.py
cargo test --locked -p pricing --test local_vol_direct_variance
```

`initial-gamma-v0.1.json` contains conditional continuation-price second
stencils at two bump widths, correlated differences, and reconstructed
first-interval Vega moments for three paths and three antithetic pairs.
A 100-digit Decimal price replay and Gaussian power-series integrals provide
independent values. Gamma/initial-Vega moments use relative tolerance `1e-8`;
small fine-minus-coarse means and standard deviations use an absolute `1e-9`
roundoff budget in their respective units. This new diagnostic does not change
any existing acceptance tolerance. See the
[initial-Gamma record](../../design/validation/local-vol-initial-gamma-v0.1.md).

```shell
python scripts/check_local_vol_initial_gamma.py
cargo test --locked -p pricing --test local_vol_initial_gamma -- --nocapture
```


`vegakt-chain-v0.1.json` composes independent Decimal price differences,
Gamma recovery, finite-strike equation-(9) integration and two-pass moments.
It checks 54 channels for three paths and three antithetic pairs, including
bucket/hedge uncertainty, quadrature and bump differences, and tail/edge audits.
See the [joint-chain record](../../design/validation/local-vol-vegakt-chain-v0.1.md).

```sh
python scripts/check_local_vol_vegakt_chain.py
cargo test --locked -p pricing --test local_vol_vegakt_chain -- --nocapture
```

`density-mass-v0.1.json` and `weak-chain-v0.1.json` independently validate
integrated log-hat density masses and the weak Gamma chain. Their references
use 120-digit Gaussian moments and 100-digit price differences, respectively.
They do not replace the original point-density fixtures or the numerical gate.
See the [validation record](../../design/validation/local-vol-weak-gamma-v0.1.md).

```shell
python scripts/check_local_vol_density_mass.py
python scripts/check_local_vol_weak_chain.py
```


- `multi-step-gamma-v0.1.json`: 4区間・正skew、6区間・負skew。第1報告満期は2遷移後の非平坦スマイル。終端年率分散作用素と較正接線を含む。先行rowは固定し、全満期Gamma受入とは扱わない。
