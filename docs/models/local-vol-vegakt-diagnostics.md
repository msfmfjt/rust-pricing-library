# Local Volatility and VegaKT diagnostics catalogue

Status: accepted catalogue for the Local Volatility/VegaKT slice

Implementation update (2026-10-04): ordinary pricing now uses signed terminal
Gamma measures for eligible spatially flat, zero-carry, single-terminal
piecewise-affine payoffs. This path retains actual paired/scramble sample
statistics. See the [scope and validation record](../../design/validation/local-vol-terminal-pathwise-v0.1.md).
Other unsupported native LV requests retain the diagnostic local-node allocation described
in the [allocation audit](../../design/validation/vega-kt-backtest-allocation-audit-v0.1.md).
Neither method emits the historical equation-(11) label; general nonterminal recovery remains open.

The [reporting-maturity correction](../../design/validation/vega-kt-reporting-maturities-v0.1.md)
allows sparse reporting dates and dates inside simulation intervals in the same
restricted model. Every quoted row must match the simulated cumulative variance,
and the final reporting date must equal the contract expiry. This fixes evaluation
failures caused solely by using fewer reporting dates than simulation dates.

The [general-smile P0 extension](../../design/validation/vega-kt-smile-acceptance-v0.3.md)
adds `terminal_payoff_replication_iv_v1`. It reports exact market-IV risk from
signed terminal static replication, with `truncation_order=not_applicable`,
zero excluded mass and zero residual. Bucket `sample_variance` and
`price_covariance` are absent; an explicitly requested full covariance matrix
contains unavailable entries. These are not estimated zero variances. Price
and local-Greek MC uncertainty is unaffected. Contract expiry may be an interior
reporting node. Unsupported payoffs retain the diagnostic allocation policy.

This catalogue covers diagnostic and warning surfaces introduced by the
Local Volatility/VegaKT roadmap. It complements the frozen European
Black-Scholes catalogue and records the stable names that tests and replay
fixtures may rely on.

## Surface and Local-variance diagnostics

Local-variance grids store explicit time nodes, log-forward-moneyness nodes,
row-major values, Floor, Cap, and deterministic repair records. Repairs are
recorded after a valid implied-volatility surface evaluation; surface invariant
failures remain typed errors and are not repaired.

Repair records preserve the time index, log-moneyness index, coordinates,
original binary64 bits, applied value, policy label, and row-major occurrence
index.

| Stable code | Condition | Applied value |
|---|---|---|
| `local_variance_nan` | Raw Local variance is NaN | Floor |
| `local_variance_negative_infinity` | Raw Local variance is `-Infinity` | Floor |
| `local_variance_positive_infinity` | Raw Local variance is `+Infinity` | Cap |
| `local_variance_non_positive` | Raw Local variance is finite and `<= 0` | Floor |
| `local_variance_below_floor` | Raw Local variance is finite and below Floor | Floor |
| `local_variance_above_cap` | Raw Local variance is finite and above Cap | Cap |

Horizontal grid use outside the log-forward-moneyness range is flat at the
nearest boundary. Runtime interpolation statistics record:

| Field | Meaning |
|---|---|
| `left_flat_count` | Number of Local-variance interpolations assigned to the left boundary |
| `right_flat_count` | Number of Local-variance interpolations assigned to the right boundary |
| `max_left_excursion` | Maximum absolute distance beyond the left boundary |
| `max_right_excursion` | Maximum absolute distance beyond the right boundary |

Public Local Volatility Delta and Gamma are reported with risk method
`central_bump`. The bump is evaluated with common random numbers, fixed
reduction order, and Spot-specific recompilation of forward normalizers and
affine dividend transforms. Public Local Vega and VegaKT output remain separate
acceptance work.

## Affine dividend diagnostics

Discrete dividends are represented by deterministic affine transforms between
the contract Spot coordinate and the continuous Local Volatility martingale
coordinate. Event timelines record pre-event and post-event coordinates. The
path engine treats a non-positive reconstructed post-dividend Spot as a typed
market error, not as a warning or clamp.

The matching-condition checker reports expected and actual call-surface values,
absolute error, and both accepted tolerances when the affine dividend surface
condition is violated.

## Positive-time path Gamma recovery

`PricingPlan.evaluate_vega_kt_gamma_recovery_diagnostic` adds a separate,
explicitly regularized source-hat convolution inverse for continuous path
payoffs in spatially flat, zero-carry pseudo-MC. Asian, fixed-history Asian and
Fixed Lookback are covered by the [component validation](../../design/validation/local-vol-path-gamma-recovery-v0.1.md).
The result reports signed Gamma masses, same-draw reconstruction differences,
MC uncertainty, inverse noise amplification, resolution trace and counts of
nonzero source-hat contributions. Empty/rare tail samples do not establish zero
risk. Initial local Vega remains explicitly unrecovered; no IV-bucket acceptance
or automatic use in normal pricing follows from this diagnostic.

## Conditional initial Gamma measure

`PricingPlan.evaluate_vega_kt_initial_gamma_diagnostic(boundaries)` supplies the
first positive-time Gamma from the conditional compiled payoff in the same
flat-space, zero-carry scope. Future shocks are held fixed, observed history is
constant, and exact signed payoff kinks are averaged over future draws. The
first interval is integrated analytically at the original kink locations.
This additional payoff information resolves the initial boundary that cannot be
identified from its scalar local Vega alone.

Returned bin masses include both tails and are not nodal densities. Changing
bin boundaries does not change total Gamma, first-step Vega, price or their
sample moments. Same-draw differences against native AD and price preserve
covariance. Nonzero contribution counts expose unobserved tails; zero sampled
mass/SE is not proof of zero risk. The diagnostic accepts a volatility term
structure but rejects nonflat variance, carry, barriers, smoothing, RQMC and
non-piecewise-affine payoffs. It does not certify ordinary path-dependent IV
buckets. See [validation and remaining limits](../../design/validation/local-vol-initial-gamma-measure-v0.1.md).

## Split-sample Gamma recovery audit

`VegaKtGammaRecoveryPolicy` fixes a descending list of candidate regularizations,
a pilot sample count, a reconstruction limit, a relative Gamma standard-error
limit and an estimated MC norm margin. `evaluate_vega_kt_gamma_recovery_audit`
uses pilot counters `[N,N+P)` to choose the strongest feasible candidate, then
validates once on the original `[0,N)` pricing counters. Antithetic pairs remain
intact. A failed validation is reported without selecting another candidate.
All candidate moments, quality measures, policies and sampling ranges remain
visible; an absent candidate is not a zero-risk result.

The same report includes the conditional initial measure on validation counters.
The initial and positive-time AD contributions agree by construction of the
sampling ranges, but no full cross-time IV covariance or equation-(4)/(9) hedge
is claimed. Estimated norm margins are not simultaneous confidence bounds, and
unobserved tails remain identified. See [selection validation](../../design/validation/local-vol-gamma-selection-v0.1.md).

## VegaKT diagnostics

VegaKT reports retain the configured maturity and log-forward-moneyness
coordinates, row-major bucket order, implied-volatility value at each reporting
bucket, raw and market-scaled units, covariance layout, and whether the full
bucket covariance matrix was requested.

| Field | Meaning |
|---|---|
| `policy_label` | Eligible flat terminal-measure pricing reports `flat_variance_terminal_measure_pathwise_v1`, exact smile replication reports `terminal_payoff_replication_iv_v1`; unsupported native LV reports `local_vol_node_allocation_diagnostic_v1`; standalone primitives retain their own policy |
| `truncation_order` | Native LV currently reports `not_applicable`; no paper convergence order is claimed |
| `active_domain` | Final source grid indices: all nodes for the terminal measure with clamped end hats; connected density domain containing the forward node for local-node allocation |
| `excluded_probability_mass` | Zero for the untruncated terminal measure; probability mass outside the density domain for local-node allocation when available |
| `signed_residual` | Zero for the fully allocated terminal measure; independently accumulated initial-time local adjoints plus positive-time density exclusions for local-node allocation; never a balancing difference |
| `pre_projection` | Independently accumulated atom IV samples for the terminal measure; local parallel Vega for local-node allocation |
| `left_edge_count`, `right_edge_count` | Reporting-IV projections assigned to edge buckets |
| `left_edge_sensitivity`, `right_edge_sensitivity` | Signed sensitivity assigned to each reporting edge |

Bucket estimates expose raw means in currency per unit absolute volatility and
market-scaled means in currency per volatility point. Per-bucket sample
variance and covariance with Price are present when sample data is supplied.
The full bucket covariance matrix is populated only when explicitly requested:
the full-matrix layout requires `full_bucket_covariance`, and the compact
price-and-bucket variance layout omits that field.

For terminal-measure pricing, signed call-replication IV derivatives are
projected at their physical strikes on each actual native path. The affine
payoff remainder has exact zero IV risk under the validated zero-carry scope.
Local parallel Vega remains a separate AD estimate and can differ, especially
with a volatility term structure or an affine payoff remainder. Price covariance
uses the actual original payoff and these IV samples. Antithetic pairs and RQMC
scramble means are reduced before estimating uncertainty.

For the local-node fallback, `pre_projection` remains local parallel Vega and
is not a certified canonical IV Vega. Both paths reconcile their independently
accumulated total against buckets plus exclusions. Edge sensitivities are
averaged over sampling units while counts retain all evaluated path assignments,
including antithetic mates. A zero terminal residual certifies allocation within
that restricted model; it does not validate arbitrary local-volatility surfaces
or a hedging strategy.

## Errors

Local Volatility and VegaKT validation failures terminate the calculation and do
not produce a partial pricing result. Important typed failures include invalid
SSVI/eSSVI admissibility, invalid Local-variance grids, unbracketed surface tail
quantiles, non-positive post-dividend Spot, invalid VegaKT density thresholds,
an inactive forward density node, equation (11) denominator failures,
transition-mass defects, reporting-basis shape mismatches, covariance shape
mismatches, and bucket-plus-residual reconciliation failures.

Python maps these failures through the existing `ValidationError` surface for
request construction and the existing runtime error surface for failed
evaluation. Successful warning ordering remains deterministic.

## References

- [Dupire, *Pricing with a Smile*](https://www.risk.net/derivatives/equity-derivatives/1500211/pricing-with-a-smile), for the Local Volatility construction from a call surface.
- [Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2033323), for SSVI admissibility and density conventions.
- [Corbetta et al., *Robust calibration and arbitrage-free interpolation of SSVI slices*](https://arxiv.org/abs/1804.04924), for eSSVI interpolation consistency.
- [Adrien et al., *Vega KT for the Local Volatility Model: An AD Approach*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4107770), for Local Vega, VegaKT and equation (11) diagnostics.
- [Guyon and Henry-Labordère, *The Smile Calibration Problem Solved*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032), for affine-dividend surface matching.

## Signed-atom IV chain and joint covariance

`evaluate_vega_kt_atom_chain_diagnostic(strikes, policy=...)` connects the initial
conditional measure and pilot-selected positive-time Gamma through equations (4)
and (9). Gaussian product integration covers all positive strikes; reporting hats
clamp the tails. The same validation counters supply bucket, hedge, total and
price moments, price covariances and optional cross-maturity bucket covariance.
A failed recovery returns `estimate=None`. Matching flat reporting IV and every
positive simulation time are required. Covariance is per sampling unit; divide
by N for covariance of estimated means. Inverse bias and full market-IV acceptance
remain outside this diagnostic. See [evidence and limits](../../design/validation/local-vol-atom-chain-v0.1.md).

## Optional local-Vega conservation

`VegaKtGammaRecoveryPolicy(..., preserve_local_vega_total=True)` uses an
explicit equality-constrained Tikhonov inverse. It preserves `1'Bq=1'b` with a
fixed linear operator, retains the full spatial residual, and keeps the same
pilot/validation separation and quality limits. Labels distinguish it from the
unchanged default. It removes the lost-integral bias in flat-model maturity-IV
shocks; it does not establish the complete smile bucket gate. See
[validation and the remaining finite-step calibration gap](../../design/validation/local-vol-conservative-gamma-v0.1.md).

The companion `TwoStepSmileCalibrator` now addresses that price-calibration gap
for listed terminal quotes on exactly two intervals, with the first marginal
fixed and either flat or nonflat second-step local variance. It independently
rebuilds both signs of quote changes before native price evaluation. The
[new evidence](../../design/validation/local-vol-discrete-smile-calibration-v0.1.md)
also projects Gamma through the same calibration tangent and retains the
remaining strike-specific bias. This does not extend the native Gamma scope
or establish full market-IV bucket acceptance.

For specified market-calibration directions, add
`local_vega_conservation_directions` to the Gamma recovery policy. The tensor
contains one matrix per positive-time recovery row. Matrices have a common
count of 1..8 directions and a width equal to the source grid (2..129 nodes),
with at most 64 rows and 16,384 coefficients. The solver preserves all specified
functionals, plus the optional total, in one fixed inverse. Rank loss fails
before sampling. Same-path projection/residual moments expose actual sampling
uncertainty; the old spatial and Gamma-noise gates are not relaxed. Defaults
omit this field and preserve previous output. See the
[quote-Gamma retest](../../design/validation/local-vol-gamma-quote-conservation-v0.1.md).


### 非平坦な2区間Gamma診断

`evaluate_vega_kt_two_step_smile_gamma_audit(strikes, policy=policy,
variance_conservation_directions=directions)` は、固定Gaussian第1分布と非平坦な
第2区間に限定した別APIである。`coordinate='annual_local_variance_nodes'`、
作用素行列、積分精度、全pilot、held-outのGamma・空間残差・方向別の実測SEを返す。
`estimate` はheld-out合格時の指定方向の感応度で、不合格時は `None`。
方向未指定なら方向別配列は空で、Gamma自体は候補の `gamma_masses` を参照する。

既存policyの `preserve_local_vega_total` と `local_vega_conservation_directions` は
ボラ感応度用のため新APIでは拒否する。必要な年率分散方向を新引数で明示する。
較正済みquote方向なら `TwoStepSmileCalibrator.terminal_iv_tangent()[0].T` を渡す。
第1区間flat・実際の2区間time gridとの完全一致、方向のrank、積分精度を満たさない
入力は標本生成前に拒否する。品質判定のMC marginは同時信頼区間ではない。

[範囲と独立価格検証](../../design/validation/local-vol-nonflat-gamma-v0.1.md)。


### 非平坦な継続価格の初期Gamma診断

`PricingPlan.evaluate_vega_kt_initial_smile_gamma_audit(boundaries,
subdivisions=32768, relative_tolerance=1e-8)` は、2区間・一定第1分散を対象にする。
出力はcoarse/fineの符号付きGamma bin質量と年率分散感応度、独立score積分、
価格・感応度のtail上界、native ADと価格の実測MC差を含む。
格子・積分精度不足は `recovery_failed`、native照合失敗は `validation_failed` として保持し、
`estimate` を返さない。自動的な許容値緩和や格子再選択は行わない。

Gammaは有限log領域内の区分線形近似。決定的な回復値にMC標準誤差を作らない。
既存flat初期Gamma APIと、第2区間の `two_step_smile_gamma_audit` の出力は維持する。
後者の `initial_gamma_recovery='not_implemented'` はそのAPIが初期Gammaを計算しないことを示し、
初期成分を評価する場合はこの別APIを明示的に呼び出す。
[検証記録](../../design/validation/local-vol-initial-smile-gamma-v0.1.md)。

### 共同年率分散方向と較正済み市場IV

`PricingPlan.evaluate_vega_kt_joint_gamma_audit(strikes, initial_bins, policy=policy,
variance_conservation_directions=basis, risk_directions=readouts,
subdivisions=32768, relative_tolerance=1e-8)` を追加。
readoutは第1一定分散の係数と第2分散節点の係数、1..16方向。従属readoutは許可するが
保存制約は独立な1..8方向（または空）に限る。共同共分散保存量は400万成分まで。

`initial` と `second` に既存診断の全結果を保持し、両方が合格した場合だけ
`joint_validation` に方向別native・回復・差と `(3D)x(3D)` row-major共分散を返す。
順序はnative D方向、回復 D方向、差 D方向。標本単位の共分散をNで割れば平均の共分散になる。
共同判定失敗でもraw統計は保持するが、最上位 `estimate` は `None`。
成功statusは `joint_direction_policy_passed`。失敗statusは
`initial_recovery_failed`、`second_recovery_failed`、`joint_validation_failed`。
native側では `market_calibration='not_verified_by_native'` と表示する。

較正済み市場IVを評価する場合はcompanionの
`rust_pricing_backtest.joint_smile.evaluate_joint_market_gamma(payload, bins, policy=...)`
を用いる。第1parallel IV、終端quote IV順の接線・request hash・native診断を返す。
IVの単位は小数表示の絶対値（1%=0.01）。通常providerへの昇格は行わない。
[範囲と検証](../../design/validation/local-vol-joint-smile-gamma-v0.1.md)。

### 非Gaussian sourceの3区間終端診断

`PricingPlan.evaluate_vega_kt_three_step_terminal_gamma_audit(strikes, policy=policy,
variance_conservation_directions=U.T)` を追加した。第3rowの全空間AD・回復Gamma・残差・
方向別の実測SE、全pilotとheld-out判定を返す。direction指定は省略可能。
`operator_accuracy` はsource領域、確率総量、株価期待値、積分細分化差とtail上界を含む。
入力方向の市場較正はnativeでは認証しない。

`recovered_variance_row=2`、`unrecovered_variance_rows=[0,1]`、
`earlier_gamma_recovery='not_implemented'` を明示する。
成功statusは `positive_time_policy_passed`、失敗時は `estimate=None`。
方向別SEに全方向間の共分散は含めず、通常のIV bucket出力も置き換えない。
[検証](../../design/validation/local-vol-three-step-gamma-v0.1.md)。

### 3区間の履歴条件付き中間Gammaと依存方向

`evaluate_vega_kt_middle_smile_gamma_audit(boundaries, subdivisions=512, relative_tolerance=1e-8)` は
最初の観測を固定した継続価格から第2分散rowを回復する。4水準の符号付き測度、
内外積分の収束、価格・分散tail、独立native AD照合を返す。履歴全区間を確率密度で重み付けして
調べるMC単位は `antithetic_stratified_sweep` または `stratified_sweep`。
`recovery_failed` と `validation_failed` を区別し、いずれも `estimate=None` とする。
報告bin質量は履歴平均した有限条件付き領域の量であり、無条件Gammaとは解釈しない。

`evaluate_vega_kt_three_step_terminal_readout_audit` は、独立な
`variance_conservation_directions` と依存を許す `variance_risk_directions` を分離する。
指定readoutが全候補の逆作用素で保存されるかをMC前に検査する。依存方向のSEは同一パスの
全節点を合成して算出し、標準誤差の単純加算は行わない。旧終端APIの出力は変えない。
[検証・受入範囲](../../design/validation/local-vol-middle-smile-gamma-v0.1.md)。

### 3区間の初期Gammaと全満期の接続

`evaluate_vega_kt_three_step_initial_gamma_audit(boundaries, subdivisions=512, relative_tolerance=1e-8)` は、
第1観測値を動かし、残り2遷移の条件付き継続価格から初期Gammaを回復する。
第1年率分散rowは平坦、第2・第3rowは非平坦でよい。4水準の符号付き傾き差、
32/64点の積分比較、価格・分散感応度のtail上界、通常native MCとの照合を返す。
第2遷移の確率質量・martingale積分が1e-10内で再現できない場合は入力領域エラーとし、再正規化しない。
回復またはMC照合が失敗した場合の `estimate` は `None`。

`rust_pricing_backtest.three_step_full_smile.evaluate_three_step_market_gamma` は、
初期・中間・終端Gammaを全満期の較正接線で合成する。座標順は第1満期の平行IV、中間quote、終端quote。
標準誤差は終端MCに由来し、初期・中間の数値許容差は別の `deterministic_functional_allowance` に残す。
第1満期の個別スマイル点、任意区間数、全座標間共分散は未対応。
[受入範囲と独立照合](../../design/validation/local-vol-initial-three-gamma-v0.1.md)。


### 可変区間数の終端Gamma

`evaluate_vega_kt_multi_step_terminal_gamma_audit(strikes, policy=policy,
variance_conservation_directions=U.T)` は2..16区間の最後の年率分散rowを回復する。
simulation gridと局所分散time gridは一致を要求する。最初のrowは非平坦でもよく、
第1報告満期を複数遷移後に置けば非平坦な第1市場スマイルも扱える。

全先行遷移の密度積分から作用素を作り、各段階の確率質量・平均spot・tail・細分化差を
`operator_accuracy.steps` に保持する。終端感応度は全履歴を固定したpayoffの解析積分、
`native_price` は従来のfull-path価格。旧APIのraw AD統計とは推定量を区別する。

`recovered_variance_row=N-1`、`unrecovered_variance_rows=[0,...,N-2]`。
成功は `positive_time_policy_passed`、pilot/held-out不合格は `estimate=None`。
保存方向は独立な最大8方向で、市場IVとの対応はcompanionで較正する。
先行Gamma・全満期IV連鎖・全方向共分散はこのAPIでは提供しない。
[4・6区間の独立受入と精度領域](../../design/validation/local-vol-multi-step-gamma-v0.1.md)。


### 可変区間の先行Gammaと同一パスの合成

`evaluate_vega_kt_multi_step_gamma_chain_audit(variance_rows, gamma_bin_boundaries,
variance_risk_directions=directions, subdivisions=64, relative_tolerance=1e-6,
gamma_relative_standard_error_limit=.25)` は、指定rowの条件付き継続価格を動かして
符号付きGamma測度を回復する。directionsの形は `[方向][選択row][局所節点]`。
2..16区間、最大32読み出し方向、従属・ゼロ読み出しも許可する。

全観測履歴を保持し、分岐状態と後続状態を動かし、最終区間を解析積分する。
市場方向のSEは全rowを同一パスで合成して算出する。各rowの有限windowのGamma bin、
節点感応度、細分化差、tail、raw native AD、方向別統計を返す。
終端ADの希少事象によるSE過小評価は、解析的な条件付き第2モーメントでも照合する。

成功は `path_gamma_chain_passed`。数値精度またはGamma・方向のMC精度が不足すると
`recovery_failed`、native照合が失敗すると `validation_failed`、いずれも `estimate=None`。
従来のpilot付き終端逆作用素とは別の診断であり、そのpolicyと出力は変更しない。
[全報告満期の受入と制約](../../design/validation/local-vol-path-gamma-chain-v0.1.md)。


### 可変区間Gammaの合同共分散

`evaluate_vega_kt_multi_step_gamma_chain_audit` は `joint_sample_covariance` を返す。
D方向に対し行優先 `(3D) × (3D)`、順序はnative AD、回復Gamma、その差。
antitheticではペア平均後の共分散であり、平均の共分散にはペア数で割る。
受入失敗時もraw診断は残るが、推定値の受入を意味しない。
[共分散受入](../../design/validation/local-vol-path-covariance-v0.1.md)。


### 中間区間のまれな支払いと条件付きnative AD

可変区間Gamma診断は、最終遷移を解析積分したnative ADとの同一標本差と、
raw AD−条件付きADの差を別々に検査する。後者は解析的な条件付き分散をSE評価に含める。
raw ADが全標本で0でも、不確実性を0とは扱わない。従来のGamma・raw AD・共分散は保持し、
旧直接照合の結果は `raw_node_validation_passed` に残す。
[数式・検証・適用範囲](../../design/validation/local-vol-conditional-ad-v0.1.md)。

4・6区間の条件付きGammaを欧州型バックテストへ接続する範囲と失敗時の動作は、
[複数満期providerの受入](../../design/validation/local-vol-multi-step-provider-v0.1.md)を参照。

Asian・Lookbackの履歴付きGammaを台帳に接続する範囲は
[観測履歴と決済の受入](../../design/validation/local-vol-path-ledger-v0.1.md)を参照。

Lookbackの当日確定値は `valuation_date_fixing` で指定し、
`lookback_valuation_date_fixing` に記録する。`future_monitoring_count` は
評価日以降の日程数を引き続き表し、確定済みの当日も含む。
[当日観測の価格・Gamma検証](../../design/validation/local-vol-lookback-fixing-day-v0.1.md)。

全区間の分散rowを同時に動かす較正と、初期遷移を含む市場方向Gammaは
[明示的な較正延長の受入](../../design/validation/local-vol-variance-extension-v0.1.md)を参照。
較正規則は価格とリスクの定義の一部であり、新APIは診断専用とする。

2つのpresetを通常台帳へ採用する境界は[全区間較正provider](../../design/validation/local-vol-extension-provider-v0.1.md)を参照。

[Put側の数値受入](../../design/validation/local-vol-extension-put-v0.1.md)は、Putを直接積分した価格差分と
全区間Gammaを比較する。既知最安値付きLookbackでは実効行使価格を積分区切りへ含め、
精度基準を維持したまま回復誤差を解消する。明示的なPut対応scopeでCallヘッジを含む台帳へ接続する。

[カスタム基底の受入](../../design/validation/local-vol-custom-basis-v0.1.md)では、時間・マネーネスの
粗い節点とパラメータ尺度を明示し、両側の独立価格差分へ照合する。
同じ市場quotesに較正しても基底の尺度で経路依存商品の感応度が変わるため、設定を監査に保持する。
低水準の診断APIを保持し、商品・観測履歴・価格分布・較正・Gamma精度を検査したproviderだけがヘッジ用Riskを返す。

[名前付き基底の台帳接続](../../design/validation/local-vol-custom-basis-provider-v0.1.md)では、
受入済みの2種類の生成規則だけを選べる。基底構成・較正・Gammaのhash一致を検査し、
Call/Putの売買・決済・途中再開へ接続する。任意の基底行列を受入済みとは扱わない。

[raw基底の入力別検証](../../design/validation/local-vol-raw-basis-v0.1.md)では、
行列を市場・格子・パラメータの尺度の意味に結び付け、独立再較正価格から全市場方向Gammaを照合する。
時間とマネーネスが結び付く行列も指定できる。合格の範囲は、検査した行列と市場・商品条件に限定する。

[結合基底の台帳接続](../../design/validation/local-vol-coupled-basis-provider-v0.1.md)で、
この2行列を `coupled_row_hats` として明示選択できる。生成行列のhash一致を必須とし、
受入済み行列・較正接線・Gammaの対応を保持する。raw行列の自動採用は行わない。

[現行版の行列別検証](../../design/validation/local-vol-current-raw-basis-v0.1.md)は、
現行の受入記録と変更のない数値チェッカーを結び付ける。事前適合確認と数値受入を区別し、
保存内容の整合性を確認して未完了パネルから再開する。通常台帳の受入範囲は拡張しない。

## 商品間の推定誤差とVegaKT損益説明

[市場拡張・合同共分散・損益説明の実装記録](../../design/validation/vega-kt-portfolio-market-v0.1.md)を参照。
`portfolio_market_risk.evaluate_portfolio` で共通経路の異商品共分散を計算し、
`explain_backtest` で保存済み台帳に対応する期首リスクから追加の損益説明を作成する。
固定strikeのIV変化にDeltaを合わせ、期限切れ・欠落・外挿が必要な区間は未説明とする。
既存の台帳と旧受入記録は保持する。
