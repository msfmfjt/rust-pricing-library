# 経路依存商品の初期Gamma測度 — v0.1

2026-10-05。空間平坦・ゼロキャリーの連続区分線形ペイオフについて、
最初の正時刻のGammaと最初の区間の局所Vegaを本体の診断APIへ接続した。
Asian、既知fixing付きAsian、Fixed Lookback、European callを独立に照合する。
通常のIVバケットをこの診断で置き換えず、経路依存商品の全面的なIV数値受入は未成立。

## 方法と初期境界

時刻0の局所Vegaは1つのスカラーであり、それだけから最初のGamma分布は識別できない。
新しいAPIは、ペイオフと将来の遷移という追加情報を使う。
将来の乱数 `z1,...,zN-1` を固定し、最初の正時刻の状態を `x` とすると、
空間平坦・ゼロキャリー下の将来の観測は `a_i*x` となる。
既知fixingと評価日当日の観測は定数として維持する。

既存の終端測度コンパイラを共用し、元のコンパイル済みペイオフを各標本で

`f_z(x) = cash_z + slope_z*x + sum_j q_j(z)*(x-K_j(z))+`

へ展開する。商品名に応じた公式を本体へ追加していない。
符号付きGammaは `sum_j q_j(z) delta_(K_j(z))`。将来乱数に関する平均が
最初の正時刻の条件付き継続価格のGamma測度となる。

最初の区間の分散 `v0=sigma0²*dt0` のみ厳密に積分し、各標本の初期Vegaを

`sum_j q_j * S0 * phi(d1_j) * sigma0*dt0/sqrt(v0)`

で計算する。現金・線形部分の初期Vegaは厳密に0。価格も同じ元の原子位置で
Blackのcall期待値と現金・線形部分を積分する。密度の除算、有限差分Gamma、
目標Vegaに合わせる再正規化、正のGammaへの切り詰めは行わない。

最初の乱数を含む元のパスも評価し、コンパイル済みペイオフの自動微分による
初期局所Vega、通常価格、条件付き期待値との差を同じ標本で蓄積する。
antitheticはペアにしてから平均・分散・SEを計算し、差の共分散を維持する。

## APIと表示区間

```python
initial = plan.evaluate_vega_kt_initial_gamma_diagnostic(
    [70., 85., 95., 100., 105., 115., 135.]
)
```

- `gamma_bin_masses`: `[0,b0),[b0,b1),...,[bn,infinity)` のGamma積分質量。
  境界上の原子は右側の区間に入る。密度や節点原子の代替ではない。
- `gamma_bin_first_step_vegas`: 各区間の原子の初期Vega寄与。元の位置で計算するため、
  表示区間を細分化・変更しても全体の価格・Vega・Gamma質量とその分散は変わらない。
- `gamma_bin_nonzero_sampling_units`: 非ゼロの質量寄与があったパス／ペアの数。
  符号の相殺を含むため、原子の存在数や完全な尾部被覆を意味しない。
- `recovered_initial_local_vega` / `native_initial_local_vega` /
  `initial_local_vega_difference`: 条件付き測度、既存AD、その標本ごとの差の統計。
- `conditional_price` / `native_price` / `price_difference`:
  最初の区間を積分した価格、元のパス価格、その標本ごとの差。
- `payoff_reconstruction_difference`: 元のパスの `S1` で展開を評価した値と元のペイオフの差。

`policy_label=flat_variance_conditional_initial_gamma_measure_v1`、
`diagnostic_only=true`、`iv_bucket_acceptance=not_established` を明示する。
MC誤差は将来乱数の標本誤差であり、非平坦モデルの近似を主張しない。
空間非平坦、キャリー・配当、早期行使、バリア、平滑化、指示関数、RQMC、
状態同士の積や状態依存の除数は拒否する。期間ごとのボラティリティの変化は許容する。

## 数値検証

[独立スクリプト](../../scripts/check_initial_gamma_measure.py)は、2観測の各商品を
`(A(R)*S1-K_eff)+` と直接表し、将来の1次元正規変数をGauss–Legendre積分する。
Gamma表示境界とLookbackの折れ曲がりで積分区間を分割する。
96点と192点の積分を比較し、コンパイラ・原子・ADを参照計算で再利用しない。

4商品×2ボラティリティ構造×2 seed×antithetic有無の32設定で、Gamma区間質量、
区間Vega、Gamma総量、初期Vega、価格、積分収束、標本ごとのAD・価格差、
ペイオフ再構成、区間和の保存を検査する。
さらに実際の最初の局所分散行を±0.0001 volatilityで作り直し、
16独立バッチ×512ペアの共通乱数による価格差分を4商品で比較する。合計484判定。

最初の131,072標本では483/484判定が合格し、既知fixing付きAsian・
volatility 0.17/0.26・seed 7・antitheticなしの遠い裾で1件失敗した。
135以上のGamma質量は参照 `3.619356231e-6` に対し標本値0、SE0だった。
0の推定SEは尾部リスクがない証拠ではない。失敗を保存し、
[細分化ポリシー](../../fixtures/backtesting/initial-gamma-measure-refinement-policy-v0.1.json)で
標本を1,048,576へ増やした。**許容値、商品、seed、表示区間を変更せず484/484判定が合格**。
問題の135以上の区間は非ゼロ寄与7件、質量 `3.679546014e-6`、
SE `1.390802026e-6` となった。70未満の区間は依然0件で、参照質量
`1.470753596e-6` は固定絶対許容値 `2e-6` 内にある。全ての裾が十分に観測されたとは扱わない。
96点／192点の積分の最大差は `3.198e-14`。

初期Vegaの代表値（volatility 0.2/0.2、seed 7、antithetic）は以下。
差とSEは標本ごとに「回復値−AD値」を求めて集計したもの。

| 商品 | 回復した初期Vega | ADとの差 | 差のSE |
|---|---:|---:|---:|
| European call | 11.794089 | -0.010906 | 0.018537 |
| Asian | 14.927025 | -0.005351 | 0.016379 |
| 既知fixing付きAsian | 10.420253 | -0.001308 | 0.012741 |
| Fixed Lookback | 14.520953 | -0.012086 | 0.016703 |

最終の照合結果・尾部寄与数・ソースとバイナリの指紋は
[機械記録](local-vol-initial-gamma-measure-v0.1.json)に保持する。

## 正時刻側に残る3件の切り分け

[前回の372判定](local-vol-path-gamma-recovery-v0.1.md)の合格と、3件の再構成品質失敗は別物。
[追加分析](../../scripts/analyze_path_gamma_bias.py)では、同一作用素 `B` と逆作用素 `R` について

`(BR-I)*b_MC = (BR-I)*b_reference + (BR-I)*(b_MC-b_reference)`

を計算する。第1項が参照入力に対する有限基底・正則化の残差、第2項が標本雑音の寄与。
ノルムは一般に加算できず、両者の方向によって強め合うことも相殺することもある。
72設定とGamma節点数17/33/65/129の参照値による細分化を保存する。

33節点Asianの参照入力では、lambda 0.01で約7.2753%、0.001で約3.1381%、
0.0001で約1.1017%の再構成誤差。高標本化だけでは強い正則化による偏りは消えない。
これを理由にlambdaを自動的に変更したり、以前の3%判定を緩和したりしない。
Gamma節点を33から129へ増やしてもこれらはほぼ不変だった。
正則化の選択と雑音増幅の両立は次の未解決課題として残す。

## 回帰と範囲

新規テストは既知履歴、当日観測、Put、現金・線形のみの支払額、符号付きcapのGamma、
非線形グラフの拒否、1区間のDirac、3観測・不均等時間区間・期間構造、
Brownian bridge、worker 1/4、notionalの分散換算、裾を含む区間変更を検査する。
Rust26テスト（条件付きコンパイラ2＋関連integration24）、Python VegaKT56テストが通過。
最終バイナリで既存の正時刻独立照合372項目も全て合格し、macOS arm64の
Local Vol replayは受入fixtureと完全一致。Clippy警告0、整形・依存方向・差分検査も成功。
新しいAPIは通常の価格・リスクレポートや既存P0ガードを変更しない。
バイナリ指紋は変わるため、以前のnativeバイナリのcheckpointは流用できない。

この段階で初期境界を回復しても、全時刻のGammaを式(4)・(9)へ結合した経路依存商品の
IVヘッジの受入にはならない。正時刻側の正則化、非平坦スマイル、バリア、
実市場バックテストの効果検証は未完了。手動CIへ本診断を追加するが、
リモートCIやWindows実機が実行済みとは扱わない。

```shell
python scripts/check_initial_gamma_measure.py \
  --policy fixtures/backtesting/initial-gamma-measure-refinement-policy-v0.1.json \
  --output benchmark-results/vega-kt-initial-measure/refined-report.json
```
