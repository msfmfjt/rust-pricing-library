# 経路依存商品のGamma回復 v0.1

日付: 2026-10-05  
状態: **空間平坦LV・正時刻のGamma回復をnative診断APIへ実装**。
Asian、既知fixing付きAsian、Fixed Lookback、対照用Europeanで独立照合372判定が合格。
これは正則化した有限基底の回復処理の検証であり、経路依存VegaKT全体の受入ではない。

## 変更点と適用範囲

[従来のweak Gamma](local-vol-weak-gamma-v0.1.md)は、局所Vegaのsource hatと
次時点Gammaのhatを対角的に対応させていた。満期の折れ点が近隣Strikeへ広がり、
MC標本数だけでは消えない誤配賦が残ることを[独立調査](vega-kt-remaining-failures-v0.1.md)で確認済み。
今回、論文式(3)をsource hat上で積分した**全畳み込み行列**を構築し、正則化した逆問題を解く。
合計Vegaだけを合わせる倍率調整、正のGammaへの切り詰め、残差の消去は行わない。

コンパイル済みpayoffの観測マップ・既知fixing・max/minを含む経路状態と、既存の
逆方向ADから実際のstep-local Vegaを取る。観測を満期Spotだけへ置き換えない。
時間方向の分散変化は扱うが、空間方向は平坦、forwardは一定、割引は1を要求する。
配当、非平坦スマイル、早期行使、平滑化、指示関数・離散/連続Barrier、RQMCは拒否する。
Barrierは境界の分布微分や直接分散感応度が必要なので、通常の状態ADで代用しない。

既存の通常評価と[受入済みP0 European経路](vega-kt-smile-acceptance-v0.3.md)は変更していない。
経路依存商品を `numerically_validated_p0` へ昇格せず、バックテストの対象制限も維持する。

## 数式・正則化・統計

Gammaを符号付き原子 `Gamma(dK)=sum_j q_j delta(K-K_j)` で表し、局所Vegaの
source-log hat積分を `b_i` とする。source累積分散を `v0`、区間分散を `v1`、
`x_j=log(K_j/S0)` とすると、畳み込み行列は

```text
B_ij = sigma*dt * K_j²*d_(t+dt)(K_j)
       * E[h_i(X_t) | X_(t+dt)=x_j]
X_t | X_(t+dt)=x_j ~ Normal(v0/(v0+v1)*x_j, v0*v1/(v0+v1))
b ≈ B*q
```

である。Gaussianの条件付きhat積分には既存の検証済み関数を使う。
source状態の裾は端のhatへflat外挿され、入力bと同じ定義を保つ。
一方、GammaのStrike領域は有限なので、領域外の質量を回復したとは扱わない。

```text
q_lambda = argmin_q ||B*q-b||² + (lambda*||B||_F)² ||q||²
```

を規格化した双対Cholesky分解で解く。lambdaは呼出し側が明示し、範囲は
`[1e-6, 1]`、source/Gamma節点は各2〜129、全作用素は100万要素以内。
正則化・作用素は観測されたAD値に応じて変更しない。Frobeniusノルムによる
逆作用素の雑音増幅上界と、回復分解能 `trace(R*B)` を出力する。
Gamma質量の単位は通貨/Spotで、節点密度とは異なる。負の質量も保持する。

各独立標本について `q=R*b` と `B*q-b` を計算してから統計を集約する。
反対変数は先にペア平均する。従って、個々のGammaと再構成差・合計のSEは、
元の局所Vega間の同時変動を含む。周辺SEを独立として加算していない。
MC標本誤差には、正則化バイアス、有限Strike領域の誤差、モデル誤差は含まれない。

時刻0の局所Vegaは、Spot一点からの1本の制約しか与えない。この入力だけで
最初のGamma曲線を一意に回復しない。`unrecovered_initial_local_vega` と
`initial_gamma_status=not_identifiable_from_initial_local_vega` を明示する。
初期境界には、既存の条件付き価格評価など追加情報の接続が必要である。

## API

```python
report = plan.evaluate_vega_kt_gamma_recovery_diagnostic(
    gamma_strikes,
    relative_regularization=0.001,
)
```

`0.001` は再現例の設定であり、全入力に適した推奨値ではない。結果はJSON互換dict。

- `policy_label=flat_variance_convolution_gamma_diagnostic_v1`
- `rows`: source/recovered時刻、Gamma質量、局所Vega、再構成差と各SE、合計
- `inverse_noise_gain_bound`、`resolution_trace`、作用素規格化
- `local_vega_nonzero_sampling_units`: 各source hatの非ゼロ寄与を持つ独立標本数
- `diagnostic_only=true`、`iv_bucket_acceptance=not_established`

このdictは通常の `pricing_result` ではなく、IVバケットやその射影残差を持たない。
リスクの数値方式とnativeバイナリを含めて利用側で識別する。
Rustでは `VegaKtConvolutionPlan` を独立に使い、回復・再構成行列も取得できる。
実装は[回復作用素](../../crates/pricing/src/risk/vegakt_convolution.rs)、
[価格本体への接続](../../crates/pricing/src/engine/risk/vegakt_convolution.rs)、
[Python境界](../../crates/pricing-python/src/vegakt_convolution.rs)を参照。

## 独立検証

[検証スクリプト](../../scripts/check_path_gamma_recovery.py)はnativeの条件付きhat公式を
再利用せず、各source hat区間を96点Gauss–Legendreで直接積分する。
逆問題もCholeskyを再実装せずNumPyのSVDで解く。既知fixing98を含む2観測Asianと
2観測Lookbackについて、条件付きvanilla価格のVegaから局所Vegaを独立に求める。
これは正則化された有限基底解の参照であり、真の非正則化Gammaそのものではない。

4商品×3格子（9/17/33）×2 seed×3正則化（0.01/0.001/0.0001）の72設定を検査。
各設定でAD値、SVD解、独立積分からの回復値、再構成差、雑音増幅の5項目を照合する。
さらに4商品×3 source hatの12価格差分を検査し、合計372判定となる。
価格差分は入力局所分散を実際に±vol bumpで作り直し、16独立バッチ×512ペアの
共通乱数差分からSEを計算する。MC比較は `6 SE` と固定絶対許容値を使用する。

### 少標本で検出した失敗

最初の8,192ペアでは372判定中1件が失敗した。Lookback、17節点、seed 7、
正則化0.0001のGamma参照比較が判定幅を超えた。尾部source hatに非ゼロ寄与がなく、
入力の推定SEが0となったことを逆作用素が増幅していた。0の推定SEは、真の尾部リスクが
0である証拠にならない。このため非ゼロ寄与数を診断へ追加した。

[元ポリシー](../../fixtures/backtesting/path-gamma-policy-v0.1.json)と失敗を保存し、
[標本数細分化ポリシー](../../fixtures/backtesting/path-gamma-refinement-policy-v0.1.json)で
131,072ペアへ増やした。**全閾値・格子・seed・正則化は変更せず、372判定が合格した**。
問題の尾部hatの非ゼロ寄与は0から18となった。それより遠い裾には未観測の領域が残る。
高標本のnative解と独立SVD解の最大差は `3.827e-9`、雑音増幅上界の相対差は `1.547e-10`。

### 再構成誤差は別に監査する

独立解との一致は、正則化バイアスが小さいことを保証しない。
参考の3%再構成監査では、72設定のうち3設定が3%を超えた。
33節点Asianのlambda=0.01（両seed）は約6.63%/7.96%、lambda=0.001（seed 41）は約3.65%。
この3件を消したり、372個の実装照合の合格を理由に品質合格へ変更したりしない。

17節点・seed 7・lambda=0.001では以下の結果となった。

| 商品 | 局所Vegaの再構成相対誤差 |
|---|---:|
| European call | 0.0317% |
| Asian | 0.2749% |
| 既知fixing付きAsian | 0.1399% |
| Fixed Lookback | 0.0159% |

同じ設定の雑音増幅上界は約62.97、分解能traceは17未知数に対して約6.28。
従って、再構成が良好でもGamma各成分が一意・高精度に識別されたとは扱わない。
正則化の選択、領域・時間格子の収束、尾部の十分な標本数を次段階で確認する必要がある。

## 回帰と再現性

- 新Rust6テスト、既存価格統合10テストが成功。
- native Python VegaKT関連48テストが成功（新規4）。worker 1/4、既知fixing、
  notionalと分散の換算、無効な正則化、非対応モデル・指示関数・RQMCの拒否を含む。
- Clippy警告0、macOS arm64 Local Vol replayは直前の受入fixtureと完全一致。
- 通常評価・P0のガード・wire形式は変更なし。Windows実機検証は未実施。
  nativeバイナリの指紋は変わるため、旧バイナリのcheckpointは流用できない。
- 手動numerical CIに独立照合を追加。リモートCIが実行済みであるとは主張しない。

機械記録は[JSON](local-vol-path-gamma-recovery-v0.1.json)。低標本の失敗、
高標本の判定、正則化別の品質監査、ソース・バイナリ指紋を保持する。
詳細は `benchmark-results/vega-kt-path-gamma/` に保存する。

```shell
cargo test --locked -p pricing --test local_vol_convolution_gamma
python -m unittest discover -s tests/python -p test_vega_kt_gamma_recovery.py
python scripts/check_path_gamma_recovery.py \
  --policy fixtures/backtesting/path-gamma-refinement-policy-v0.1.json \
  --output benchmark-results/vega-kt-path-gamma/refined-report.json
```

Pythonは今回のnativeライブラリとNumPyを使う。追加のSciPy依存はない。
原資料はユーザー提供のSSRN 4107770 PDF、式(3)・(10)・(11)。
