# 3区間・非正規分布からの終端Gamma回復

2026-10-06。**984 / 984 checks合格、失敗0**。
第1区間の年率分散が空間一定で、第2・第3区間が非平坦なlog-Eulerモデルを検証した。
第2時点の分布を実際の2回の遷移から積分し、第3区間の年率分散節点ADからGammaを回復する。
早い2区間を固定した終端IV 5方向の再較正価格差分までが今回の受入範囲。
初期・中間の継続価格Gammaと全満期IV chainは未実装。通常providerの全面受入も未確立。
[機械可読記録](local-vol-three-step-gamma-v0.1.json)。

## 追加した処理

`VegaKtConvolutionPlan::from_two_transition_variance_source` は、
`X1 ~ N(-V1/2,V1)` と
`X2 | X1=x ~ N(x-v2(x) dt2/2, v2(x) dt2)` を積分する。
第2時点の分布を単一正規分布で近似しない。第3区間の分散v3にも実際の線形補間と境界一定延長を使う。
作用素は `B[j,k] = E[hat_j(X2) d Black(S2,Kk,v3(X2)dt3)/d v3]`。
第2区間が平坦な極限では、既存の厳密Gaussian source作用素に一致する。

32点と64点のGauss–Legendre積分を各native格子区間で行い、作用素だけでなく
確率総量と株価の第1モーメントも確認する。積分後の再正規化は行わない。
総積分分散の上界Aを用い、source log領域を `[-A/2-12 sqrt(A), A/2+12 sqrt(A)]` とする。
適合的なlog-Euler過程の指数モーメントに対するChernoff上界から、source両端の確率と
株価重み付きtailを評価する。第1積分の±12標準偏差外は別に加算する。
最小の第3年率分散を使って株価tailを作用素各要素の上界へ変換する。
積分次数間の差は収束診断であり、厳密な総近似誤差上界ではない。

`PricingPlan.evaluate_vega_kt_three_step_terminal_gamma_audit` は実際のcompiled pathから
第3分散rowのADを抽出する。先行2区間の非平坦性と商品の履歴はnative価格・ADに保持される。
既存の候補評価処理を共有化し、pilotとheld-outを分離した判定と全空間残差を維持した。
診断に `recovered_variance_row=2`、`unrecovered_variance_rows=[0,1]` を明記する。
`unrecovered_initial_variance_sensitivity` は第1rowのみの既存補助統計で、中間rowの回復値ではない。

`ThreeStepTerminalSmileCalibrator` は、先行2区間とそのquoteを固定し、第3分散rowで
終端call quoteをminimum-norm Newton較正する。元requestを変更せず、新requestと監査情報を返す。
独立終端接線Uを `variance_conservation_directions=U.T` に設定し、年率分散方向を保存する。
native API単独は渡された方向が市場較正に由来することを認証しない。

## 固定した検証条件

正・負skewの2プロファイル、call / Asian / 既知平均付きAsian / lookbackの4商品、
seed 7 / 41の計16条件。局所分散33節点、終端quote 5節点、3回の観測・遷移。
第2時点の市場quoteは2区間較正、第3時点の合成市場quoteは独立な条件付き価格積分で設定した。
各quoteの再較正時に前2rowが完全に保存されることも検証する。

- pilot 262,144 pairs、held-out 524,288 pairs。
- 正則化候補 `[.01,.003,.001,.0003,.0001]`、空間誤差3%、Gamma相対SE25%、MC margin 2。
- 各quote IV bump 1e-5 / 5e-6、価格微分収束0.002。
- 独立native価格はseed 4001..4008、各16,384 pairs。比較基準は6 SE＋0.002。
- Gamma母集団射影biasは0.002＋1%。有限±0.003の終端IV移動も独立価格で較正確認。

既存の判定閾値・標本数・正則化候補を変更していない。
callとlookbackは0.01、Asianと既知平均付きAsianは0.0001をpilotが選択し、held-outも合格した。
較正接線の節点差分テストでは、1e-5→5e-6→2e-6→1e-6の二次収束を確認し、
最細bumpで2e-6以内を要求する。価格差分の上記bump幅・閾値は維持している。

## 独立照合の結果

| 検査 | 最大値 |
|---|---:|
| native作用素と96点独立積分の差 | 3.34e-14 |
| 母集団接線感応度と再較正価格差分の差 | 3.55e-5 |
| 独立価格差分のbump細分化差 | 2.66e-5 |
| 独立payoff価格積分の64→96点差 | 2.31e-14 |
| 独立payoff年率分散微分の64→96点差 | 1.87e-14 |
| 独立KKTによるGammaとの差 | 1.35e-7 |
| 母集団の終端quote Gamma射影bias | 5.94e-13 |
| held-out空間誤差の判定値 | 2.623%（上限3%） |
| held-out Gamma相対SE | 8.815%（上限25%） |

payoff oracleは第1状態ごとに第2状態の条件付き領域を積分し、native分散節点と履歴依存の
payoff境界で分割する。較正のsource密度生成やnative ADは使わない。
作用素の独立検証にはnativeより広い領域と96点積分を使う。
同一経路の第3row price bumpも別の単体テストで照合した。

保存制約の母集団biasは有限標本の価格精度ではない。
実測Gammaとnative再較正価格微分の最大差は1.186で、双方の独立SEを合成した基準内だった。
方向ごとのSEは同一経路の節点相関を保持するが、今回のAPIは全方向間共分散を返さない。
SEはpilot選択を条件としたMC推定で、正則化biasや積分誤差は含まない。

## 失敗・互換性・再実行

pilotに合格候補がない場合とheld-out失敗は `estimate=None` とし、候補や選択結果を保持する。
sourceが狭すぎて積分で解像できないモデル、非有限値、rank不足、追加観測によって
実際の区間数が3でなくなるモデルは拒否する。負のGamma質量を除去しない。
旧40ケースは前回nativeの出力とbyte単位で一致した。

関連Rust 36件、Python VegaKT 111件、companion較正16件、wheel contract 4件が合格。
wheelの実行時APIと型定義も照合した。

```sh
python -m pip install . "./backtesting[validation,constraints]"
python scripts/check_three_step_gamma.py \
  --policy fixtures/backtesting/three-step-gamma-policy-v0.1.json \
  --output benchmark-results/vega-kt-three-step/report.json
```

SciPyは独立検証の正規CDFに用いる。calibratorとnative API自体は新たなSciPy依存を持たない。
手動numerical CIとsource archive必須ファイルに検証一式を追加した。

残件は3区間の初期・中間Gammaと満期間再較正接線の結合、任意区間数、非平坦な第1市場満期、
carry・配当・バリア・早期行使、および通常providerの全面IV受入。
