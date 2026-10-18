# 2区間の離散スマイル再較正とGammaのquote別残差

2026-10-05。市場IVを変更した後、実際に使う2区間のlog-Euler価格へ
vanilla quoteを合わせる非線形較正を追加した。**再較正・価格差分・演算整合性は
222/222チェック合格**。一方、較正後の市場quote方向へGammaを射影すると、
**40比較中30比較が精度目標に未達**だった。全体の受入判定は `failed` を維持する。
これは4商品×2 seed×5 quoteの比較で、未達は商品×quoteで15種類ある。

## 実装した範囲

`rust_pricing_backtest.discrete_smile.TwoStepSmileCalibrator` は、
ゼロキャリー・割引係数1、時刻 `[0,t1,t2]` の局所分散モデルを扱う。
第1区間の分散と第1満期の平坦IVを固定し、第2区間の局所分散を変更して
第2満期の**指定されたquote節点**の価格を合わせる。
初期状態は、第2区間が平坦でも非平坦でもよいが、変更前のquoteと離散価格が
あらかじめ較正されていなければ拒否する。

```python
from rust_pricing_backtest.discrete_smile import TwoStepSmileCalibrator

calibrator = TwoStepSmileCalibrator(payload)
recalibrated_request, audit = calibrator.calibrate(terminal_iv_shifts)
local_variance_quote_tangent, tangent_audit = calibrator.terminal_iv_tangent()
```

`terminal_iv_shifts` は最終満期の各quote節点に対する絶対IV変更量である。
元のrequestは変更しない。再較正したrequestをnativeで使う場合も、
同じ2区間の時間格子を維持する必要がある。契約イベント等で内部時刻を追加した
別の離散過程に対して、この較正結果をそのまま正しいとは扱わない。

実際の較正価格は

`C(K) = E[BlackCall(S1, K, local_variance_1(log(S1/S0)) * (t2-t1))]`

である。第1状態をGauss–Legendreで積分し、第2状態を条件付きBlack価格で
積分する。局所分散の線形補間と端点外の一定延長はnativeと一致させる。
局所分散節点の価格微分を解析的に計算し、各Newton増分のEuclidean normが
最小となる解を使う。これは**局所分散をどのように延長するかという明示的な選択**であり、
有限個のquoteからスマイル全体を一意に復元したという意味ではない。

quote行の特異値とランク、全反復の価格残差、刻み縮小係数を保存する。
分散floor/cap、quoteの単調性・凸性・第1満期に対する価格下限、ランク不足、
収束失敗はエラーとし、クリップで隠さない。既定の価格許容誤差は `1e-10`。
積分点数を32から64へ増やし、±12標準偏差外の解析的な価格上限を加えて確認する。
第1満期の非平坦スマイルは、1回のGaussian遷移で到達できないため対象外である。

## 独立な価格検証

[固定ポリシー](../../fixtures/backtesting/discrete-smile-calibration-policy-v0.1.json)は、
局所分散33節点、quoteのlog-moneyness `[-0.2,-0.1,0,0.1,0.2]` を使用する。
平坦な初期IV 20%と、これに `0.015*x+0.02*x²` を加えて再較正した非平坦モデルを検証した。
各モデルについてスキュー、曲率、5個の単独quote変更の7方向を扱う。

- 有限の変更幅 `0.003` の両符号で、別実装・96点の価格積分に照合。
- 微分の変更幅 `1e-5` と `5e-6` の両符号をそれぞれ非線形再較正。
- 通常call5ストライクと、ATM call、Asian、既知fixing付きAsian、Lookbackの価格を比較。
- native価格差分はseed 4001〜4008、各16,384 antitheticペア。同じseed内の
  符号・幅で共通乱数を使い、batch間から差分のSEを算出する。
- 独立参照は較正器の価格関数を呼ばず、商品の境界で分割した条件付きペイオフ積分を使う。
  64点と96点の比較、変更幅の細分化も確認する。
- 回復Gammaやエキゾチックの参照価格を較正器に入力しない。

最大quote価格残差は **4.793e-11未満**、市場quoteの価格応答との差は
**2.132e-9未満**。元の「連続Dupireを2区間へ標本化するとスキュー応答が
50%不足する」診断は履歴として残し、この限定された離散較正方式を別に検証した。
すべての満期・ストライクで元の連続スマイルを再現するとの主張はしない。

## Gammaとの接続で残る失敗

平坦な基準点では、第2区間の年率局所分散に関するquote IVのJacobianは

`A_(k,j) = dt1/(2*IV_T*T) * E[hat_j(X1) | X_T=log(K_k/S0)]`

となる。条件付き平均は `V1/V_T*log(K_k/S0)`、分散は `V1*(V_T-V1)/V_T`。
この式を別の単体テストで積分して、較正器の右逆行列 `U` を確認した。
第1区間を固定するので、quote感応度は局所Vega `b` から `U'b/(2*sigma1)`、
回復Gamma `q` からは `U'Bq/(2*sigma1)` となる。両者の差は
`U'(Bq-b)/(2*sigma1)` であり、合計Vegaの保存だけでは消えない。

既存のpilot262,144ペア・評価131,072ペア、空間残差3%、Gamma相対SE25%、
MC余裕2を維持した。4商品×seed 7/41の全8設定で従来の評価条件は通過する。
quoteについては別途 **`0.002 + 1% * abs(独立母集団quote感応度)`** の精度目標を
設け、母集団の正則化偏りを測った。MC誤差でこの偏りを隠さない。

| 商品 | 選ばれたlambda | 最大quote感応度の偏りの絶対値 | 未達数／比較数 |
| --- | ---: | ---: | ---: |
| Call | 0.01 | 0.817302 | 10/10 |
| Asian | 0.0003 | 0.051605 | 0/10 |
| 既知fixing付きAsian | 0.003 | 0.401410 | 10/10 |
| Lookback | 0.01 | 0.498948 | 10/10 |

上表はMonte Carloの観測誤差ではなく、同じ選択lambdaを使った母集団参照での偏りである。
native標本の回復Gammaは、その正則化された母集団参照と整合する。
表示するquote SEは節点の周辺SEから三角不等式で得た**上限**であり、
未知の共分散を独立と仮定したSEではない。
また、較正した局所ADの感応度は独立なエキゾチック価格差分と最大 **1.974e-5未満**で一致した。
したがって、この比較で次に対処すべき偏りはquote較正ではなくGammaの逆問題側にある。

次はquote方向の再構成残差も含むpilot選択、または複数の感応度方向を保存する
固定線形制約を検討する。非平坦基準でのGamma回復、多区間の非線形較正、
第1満期スマイルの到達可能性、キャリー・配当等は引き続き別の未完了項目である。
通常リスク計算と受入済みP0 providerを昇格させない。

## 再現と判定

```sh
PYTHONPATH=backtesting python scripts/check_discrete_smile_calibration.py \
  --policy fixtures/backtesting/discrete-smile-calibration-policy-v0.1.json \
  --output benchmark-results/vega-kt-discrete-smile/report.json
```

全体の実行はquote-Gamma失敗を含めて**終了コード1**を返す。
`--calibration-only` を明示した場合だけ、222件の再較正・整合性検証を終了コードの
対象にする。その場合もJSON内の30件の未達と全体の `status='failed'` は保持する。
手動CIには較正範囲のジョブを追加し、全レポートを成果物に保存する。リモートCIは未実行。

新しい7件を含むcompanionの **223/223単体テスト合格**。
数値検証では実行中のソース変更を検出し、レポートに依存ソースとnative binaryの
SHA-256を記録する。[機械可読記録](local-vol-discrete-smile-calibration-v0.1.json)にも
判定・残る失敗・参照先の指紋を保存する。
