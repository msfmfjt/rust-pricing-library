# VegaKTの満期Gamma点質量と正確な伝播 v0.1

日付: 2026-10-04  
状態: 制限付きの満期境界をRust/Pythonに実装。24ケース・96検証の失敗0件。
通常の `evaluate()` に対する既存154項目の数値受入は54件の失敗が残る。

## 修正した問題

[前回の調査](vega-kt-remaining-failures-v0.1.md)では、積分Gamma近似の合計Vegaが
許容範囲でも、満期の点状Gammaが近隣Strikeに広がる偏りを確認した。
8区間・17節点のATMバケットは、標本誤差を取り除いた母集団計算でも
15.9697997652となり、正解23.5834379360から32.28%不足していた。

今回の境界処理は、コンパイル済みペイオフの区分線形表現から傾きの跳びを抽出し、
符号付き点質量として保持する。商品名によるcall専用分岐や解析Vegaへの合計合わせは
行わない。call・put・straddle・capped call/call spread・負の倍率・線形相殺を検証した。

```text
payoff(S) = cash + slope*S + sum_a q_a*(S-K_a)+
Gamma_T = sum_a q_a*Dirac(K_a)
```

cashとslopeも保存し、元のペイオフとの値の一致を検査する。同一Strikeの質量は
補償加算で合成し、相殺された質量を除く。min/maxの区間内交点も新たなStrikeとして
抽出する。連続性を数値的に確認できない場合、過大な計算量、非有限値はエラーとなる。

実装は[点質量と伝播](../../crates/pricing/src/engine/risk/vegakt_measure.rs)、
[ペイオフの変換](../../crates/pricing/src/engine/risk/vegakt_terminal.rs)、
[PricingPlanへの接続](../../crates/pricing/src/engine/risk/vegakt_terminal_pricing.rs)に分けた。

## 伝播と時刻0

空間平坦な分散vについて、点質量qの伝播後Gammaは

```text
q * phi(log(S/K)/sqrt(v) + sqrt(v)/2) / (S*sqrt(v))
```

となる。v=0では点質量のままなので、密度としての評価は拒否する。
分散v1、v2を順に伝播した結果と、v1+v2で直接伝播した結果の一致を独立積分で検証した。
コンパイル済みペイオフを正規分布で積分した価格の二階差分とも照合した。

source時点の積分分散u、満期までの残存分散w、V=u+wの場合、元のlog-hat別局所Vegaは

```text
A_ij = sigma_i * dt_i * sum_a q_a*K_a^2*d_T(K_a)
       * E[h_j(X_i) | X_T=log(K_a/S0)]
conditional mean     = (u/V)*log(K_a/S0)
conditional variance = u*w/V
```

で計算する。hatはnative ADと同じlog座標・端点定数外挿を使う。
u=0ではh_j(0)を直接評価するため、初期密度の割り算や除外感応度の足し戻しは不要。
条件付きhatの計算に加え、source状態の無条件密度×伝播後Gammaを独立に積分して検証した。

## 満期別・Strike別の投影

この制限付き終端ペイオフでは、各中間時点のGammaは同じ満期測度の伝播となる。
半群性により式(4)の `H_i*Gamma_(i+1)-Gamma_i` は中間時点で0となり、
満期のヘッジ測度 `-Gamma_T` が残る。式(9)では各点のVegaを掛けてから報告IVのhatへ
投影する。Strikeが格子間にある場合は、隣接IVバケットへそのhat比率で配賦する。
ヘッジ数量はさらに報告節点のVegaで割るため、単なる点質量の線形補間とは異なる。

| 検証 | 件数 | 失敗 | 最大絶対差 |
|---|---:|---:|---:|
| 満期・Strike別バケットと独立参照 | 24 | 0 | 0 |
| source-hat別Vegaと無条件状態積分 | 24 | 0 | 7.11e-15 |
| 期待局所parallel Vegaと一定vol参照 | 24 | 0 | 2e-11未満 |
| native局所Vegaとの比較、6SE | 24 | 0 | 各ケースの6SEを上限として判定 |

ATMケースの新しい満期バケットは23.5834379360、その他のバケットは0。
nativeのcall/put、K=95/100/105、価格のvol差分、seed・標本数・worker変更も検査した。
時間方向にvolが変わる場合、IV parallel Vegaと局所vol parallel Vegaは一般に異なるため、
別々の値として返し、差を投影残差として埋めない。

## APIと対象範囲

```python
diagnostic = plan.evaluate_vega_kt_terminal_diagnostic()
atoms = diagnostic.gamma_atoms
buckets = diagnostic.raw_buckets
audit = diagnostic.to_dict()
```

Rustには同名のPricingPlanメソッドと、任意のコンパイル済み式を入力する
`terminal_payoff_gamma_measure` を追加した。後者はcall spreadなどの合成式も扱う。

対象は、単一満期・単一原資産の連続区分線形ペイオフ、空間平坦な局所分散、一定forward、
単位割引、恒等配当座標。報告満期は全正時刻のシミュレーション節点に一致し、IVは
実際の左端分散で積算したモデル分布と整合する必要がある。
多時点観測、indicator、spot同士の積やspotによる除算、smoothing、barrier、早期行使、
Bassモデルはこの経路の対象外。定数による乗除算と負の倍率は扱う。

計算は決定論的であり、MC標本数・乱数・共分散指定を利用しない。
`calculation_kind: deterministic` と `diagnostic_only: true` を付け、
標準誤差・標本数・受入済みフラグは作らない。端点外へのIV配賦は別途監査値として返す。
`terminal_measure_projection_residual` は渡した終端測度の配賦残差だけを表す。

## 既存受入との関係と残作業

[受入ポリシー](../../fixtures/backtesting/numerical-policy-v0.1.json)は変更していない。
通常の154項目を再実行し、24観測と154判定は前回と完全一致、失敗は54件のまま。
既存の弱Gamma経路も保持し、今回の境界処理と区別する。弱Gammaの63比較も失敗0件。

新しい正確なVegaでも、元の「native MC値との差3%」をそのまま当てると、
seed 97・1024ペア・8区間・17節点の1件が5.6234%で不合格となる。
差は1.9108SEである。この事実を保存し、96項目の診断結果で元の判定を置き換えない。

次はこの終端境界を一般のGamma回復に接続し、source-hat作用素の条件数と標本誤差を
評価する。通常レポートへ統合するには、同一標本の差・価格との共分散・全バケット共分散、
局所volとIVのリスク定義の区別、統計的比較の別バージョン化、ヘッジ数量安定性、
wire/replayをまとめて検証する必要がある。今回だけで本体受入の解消とはしない。

## 再実行と記録

```shell
python scripts/check_vega_kt_terminal_measure.py \
  --output benchmark-results/vega-kt-terminal-measure/report.json
python -m unittest discover -s tests/python -p 'test_vega_kt_terminal.py'
```

Rustの関連integration 74件、local-vol lib 46件、PythonのVegaKT関連28件が成功。
Clippy、型定義とnative公開APIの厳密照合も成功した。
全24ケースの値と比較は `benchmark-results/vega-kt-terminal-measure/report.json`、
通常受入は `benchmark-results/vega-kt-numerics/terminal-measure/report.json` に保存した。
追跡可能な指紋と要約は[検証記録](local-vol-terminal-measure-v0.1.json)に保持する。
