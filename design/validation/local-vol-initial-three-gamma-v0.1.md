# 3区間の初期Gammaと全満期の較正連鎖

2026-10-06。第1年率分散が平坦、第2・第3rowが非平坦な3区間log-Eulerモデルを対象とする。
第1満期の平行IVと中間・終端の各5市場quote、計11方向を接続した診断。
任意区間数、第1満期の個別スマイル点、通常providerの全面的IV受入は対象外。
[機械可読記録](local-vol-initial-three-gamma-v0.1.json)。

## 初期Gammaの定義

`F(s1)=E[payoff(s1,S2,S3)|S1=s1]` のspotに関する分布2階微分を回復する。
Asianの最初のfixing、既知平均、lookbackの過去最大を正しく残し、s1とともに変える。
第2遷移の分散は実際の `v2(log(s1/S0))`、第3遷移も実際の `v3(log(S2/S0))` を使う。
第3遷移をcompiled payoffの区分線形表現に対するBlack積分で消去する。
第2遷移の条件付き±12標準偏差を、分散節点とpayoff切替点で分割して32/64点積分する。

初期Gaussianの±12標準偏差に、分散節点、bin境界、compiled payoffから得た切替点を加える。
背景格子をN,2N,4N,8Nで細分化し、物理spotの傾き差を符号付きGamma原子として集計する。
2組の隣接3水準に `(a-6b+8c)/3` を適用する。負の原子や境界原子を切り捨てない。
Gammaを第1区間のBlack年率分散kernelで積分すると、最初のrowの平行年率分散感応度になる。

第2遷移の積分は、32/64点の両方で確率質量1と条件付き平均spotを1e-10以内に再現することを要求する。
極端に狭い遷移を未解像のままゼロ継続価格と扱うことはせず、明示的に失敗させる。再正規化はしない。
初期分散が小さすぎて浮動小数点上の積分格子が潰れる場合も、添字エラーを起こす前に明示的に拒否する。
compiled payoffの線形成長上界から、内側と外側の価格・年率分散感応度のtail上界を導く。
32/64点差の最大相対値も成長上界で伝播させ、価格・分散感応度の収束許容差として別に検査する。
有限領域のGamma質量を全領域の質量とは呼ばず、細分化差も厳密な総誤差上界とは呼ばない。

N=512、相対汎関数許容差1e-8を既定とする。通常native MCの価格と第1分散rowのAD和を別に計算し、
価格と感応度をそれぞれ6 SE＋決定論的許容差で照合する。
第1分散は一つの平行座標なので通常のpathまたはantithetic pairが標本単位となる。
回復またはMC検査の失敗で `estimate=None`。中間Gammaの既存の層別一巡抽出は変更しない。

## 独立参照

4商品の解析的な第3遷移継続価格に対し、最初のfixingの明示的微分と
第2遷移密度の初期spot微分を積分して `F'(s1)` を求める。
密度の微分には第2分散の局所的な傾きも含める。
そのDeltaに対する部分積分 `∫K dF'=[KF']−∫K_x F' dx1` を初期Gammaの独立参照とする。
64/96点を比較する。nativeの傾き差、初期score価格微分、AD、較正接線を参照Gamma生成に使わない。
SciPyは独立参照スクリプトに限定し、本体・companionの新たな実行依存にはしない。

## 3満期の較正連鎖

`calibrate_all_maturities(first_iv_shift, middle_iv_shifts, terminal_iv_shifts)` は、
第1平行IV、第2市場slice、第3市場sliceの順に再較正する。
後続quoteのshiftがゼロでも、状態分布の変化を相殺するよう後続の局所分散を動かす。
すべてのquote価格検査が通るまで、未完成のrequestは返さない。入力requestは変更しない。
最小Euclideanノルムの年率分散増分という既存のモデル選択を維持する。

`all_maturity_iv_tangent()` の列順は、第1平行IV、中間5点、終端5点。
行順は、第1平行年率分散、中間33節点、終端33節点。
第1IV→第1分散をa、第1IV→中間分散の補償をbとし、終端価格の第1・第2分散微分をH0,H2とする。
終端quote Vega V3、終端単独接線U3により、第3分散の補償は
`c=U3[-diag(V3)^(-1)(H0 a+H2 b)]`。
新しい第1列は `[a,b,c]`、残る10列は既存の中間・終端接線そのもの。
H0は第1Gaussian密度の微分から独立に積分し、32/64点差とtailを検査する。

初期・中間の決定論的Gamma寄与と終端Gammaの11方向を加える。
終端の独立保存制約はU3のみで、依存する補償方向を重複制約にしない。
同一パス上で合成した終端方向統計のSEを使い、初期・中間の数値許容差は別出力に残す。
SEはpilot選択を条件とし、数値誤差・正則化bias・方向間共分散を含まない。

## 数値受入と以前の証跡

正・負skew × call / Asian / 既知平均付きAsian / lookback × seed 7 / 41の16条件。
33局所節点、各満期5 quotes、初期・中間GammaのN=512、相対許容差1e-8。
既存の終端条件はpilot 262,144 pairs、held-out 524,288 pairs、
正則化 `[.01,.003,.001,.0003,.0001]`、空間誤差3%、Gamma相対SE25%、MC margin 2。
較正quoteの価格誤差は1e-10。第1平行IVの±0.003でも後続market quotesを保持する。

新しい第1方向を1e-5 / 5e-6動かした再較正価格を、独立積分と通常native価格の双方で新たに検査する。
native価格は独立seed 4001..4008、各16,384 pairsを使う。
市場Gamma比較は6 SE＋決定論的許容差＋0.002、母集団biasは0.002＋1%、bump細分化差は0.002。

前回の中間・終端10方向の価格差分は、[1,344検査の証跡](local-vol-middle-smile-gamma-v0.1.md)から引き継ぐ。
引継ぎ前に同一policy・同一request・参照実装のhashを検査し、全bumpの較正requestの完全一致を確認する。
さらに全16条件の中間auditと終端Gamma統計を現在のbinaryで再計算し、前回結果との完全一致を要求する。
10方向の価格差分を今回新たに再計算したとは扱わず、レポートの `fresh_coordinates` と `retained_coordinates` で明示する。
前回証跡のSHA-256と今回実装のSHA-256を記録し、実行中の変更も拒否する。
前回のsource archiveとも照合し、既存calibratorの全コードがbyte単位でそのまま残り、
今回の新しいメソッドだけを末尾に追加したことを確認する。

**1,480 / 1,480 checks合格、失敗0。** 全16条件で初期・中間・終端の検査が合格。

| 検査 | 最大差 |
|---|---:|
| 初期Gammaと独立Delta部分積分 | 6.14776e-09 |
| 初期価格と独立積分 | 2.52308e-10 |
| 11方向の接線と再較正価格差分 | 3.54129e-05 |
| 第1方向の独立価格bump細分化差 | 1.32339e-08 |
| 11方向の同一パス保存誤差 | 8.2704e-13 |
| 11方向のGamma母集団bias | 2.43328e-08 |
| native再較正価格差分との最大差 | 1.30241 |

native再較正価格との比較は、方向ごとの6 SEと決定論的許容差を含む固定基準内。
母集団biasの小ささを有限標本精度とは解釈しない。

Rust 40件、Python VegaKT 125件、companion較正22件、wheel契約5件、合計192件が合格。
clippy、format、依存方向、差分、実行時wheel契約も合格。
旧40ケース＋旧3区間終端8ケースはbyte単位で同一。
今回全16条件で再計算した中間auditと終端のGamma・残差・品質統計も前回と完全一致した。

第2分散の傾きが折れ曲がる点の負の初期Gamma原子を解析値で照合した。
worker数変更、notional倍率、通常価格の不変性、antitheticなし＋Brownian bridgeも検査した。
格子不足と2標本のMC不足は失敗を維持して推定値を返さない。
極端に狭い第2遷移は確率質量・平均spotの検査で拒否し、ゼロ価格として受け入れない。
17読み出し方向を許容し、18方向を拒否することも確認した。
極小の第1分散で積分格子が潰れる入力は、処理を落とさず明示的な数値領域エラーで拒否する。

ソース配布物は必須490件・計869ファイルを検査した。nativeとcompanionの検証済みwheelを
作業用 `.venv` にインストールし、native binaryのSHA-256一致、新規較正接線、境界条件の拒否を再確認した。


## 再実行

Python native wheel、NumPy、検証用SciPy、backtesting packageを利用できる環境で実行する。
先に `check_middle_smile_gamma.py` の同一policyによる合格レポートが必要。

```sh
python scripts/check_initial_three_gamma.py \
  --policy fixtures/backtesting/initial-three-gamma-policy-v0.1.json \
  --previous-report benchmark-results/vega-kt-middle-smile/report.json \
  --output benchmark-results/vega-kt-initial-three/report.json
```

CIは中間・終端の検証後にこの検査を実行し、両方の生レポートをartifactに保存する。
原本の数値証跡は `benchmark-results/vega-kt-initial-three/`、版管理可能な集約値は上記JSONに置く。
`full_market_iv_bucket_acceptance=not_established` を維持する。
任意区間数、第1満期の非平坦スマイル、carry・配当・barrier・早期行使、全quote共分散、通常providerでの受入は残る課題。
