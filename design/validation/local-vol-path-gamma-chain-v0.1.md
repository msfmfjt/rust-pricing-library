# 可変区間数の条件付きGammaと全報告満期のスマイル

日付: 2026-10-06。機械可読な集約値は[JSON](local-vol-path-gamma-chain-v0.1.json)。

## 今回の対応範囲

4区間・2報告満期、6区間・3報告満期の各5市場quotesについて、先行満期の変更と後続満期の補償を接続する。
第1市場満期は2遷移後で、約0.348 vol percentage pointsの非平坦スマイルを持つ。
各報告満期の直前の分散rowだけを較正し、報告のない区間の分散rowは固定する。
4区間ではrow `[1,3]`、6区間では `[1,3,5]` を回復する。row番号は0始まり。
この延長規則に従う全市場10・15方向を検証する。任意の較正规則への不変性は主張しない。

新しいnative APIは `evaluate_vega_kt_multi_step_gamma_chain_audit`。
2..16区間、2..33局所節点、選択した昇順・重複なしの分散row、1..32読み出し方向を受け付ける。
simulation gridと分散time gridの一致、unit discount・zero carry、連続PWL payoffを要求する。
全区間に対してGammaを自動回復するAPIではなく、指定rowのGammaを同一パスで合算する診断である。

## 履歴・後続経路を動かすGamma

row rより前の観測履歴と、rより後の標準正規乱数を固定する。
状態 `S_(r+1)=s` を動かすと、その観測値、後続の状態、後続の状態依存分散をすべて再計算する。
最後の区間はcompiled payoffの符号付きPWLをBlack積分する。
こうして得る継続価格 `F(s; history, future shocks)` を、現在の遷移の条件付きwindowで補間する。
Asianの全fixing、既知fixing、lookbackの過去最大値を保持し、履歴をsource spotだけに縮約しない。

物理spot方向のPWL傾き差が符号付きGamma原子となる。
各原子 `(K,mass)` の寄与を `mass*S_source*phi(d1)*dt/(2*sqrt(v*dt))` として積分し、
source位置のnative補間hatに配分する。負の原子を削除せず、Gammaを価格差分そのもので代用しない。
末尾のrowはpayoffの解析Gamma原子をそのまま使い、補間を行わない。

条件付きwindowは `log(S_source)-v*dt/2 ± 12*sqrt(v*dt)`。
局所補間節点とGamma bin境界を追加し、N・2N・4Nの入れ子格子で傾き差を回復する。
粗・細のRichardson測度を `(4*b-a)/3` で作る。各パスの粗細差の絶対値を平均し、
その6 SEを加えて数値精度を判定する。符号を平均して数値誤差を相殺しない。
bin質量は有限window上の履歴・後続乱数平均であり、全領域のGamma総量とは扱わない。
binごとの質量細分化差は別途報告し、価格・分散汎関数の受入とは区別する。

compiled payoffから `A + sum B_i*S_i` の成長上界を作る。
将来の固定乱数に対する最大log-Euler倍率を使い、各条件付き曲線の `a+b*s` 上界を求める。
曲線端の線形外挿との差を、Gaussian tailとstock加重tailで評価する。
分散微分にはGaussian scoreの絶対値上界を使う。
精度が不足するwindow、非有限値、作業量上限を超える入力は明示的に拒否する。
最大262,144 sampling units、還元用保存量400万成分、推定継続計算量100億演算までとする。
入れ子格子の収束診断やMCで平均したtail許容差は、総誤差の厳密保証ではない。

## 標本誤差と失敗の扱い

各区間のGamma、raw native AD、同一パス上の差を保存する。
市場方向は全row・全節点を同じパスで合算し、antithetic平均後に標準誤差を計算する。
区間別SEの単純加算を行わない。全方向間の共分散行列は返さない。

終端raw ADは、遠いsource節点でpayoffの発生がまれだと、観測分散を過小評価しうる。
PWL payoffの各傾き区間でraw variance ADの条件付き第2モーメントを解析積分する。
`Z` を `2*sqrt(v*dt)` だけshiftした切断正規の0・1・2次モーメントから求める。
独立Gaussian積分で照合し、antithetic組にはCauchy-Schwarzによる分散上界を使う。
節点別AD照合のSEは、実測SEと、この条件付き分散平均＋6 SEから得るSEの大きい方とする。
raw統計自体は変更しない。読み出し方向は別途、同一パス差の実測SEで検査する。

数値回復・各rowのGamma bin相対SE・合算方向の相対SE・native照合のいずれかが不合格なら、
最上位の推定値は `None`。raw統計と失敗した判定は保持する。
相対SEは各rowのbinベクトルと全読み出し方向のベクトルについて `L2(SE)/L2(mean)` を使う。
個々のquoteの相対誤差を25%以内に保証する判定ではない。
有限標本の6 SE基準は、まれな事象を含む同時信頼区間の保証ではない。

## 全報告満期の較正

`MultiStepSmileCalibrator.calibrate_all_maturities(shifts)` は、先行満期から順に、
その直前rowを最小ノルムの年率分散増分で較正する。変更後のsource分布で次の満期を較正する。
後続shiftを0にすれば、その市場quotesを保持する。分散の上下限制約と価格精度を満たすまで
backtrackingし、clippingで失敗を隠さない。最後に全報告満期を再検証する。
第1報告満期が1回の遷移直後の場合は、個別smileが到達不能なのでこの入口では拒否する。

`all_maturity_iv_tangent()` は、実際の遷移密度の分散微分から各満期の価格Jacobianを求める。
後方call継続価格と前方source密度を組み合わせ、下三角の補償接線を構成する。
32/64点の細分化水準で全市場IV応答が単位行列になることを1e-9以内で検査する。
この較正には商品Gammaや商品価格MCを使用しない。

## 数値受入

正skewの4区間・負skewの6区間 × 4商品 × seeds 7 / 41の16条件。
各Gamma計算は4,096 antithetic pairs、N=512（最大4N）、相対汎関数許容差1e-6。
Gamma binと合算方向の相対SE上限は25%。従来の終端逆作用素APIの3%・25%のpolicyは変更しない。
今回の条件付きGamma測度は、正則化・pilot選択を用いる従来の終端逆作用素とは別の推定法である。

全10・15方向のIVを±1e-5 / ±5e-6動かし、先行・後続をすべて再較正する。
再較正後の全市場価格を、独立96点積分で1e-10以内に照合する。
第1満期を±0.003平行移動した場合も後続のquotesが保持されることを検査する。
商品価格差分は、通常native価格（8 seeds × 16,384 pairs）と、
独立scrambled Sobol＋解析終端積分（8 scrambles × 65,536経路）の両方を新規計算する。
Gammaとの比較は6 combined SE＋0.002。異なるbump幅も同じ乱数を用いて照合する。
較正Jacobianやnative ADから独立参照リスクを生成しない。

**2,704検査合格、失敗0。全16条件で10・15方向のGamma連鎖が合格。**

| 検査 | 最大値 |
|---|---:|
| 全満期接線のIV応答誤差 | 7.33882e-15 |
| 独立全満期quote価格誤差 | 8.8145e-12 |
| 第1満期stress後の全quote価格誤差 | 3.63931e-13 |
| Gamma binと節点感応度の合計差 | 3.55271e-15 |
| 同一パス方向の合成誤差 | 3.01981e-14 |

Gamma bin相対SEの最大は2.087%、合算方向の相対SEは16.645%。
native再較正価格差分との最大差は6.12083、方向別許容幅に対する最大比率は0.399186。
標本誤差を含む受入であり、全方向の絶対誤差が0.002以内という意味ではない。

Rust 46件、Python VegaKT 136件、companion較正28件、wheel契約5件、合計215テスト合格。
旧48ケースと前回の可変区間終端16ケース、計64ケースの出力はbyte単位で同一。
worker数、notional倍率、通常価格、従属・ゼロ読み出し、antitheticなし＋Brownian bridge、
粗い格子・MC精度不足での推定値抑止、精度領域と計算量上限も検査した。

clippy、format、依存方向、差分、実行時wheel契約が合格。
検証済みnative/companion wheelを作業用 `.venv` に反映し、binary hashと較正接線を確認した。

ソース配布物は必須518件・計897ファイルを検査し合格。文書のローカルリンク検査も合格。


## 再実行と残る範囲

```sh
python scripts/check_multi_step_gamma_chain.py \
  --policy fixtures/backtesting/path-gamma-chain-policy-v0.1.json \
  --output benchmark-results/vega-kt-path-chain/report.json
```

native wheel、companion、NumPyと検証用SciPyが必要。実装本体にSciPy依存は追加していない。
CIの手動数値受入に追加し、生レポートをartifactに保存する。実行中のsource変更はhashで拒否する。

今回受け入れたのは、指定した較正延長規則と4・6区間のモデルである。
2..16区間という入力上限を、全区間数・全モデルの数値受入とは扱わない。
報告されない区間の分散変更、異なる較正延長、全方向共分散、carry・配当・barrier・早期行使、
通常providerへの接続は対象外。`full_market_iv_bucket_acceptance=not_established` を維持する。
