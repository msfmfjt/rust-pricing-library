# 可変区間数の終端Gammaと非平坦な第1市場満期

日付: 2026-10-06。集約結果は[JSON](local-vol-multi-step-gamma-v0.1.json)。

## 対応範囲

2..16区間のlog-Eulerモデルに、実際の先行遷移分布を使う終端Gamma診断を追加した。
数値受入の対象は4区間・正skew、6区間・負skewの2モデル、各4商品、2 seedsの16条件。
局所分散は33節点、終端の市場IVは5点。第1市場満期は2回の遷移後に置き、
約0.348 vol percentage pointsの非平坦スマイルを持つ。
call、全観測を使うAsian、既知fixing付きAsian、lookbackを調べる。

回復するのは最後の年率分散rowだけで、先行rowと先行市場quotesは固定する。
最初の分散row自体も非平坦でよいが、第1遷移が参照する値はlog spot=0の補間値だけである。
第1市場満期の非平坦スマイルは、2回目以降の状態依存遷移から生じる。
第1満期の各quoteを動かすGamma、全区間のGamma連鎖、通常providerでの全面受入は未実装。

## 再帰的なsource分布

状態を `X_i=log(S_i/S_0)`、年率局所分散を `v_i(x)` とすると、各遷移は
`X_(i+1)=X_i-v_i(X_i) dt_i/2+sqrt(v_i(X_i) dt_i) Z_i`。
Gaussian遷移密度を順に積分し、最後の区間のsource密度を構成する。
モーメントによるGaussian近似や密度の再正規化は行わない。

第1sourceは平均 `-A/2` の±12標準偏差。後続sourceは、その時点までの積分分散上界Aにより
`[-A/2-12 sqrt(A), A/2+12 sqrt(A)]` を使う。
最初のGaussian tailに加え、以降は確率tailの上界 `2 exp(-72)` と、
stock加重tailの上界 `S0 (1+exp(lower)) exp(-72)` を各段階で累積する。
後続のmartingale遷移は失われたstock第1モーメントを増やさない。

局所分散の補間節点で積分区間を分け、各cell幅をその遷移の最小標準偏差の4倍以下にする。
32/64点のGauss-Legendre積分を比較する。各段階・両水準で確率質量1と平均spot=S0を検査し、
絶対確率誤差1e-10、spot相対誤差1e-10にtail上界を加えた範囲外なら拒否する。
2..33局所節点、2..65 Gamma節点、各段階8192積分点、各水準4億kernel評価までとする。
潰れたwindow、極端に狭い遷移、数値領域や作業量の超過は明示的なエラーになる。

終端作用素はsource密度で `hat_j(x) S phi(d1) dt/(2 sqrt(v(x)dt))` を積分する。
作用素の32/64点差とtail上界を、それぞれ最大要素の1e-10以内とする。
細分化差は収束診断であり、積分誤差全体の厳密上界とは解釈しない。
2・4・6・16区間の平坦分散極限は解析Gaussian作用素とも照合する。

## 終端payoffの条件付き積分

通常の経路ADをそのまま使った探索では、4区間Asianはheld-outのGamma相対SEが約26.0%、
6区間Asianは全pilot候補が不合格だった。失敗したraw診断も保存している。

新しい診断は、それまでの全観測履歴を固定し、compiled payoffを終端spotの符号付きPWLにする。
Gamma原子 `(K_a,m_a)` に対し、最後の年率分散感応度は
`sum_a m_a S_source phi(d1_a) dt/(2 sqrt(v dt))`。
これをsource位置の補間hatに配分する。cash・線形部分の条件付きVegaは厳密に0となり、
最終乱数の積分によって相殺ノイズを減らす。負のGamma原子は保持する。
履歴をsource spotだけに集約せず、Asianの全fixingやlookbackの過去最大値を保持する。

通常のnative価格は従来のfull-path payoffのまま計算する。
新APIの分散感応度はこの条件付き推定量であり、旧APIのraw pathwise AD統計とは区別する。
同一パスで回復Gamma・残差・各保存方向の統計を作り、antithetic平均後にSEを計算する。
方向間の全共分散や、pilot選択を含む無条件の信頼区間は返さない。

## 較正と独立受入

`MultiStepTerminalSmileCalibrator` は、報告満期がsimulation knotsの部分集合であり、
最後の満期を含むrequestを受け付ける。全報告満期の基準価格一致を検査する。
最後のrowだけを動かして終端IVを較正し、先行rowと先行quotesを保持する。
Newton/backtrackingとVega正規化逆行列を使い、分散制約違反をclippingで隠さない。
接線の精度は独立した細分化水準のJacobianで検査する。

pilot 262,144 pairs、held-out 524,288 pairs、正則化 `[.01,.003,.001,.0003,.0001]`、
空間誤差3%、Gamma相対SE25%、MC margin 2は前回と同じ。
全pilot不合格・held-out不合格のいずれも `estimate=None` を維持する。

独立した96点の密度積分と64点の細分化比較で作用素を検査する。
商品側は8種類のscrambled Sobol系列、各65,536経路と独立した条件付きBlack積分で参照値を作る。
この参照はnative AD、native source builder、較正Jacobianから生成しない。
SEは8回のscramble平均から推定する。6 SEの判定は同時信頼保証ではない。

終端5 quoteすべてについて、IVの±1e-5 / ±5e-6を再較正し、通常native価格を新たに計算する。
8 seeds、各16,384 pairs。較正価格誤差は1e-10、Gammaとの比較は6 combined SE＋0.002。
同一パス方向保存は `abs(mean)+6 SE <= 1e-8`、独立primal KKTとの比較は2e-6。
±0.003の平行終端IV stressでも再較正と先行quotesの保持を検査する。
今回の価格差分は全方向を新規計算し、以前の価格差分証跡は流用しない。

**1,402 / 1,402検査合格、失敗0。** 全16条件で終端Gammaのheld-out受入に合格。

| 検査 | 最大差 |
|---|---:|
| 作用素と独立96点積分 | 2.84217e-14 |
| 較正quoteの価格誤差 | 1.1724e-13 |
| ±0.003 stressの価格誤差 | 2.0723e-11 |
| 独立primal KKTとのGamma差 | 5.15225e-08 |
| 5方向の同一パス保存誤差（6 SE込み） | 1.44924e-12 |

held-outの空間誤差margin比の最大値は2.726%、Gamma相対SEの最大値は3.103%。
native再較正価格差分との比較は最大2.236、許容幅に対する比率の最大値は0.634未満だった。
方向別の標本誤差を含む受入であり、価格差分との絶対誤差が一律0.002以内という意味ではない。

Rust 43件、Python VegaKT 131件、companion較正25件、wheel契約5件、合計204テスト合格。
旧40ケース＋旧3区間終端8ケースの出力はbyte単位で同一。
worker数変更・notional倍率・通常価格の不変性・antitheticなし＋Brownian bridgeを検査した。
厳しすぎるpilot基準と2標本のheld-out不足は不合格を維持し、推定値を返さない。
潰れた第1遷移と極端に狭い第2遷移は数値領域エラーで拒否する。

clippy、format、依存方向、差分、実行時wheel契約も合格。
検証済みnative/companion wheelを作業用 `.venv` へ反映し、native binaryのhash、
companionソース、新しい33×5接線、精度領域の拒否を確認した。

ソース配布物は必須506件・計885ファイルの検査に合格。文書の内部リンクも検査済み。


## 再実行と残る範囲

native wheel、NumPy、検証用SciPy、companion packageを利用できる環境で実行する。
native/companion本体にはSciPy依存を追加していない。

```sh
python scripts/check_multi_step_gamma.py \
  --policy fixtures/backtesting/multi-step-gamma-policy-v0.1.json \
  --output benchmark-results/vega-kt-multi-step/report.json
```

CIの手動数値受入に追加し、生レポートをartifactへ保存する。
この検査は実行開始時のsource SHA-256を固定し、実行中の変更を拒否する。
生証跡は `benchmark-results/vega-kt-multi-step/`、集約値は上記JSONに置く。

2..16区間は入力上限を表す。全モデル・全区間数の数値受入を意味しない。
先行Gammaと全満期再較正、非ゼロcarry・配当・barrier・早期行使、全方向共分散、
通常providerでの一般スマイル全面受入は引き続き対象外。
`full_market_iv_bucket_acceptance=not_established` を維持する。
