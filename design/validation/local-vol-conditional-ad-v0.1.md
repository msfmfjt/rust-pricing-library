# 経路依存Gammaの中間区間AD照合

日付: 2026-10-06。集約結果は[JSON](local-vol-conditional-ad-v0.1.json)。
対象は[全方向共分散の受入](local-vol-path-covariance-v0.1.md)に残った2条件。
以前の失敗記録は変更せず、新しい結果を別に保存する。

## 問題と変更

6区間Asianと既知平均付きAsianのseed 7では、row 3・node 11のraw native ADが
全標本で0だった。Gammaは非ゼロであり、実測ADだけから標本誤差を評価すると
まれな支払いの不確実性を捉えられず、増標本でも照合が失敗した。

最終遷移の正規乱数を解析積分したnative ADを照合参照に追加する。
Gammaの回復、raw native AD、同一経路の方向統計と共分散は変更しない。
新しいAD参照はnativeの観測値勾配と逆伝播から作り、回復Gammaや較正接線から生成しない。
相対SE上限25%、汎関数許容差1e-6、6 SEの照合基準は維持する。

## 解析積分

最終遷移より前の状態と乱数を固定する。終端状態をsとしてcompiled payoffを分岐に分ける。
最終出力のGamma原子だけでなく、全中間演算の分岐境界を保持する。
例えば同じ終端形状のcallを差し引いた場合、出力の傾き差は消えても、異なる過去観測値への
ADは残りうる。最大256境界を許容し、処理範囲を超える場合は明示的に拒否する。

連続な区分線形payoffの各分岐では、観測値に対する勾配が一定となる。
最終遷移を除いたnative逆伝播を使い、各分散節点のraw ADを

`X(Z) = A + exp(-V/2 + sqrt(V) Z) × (B + C Z)`

と表す。Vは最終区間の積分分散。係数には過去観測の直接感応度、
後続状態の連鎖、および状態依存の局所分散の微分を含める。
分岐に対応する切断正規区間ごとに、平均と第2モーメントを解析積分する。

`E[e^(tZ) f(Z)] = e^(t²/2) E[f(Z+t)]`

により、必要なのは切断正規の0・1・2次モーメントである。
全分岐を合算し、条件付き分散を `E[X²] - E[X]²` から求める。
大きな負の分散や非有限値は拒否し、丸め誤差の範囲の負値だけを0にする。

指数多項式の積分を独立Gaussian積分と照合する。
さらに、非平坦な4遷移の全分散節点について、終端乱数を数値積分しながら
native逆伝播を直接実行した平均・分散と照合する。符号付きpayoffと過去観測も含む。

## 二段階の照合

1. 回復Gammaと、最終乱数を積分したnative ADの同一標本差を、実測の6 SE＋汎関数許容差で比較する。
2. raw native ADと条件付きnative ADを比較する。この差のSEは、実測SEと
   `sqrt((平均条件付き分散 + 6 SE) / N)` の大きい方を使う。

antitheticの場合はペア平均を1標本とする。条件付き分散のペア平均はCauchy–Schwarzによる
ペア残差分散の上界として使い、2経路を独立とは扱わない。
終端rowでは従来のGamma/raw AD照合も引き続き必須とする。
raw統計を解析値で上書きしたり、失敗節点を除外したりしない。

条件付き分散を外側のMCで平均するため、平均＋6 SEは厳密な総誤差上界ではない。
遠い節点のAD参照はなお標本誤差が大きい場合がある。個別節点の高精度を保証するものではない。
Gammaの回復精度・方向の標本精度・価格差分との照合を別途要求する。

## 追加した診断

各rowに以下を追加する。従来の統計フィールドは保持する。

| フィールド | 内容 |
|---|---|
| `conditional_native_variance_nodes` | 最終乱数を積分したnative ADの外側MC統計 |
| `conditional_node_differences` | Gamma−条件付きnative ADの同一標本差 |
| `raw_ad_conditional_variances` | raw ADの条件付き分散の外側MC統計 |
| `raw_conditional_node_differences` | raw AD−条件付きnative ADの同一標本差 |
| `raw_conditional_standard_errors` | 第2段階の照合に使うSE |
| `raw_node_validation_passed` | 従来の直接照合結果を診断として保持 |
| `conditional_gamma_validation_passed` | 第1段階の合否 |
| `raw_conditional_validation_passed` | 第2段階の合否 |

rowの `validation_passed` と全体の受入には、新しい二段階の照合を用いる。
精度不足時の `estimate=None`、市場向け共分散の抑止、raw統計の保持は継続する。

## 数値受入

全市場10・15方向の16条件と、合成4方向の16条件を検査する。
全市場方向は4,096 pairs・N=512。合成方向は既存の10条件を同じ設定で、
追加6条件を8,192 pairs・N=256で評価する。
以前失敗した16,384 pairs・N=128の既知平均付きAsianも別途再実行する。

過去のGamma・raw AD・共分散の完全一致と、価格/較正コードのhash一致を条件に、
前回のnative価格差分・独立Sobol価格差分の標本を引き継いで比較する。
価格差分自体を今回新規計算したとは扱わない。

結果: **32パネル・5,904検査合格、失敗0**。全市場方向16/16、合成方向16/16。
関連テスト206件（Rust 33、Python 140、companion 28、wheel 5）も合格。
最終ソースを固定した再実行の数値・検査結果は初回と完全一致した。

| 以前失敗した条件（6区間・seed 7、row 3・node 11） | Gamma | 条件付きAD | 差のSE | 新しい二段階照合 |
|---|---:|---:|---:|---|
| asian | 0.0026675463 | 0.0018619046 | 0.0014151947 | 合格 |
| fixed_asian | 0.0019428062 | 0.0012997593 | 0.0010347890 | 合格 |

両条件のraw AD平均・標本分散は引き続き0。従来の直接照合はfalseとして残す。
新しい照合はGamma対条件付きAD、およびraw AD対条件付きADの両方で合格した。
16,384 pairs・N=128の既知平均付きAsianも合格し、従来のGamma・raw AD・共分散は完全一致。

静的検査・整形・依存方向・wheel契約を確認し、更新wheelを作業環境に導入した。
導入後も以前失敗した条件を再実行し、受入時の出力と完全一致した。
ソース配布物908ファイル（必須529ファイル）の構成検査と文書リンク検査も合格した。

## 再実行

```sh
python scripts/check_path_gamma_conditional_ad.py \
  --covariance-report benchmark-results/vega-kt-path-covariance/report.json \
  --precision-report benchmark-results/vega-kt-path-covariance/precision-report.json \
  --price-report benchmark-results/vega-kt-path-chain/report.json \
  --output benchmark-results/vega-kt-conditional-ad/report.json
```

CIでは価格差分を含むGamma連鎖、共分散、合成方向の増標本試験、この照合の順に実行する。
失敗した判定は非ゼロ終了コードと生レポートに残す。

## 残る範囲

今回の受入は4・6区間、所定の較正延長と報告されない分散rowの固定を前提とする。
一般の較正延長、異なる商品の合同共分散、carry・配当・barrier・早期行使、
通常バックテストproviderへの接続は対象外。
`full_market_iv_bucket_acceptance=not_established` を維持する。
