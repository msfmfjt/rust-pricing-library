# 市場条件の拡張・商品間共分散・VegaKT損益説明

日付: 2026-10-13。全6市場・10商品・2 seedの120パネル、4,038検査が合格し、失敗は0件。
最終ソース・同一nativeバイナリに結び付けた [受入JSON](vega-kt-portfolio-market-v0.1.json) に結果を保存した。

## 実行結果

| ケース | 商品×seed | 検査数 | 失敗 |
|---|---:|---:|---:|
| regression_4 | 20 | 643 | 0 |
| regression_6 | 20 | 943 | 0 |
| short_3_left_skew | 20 | 403 | 0 |
| uneven_3_right_skew | 20 | 643 | 0 |
| uneven_5_smile | 20 | 643 | 0 |
| long_8_smile | 20 | 763 | 0 |
| 合計 | 120 | 4,038 | 0 |

6市場それぞれで6種類、計36のスポット・IV損益シナリオを確認した。
従来4/6区間の40件は、保存済みのnative数値監査と全数値が完全一致した。
比較から除くのは入力・計画の識別hashと新しい標本列の説明だけである。
価格、Gamma、条件付きAD照合、方向感応度、共分散、精度検査の値は除外していない。

追加したポートフォリオ評価は商品間の推定誤差共分散を返し、同一商品の正負保有で
価格・VegaKT・標準誤差が完全相殺される。保存済みバックテストの損益説明は
APIとCLIが完全一致し、対象2区間とも説明レポートを生成した。
元の台帳への照合誤差は日次最大 `3.33e-15`、累積最大 `4.22e-15`。
元の結果ファイルは変えていない。この照合誤差は台帳の会計一致であり、感応度近似の残差とは区別する。

Rustのライブラリ試験460件、companion試験274件、新規native試験6件、
既存診断ツールの互換性試験9件が合格した。Rustの既存5件の除外はそのままである。
型・静的検査、生成schema整合も確認した。途中のseed 7までを保存した結果から
seed 41を再計算した再開結果は、連続実行と完全一致し、完了済み結果からの再開もバイト単位で一致した。
ソース・バイナリ・保存内容の不一致と入力への上書きは拒否する。

既存の行列診断CLIは本受入記録へ結び直し、4/6区間の実際の事前検査も通過した。
これは旧CLIの2,770検査を再実行したという意味ではない。今回の40件の数値完全一致、
変更していない独立価格計算・検査方針、CLI互換性試験と事前検査で接続を確認している。
再現用のソースアーカイブには実装・方針・試験・本受入記録を含め、内容検査を行った。

## 実装

`portfolio_market_risk.evaluate_portfolio` は、同一市場・時間格子・乱数設定の
European、算術平均Asian、固定行使価格Lookbackを共通の乱数経路で評価する。
数量は符号と契約倍率を反映済みの単位数で指定する。零数量のヘッジ候補も検査する。
全商品の較正・Gamma回復・精度検査が通ったときだけ、商品×市場IVの合同共分散を返す。

Rustの既存Gamma監査に `retain_samples=True` を追加した。
保存列は価格と回復した市場方向感応度、行は乱数カウンタ順の独立標本である。
通常MCのantitheticでは二経路の平均を一標本とし、共分散を標本数で割って平均の共分散にする。
RQMCではSobol点ではなく、独立scramble内の平均を一標本とする。
同じscramble seed・個数・antithetic設定を共有すれば、商品ごとの点数やBrownian bridge設定を変えられる。
各商品は同一scramble列の先頭から指定点数を使い、共分散はscramble番号で対応付ける。
異なるBrownian変換も同一scrambleの決定論的関数なので、商品間で組にした平均値は独立な反復になる。
点数と完全なengine設定は商品別に保存する。通常MCの標本数不一致は拒否する。
従来の集計順序・精度条件は保持する。標本保存にはメモリ予算を設けた。
計算量上限にはscramble数×点数を使う。raw ADとの検査に残る条件付き分散の下限は
物理的な経路対数によるMC参考値と明記し、RQMC推定量の標準誤差とは区別する。

RQMCの非終端行の条件付きAD照合には、同じscrambleのより長いSobol列を使う。
経路数はGamma側の最大64倍、全scramble合計262,144対までとし、別の計算量上限も検査する。
Gammaの価格・感応度・推定共分散を変えず、低確率領域を含む比較用ADの精度を上げる。
GammaとADの差の標準誤差は、両者をscrambleごとに対応付けて計算し直す。
AD側の点数は `conditional_ad_points_per_scramble` に保存する。
終端行ではGammaとADが同じ最後のショックを積分するため、同一経路上の恒等式として直接照合する。
ここだけはGamma側と同じ点数を使い、異なる標本平均の差による余計なばらつきを入れない。
行ごとの実際のAD点数も保存する。

合同共分散には異商品の非対角ブロックを含める。保有数量を掛け合わせた後の
標準誤差を出し、同一商品・同一計算設定の正負ポジションでは価格・VegaKT・標準誤差が相殺される。
市場リターンの共分散ではなく、Monte Carlo推定誤差の共分散である。
較正・離散化の偏りは含まない。

## 損益説明とDelta convention

`explain_pnl` は期首の数量と感応度を使い、絶対満期・固定strikeに市場IVを合わせる。
現在の格子からはlog-moneyness上で線形補間し、外挿や満期の代替はしない。

凍結LVでスポットを動かした際の固定strike IV応答を `c` とし、
`Delta_market = Delta_frozen − VegaKT · c` とする。
説明額は `Delta_market × ΔS + VegaKT · ΔIV_fixed_strike`。
`c` はスポット差分と半幅差分を照合して収束を検査する。
これにより、凍結LVのDeltaにすでに含まれるIVの座標移動を重複計上しない。

固定LVのGammaを市場Deltaと混在させず、このレポートのGamma項は未計上とする。
Theta、Gamma、曲線、配当、観測値の確定、交差・高次項、数値誤差は実測損益との差に残る。
VegaとDelta補正項の標準誤差には合同共分散を使うが、凍結Delta自身の推定誤差は含まない。

`explain_backtest` は既存バックテストの保存入力、期首ポジション、採用済みリスクを検査し、
対象・オプションヘッジ・株式と現金成分を実測台帳へ照合する追加レポートを作る。
選択した較正拡張と期首の採用済みVegaKTが一致しない場合は拒否する。
満期済み・欠落・strike域外の区間は理由付きで未説明に残し、完全な説明と表示しない。
既存の台帳・result schemaは変更しない。

```sh
python scripts/explain_vega_kt_pnl.py \
  --result benchmark-results/vega-kt-portfolio-market/pnl-replay-result.json \
  --extension all_rows --workers 4 \
  --output benchmark-results/vega-kt-portfolio-market/pnl-replay-cli.json
```

## 数値受入の構成

従来の4/6区間・結合基底を回帰ケースとして再計算する。
拡張ケースは短期3区間、不均等3区間、不均等5区間、長期8区間。
17/25/33分散節点、3/5市場strike、異なる満期構成・歪み・曲率・行使価格・
非均等Asian観測重みを含める。合格は宣言したケースと精度方針に限る。

| ケース | 区間数 | 分散節点数 | 市場IV数 | 較正拡張 |
|---|---:|---:|---:|---|
| regression_4 | 4 | 33 | 10 | coupled_row_hats |
| regression_6 | 6 | 33 | 15 | coupled_row_hats |
| short_3_left_skew | 3 | 17 | 6 | all_rows |
| uneven_3_right_skew | 3 | 25 | 10 | all_rows |
| uneven_5_smile | 5 | 33 | 10 | all_rows |
| long_8_smile | 8 | 33 | 12 | all_rows |

精度・商品別RQMC設定は [policy JSON](../../fixtures/backtesting/portfolio-market-policy-v0.1.json) に固定する。
European、Asian、既知fixing付きAsian、LookbackのCall/Putと、過去・当日fixing付きLookback Putの
計10商品を、seed 7と41で検査する。

独立した密度積分で市場価格を確認し、再較正したIVの正負差分を
独立したSobol価格積分に入力して各商品のVegaKTと照合する。
商品間共分散は直接加重した経路標本の分散と照合する。
スポットのみ・IVのみ・同時変化をそれぞれ全幅と半幅で再評価する。
単なる損益の算術一致を感応度の正しさの根拠にはしない。

短期Lookbackでは積分分割を増やし、不均等3/5区間ではIV差分を細かくして
較正精度を上げる。拡張市場にはRQMCを使い、受入許容差は緩めない。
積分分割数も商品別に指定できる。各商品の分割・倍分割・四倍分割の照合は維持し、
滑らかなAsianとLookbackの精度要求に合わせて、積分と経路数へ計算を配分する。

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_portfolio_market.py \
  --case uneven_3_right_skew \
  --output benchmark-results/vega-kt-portfolio-market/uneven_3_right_skew.json
```

ケース別の部分結果からは `--resume` で再開できる。
入力・方針・ソース・nativeバイナリと保存内容のhashが一致し、独立価格計算が完了していることを要求する。
hashは取り違え・破損検出用で、署名ではない。

```sh
python scripts/summarize_portfolio_market.py \
  --directory benchmark-results/vega-kt-portfolio-market \
  --output benchmark-results/vega-kt-portfolio-market/matrix.json
```

集約は全ケースの同一ソース・nativeバイナリ・実効方針・検査網羅を要求する。
旧数値との完全一致比較には、保存済みの
`benchmark-results/vega-kt-current-raw-basis/{4,6}.json` も必要である。
手動CIは各市場ケースを独立ジョブで再検査し、部分結果も保存する。

## 範囲

配当・carryなし、同一原資産・通貨・時間格子・乱数設定を対象とする。
実市場でのヘッジ効果、任意市場・行列の全面受入、非対応商品への自動採用は主張しない。
既存の通常providerの受入条件は継続し、拡張市場は明示的なポートフォリオ評価APIで利用する。
本更新はnativeバイナリを変更するため、旧バイナリに結び付いたcheckpointは
保存入力から再実行する。過去の数値受入記録は変更しない。
