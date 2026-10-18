# VegaKT バックテストの共通シナリオ比較・尾部統計 v0.1

日付: 2026-10-03  
状態: 集計・比較の機能検証は合格。VegaKT本体の数値受入は未合格。

## 対象と追加機能

[要件定義](../vega-kt-hedging-backtest-requirements-v0.1.md)の第9章・第10章に沿い、
保存済みの四戦略の実行結果から、最終ヘッジ誤差を共通シナリオ上で集計します。
対象は同じ契約・建玉・当初取得価格・観測日程の市場シナリオ群です。
異なる満期の取引群、重複する実履歴のblock bootstrap、異種ポートフォリオの正規化は今回の対象外です。

`rust_pricing_backtest.ensemble` に以下を追加しました。

- `EnsembleConfig`: 通常／割引後の誤差、標本設計、分位水準、信頼水準、bootstrap設定、最小標本数、分母閾値。
- `ScenarioResults`: シナリオID、独立反復ID、生成元、外側乱数情報、四戦略の結果。
- `aggregate_scenarios`: 同条件検査、共通標本抽出、統計集計、診断状態を引き継ぐレポート。
- CLI、version 1の集計要求／結果JSON、全シナリオ・戦略を含むCSV。
- 公開ネイティブAPIを使う24シナリオの合成実行例と、その保存結果からの再集計。

個別実行の入力version 5、結果version 9、checkpoint version 7は変更しません。
既存の会計・最適化ロジックも変更していません。ソースhashによる再開時の同一性検査は従来どおりです。

## 共通標本と比較条件

全四戦略が `completed`、全契約・支払・ヘッジが精算済みで、選択した最終誤差が取得できるシナリオだけを採用します。
一戦略でも `partial` / `failed` / missing / 未精算 / 指標欠損なら、そのシナリオを全戦略の集計から除外します。
停止後の損益や欠けた誤差をゼロで埋めません。四戦略すべてが欠けた予定シナリオも件数に残せます。
ただし、コホートを識別するため、要求全体には少なくとも一つの実行結果が必要です。

各行には状態、除外理由、完了snapshot数、全snapshot数、coverage、最終処理時刻、failure code、
実行コスト、売買金額、結果hashを保存します。戦略別の完走・部分終了・失敗・欠損件数と、
その戦略単独で最終誤差が得られる件数も残します。比較の分母は常に共通標本数です。
除外が市場状態に依存すればcomplete-case集計に選択バイアスが生じ得ることをmanifestへ明記します。

戦略名以外の設定、契約・対象建玉・entry費用、観測日程、価格評価・約定・資金・精算規則、
providerと実装hash、記録された実行環境を揃えます。
ネイティブpricing requestのengine設定・内側seed・path数・risk要求・grid軸もシナリオ間で固定します。
custom providerの設定は、そのprovider自身のidentity規約で識別する必要があります。

同一シナリオ内の四戦略には、全入力履歴hashの一致を要求します。
異なるシナリオでは市場値と後続numeraireは変わりますが、資本移動と初期numeraireの正規化は共通にします。
期間や費用条件が違う結果は、除外して黙って続行するのでなく入力不一致として拒否します。
異なるIDで同一履歴を二重投入することも拒否します。

## 統計と推定方式

誤差 `e` に対し、平均、標本標準偏差（ddof=1）、`sqrt(mean(e²))` を出します。
ゼロ件なら全指標null、1件の標準偏差はnullです。通常と割引後の誤差を混ぜません。
実行コストと売買金額の平均は通常通貨額として別名で出します。

損失 `L=-e` を昇順に並べた経験分布について、分位点はinverse CDFとし、
ESは分位関数の上側区間を積分して `1-alpha` で割ります。
境界にある観測は必要な質量だけ加え、同値の損失をすべて尾部へ入れたり、件数を切り上げたりしません。
利益しかない標本では負の損失分位点・ESをそのまま保存します。

例: 誤差 `[-10,-4,2,8]` は平均−1、標本分散60、RMSE `sqrt(46)`。
損失は `[-8,-2,4,10]`。62.5%分位点は4、上側質量1.5件なのでESは `(10+0.5×4)/1.5=8`。

Deltaを基準に `1−Var(e_strategy)/Var(e_delta)` と同一シナリオの平均誤差差を出します。
基準分散が1e-12以下、または標本不足なら改善率はnullです。
この閾値は誤差単位の二乗であり、通常／割引後のどちらを選んだかで単位が変わります。
負の改善率も保持し、改善を機能テストの合格条件にしません。

デフォルトの `dependent_or_unknown` は信頼区間を出しません。
`independent_simulation` では生成元・乱数algorithm/domainを共通にし、反復IDと外側seedの重複を拒否します。
これは利用側の独立性宣言であって、独立性の統計検定や証明ではありません。
同一日程の独立simulationは許容し、重複暦期間の組数も表示します。
従属／不明モードでは反復ID重複を件数として表示し、IIDの区間計算を有効にしません。

独立simulationで20件以上の共通標本があるとき、2,000回のpaired bootstrapを行います。
全戦略に同じscenario index列を使用し、PCG64 seed 20261003、scenario ID昇順、95% percentile区間を固定します。
区間端点の分位数は線形補間です。外側市場変動に関する区間であり、固定した内側pricing乱数に条件付きです。
内側Monte Carlo誤差の評価や、従属系列のblock bootstrapを実施したという意味にはなりません。
bootstrapの一部で基準分散が定義できなければ、改善率の区間全体をnullとし、該当回数を残します。
定義できた反復だけを選んで区間を狭めません。

尾部の有効件数 `n×(1-alpha)` が20未満なら、分位点・ESを `reference_only` とし、尾部区間を出しません。
デフォルト95%/99%水準では、少なくとも400件/2,000件が必要です。
これらは明示した研究用デフォルトであり、十分な推定精度を保証する閾値ではありません。

## 合成実行と再現性

追加例は同じ初期市場から、暦日ACT/365Fの独立な正規乱数増分でspotとsurfaceのvol水準を変えます。
spotの年率変動係数0.22、vol水準の対数変動係数0.5を使用し、歴史データからの推定は行っていません。
外側乱数は `SeedSequence([0x56454741, market_seed])` / PCG64、内側pricing seedは全戦略・全シナリオで7です。
bootstrap乱数とも分離します。生成前に全予定シナリオを集計要求JSONへ記録します。

```shell
python -m unittest discover -s backtesting/tests -v
python -m unittest discover -s tests/python -p test_vega_kt_backtest.py -v
python examples/python/vega_kt_scenarios.py --scenarios 24 --paths 256 --diagnostic-only --numeraire-rate 0.01 --output benchmark-results/vega-kt-scenario-example
python -m rust_pricing_backtest.ensemble benchmark-results/vega-kt-scenario-example/ensemble-input.json --output benchmark-results/vega-kt-scenario-example/reaggregated
```

CLIは共通標本が全件なら0、除外のある結果を保存した場合1、入力拒否は2です。
0は集計完了を表し、VegaKTの数値受入を表しません。

## 検証結果

- companionテスト **147件合格**。今回、手計算・部分／失敗／欠損・異条件・乱数・CLI等の20件を追加。
- ネイティブ連携 **20件合格**。内側seed・path数・risk要求の一致検査と、外側乱数の再現性／分離の2件を追加。
- 24シナリオ × 四戦略、各11セッションを完走し、全建玉・支払を精算。共通標本24件、除外0件。
- seed 1000–1023、内側256 sampling units、共通numeraire rate 0.01で実行。
- 保存した割引後最終誤差から、NumPyによる平均・ddof=1標準偏差・RMSEと、別の尾部計算で全戦略の集計値を照合。
- Delta基準の三つの分散比を別計算と照合。96件のscenario/strategy CSVを再読込み。
- 保存済み結果をCLIで再集計し、JSON・CSVが初回の集計出力と一致。
- 95%/99%の尾部有効件数は1.2件/0.24件。すべて参考値で、尾部信頼区間はnull。
- 通常統計・比較のbootstrap区間は出力。今回の24件の例では、分母未定義のbootstrap反復は0件。
- 少数・ゼロ分散・near-zero分散のfixtureでは、nullと理由を保持。重複履歴、戦略間の市場差、期間・費用・実装差は拒否。
- companion wheelの構築と全16ソースモジュールの作業ツリーとの一致を確認。
- Markdownリンク、差分空白、Python構文を確認。

ネイティブ計算コードは今回変更していません。バイナリSHA-256:
`93a5c068e81bc4d056634f403cb3616b4865962203d4073f0ebf818732c6741c`

## 限界と残件

集計結果にも `risk_validation.status=unvalidated`、`diagnostic_only=true` と既存blockerを保持します。
合成市場とmodel由来の気配に基づく比較で、VegaKTが有効であることや実市場の成績を示すものではありません。
実市場データ、異期間・異種取引コホート、従属系列の区間推定、外側／内側の数値収束検証は残件です。

今回、数値受入パネル・Rust全体のゲート・リモートCIは再実行していません。
[直近の数値検証](vega-kt-backtest-allocation-audit-v0.1.md)は154項目中54項目が不合格です。
paper gamma recovery/propagation、初期時点、IV-shock解釈とDelta conventionの整合は未解決です。
