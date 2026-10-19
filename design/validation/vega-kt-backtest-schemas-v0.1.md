# VegaKT バックテストのJSON Schema・オフライン構造検証 v0.1

日付: 2026-10-03  
状態: 現行文書の構造検証・配布の機能検証は合格。VegaKTの数値受入は未合格。

この記録はSchema初回導入時のv5/v9/v7/v1の検証結果です。後続の
[資金・総額制約の実装](vega-kt-backtest-budget-v0.1.md)で新しい形式を追加しました。
ここで検証した六つのSchemaと五つのgoldenは変更せず、引き続き配布・検証します。

## 実装範囲

[要件定義](../vega-kt-hedging-backtest-requirements-v0.1.md)第9章・第10章に沿い、
現行の保存形式をDraft 2020-12 JSON Schemaで定義しました。

| 文書 | 対象version | セレクタ |
| --- | --- | --- |
| 実行入力 | 5 | `request-v5.schema.json` |
| 実行結果 | 9 | `result-v9.schema.json` |
| checkpoint | 7 | `checkpoint-v7.schema.json` |
| シナリオ集計入力 | 1 | `ensemble-request-v1.schema.json` |
| シナリオ集計結果 | 1 | `ensemble-v1.schema.json` |

共通定義 `definitions-v1.schema.json` と合わせて六ファイルをcompanion wheelへ同梱します。
型・必須項目・既知の構造を明示した仕様は
`scripts/generate_backtest_schemas.py` で管理し、実行例から型を推測して自動更新する方式にはしていません。
`--check` は仕様と配布ファイルの不一致を検出します。

既存の入力／結果／checkpointのversionは変更しません。
replay・集計の既存動作も変更せず、`validation` extraを追加した任意の構造検査として提供します。
Schemaを使わない通常利用の追加runtime依存はありません。

## 検査内容と境界

入力では契約、建玉、気配、感応度基底、fixing、支払割引、numeraire、外部資金移動を定義しました。
結果では11表とsummary、manifest、失敗情報、checkpoint内の同じ表・原価簿・未払状態を定義します。
シナリオ集計では母集団情報、各実行の欠損・除外理由、統計量、尾部指標、信頼区間の形を定義します。

core recordの未知項目を拒否し、型・数値範囲・nullを検査します。
providerのvalidation/policy metadata、外側生成器のmetadata、ネイティブpricing controlの内部metadataは拡張点として保持します。
現行readerで省略できる入力のデフォルト項目も受け付けます。

状態に関する主な検査:

- failed結果にはfailure詳細が必須。成功／部分終了ではfailureはnull。
- 部分終了・失敗・未精算では最終ヘッジ誤差をnullとする。
- explicit fixing規則には対応するfixing ID／観測情報が必須。
- 未計算Gammaにbump metadataだけを付ける、IVだけを供給してforwardを欠く、といった記録を拒否。
- 未対応IV nodeのshockをゼロで埋める、診断内積を `pnl_adopted=true` とする記録を拒否。
- 参考値／算出不能の尾部統計に信頼区間を付ける記録を拒否。
- 独立simulationを指定した集計入力には、外側乱数のalgorithm/domain/seedを要求。

`loads_document` は重複キー、NaN／Infinity、floatへ変換した際に無限大となる数を拒否します。
JSONから読んだ値を `validate_document` へ渡します。dataclassやNumPy scalarの暗黙変換はしません。
日付は実在する暦日、日時は `T` と `Z` または `±HH:MM` offsetを持つISO表現を検査します。
外部validatorを使う場合もformat検査を有効にする必要があります。

pricing requestは既存どおりJSON文字列です。CLIではその文字列のJSON構文も検査します。
`contentMediaType` はSchema上の注釈に過ぎず、外部validatorが内側JSONを自動検査するとは限りません。
内側のproduct/model/riskの内容はネイティブのSchema・adapter・pricing planの責任範囲です。

**構造検証は計算検証ではありません。**
損益照合、実際のhash一致、基底とbucketの長さ整合、行を跨ぐID一意性、時刻順序・availability、
戦略間の条件・比較母集団・独立性は、既存のreader／engine／集計処理が検査します。
例えば形式の正しい64桁hashを別の値に変えた文書は、構造検査だけでは拒否しません。
テストでもこの境界を明示しています。

検査出力は `validation_scope=structure_and_embedded_json_syntax`、`semantic_checks=not_run`、
`financial_acceptance=not_evaluated` を持ちます。
元の `risk_validation` をそのまま表示し、`unvalidated` を昇格させません。
schema IDと、実際に検証へ使用した六ファイルの内容hashを記録します。

## 配布・実行

```shell
python -m pip install "./backtesting[validation]"
python scripts/generate_backtest_schemas.py --check
python -m rust_pricing_backtest.validation fixtures/backtesting/wire/request-v5.golden.json fixtures/backtesting/wire/result-v9.golden.json
python -m rust_pricing_backtest.validation --export-schemas benchmark-results/backtest-schemas
```

検証用の任意依存は `jsonschema>=4.23,<5`。今回ローカルでは4.26.0を使用しました。
明示したDraft 2020-12 validatorとローカルのresource registryを使い、リモートSchemaを取得しません。
ensemble入力中の結果ファイルへもアクセスしません。ファイル参照の実在・内容は集計実行時の検査対象です。
exportだけなら任意依存は不要です。

CLIは文書ごとにJSON検査結果を出し、全件合格で終了コード0、一件でも検証できなければ1です。
ネイティブ拡張を読み込んで価格計算する処理はありません。

五種類のgoldenファイルを `fixtures/backtesting/wire/` へ保存しました。
手計算fixtureに基づく結果には実際の売買、外部資本、後日支払、未払claim、numeraire、11表を含みます。
ensembleのgoldenは一戦略だけが供給された不完全な母集団として保存し、欠損を隠していません。
保存したsource hashは取得時点の識別情報です。将来の作業ツリーのhashへ置換しません。

## 検証結果

- companionテスト **172件合格**。今回25件を追加。任意依存を有効にし、skipなし。
- ネイティブ連携 **21件合格**。今回1件を追加し、四戦略の入力・結果・checkpointを実際のネイティブ出力で検査。
- complete／partial／failedを検査。初回commit前の失敗、未払期間中の状態、numeraire有無、四戦略を含む。
- 五種類のgolden、未知項目、必須項目欠落、boolと数値の混同、不正暦日、timezone欠落、破損した内側JSONを検査。
- 旧version 8結果は既存readerで読み込める一方、現行Schemaはversion違いとして拒否することを確認。
- 従来保存した実物のversion 8結果も読み込み、新しいIV監査履歴を捏造せず空表とすることを確認。
- 配布用wheelを作成し、**17ソースモジュール・6 Schemaファイル**が作業ツリーと一致することを確認。
- リポジトリ外からwheelそのものをimportし、golden5件、IV監査例の四戦略×入力／結果／checkpointの12件、
  シナリオ例の集計入力／結果2件、計**19文書**を検証。
- ネットワーク取得を禁止したテストで構造検証が成功。
- Schema生成差分、既存ネイティブSchema checker、Markdownリンク、差分空白、Python構文を確認。

19文書の検査結果は `benchmark-results/vega-kt-schema-validation/validation.json` に保存しました。
backtesting CIの三OSジョブへvalidation extraとSchema生成差分の検査を追加しました。
リモートCIは今回実行していません。

## 残件

旧文書のformal Schemaやversion移行機能は追加していません。既存readerは維持します。
旧checkpointのhash制約を緩めたり、versionだけ付け替えて現行として扱ったりはしません。

ネイティブ計算コード、会計、最適化、統計の計算式は今回変更していません。
数値受入パネル・Rust全体のゲートは再実行していません。
[直近の数値検証](vega-kt-backtest-allocation-audit-v0.1.md)は154項目中54項目が不合格です。
paper gamma pipeline等の残件と、実市場データによる評価は引き続き未完了です。
