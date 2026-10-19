# 固定市場ショックを日次バックテストへ接続 v0.1

日付: 2026-10-04  
状態: 設定、日次最適化、残余リスク監査、保存・再開・集計へ統合。
companion 212件、native VegaKT関連35件が通過。
制限付き数値受入v0.2は307項目通過、旧v0.1は6件失敗を保持する。

## 実装した動作

[前回の市場ショック変換](vega-kt-market-shocks-v0.2.md)を、通常の日次バックテストの
VegaKT最適化で明示的に選択できるようにした。

```python
config = BacktestConfig(
    strategy="vega_kt",
    market_shock_log_moneyness=(-0.4, 0.0, 0.4),
)
```

市場のlog-forward-moneyness節点を実行全体で固定する。各日の報告格子には、この全節点と
同じ端点が必要。期日列はその日の報告期日と厳密に一致させる。絶対Strikeの固定や
期日間の補間はしない。forwardが動けば節点に対応するStrikeも動く。
この設定は同一日の報告Strike格子から独立したショック定義であり、期日方向の格子変更や
異なる日のリスクノルムを同一条件で比較する仕様ではない。

設定が `None` の場合は従来の最適化と出力形式を使う。有効な場合、VegaKT戦略の
候補選定の最小Vega判定と最適化の両方に、変換済み市場リスクを用いる。
保有中の候補外契約は従来どおり最適化に含め、取引可能でなければ数量を固定する。
unhedged・Delta・scalar Vega戦略は従来の注文を維持し、同じ市場ショックでの監査を追加する。

箱型制約のソルバーと、現金・グロス建玉・Delta中立株数を同時に扱うソルバーの両方に
対応した。リスクの目的関数だけを変換し、手数料、資金、数量上限、ロット丸め、
翌日のbid/ask約定、実際の約定時の制約判定は維持する。

## 日々保存する監査

`risks[*].market_shocks` には、変換ID、元の報告basis ID、市場basis本体とID、
targetおよび約定後の実保有ポジションの市場バケットと補空間リスクを保存する。

`decisions[*].market_shocks` には、同じ定義に加え、実際にソルバーへ渡した
target/matrix、連続解のリスク、ロット丸め後の発注予定ポジションのリスクを保存する。
元の細格子の残差も保持する。Delta/unhedged戦略にはVega最適化の連続解を作らない。
翌日の市場や約定価格を使って前日の監査を上書きしない。

補空間のノルムと最悪方向のショックは従来の独立監査を使う。
これは報告quote座標のユークリッド指標であり、実測市場の共分散や戦略の有効性ではない。
日次RiskProviderは全バケット共分散を保持していないので、市場バケットの標準誤差を
限界標準誤差から合成することはしない。元のnative品質検査はそのまま実行する。

## 保存・再開と互換性

| 設定 | request | result | checkpoint | ensemble |
|---|---:|---:|---:|---:|
| 未指定・None | 6 | 10 | 8 | 2 |
| 市場ショック節点を明示 | 7 | 11 | 9 | 3 |

有効時だけ新形式を使う。既存11個のJSON Schemaはbyte単位で不変で、新たに5個を追加した。
新schemaは市場ショックの設定、ポリシー、各日の監査を必須とする。
旧版のversion番号のまま新設定を渡す入力、新版なのに設定がない入力は拒否する。
オフライン検証と4戦略のensembleにも新形式を接続し、異なる市場ショック定義の結果を
同一条件の戦略比較へ混ぜることを拒否する。

設定とポリシーは実行指紋に入り、日々のbasisと監査はcheckpointの状態指紋に入る。
節点の変更、設定の無効化、保存状態の改変は再開時に拒否する。
コード指紋も従来どおり再開条件に含むため、更新前の実行のcheckpointをそのまま
新しい実装へ流用できるという意味ではない。

格子の適合性は、その日の処理を始める時点で検証する。未来の日に不適合な格子があっても
それ以前の計算には使わない。失敗日は資金移動・決済・約定・監査をまとめて取り消し、
完了済みの履歴と未約定注文から再開する。失敗日以降の入力を修復すると、最初から
正しい入力で実行した結果へ完全一致した。

## 検証結果

手計算のケースでは、細格子上で分離したtarget/hedgeを粗い市場ショックへ変換し、
ヘッジ連続解2.0、lot=0.3で発注1.8となることを検証した。元の細格子での解は0であり、
設定を有効にした時だけ意図した目的関数へ変わる。5節点と9節点の報告格子で注文は一致。
連続解と丸め後のリスクが別に保存され、翌日の実保有監査と丸め後監査も一致する。

現金・グロス制約ケースでは `gross=150+80*h<=250` から連続解1.25、lot=0.1で
発注1.2、丸め後gross=246を確認した。保有中だが候補外になった契約、選択した市場
ショックへの感応度が0の候補、将来の格子不正、JSON往復、再開、形式違反も検証した。

nativeライブラリを使った5日・4戦略の合成データでは、全戦略が完了し、2日後の
checkpointからの再開がJSON文字列で完全一致した。request・result・checkpoint・
ensembleはいずれも新schemaを通過した。全出力は `unvalidated` を維持している。
この合成データを実市場のヘッジ成績の根拠にはしない。

| 検証 | 結果 |
|---|---|
| companion全テスト | 212件通過 |
| native VegaKT関連 | 35件通過 |
| 固定市場ショック数値受入v0.2 | 307項目通過 |
| 元の数値受入v0.1 | 154項目中6件失敗、従来の観測・判定を保持 |
| native再開 | 4戦略とも一括実行と完全一致 |
| 旧JSON Schema | 11個すべて不変 |

新規テストは[日次統合9件](../../backtesting/tests/test_market_shock_replay.py)と
[native統合1件](../../tests/python/test_vega_kt_backtest.py)。本体Rustの計算は変更していない。
今後の課題は一般smileの数値受入と市場ショックの実証的な選択であり、nativeの
`diagnostic_only` ガードは維持する。

## 再実行と記録

```shell
python -m unittest discover -s backtesting/tests -p 'test_market_shock_replay.py'
python -m unittest discover -s tests/python -p 'test_vega_kt_*.py'
python -m rust_pricing_backtest.market_shock_numerics \
  --policy fixtures/backtesting/market-shock-policy-v0.2.json \
  --output benchmark-results/vega-kt-market-shock-replay/numerics
```

検証したnativeの入力・結果・CSV・途中checkpointは
`benchmark-results/vega-kt-market-shock-replay/native/` の各戦略ディレクトリへ保存した。
書き出された入力は従来のCLIで再実行できる。

```shell
python -m rust_pricing_backtest \
  benchmark-results/vega-kt-market-shock-replay/native/vega_kt/input.json \
  --checkpoint benchmark-results/vega-kt-market-shock-replay/native/vega_kt/partial-checkpoint.json \
  --output benchmark-results/vega-kt-market-shock-replay/replayed --diagnostic-only
```

ソース指紋、旧schema指紋、native実行の要約は[検証記録](vega-kt-market-shock-replay-v0.1.json)を参照。
