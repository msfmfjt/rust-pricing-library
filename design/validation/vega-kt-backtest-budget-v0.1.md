# VegaKT バックテストの資金・総額制約 v0.1

日付: 2026-10-03  
状態: 任意の制約付き意思決定・保存・再実行を機能検証済み。VegaKTの数値受入は未合格。

## 実装範囲

[要件定義](../vega-kt-hedging-backtest-requirements-v0.1.md)のSTRAT-02/03に沿い、
`constraint_mode="decision_cash_gross"` を追加しました。
既定の `execution_cash` は従来のNumPy solverと約定時の資金検査を維持します。
新モードだけ `rust-pricing-backtest[constraints]` のSciPy依存を使います。

総額は時点の原資産midに、対象・ヘッジの生存オプションの絶対数量×multiplierと
株式の絶対数量を掛けた合計です。同一契約・同一book内では数量をnetしますが、
対象とヘッジのbook間では相殺しません。満期で固定した未払金は除外します。
オプション時価の合計、証拠金、リスク加重額とは異なります。

連続解では `stock = -target_delta - option_deltas @ h` とし、株式の資金使用・費用・数量上限を
オプションの選択と同時に制約へ入れます。現金支出は意思決定時のbid/ask、slippage、手数料を使い、
将来の価格・入出金・受払を先取りしません。Vega目的関数の1 vol point尺度、ridge、
オプション売買だけに掛かるturnover penaltyは従来と同じです。

絶対値を補助変数で線形化し、HiGHSで実行可能性を調べ、SLSQPで二次目的関数を最小化します。
solverの終了code/message/success、目的関数、SVD、連続数量を保存します。
別計算の非負乗数でKKTのprimal・stationarity・complementarityを検査し、
すべてが `solver_tolerance` 以下の場合だけ採用します。行は最大絶対係数と1の大きい方で割り、
目的関数は保存した `objective_scale` で割っています。残差はこの規格化後の値です。
この方式の `projected_gradient` はnullであり、KKT残差を別名で偽装しません。

連続数量から現金・総額を再計算した後、オプションをlot丸めし、株式Deltaを再計算・上限適用・lot丸めします。
丸め後の現金・総額も再検査し、残存Deltaを保存します。丸めで実行不能なら
`rounded_budget_infeasible` でsnapshotをrollbackします。整数解を探索・修復する実装ではありません。
連続問題の実行不能は `budget_infeasible`、KKT不合格は `solver_not_converged` です。
数値的に財務上限を満たせない連続解も採用しません。

翌snapshotの約定価格でバスケット全体を再検査し、超過なら
`rejected_cash_limit` / `rejected_gross_limit` として約定させません。
約定候補の資金・総額を注文に記録します。通常注文の数量制約を免除する終端清算も、
現金・総額の検査対象であり、超過時はrun失敗です。
保有総額はリバランスのない日も受払・約定後に検査します。価格変動による超過を
既発注の取引で解消できなければ `gross_limit` でそのsnapshotをrollbackします。
自動ロスカット、猶予期間、約定前の現金不足を売却で補う処理は追加していません。

予測現金・総額の比較にはbinary64の16 ULPだけを許容し、solver toleranceに応じて資金枠を拡大しません。
実約定の現金下限は従来どおり厳密比較です。
候補順は契約IDで固定し、SciPy/NumPyのversionをcheckpoint・集計の同一性に含めます。
ridge=0で解が非一意の場合の、異なるsolver version間で共通の正準tie-breakは残件です。

## 保存形式と互換性

| 文書 | 新しいversion | 変更 |
| --- | --- | --- |
| request | 6 | constraint_mode / gross_notional_limitを明示 |
| result | 10 | solver・丸め・約定のbudget監査、backend version |
| checkpoint | 8 | 新しい注文・decision監査とbackend同一性 |
| ensemble結果 | 2 | 新しい制約条件を共有条件へ保持 |
| ensemble入力 | 1のまま | 集計入力の形は変更なし |

新定義 `definitions-v2.schema.json` とセレクタ4枚を追加し、Schemaは計11ファイルです。
[前回の六ファイル](vega-kt-backtest-schemas-v0.1.md)のbundle hash
`5ad1c06bd94ba9403ea97142b6a50d098f68e09a2189738f8336a7a9ff9401d3` が変わっていないことをテストします。
五つの既存goldenも変更していません。validatorは新旧両方をdispatchします。
request v1–5は従来モードに読み込み、result v1–9も読み取りを維持します。
旧checkpointの新コードへの移行はせず、再実行します。
ensemble v2は同一の旧result v9群からも作れますが、新旧条件の混在比較は拒否します。

## 検証結果

環境はmacOS、Python 3.13.1、NumPy 2.5.3、SciPy 1.16.1、jsonschema 4.26.0です。

| 検査 | 結果 |
| --- | --- |
| Companion全体 | 193 passed。制約依存・Schema依存のskipなし。展開したwheelをrepo外からimport |
| Native adapter/integration | 22 passed。新モードで4戦略のJSON replay・checkpoint再開を含む |
| 制約oracle | 1変数の資金／総額／株式上限の解析解、費用付き2変数の独立dense gridとの比較が合格 |
| 失敗・因果性 | 連続・丸め後の実行不能、偽solver成功、翌価格での約定拒否、相場変動超過、snapshot rollbackが合格 |
| 互換性 | 緩い新制約で旧モードと台帳・約定・損益が一致。旧request/schema/goldenを検証 |
| Native合成例 | 11時点×4戦略が完了・精算済み。4入力の再実行と途中からの再開がそれぞれ保存結果と完全一致 |
| 配布 | 18 Python module、11 Schemaを含むwheel作成。19文書をrepo外から構造検証 |

合成例はgross cap 2,000、256 paths、診断用nativeで実行しました。
VegaKT戦略の最大連続grossはbinary64丸めを含む `2000.0000000000002`、
最大lot丸め後grossは `1999.413677287849` でした。上限が実際に効くケースです。
この数値は資金・総額処理の確認用であり、戦略の有効性を示しません。
VegaKTの1意思決定はSLSQP status=8 / success=falseでしたが、別計算のKKT検査を満たしたため採用しました。
このrunの最大KKT残差は `2.915719404842275e-9`、設定許容値は `1e-8` です。
終了codeとKKT検査を分けて記録し、solverの成功と表示しません。

生成物と再実行・構造検証の記録は `benchmark-results/vega-kt-budget-example/` にあります。
`verification.json` は使用wheel、Schema bundle hash、4戦略の検査と19文書の検証結果を記録します。
新旧11ファイルのbundle hashは `ffa9e53bfb1cfb48cedd09ff7a2b188456a5d7dfe9a372c5e37210f2c5b6f52e` です。
wheelは `/tmp/vega-kt-budget-companion-wheel/rust_pricing_backtest-0.1.0-py3-none-any.whl`、
SHA-256は `06b4bee57ae4e27937db823e475a173c20334a917c4b8ea223e49ebda38bc22a` です。

```shell
python -m pip install "./backtesting[constraints,validation]"
python -m unittest discover -s backtesting/tests -v
python -m unittest discover -s tests/python -p test_vega_kt_backtest.py -v
python scripts/generate_backtest_schemas.py --check
python examples/python/vega_kt_backtest.py --paths 256 --diagnostic-only --gross-notional-limit 2000 --output benchmark-results/vega-kt-budget-example
```

Nativeは既存の診断用ビルドを使用しました。binary SHA-256は
`93a5c068e81bc4d056634f403cb3616b4865962203d4073f0ebf818732c6741c` です。
今回はRust再ビルド・全Rustテスト・独立数値panel・remote CIを再実行していません。
前回の独立数値gateは154検査中54失敗のままです。
paperのGamma回復・伝播、初期時刻の欠落が未解決であり、native出力は引き続き
`local_vol_node_allocation_diagnostic_v1` / `unvalidated` です。

整数最適化、証拠金、ストレス時の資金先読み、正準tie-break、実市場検証、
独立VegaKT数値受入は残件です。機能テストの合格でH2全体を完了とはしません。
