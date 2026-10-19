# 商品間共分散と損益説明の通常バックテストへの接続

日付: 2026-10-13。

## 実行結果

| 確認 | 結果 |
|---|---|
| companion試験（今回の手計算9件を含む） | 283件合格 |
| 実nativeのポートフォリオ試験 | 5件合格 |
| 実nativeの通常バックテスト統合 | 1件合格 |
| 現行行列診断CLIの互換性試験 | 9件合格 |
| 4/6区間の実CLI事前検査 | 両方合格。数値再受入ではない |
| 従来の台帳・注文・約定 | 完全一致 |
| 別providerによるAPI途中再開 | 結果JSONがバイト単位で一致 |
| 配布wheelからの途中CLI再開・完了結果からの再開 | 合格。完了再開はバイト単位で一致 |
| worker 4のAPIとworker 1の配布CLI | 台帳・注文・約定・個別リスク値が完全一致 |
| 配布内容 | Python 37ファイル・schema 26ファイルを元ソースと照合 |

計298件の試験が合格した。既存21 schemaとRust 360ソースファイル、nativeバイナリは同一。
新規CIジョブ内の試験手順も追加したが、リモートCIを実行したという結果ではない。

検証記録は次のコマンドで、先行証跡と今回の結果を分けて集約する。

```sh
python scripts/check_joint_replay_compatibility.py \
  --directory benchmark-results/vega-kt-joint-replay \
  --prior-source-archive benchmark-results/vega-kt-portfolio-market/source.tar.gz \
  --output design/validation/vega-kt-joint-replay-v0.1.json
```

## 使い方

`BacktestConfig.max_joint_bucket_standard_error` に、数量・契約倍率反映後の
市場IV bucket標準誤差の上限を明示すると有効になる。単位は通貨／絶対IV。
上限は実験ごとに決める。既定値 `None` では従来の売買・保存形式を使う。

```python
config = BacktestConfig(
    strategy="vega_kt",
    max_joint_bucket_standard_error=10.0,  # この検証fixtureの設定。一般の推奨値ではない。
)
provider = RustPricingProvider(
    numerical_scope="multi_step_extension_put_gamma",
    variance_extension="all_rows",
)
result = run_backtest(config, history, contracts, portfolio, provider)
```

保存された入力は通常のCLIで実行・再開できる。

```sh
python -m rust_pricing_backtest input.json \
  --validated-put-extension all_rows --output results
python -m rust_pricing_backtest input.json \
  --validated-put-extension all_rows --checkpoint results/checkpoint.json \
  --output resumed-results
```

CLIは既定のworker数1を使う。再開には元のprovider設定を含む同じ実装・入力・方針が必要。
異なるworker数で作ったAPIのcheckpointをCLIへ渡すと、providerの識別情報不一致として拒否する。

## 売買判断と精度検査

通常providerの較正・価格・Gamma・契約履歴の検査を維持したまま、評価済みの全商品を
同一の独立標本番号で対応付けた合同リスクを計算する。数量0の評価済み候補も含める。
価格・Delta・VegaKTが実際の売買判断に用いる値と一致しない合同リスクは拒否する。
一つの合同リスクから、現在保有とlot丸め後の発注予定をそれぞれ再加重する。
符号と契約倍率を一度だけ掛け、商品間の共分散も使って標準誤差を判定する。
市場ショック座標で最適化していても、検査対象は元の全市場IV bucketとする。

精度上限を超えた場合、注文を確定せず当日の処理全体を取り消す。
前日の注文の約定、cash、台帳、損益行も同じ保存境界まで戻す。
失敗した日を修復して再開すると、連続実行と同じ結果になる。
共分散は推定誤差であり、市場リターン共分散として最適化目的へ流用しない。
既存の最適化目的と凍結LVの発注用Deltaは維持する。

`risks[*].joint_market.report` に合同リスク、`.held` に保有精度検査、
`decisions[*].joint_market` に丸め後の発注精度検査を保存する。

## 損益説明と保存・再開

前日の保有と保存済み合同リスクを使い、期首の固定strike・絶対満期へ当日のIVを合わせる。
前日のリスクを再計算しない。固定strike IVへのDelta補正、対象・ヘッジ・株式の説明額、
説明できない残差、会計照合を `greek_pnl[*].market_iv` に保存する。
集計は `summary.joint_market` で確認する。
既存の `greek_pnl` のDelta／Gamma説明とは座標の異なる別の説明であり、両者を加算しない。
Gamma・Theta等は引き続き残差に含まれ、標準誤差は凍結Delta自身の推定誤差を含まない。

満期、IV欠落、固定strikeの域外では理由を保存し、該当区間を未説明として残す。
保有が相殺していても必要な入力の欠落を隠さない。
同一スポット・同一forwardのstrike境界が浮動小数の計算順序で域外になる問題は、
対数差を先に計算することで解消した。外挿の許容幅は広げていない。

新機能を有効にした場合だけ request v9、result v13、checkpoint v11、ensemble v5を使う。
旧21 schemaファイルは変更せず、新たに5ファイルを追加した。
checkpointは実装・設定・処理済み履歴・保存状態に結び付き、各日の合同リスクも
独自の内容hashと当日の入力へ照合する。破損や異なる精度方針での再開は拒否する。

## 検証と受入範囲

手計算の共分散fixtureで、完全相殺、符号・契約倍率、保有は合格しても発注予定が不合格になる例、
約定後の停止と巻き戻し、途中再開、保存内容の破損、異なるリスクの差し替え、
四戦略・固定市場ショック・新旧schemaの整合を検査する。
実nativeの6区間AsianとEuropeanヘッジでは、従来との台帳・注文・約定の一致、
保存リスクによる損益説明と既存の別レポートによる再計算の一致、別providerでの途中再開を検査する。
満期前の1区間は説明でき、満期日の区間は理由付きで未説明とする。

本接続は通常providerが既に扱う4/6区間の明示的較正拡張・名前付き基底を対象とする。
[6市場の数値受入](vega-kt-portfolio-market-v0.1.md)を通常providerへ一括昇格する変更ではない。
価格・Gamma・較正を行うRustバイナリと計算コードの同一性を保存済みソースへ照合し、
先行の4,038検査の証跡を保持する。今回その全検査を再実行したという意味ではない。
変更した日次実行・損益説明の受入は [今回のJSON](vega-kt-joint-replay-v0.1.json) に分ける。
旧checkpointは実装変更により再開できず、保存入力から再実行する。

未使用seed・追加市場での独立確認、受入精度での252営業日性能測定、実市場データでの有効性検証は次段階。
