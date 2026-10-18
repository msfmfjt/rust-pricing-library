# Asian・Lookbackの観測履歴とバックテスト決済

日付: 2026-10-07。集約結果は[JSON](local-vol-path-ledger-v0.1.json)。
[複数満期provider](local-vol-multi-step-provider-v0.1.md)に、
日付と出所を持つ観測履歴、経路依存payoffの評価と決済を追加する。

## 契約と観測値

`PathContract` は従来の欧州型 `Contract` と別の契約型である。
`product_type` は `arithmetic_asian` または `fixed_lookback`、sideはCall/Put。
`PathObservation` に観測日、fixing ID、weightを指定する。
観測日は重複なく昇順、最終日は満期。Asianの正のweightsの合計は1、Lookbackのweightsは1とする。

fixingは既存の `Snapshot.settlement_fixings` に `SettlementFixing` として渡す。
観測日、観測時刻、入手時刻、source、revision、価格を必須とする。
評価日までに確定する全観測を要求し、未来の観測を既知値にすることは認めない。
最初のsnapshotより前の履歴も同じ確認を行う。

使用済みfixingはcheckpointの `state.path_fixings` に保存する。
次のsnapshotで再送する必要はない。再送する場合は値・日付・出所・revisionを含めて一致が必要。
異なるrevisionを自動採用したり、修正値で過去を再評価したりしない。
同一snapshotの他の処理が失敗した場合、fixingの取込みも現金・注文と一緒に取り消す。
未処理のsnapshotを修復し、最後に成功した状態から再開できる。

## Native評価との接続

```python
provider = RustPricingProvider(numerical_scope="multi_step_path_gamma", worker_threads=4)
```

CLIでは `--validated-path` を指定する。欧州型ヘッジも同じproviderで評価する。
path契約をP0や欧州型専用モードへ渡すと拒否する。
結果には `numerically_validated_multi_step_path` を記録する。

`path_history.native_product(contract, snapshot)` は、利用可能な観測値からnative productを組み立てる。
Asianは既知値と未知値を各観測に明示し、Lookbackは過去のmax（Call）/min（Put）を
`historical_extremum` に渡す。入力requestのproductはこれと完全一致を要求する。
履歴の価格、日付、weights、未来の既知値が改変されている場合は評価を止める。

評価時点の市場面は4・6区間、33分散節点、2遷移ごとに5市場quotesを報告する較正延長を使う。
割引係数1、carry・配当なし、満期同日払い、4,096以上の独立標本を引き続き要求する。
通常価格・frozen-grid Deltaと、条件付きGamma/AD・市場較正の各判定を維持する。
相対許容差1e-6、積分分割512、方向の相対SE上限25%を緩和しない。

Lookbackの確定日当日のnative評価は明示的に拒否する。
現在のnative APIは当日の観測をspotとして扱い、確定済みの値としてDeltaから外せないためである。
履歴を架空の過去日へ移動して回避しない。今回のLookback評価は観測日の翌日以降を対象とし、
満期当日はnative評価を経ず、全fixingから直接決済する。

## 決済と再現性

Asianは契約weightsで計算した平均、Lookbackは全観測の最大・最小値を使う。
最終snapshotのspotで置き換えない。数量と契約倍率はpayoff計算後に台帳が一度だけ適用する。
決済記録には各観測とfixing、fixing hashを保存する。

同一strike・満期でも欧州型と経路依存型は異なるpayoffとして扱い、欧州型ヘッジを誤って除外しない。
入力はrequest v8、結果はresult v12、checkpoint v10、ensemble v4を追加。
path契約がない場合の既存形式は継続し、以前の16スキーマはbyte単位で維持する。

## 検証

手計算では異なるweightsを持つAsian、Call/PutのLookbackを検査する。
価格98・110・94・106、weights 0.1・0.2・0.3・0.4、strike 100のAsian Callは2.4。
Lookback Callは10、Putは6。最終spotがこれらと異なっていても同じ決済となる。
欠損・未来情報・日付不一致・使用済みfixingの改訂は失敗とし、snapshotの取り消しと修復再開を確認する。

Native試験は、4区間の各商品・sideを4つの評価日で計算する。
観測が確定するたびに履歴を増やし、残る観測日を含む4区間の新しい市場面を合成する。
全ての市場quotesを再計算して較正を確認し、価格とDeltaを通常native評価と完全一致で比較する。
4戦略を満期まで進め、契約ごとの手計算payoff、スキーマ、全出力の中断再開一致を検査する。
6区間は既知履歴を持つAsian・LookbackのCall/Putを追加検査する。
合成市場による接続試験であり、実市場のヘッジ効果の受入とは扱わない。

結果: **20評価条件、16通りの満期バックテストが合格**。
関連テストは**290件合格、失敗0**（native履歴3、既存native連携39、companion 248）。
4区間ではAsian/LookbackのCall/Putを各4評価日、6区間では各商品の初期履歴付き評価を確認した。
4商品 × 4戦略の全てで、決済額・結果スキーマ・中断再開の完全一致が合格した。

導入したwheelだけを使い、新しいプロセスで履歴付きLookback CallのVegaKT戦略を満期まで再実行した。
保存済み入力からの全結果は受入時と完全一致し、新しいスキーマで検証できた。
以前の16スキーマは、前回のwheelとのbyte比較でも完全一致した。
Rust Gammaのソース・バイナリは変更しておらず、以前の5,904検査全行列を再実行したとは扱わない。


## 再実行

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_path_ledger.py \
  --output benchmark-results/vega-kt-path-ledger/report.json
python -m rust_pricing_backtest benchmark-results/vega-kt-path-ledger/input.json \
  --output benchmark-results/vega-kt-path-ledger/replay --validated-path
```

## 未受入の範囲

Lookback確定日当日のnative評価、一般の較正延長、carry・配当、barrier・早期行使、
異なる商品の合同共分散、共分散を使ったヘッジ最適化、実市場の履歴リプレイは未受入。
`full_market_iv_bucket_acceptance=not_established` を維持する。
Rust Gammaの計算は変更せず、従来の数値受入を参照する。
