# 名前付きカスタム基底とヘッジ台帳の接続

日付: 2026-10-12。[集約結果](local-vol-custom-basis-provider-v0.1.json)。
[独立数値受入](local-vol-custom-basis-v0.1.md)済みの2つの基底を、Call/Putの通常台帳へ接続する。

## 選択方法

```python
provider = RustPricingProvider(
    numerical_scope="multi_step_custom_basis_gamma",
    variance_basis_policy="row_hats",  # または weighted_time_hats
    worker_threads=4,
)
result = run_backtest(config, history, contracts, portfolio, provider)
audit = provider.gamma_audit(snapshot, contract)
```

```sh
python -m rust_pricing_backtest input.json --output result \
  --validated-custom-basis weighted_time_hats
```

`variance_extension` は指定しない。既存のCall専用scopeとPut対応preset scopeの選択方法は同じ。
任意行列や任意の節点・尺度をこのproviderへ渡す入口は設けない。

`row_hats` は遷移中点ごとの時間節点と、元の33分散節点から1つおきの17節点を使う。
`weighted_time_hats` は市場報告満期を時間節点とし、同じ17節点を使う。
後者の列尺度は時間方向の0.8〜1.3と `1+0.3*k` の積で固定する。
いずれも前回の独立数値受入と同じ行列を生成する。較正時に尺度を正規化しない。

監査の `basis_construction` に節点・尺度・行列hashを保持する。
較正とGammaの基底hashが一致しなければRiskを返さない。
基底名と生成・較正実装をproviderの識別子に含めるため、異なる基底やpresetでのcheckpoint再開は拒否される。
ソース更新前のcheckpointは保存入力から再実行する。

## 受入の境界

European・算術平均Asian・固定行使価格LookbackのCall/Put、4/6区間、33分散節点、
2区間ごとの市場満期と各5 quotes、carry・配当なし、満期払いを対象とする。
価格とDeltaは通常評価、Vegaは全区間Gammaと市場IV較正接線から得る。
Put Lookbackでは既知の最安値に対応する実効行使価格を積分区切りへ含める。

標本数は4区間4,096組以上、6区間2,048組以上、antitheticを要求する。
分割数512、相対許容差1e-6、相対SE上限25%、nativeの計算量上限を維持する。
精度検査に失敗したsnapshotではRisk・cacheを確定せず、発注を停止する。
観測値の欠落・未来の公表・採用済み値の改訂を拒否し、直前の保存状態から修正後に再開できる。

低水準のGamma・基底APIは診断APIを維持する。provider側の限定的な受入情報と区別し、
任意基底の全面受入、異商品の合同共分散、実市場でのヘッジ効果は未受入とする。
`full_market_iv_bucket_acceptance=not_established` を継続する。

## 検証内容

独立価格差分を含む前回の80パネル・5,540検査を数値基準とする。
Rust本体・較正・基底生成・価格基準・方針のソースとnativeバイナリが同一であることを確認する。
今回変更したadapter・provider・CLIの3ファイルを、引継ぎ対象の数値ソースから明示的に区別する。
新しいprovider経由で両基底、4/6区間、10商品履歴、seed 7の40パネルを再計算する。
基底構成・較正接線・Gamma・共分散を前回の監査へ完全一致で照合する。
request/plan識別子だけを除外し、bin入力ラベルの丸めは別に1 ULP以内を検査する。
Gammaや共分散の計算値に丸めの許容差を適用しない。

両基底・両区間数・Call/Put・Asian/Lookbackについて、4戦略ずつ計64件の満期決済を検証する。
4区間では当日観測を含む4評価日、6区間では2評価日から進め、96件の契約評価監査を保存する。
観測値から直接求めた支払額、売買、schema、途中保存・再開の完全一致、別基底での再開拒否を確認する。
精度失敗と基底hash不一致を注入して、発注とcacheが残らず、修正後に同じ結果へ戻ることも検査する。

生の監査とログは `benchmark-results/vega-kt-custom-basis-provider/` に保存する。

## 検証結果

40パネルの基底構成・較正接線・native監査が前回の数値受入と完全一致した。
64戦略の決済・再開、96件の契約評価、32件の失敗・修復検査が合格した。
補助ライブラリ262件、既存モードの回帰3件、新providerの5テスト×4組、計285テスト実行が合格した。
前回の数値・価格基準のソース403ファイル（Rust 360ファイルを含む）とnativeバイナリを維持した。

配布wheelから4区間row_hatsと6区間weighted_time_hatsのLookback Put＋Callヘッジを再実行した。
別providerでの途中再開と、実CLIからの再開が一致し、別基底での再開は拒否された。
同じ実行設定の4区間は結果・checkpoint全体がソース実行と完全一致した。
1 workerで実行した6区間も4 workersのソース実行と全計算値・売買・決済・損益が完全一致し、
実行設定を含む識別子だけを別扱いとした。損益表のリスクhashは元のリスク行から再計算して検証した。
配布版の12件のnative監査も完全一致し、Python/schema 56ファイルがソースと一致した。

配布物の前版との差分はadapter・provider・CLIの3ファイルと、名前付き基底モジュールの追加だけである。
初回のテスト入力ではPutの商品名変換に不備があり中断した。20種類の入力を独立受入済みrequestと
完全一致で確認して全4組を再実行した。数値計算や許容誤差は変更していない。初回ログも保存する。

集約JSONにソース・監査結果・配布wheelのSHA-256を記録し、配布ソースの必須ファイルも検査する。

## 再実行

同じnativeバイナリによる前回の数値受入を指定する。4分割を順次または並行して実行し、集約する。

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_custom_basis_provider.py \
  --intervals 4 --basis row_hats \
  --output benchmark-results/vega-kt-custom-basis-provider/4-row_hats.json
```

`--intervals 6` と `--basis weighted_time_hats` の組合せも実行する。

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_custom_basis_provider.py \
  --aggregate benchmark-results/vega-kt-custom-basis-provider/4-row_hats.json \
              benchmark-results/vega-kt-custom-basis-provider/4-weighted_time_hats.json \
              benchmark-results/vega-kt-custom-basis-provider/6-row_hats.json \
              benchmark-results/vega-kt-custom-basis-provider/6-weighted_time_hats.json \
  --output benchmark-results/vega-kt-custom-basis-provider/report.json
```

合格したruntime結果の再照合には `--recheck-report` を指定できる。
ソース・nativeバイナリ・基底・区間数・参照結果が同一の場合だけ再利用する。
