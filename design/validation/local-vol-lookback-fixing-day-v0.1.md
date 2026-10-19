# Lookback観測日当日の価格・Gammaと台帳

日付: 2026-10-07。集約結果は[JSON](local-vol-lookback-fixing-day-v0.1.json)。
[前回の経路依存台帳](local-vol-path-ledger-v0.1.md)で残した、
Lookbackの確定日当日のnative評価制限を解消する。

## 当日確定値の意味

Rust `FixedLookbackSpec::with_valuation_date_fixing`、Python
`Product.fixed_lookback(..., valuation_date_fixing=...)` とnative request v3に
任意の正値 `valuation_date_fixing` を追加した。
評価日と同日の観測がある場合だけ指定でき、payoffでは定数として使う。
価格の再評価、Delta・Gamma・Vega、条件付きGammaでも、この値はスポットや分散と一緒に動かさない。
将来観測は従来どおり模擬経路から計算する。

`historical_extremum` は評価日より前の観測の最大値（Call）または最小値（Put）を表す。
当日初めて観測する商品は `historical_extremum` を指定する必要がない。
当日の確定値が過去の極値より大きい・小さい場合も、両方をpayoffに含める。
当日の観測を架空の過去日に変更しない。

省略した場合の既存動作は保持する。当日の観測は評価時点のスポットを使い、感応度にも含まれる。
native v1/v2で新しいフィールドを受け取った場合は拒否し、既存のv1/v2スキーマは変更しない。
v3では未指定のフィールドを出力しない。

診断 `lookback_valuation_date_fixing` に当日確定値を記録し、結果JSONも往復できる。
既存の `past_monitoring_count` は評価日より前、`future_monitoring_count` は評価日以降の
**日程数**という定義を保持する。後者には確定済みの当日も含むため、未知観測数ではない。

## 台帳への接続

`path_history.native_product` が日付・入手時刻を確認したfixingを、過去の極値と当日値に分ける。
入力native requestと完全一致を要求し、当日値の欠落・改変を拒否する。
`multi_step_path_gamma` / `--validated-path` で利用する。
観測がある日は、そのsnapshotまでに利用可能な確定値が必要となる。
確定前の同日snapshotで代わりにスポットを使う機能は追加していない。

使用済みfixingの不変性、失敗snapshot全体の取消し、修復後の再開、全観測からの満期決済を維持する。
台帳のスキーマはrequest v8 / result v12 / checkpoint v10 / ensemble v4を継続する。
providerの実装とnativeバイナリはcheckpointに結び付くため、更新前のcheckpointを更新後の
実装でそのまま再開せず、保存入力から再計算する。

## 独立照合と数値受入

最後に残る未知観測が満期だけのとき、確定最大値HのCallは
`max(H-K,0) + max(S_T-max(H,K),0)`、確定最小値LのPutは
`max(K-L,0) + max(min(L,K)-S_T,0)` に一致する。
当日確定値、過去の極値の優先、strikeを越えない履歴の計6条件で、
別の欧州型payoffから計算した価格・Delta・Gamma・Vegaを照合する。
価格はBlack–Scholes解析値との6標準誤差以内も要求する。

全観測が当日までに確定した8条件は、異なるスポットでも決済額が同じで、
標準誤差・Delta・Gamma・Vegaが厳密にゼロとなることを確認する。
当日値を省略したCall/Putは従来どおりスポットに連動する。

非平坦な市場面では4・6区間、Call/Put、過去履歴あり・なしの8条件を検査する。
別途、4区間で観測日当日に評価するCall/Putを満期まで動かす。
通常native価格・Deltaとの一致、条件付きGamma/AD、市場較正、4戦略の決済と完全再開を要求する。
欠損・入手前・過去観測の改訂を注入し、注文とfixingの取込みを含めた取消しと修復再開を検査する。
前回の20評価条件、16戦略リプレイも現在のnativeバイナリで再実行する。

相対許容差1e-6、積分分割512、Gamma方向の相対SE上限25%は変更しない。
Gamma推定・条件付きAD・共分散の実装ソースは前回と同一であることを確認する。
payoffのコンパイル、商品/API、スキーマ、台帳接続には変更がある。
以前の5,904検査全行列を今回再実行したとは扱わない。

結果: **新規16評価条件・8戦略リプレイが合格**。
前回の20評価条件・16戦略リプレイも再実行し、合わせて**36評価条件・24リプレイが合格**した。
独立の欧州型照合6条件、全確定8条件と旧当日動作2条件も合格した。
関連テストは**805件合格、失敗0**（今回と前回のpath試験8、Python回帰89、台帳248、Rust本体460）。
Rustの既存5件のrelease専用重い試験は既定どおり除外し、合格件数には含めていない。
静的検査、型/API照合、スキーマ生成整合性も合格した。

以前のnativeバイナリとの比較は12条件の全request/result JSONが完全一致した。
全21台帳スキーマは前回wheelとbyte同一、native v1/v2スキーマも変更なし。
更新したnativeと台帳wheelを導入し、新しいプロセスで当日観測を含むLookback CallのVegaKT戦略を
保存入力から満期まで再実行した。全結果は受入時と完全一致した。


## 再実行

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_lookback_fixing_day.py \
  --output benchmark-results/vega-kt-lookback-fixing-day/report.json
python -m rust_pricing_backtest benchmark-results/vega-kt-lookback-fixing-day/input.json \
  --output benchmark-results/vega-kt-lookback-fixing-day/replay --validated-path
```

`--legacy-native` に以前のnative wheelを展開したディレクトリを渡すと、
別プロセスで旧API入力12条件の全request/result JSONの完全一致も検査する。

## 継続する制約

較正延長は4・6区間の各報告満期直前の分散行だけを動かす既存の範囲である。
バックテストは割引係数1、carry・配当なし、満期同日払い、4,096以上の独立標本を要求する。
一般の較正延長、barrier・早期行使、異なる商品の合同共分散とその最適化、実市場のヘッジ効果は未受入。
`full_market_iv_bucket_acceptance=not_established` を維持する。
