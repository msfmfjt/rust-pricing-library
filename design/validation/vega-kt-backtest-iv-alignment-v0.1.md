# VegaKT バックテストのIV座標合わせ・未対応リスク監査 v0.1

日付: 2026-10-03  
状態: IV座標合わせと診断出力の機能検証は合格。VegaKTをIV変化へ適用する数値的解釈・Deltaとの整合は未検証。

## 追加した機能と適用範囲

[要件定義](../vega-kt-hedging-backtest-requirements-v0.1.md)のVKT-01/02/04とPNL-04に沿い、
前時点のbucketと当日のIVを同じ絶対満期・固定strikeで比較する監査を追加しました。
前時点の建玉・raw感応度を使い、当日の保有量やリスクを過去の区間へ適用しません。

ただし、要件はLV paper decompositionを独立した市場IV quoteの勾配と区別しています。
現行DeltaはLocal-variance gridを固定したbumpであり、固定strikeのIV変化との整合も未検証です。
したがって、座標を揃えた内積は**診断値**として保存し、損益説明へ加えません。
全行で `pnl_adopted=false`、`application=diagnostic_only_not_added_to_greek_pnl` とします。
幾何的に全ノードが対応しても、採用可能という意味にはなりません。

## データと座標規則

`Risk.reporting_iv` にbucketと同じ順序のcanonical IVを、`reporting_forwards` に絶対満期ごとのforwardを追加しました。
両方を同時に指定するか、両方を未指定にします。shape・正の有限値・RiskBasisとの整合を検査します。
同じsnapshotで実際に評価した契約の提供値が食い違えば、typed errorで停止します。

ネイティブadapterは公開VegaKT座標からIVを取得し、公開DiscountCurveの評価で `F(T)=spot/D(T)` を計算します。
このadapterはゼロ配当を要求します。既存のnative forwardのcarry恒等式とも照合しました。
後日支払の価格・リスク調整比率を、IVやforward水準へ掛けません。
新たな市場入力形式は追加せず、元pricing requestとprovider出力を監査に使います。
観測時刻・availabilityと入力履歴の検査は従来のsnapshot処理を維持します。

前時点の `(T,k_old)` に対して:

```text
K = F_old(T) × exp(k_old)
k_new = k_old + log(F_old(T)) − log(F_new(T))
ΔIV = IV_new(T,k_new) − IV_old(T,k_old)
```

絶対満期 `T` が当日のgridに完全一致し、かつ当日より後であることを要求します。
その満期の当日log-moneyness軸上でIVを線形補間し、参照indexと重みを保存します。
正確にnodeへ一致した場合は重み1です。満期補間・外挿・端へのclamp・近い満期への代用は行いません。
範囲外や欠損は `delta_iv=null` とし、次の理由を記録します。

- `expired_maturity`: 前時点の満期nodeが期限に到達した。
- `missing_absolute_maturity`: 当日に同じ絶対満期がない。
- `outside_strike_domain`: 固定strikeが当日のlog-moneyness範囲を外れた。
- `missing_opening_surface` / `missing_current_surface`: 対応する時点でsurfaceが提供されていない。
- `unsupported_coordinate`: canonical f-IV以外。座標変換を推測せず、node計算を行わない。

providerが既に評価した保有銘柄・候補銘柄からsurfaceを得ます。
監査を埋める目的だけで追加価格計算をしないため、オプションが消滅した後などはsurfaceがない場合があります。
過去のsurfaceを使い回しません。同じ市場入力でも、実際の評価対象が異なる戦略間では提供範囲が異なり得ます。

## 内積とカバレッジ

対象とオプションヘッジを別々に集計します。

```text
net_bucket[b] = sum_contract(quantity × multiplier × raw_bucket[b])
gross_bucket[b] = sum_contract(abs(quantity × multiplier × raw_bucket[b]))
covered_bucket_dot = sum_matched(net_bucket[b] × ΔIV[b])
uncovered_absolute_risk = sum_unmatched(gross_bucket[b])
```

異なる契約の買い・売りを相殺する前の絶対感応度で未対応量を測ります。
ネット感応度がゼロでも、未対応リスクを消しません。
未対応の絶対感応度が非ゼロなら `complete_bucket_dot=null` とします。
ゼロリスクのgroupは `no_opening_risk`、カバレッジ比率はnullです。この場合の内積ゼロは、データが揃った証明ではありません。
node数のカバレッジと感応度によるカバレッジは別に保存します。

`blockers` は未検証のIV-shock解釈・Delta convention、該当する場合のnative VegaKT未検証、未対応リスクを明示します。
診断内積で既存Greek residualを小さくしません。`greek_pnl.vega_pnl` は引き続きnullです。
通常損益・実現／未実現損益・Delta/Gamma説明額・割引後損益の計算は変えていません。

## 出力と再開

`result.iv_alignment` と11種類目の `iv_alignment.csv` に、一つの確定snapshotにつき一行を保存します。
初日は `no_interval` です。各行は開始／終了、両時点のbasis ID・surface hash・提供銘柄、node対応表、
補間重み、対象／ヘッジ別の内積と未対応量を持ちます。
manifestは `absolute_maturity_fixed_strike_iv_audit_v1` を固定します。

任意のsurface未提供や幾何的な対応不足は監査statusとして残します。
供給されたsurface同士の不一致や非有限な計算は、その日を巻き戻して停止します。
巻戻しは約定・現金・原価・Greek説明・新しい監査表を含みます。
未来のsurface変更は過去の対応表へ影響しません。

結果version 9、checkpoint version 7へ更新しました。入力はversion 5のままです。
旧結果version 1–8は読めます。存在しない監査履歴を生成せず空テーブルとします。
旧checkpointは再利用できず、元入力からの再実行が必要です。

## 検証結果

- 台帳・最適化・入力・再開: **127件合格**。今回13件追加。
- ネイティブ連携: **18件合格**。公開IV／forwardと支払調整の分離、四戦略の座標監査／再開を2件追加。
- 四戦略で各18セッションを完了し、全建玉・未収未払を精算。
- 前回保存した実物の結果version 8と比較し、orders・fills・cashflows・ledger・pnl_components・settlements・decisions・cost_basisが一致。
- Greek説明の金額も一致。更新したrisk hashと未採用理由のmetadataだけが異なることを確認。
- CSVを再読込みし、保存したIV・index・重みから補間値、IV差、対応部分の内積、未対応絶対リスクを再計算。
- 11種類のCSV、JSON再読込み、checkpoint再開、実物の旧結果version 8・入力version 5の読込みを確認。
- companion wheelの構築が合格。全15ソースモジュールと作業ツリーの一致を確認。
- Markdownリンク、差分空白、Pythonソースの構文確認が合格。

手計算と異常系の主な結果:

| 条件 | 確認結果 |
| --- | --- |
| 固定strikeのIV関数を変えず、forward100→110でk軸のサンプル値が移動 | node値の直接差は非ゼロでも、同じstrikeへ戻したIV差はゼロ |
| 全固定strikeでIVを0.01上昇、raw bucket1〜6、short2・倍率10 | 対応部分の内積−4.2。実測損益とGreek説明には加えない |
| 同じ残存tenorでも絶対満期が翌日へずれる | `missing_absolute_maturity`。同一indexで代用しない |
| 6node中3nodeが期限到達 | 期限到達部分はnull。残りだけ比較し、未対応絶対感応度120を保持 |
| forward100→200で当日strike範囲を外れる | 外挿せず全node未対応。部分内積0でも完全内積はnull |
| 異なる契約のlong/shortでnet bucketゼロ、全node未対応 | 未対応絶対感応度420を保持し、完全内積はnull |
| exact nodeと端点 | 重み1で一致。不要な補間や端への寄せなし |
| surface欠損・非対応座標 | 欠損理由を出し、既存会計を継続。架空のIV差を生成しない |
| 約定後に契約間のIVまたはforwardを改変 | 不一致を検出し当日を巻き戻す。改変を外した再開JSONが中断なしと一致 |
| strike復元のoverflow | typed errorで停止し、前の確定snapshotまでを保存 |

合成例のnode対応数は、無ヘッジ／Deltaで各1,608、scalar Vega／VegaKTで各1,908でした。
期限到達nodeは各299。範囲外は前者各134、後者各159で、残りは前後surfaceの未提供として明示しました。
これは実際に評価した銘柄から得たデータの範囲であり、戦略の優劣やVegaKTの精度を比較する指標ではありません。

出力先は `benchmark-results/vega-kt-iv-alignment-example/` です。

```shell
python -m unittest discover -s backtesting/tests -v
python -m unittest discover -s tests/python -p test_vega_kt_backtest.py -v
python examples/python/vega_kt_backtest.py --paths 512 --diagnostic-only --payment-lag-days 2 --numeraire-rate 0.01 --capital-flow-demo --gamma-relative-bump 0.0001 --output benchmark-results/vega-kt-iv-alignment-example
```

ネイティブ計算コードは変更していません。使用したバイナリのSHA-256:
`93a5c068e81bc4d056634f403cb3616b4865962203d4073f0ebf818732c6741c`

## 残件

VegaKTのIV-shock解釈、frozen-grid Deltaとの整合、満期node消滅の扱い、独立したリスク数値受入、完全なPNL-04は未完了です。
VegaKT本体の `diagnostic_only` と `unvalidated` を維持しています。
今回、数値受入パネル・Rust全体のゲート・リモートCIは再実行していません。
[直近の数値検証](vega-kt-backtest-allocation-audit-v0.1.md)は154項目中54項目が不合格です。
