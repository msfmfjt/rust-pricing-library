# VegaKT バックテストのDelta・Gamma損益説明検証 v0.1

日付: 2026-10-03  
状態: 前時点Delta・任意Gammaによる説明と残差表示の機能検証は合格。完全なPNL-04と論文VegaKTの数値受入は未完了。

## 変更した動作

[要件定義](../vega-kt-hedging-backtest-requirements-v0.1.md)のPNL-04について、
前の確定snapshotで保有していた対象・オプションヘッジ・株式のDeltaと、取得できる場合のGammaによる損益説明を追加しました。
通常通貨額の説明であり、割引後のGreeks attributionではありません。

```text
ΔS = 当日のspot mid − 前の確定snapshotのspot mid
Delta寄与 = 前時点の保有Delta × ΔS
Gamma寄与 = 0.5 × 前時点の保有Gamma × ΔS × ΔS
残差 = 資産別の実測損益 − 計算できたGreek寄与
```

保有リスクは契約単位のリスクに符号付き数量と倍率を掛けます。
当日の保留注文の約定や最終ヘッジ清算による数量を、過去の期間へ適用しません。
stockのDeltaは前時点の株数で、Gammaはゼロです。
対象とヘッジは別々に説明し、満期前に保有したリスクは満期へ至る期間に使います。
確定未収未払には新たなspotリスクを付けず、割引評価差を残差に残します。

初日には比較期間がないため、Greek寄与はnullです。取得価格と初日評価額の差は残差に、取得時手数料は費用に表示します。
Theta、ボラティリティ・曲線変化、配当によるオプション価値への影響、交差項・高次項、数値誤差、
model markとmarket markの差、fixingの影響は、個別に測定したと装わず残差へ残します。
実際の配当・funding・借入費用・取得手数料・売買手数料・spread/slippageは、既存の正確な寄与を別建てで保存します。
外部入出金は利益へ含めません。

## Gammaの取得と未計算の扱い

`Risk.gamma`、`gamma_method`、`gamma_spot_bump` を追加しました。
Gammaは契約単位あたりのraw値で、Deltaと同じspot conventionを要求します。
非有限値や不正なbump幅は拒否します。有限な負の推定値はそのまま残します。
ネイティブproviderは公開 `gamma_raw` とmethod/bump情報を読み、元requestにGammaがなければ追加計算しません。
後日支払では価格・Deltaと同じ確定曲線の支払調整比率をGammaへ適用します。spotのbump幅はこの比率で変えません。

ある資産グループの非ゼロ保有オプションにGamma欠損があれば、グループ全体のGamma寄与をnullにし、
欠損契約を `gamma_missing_contracts` に記録します。取得できた一部だけを全体のGammaとは表示しません。
他のグループで完全に取得できたGammaは、そのグループの寄与へ含めます。
合計の `available_greek_pnl` は計算できた項の和であり、全Greekを網羅した値ではありません。
spotが動かなかった場合も、欠損Gammaをゼロへ変えません。
保有オプションがないグループとstockのGammaゼロは、構造上ゼロと分かる場合です。

VegaKTとIV変化の内積は未実装です。日を跨ぐ絶対満期・strike/moneynessの整合を実装していないため、
同一indexのIVを引いて寄与を生成しません。`vega_pnl=null` と理由を出します。
座標合わせ、Delta conventionとの整合、VegaKT本体の受入は引き続き必要です。

## ネイティブのbump幅による影響

現行ネイティブLVはDeltaとGammaでspot bumpを共有しています。
Gammaに指定した幅はDelta計算にも使われるため、Gammaの有効化・幅変更によってヘッジ数量も変わり得ます。
コンパイル時の幅決定は `crates/pricing/src/engine/compile/simulation.rs`、
共通bumpの使用は `crates/pricing/src/engine/risk/pathwise_local_vol.rs` で確認しました。

例の `--gamma-relative-bump 0.0001` は、このゼロ配当adapterの既存Deltaデフォルトと同じ幅です。
この条件で従来の売買・会計結果が変わらないことを確認しました。
0.01へ変更すると合成例のDeltaとヘッジ数量が変化しました。幅は元requestとGamma診断に保存します。
これは数値入力の違いであり、説明レポートの追加だけによる取引変更ではありません。
0.0001を精度受入済みの幅とは扱いません。
ここで取得する価格のspot Gammaは、未完成の論文VegaKTガンマ復元・伝播処理とは別です。

## 出力・照合・再開

`result.greek_pnl` と10種類目の `greek_pnl.csv` に各snapshotの説明を保存します。
行は区間開始・終了、暦日数、spot変化、前時点のrisk/inventory hash、Delta convention、risk warning、
資産別寄与・欠損状況・残差、現金／費用寄与、累積値、照合差と許容差を含みます。
`summary.greek_pnl` は観測数、区間数、Gammaが全グループで取得できた区間数、累積内訳を持ちます。
確定観測がゼロなら累積額はnullです。
manifest policyは `opening_holdings_delta_optional_gamma_v1` に固定します。

```text
計算済みGreek寄与 + 残差 + 正確な現金／費用寄与 = 通常の台帳損益
```

この日次／累積等式は算術の整合確認です。残差が小さいことやGreekの精度・戦略の有効性を証明しません。
残差を補填項にしない既存の `pnl_components` と台帳の独立照合は先に実施します。
ここでは既存のPNL日次／累積許容差を使い、差額を記録します。
計算の非有限化や照合不一致はsnapshot全体を巻き戻し、約定・原価・説明を二重計上せず再開します。

入力version 5は変更せず、結果はversion 8、checkpointはversion 6です。
旧結果version 1–7は読めます。説明がない場合は空テーブルとし、過去のGreekを作りません。
旧checkpointは再利用できず、元入力から再実行します。

## 検証結果

- 台帳・最適化・入力・再開のテスト: **114件合格**。今回12件追加。
- ネイティブ連携: **16件合格**。公開Gamma・支払調整・bump metadataと、四戦略の説明／再開を2件追加。
- 四戦略で各18セッションを完了。各17区間でGamma欠損なし（保有なしの既知ゼロを含む）。全建玉・未収未払を精算。
- 既存Deltaと同じbump幅にて、前回保存したorders・fills・cashflows・ledger・pnl_components・settlements・decisions・cost_basisとの一致を確認。
- CSVを再読込みし、前時点の保有量・リスクからDelta/Gamma寄与を独立に再計算。
- 日次／累積損益の再合計を台帳へ照合。差は日次最大約9.2e-14、累積最大約1.5e-13。
- 10種類のCSV、JSON再読込み、checkpoint再開、実物の旧結果version 7・入力version 5の読込みを確認。
- 配布用wheelを構築し、全14ソースモジュールと作業ツリーの一致を確認。通常のGamma未指定ケースも既存テストで動作を確認。
- Markdownリンク、差分空白、Pythonソースの構文確認が合格。

主な手計算・異常系:

| 条件 | 確認結果 |
| --- | --- |
| 二次価格 `10 + 0.4x + 0.05x²`、`x=S−100`、spot100→102、short2・倍率10 | Delta寄与−16、Gamma寄与−4、実測−20、残差ゼロ。longでは逆符号 |
| 上の価格へ日次0.3の時間変化を追加 | short2・倍率10の残差−6。Delta/Gammaへ押し込まない |
| model価格10→11、market mark9→13、short2・倍率10 | 実測−80、Greek寄与−20、残差−60 |
| 翌時点で株式ヘッジを開始し、最終時点で清算 | 開始前の価格変動へ新規株数を適用せず、清算前の期間は前時点の保有株数で説明 |
| 当日以降のDelta/Gammaのみを改変 | その日に至る区間の説明は変わらず、次の区間から変化 |
| 一部契約のGamma欠損、spot不変 | 欠損グループのGammaはnull。欠損をゼロへ置換しない |
| payoff10の後日支払、割引係数0.98→0.99→支払 | 支払待ちと支払日の評価差各0.1は残差に残り、架空のspotリスクを作らない |
| 初日・約定後の説明不一致 | その日を巻き戻す。不一致の注入を外した再開JSONは中断なしと一致 |
| 過大な前時点Gammaによる非有限な説明計算 | 当日を巻き戻して停止。処理済みriskを修正する場合は新規runが必要 |

出力先は `benchmark-results/vega-kt-greek-pnl-example/` です。
この例は会計・リスク接続の確認であり、実市場でのヘッジ有効性を示しません。

再現コマンド:

```shell
python -m unittest discover -s backtesting/tests -v
python -m unittest discover -s tests/python -p test_vega_kt_backtest.py -v
python examples/python/vega_kt_backtest.py --paths 512 --diagnostic-only --payment-lag-days 2 --numeraire-rate 0.01 --capital-flow-demo --gamma-relative-bump 0.0001 --output benchmark-results/vega-kt-greek-pnl-example
```

ネイティブ計算コードは変更していません。使用したバイナリのSHA-256:
`93a5c068e81bc4d056634f403cb3616b4865962203d4073f0ebf818732c6741c`

## 残件

整合したIV変化とVegaKT寄与、Greek精度・収束の独立受入、完全なPNL-04、実市場検証は未完了です。
今回の確認はPython companionとネイティブ接続が対象です。Rust全体のゲートとリモートCIは再実行していません。
VegaKT本体の `diagnostic_only` と `unvalidated` を維持しています。
数値受入パネルも今回再実行していません。
[直近の数値検証](vega-kt-backtest-allocation-audit-v0.1.md)は154項目中54項目が不合格です。
