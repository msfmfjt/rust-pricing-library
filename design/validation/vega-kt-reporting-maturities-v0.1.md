# VegaKTの報告満期と計算時間格子の分離 v0.1

日付: 2026-10-04  
状態: 空間平坦・zero carry・単一満期の既存対象範囲で、疎な報告満期による評価失敗を修正。
一般smileの数値受入と旧基準の6件は未解決のまま保持する。

## 解消した失敗

以前の満期Gamma経路は、報告満期がシミュレーションの全正時刻と一致することを
要求していた。8区間の価格計算に対して報告満期を64日・128日の2点にすると、
入力の報告basisとrisk要求が一致していても、この経路の対象外になった。
従来の局所節点配賦へ戻った後、評価時に
`Local Volatility VegaKT request nodes must match the model reporting-IV basis`
というエラーで終了した。

同じ入力を修正後のnativeライブラリで再実行すると、通常評価が完了する。
seed 7・1024反対変数ペア・8区間・17 Strike節点のATM callでは:

| 項目 | 値 |
|---|---:|
| VegaKTバケット合計 | 23.248972713636547 |
| native local parallel Vega | 23.248972713636544 |
| 射影残差 | 0 |

これは今回追加で確認した評価失敗であり、旧154判定の6件を合格へ変更したものではない。

## 計算と適用条件

シミュレーション時刻、乱数、価格パス、満期点質量の微分は変更しない。
各報告満期 `u` で、実際のLog-Euler計算が使う左端局所分散から

```text
V(u) = sum_i sigma_i^2 * max(0, min(u, t_(i+1)) - t_i)
IV(u) = sqrt(V(u) / u)
```

を求め、報告行の全StrikeのIVと従来と同じ相対許容値 `1e-10` で照合する。
報告満期が時間格子の途中にある場合はその区間の一部分を積算する。
時間格子と一致する場合は、元と同じ累積和を使う。

正の報告満期は2点以上必要で、最後の報告満期は商品の満期と厳密に一致する。
契約満期が報告満期間にあるケースや、契約満期後の報告節点は今回拡張しない。
元の範囲と同じく、全報告行が局所分散と整合しなければ対象外となる。
満期IVだけを照合して、不整合な中間IVを受け入れることはしない。

時間方向のvol変化は扱う。空間方向のsmile、非単位割引、変動forward、配当、
多時点観測、非区分線形ペイオフなどの従来の適用制約は維持する。
日次バックテストの設定形式や市場ショックの定義は変更しない。

## 検証

- Rustの価格統合10件と同一パス点質量テスト1件が成功。今回の追加は2件。
- native Python VegaKT関連38件が成功。今回の追加は3件。
- call/put、一定vol/時間方向のvol変化について、報告日 `[64, 128]`、
  `[11, 43, 128]`、`[1, 127, 128]` を検査した。価格・Delta・local Vega、
  満期バケット、標本分散、価格共分散、全バケット共分散の満期部分は、同じパスを使う
  元の報告格子と完全一致した。満期前のバケットは0を保持した。
- 固定した報告日を使い、時間区間を2・8・16と変えても決定論的な満期測度が
  独立したBlack–Scholesのquote別価格差分と絶対誤差 `2e-8` 以内で一致した。
- RQMCの4スクランブルでも標本分散を保持し、バケット共分散の合計が
  local VegaのSE二乗×4と一致した。単一パスを独立標本と扱っていない。
- worker 1/4、v3 wire往復とJSON Schema、途中再開を含む既存nativeバックテストも成功。
- 固定市場ショックv0.2の307判定・27観測は前回と完全一致して合格。
  旧v0.1の154判定・24観測も完全一致し、6件の失敗を維持した。
- Clippyの警告は0件。nativeライブラリを再構築して検証した。
- 非平坦なLocal VolのmacOS arm64再現結果は、保存済みfixtureと完全一致した。

追加の検証は[Pythonテスト](../../tests/python/test_vega_kt_terminal.py)と
[Rustテスト](../../crates/pricing/tests/local_vol_vegakt_pricing.rs)、修正は
[満期モデルの適用判定](../../crates/pricing/src/engine/risk/vegakt_terminal_pricing.rs)を参照。

## 互換性と記録

既存の計算方式名、公開API、wire形式、受入ポリシー、判定閾値は変えていない。
以前から対象だった入力は同じ演算を使う。新たに対象となった入力には、既存の
方式識別がplan指紋へ入る。native binaryも更新されるため、過去のバックテストの
checkpointを新しいバイナリへそのまま流用することはできない。
`diagnostic_only` と `unvalidated` は維持する。

検証結果とソース指紋は[JSON記録](vega-kt-reporting-maturities-v0.1.json)に保存する。
修正前後の実入力と結果は `benchmark-results/vega-kt-reporting-maturities/before-after.json`、
既存受入の再実行は同ディレクトリの `numerics/report.json` にある。

```shell
cargo test --locked -p pricing --test local_vol_vegakt_pricing
cargo test --locked -p pricing --lib terminal_pathwise
python -m unittest discover -s tests/python -p 'test_vega_kt_*.py' -v
python -m rust_pricing_backtest.market_shock_numerics \
  --policy fixtures/backtesting/market-shock-policy-v0.2.json \
  --output benchmark-results/vega-kt-reporting-maturities/numerics
```

Python検証は今回再構築したnativeライブラリを読み込んで実行する。
