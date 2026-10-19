# VegaKT Gamma伝播の数値検証 v0.1

日付: 2026-10-03  
状態: Gamma伝播部品の回帰検査は合格。VegaKT本体のH2/H3数値受入は未合格。

## 修正した問題

Rust本体の [Gamma伝播処理](../../crates/pricing/src/engine/risk/vegakt.rs) に、
独立参照値で再現する数値誤差がありました。

1. 正規分布の区間確率で、下側CDF差と上側確率差の小さい方を選んでいました。
   両CDFが1に丸まると、表現可能な裾確率まで0になります。
2. 最終セルを `1 - それ以前の確率合計` で求めるため、小さな右裾が失われていました。
3. `log(K) - (log(S) + 1.5*v)` と近接節点の差によって、短い時間刻みでは
   ドリフトと区間幅の精度が低下していました。線形Gammaの切片・一次モーメント式も、
   狭いStrike間隔では大きな項同士の差となり、桁落ちしました。
4. 有限の入力から一次モーメントがoverflowしても、成功として返していました。

正側区間は上側確率差、負側区間は下側確率差を使用し、両端セルも直接計算します。
確率合計の丸め誤差だけを最大確率のセルに戻すため、微小な裾を消しません。
正規裾確率にはLaplaceの連分数を使い、近接Strikeには `log1p`、狭い正規区間には
Gaussian密度の中点展開を使います。桁落ちする線形Gamma積分は、`expm1` で
Strikeの補間重みを計算し、正規座標の8点Gauss–Legendre積分に切り替えます。
非有限モーメントは既存の型付きエラーで拒否します。

分岐条件・積分パネル数・丸め処理は
[計算仕様8節](../../docs/models/local-vol-vegakt-calculation-specifications.md) に記録しました。
正規確率の定義と連分数は [NIST DLMF §7.2](https://dlmf.nist.gov/7.2) と
[§7.9](https://dlmf.nist.gov/7.9) に基づきます。

## 独立参照値と回帰検査

[参照値生成・検査スクリプト](../../scripts/check_local_vol_gamma_numerics.py) は、
120桁のPython Decimalで正規密度の収束冪級数を積分します。
Rust実装、SciPy、Rustと同じCDF近似・連分数・数値積分を使用しません。
入力はbinary64の値を厳密にDecimalへ変換し、出力だけをbinary64へ丸めます。
120桁は計算精度の設定であり、参照値120桁すべての保証を意味しません。

[固定参照データ](../../fixtures/local-vol/gamma-numerics-v0.1.json) は、中心、左右の裾、
分散 `1e-20` の短時間、狭い中心hat、狭い右裾hat、計算方法の境界付近のhatの8ケースです。
31セルの確率・一次モーメントと8個のGamma積分を、相対誤差 `1e-10`、絶対許容値0で比較します。
微小な期待値を絶対誤差の下限で見逃すことはありません。

| 観測対象 | 独立参照値 | 修正前 | 修正後 |
| --- | ---: | --- | --- |
| 右裾の有限セル確率 | `1.3251756975695472e-19` | 0、不合格 | 相対誤差 `1e-10` 以内 |
| 右裾ケースのGamma積分 | `2.8419445482866904e-20` | 0、不合格 | 相対誤差 `1e-10` 以内 |
| 最右端の非有界セル確率 | `5.385283668034879e-24` | 補集合計算で消失する対象 | 相対誤差 `1e-10` 以内 |
| 短時間ケースのGamma積分 | `2.315626908529386` | 桁落ちする対象 | 相対誤差 `1e-10` 以内 |
| 有限入力でのモーメントoverflow | 型付きエラー | 成功として返却、不合格 | エラーとして拒否 |

[Rust受入テスト](../../crates/pricing/tests/local_vol_gamma_numerics.rs) は、参照値との比較に加え、
符号・倍率への整合、非負Gamma、定数Gammaの `exp(v)` 保存、通貨単位の
`1e-100`～`1e100` への変更を検査します。最初の3検査が修正前に失敗することを確認し、
修正後は追加2検査を含む5検査が合格しました。
境界付近のhatも追加し、中間実装で相対誤差 `1.0822e-8` の失敗を確認したうえで、
積分方法の切替条件を修正しました。元の6ケースの参照値・許容誤差は変更していません。
参照データ再計算を既存CIへ追加しました。通常実行はファイルを変更せず、
`--write` を指定した場合だけ参照データを再生成します。

## ローカル検証

環境はmacOS arm64、Rust 1.98.1、Python 3.13.1、NumPy 2.5.3です。

| 検査 | 結果 |
| --- | --- |
| 新しいGamma独立参照 | 8ケース、70個の数値比較が合格 |
| 新しいRust回帰テスト | 5 passed |
| 既存VegaKT部品テスト | 37 passed |
| Local Vol関連libテスト | 46 passed。上記37件と重複あり |
| 既存Local Vol式参照チェック | 67 checks passed |
| 変更範囲のClippy | passed、warnings denied |
| release native wheel | build passed |
| 文書リンク・変更ファイルのformat・diff空白 | passed |
| 作業ツリーのsource archive | passed。companionのソース・Schemaも検査対象に追加 |
| 新しいnativeでのPython連携テスト | 22 passed |
| 既存の本体数値panel | **154検査中54失敗** |

```shell
python scripts/check_local_vol_gamma_numerics.py
python scripts/check_local_vol_reference_fixture.py
cargo test --offline --locked -p pricing --test local_vol_gamma_numerics
cargo test --offline --locked -p pricing --lib vegakt
cargo test --offline --locked -p pricing --lib local_vol
cargo clippy --offline --locked -p pricing --lib --test local_vol_gamma_numerics -- -D warnings
python -m maturin build --release --locked --offline --interpreter .venv/bin/python --out /tmp/vega-kt-gamma-native-wheel
python -m unittest discover -s tests/python -p test_vega_kt_backtest.py
python -m rust_pricing_backtest.numerics --policy fixtures/backtesting/numerical-policy-v0.1.json --output benchmark-results/vega-kt-numerics/gamma-stability
```

最後の2コマンドは、新しくビルドしたwheelの展開先とcompanion・依存ライブラリを
`PYTHONPATH` に指定して実行しました。共有環境のnative拡張は置換していません。
新しいwheelのSHA-256は
`f0502360406e9c0a0ba777df102203d8778e10084e74e398d5419dbebee957e0`、
実行したnativeバイナリのSHA-256は
`811d2ee04b4bbadad78c418f1df280b66a2182e083943879cd874ff3f88505e2` です。
ビルドは未コミット変更を含む現在の作業ツリーから行いました。
テストログとハッシュは `benchmark-results/vega-kt-gamma-numerics/`、
本体panelは `benchmark-results/vega-kt-numerics/gamma-stability/` に保存しています。
全workspaceのRustテスト・remote CIの実行結果ではありません。

## 互換性と受入の境界

連続時間のカーネル、区分線形Gamma、公開関数signature、wire formatは変更しません。
有限精度アルゴリズムの欠陥修正であり、修正対象の部品出力は過去のbinary64結果から変わります。
共有の正規CDFやMCの計算順序は変更しません。過去の参照データを上書きして
合格させることもしていません。

この検査は8ケースと記載した不変量の保証であり、任意の極端入力、適応積分の誤差上界、
全面的なVegaKT論文整合を保証するものではありません。

本体の公開実行経路は、引き続き局所Volの節点随伴を直接IVバケットへ配賦しています。
修正したGamma伝播部品と式(11)のGamma回復は、その経路にまだ接続されていません。
したがって、今回の修正だけでは [既存panel](vega-kt-backtest-numerics-v0.1.md) の失敗数は減りません。
seed=7、4,096 sampling units、8時間区間、17 Strike節点では、Local Vegaは23.578941、
バケット合計は20.747075、残差は2.831866（12.0101%）です。
合否閾値3%は変更していません。

`local_vol_node_allocation_diagnostic_v1` / `unvalidated` とcompanionの診断モード制限を維持します。
過去の [L0–L8受入記録](local-vol-vegakt-conformance-v0.1.md) にも、後続の独立検証が
その受入判断を覆したことを追記しました。

残件は、原論文の式と初期時刻の扱いを照合したGamma回復・伝播・IV配賦の接続、
満期／Strikeの局在性、非平坦面、時間・空間細分化、ヘッジ誤差収束です。
今回 [SSRN原論文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4107770) と
[出版元](https://www.risk.net/cutting-edge/7955239/vega-decomposition-for-the-lv-model-an-adjoint-differentiation-approach)
から本文を取得できず、接続部分の式は未照合です。残差の任意配賦では代替しません。
