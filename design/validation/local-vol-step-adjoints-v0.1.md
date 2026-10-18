# VegaKTシミュレーション各ステップの随伴保存 v0.1

日付: 2026-10-03  
状態: 本体のreverse sweepに保存処理を実装。独立した価格差分とGamma回復への入力接続を検証。
報告IVバケットへの本体接続・H2/H3受入は未完了。

## 改善した点

従来の [native reverse](../../crates/pricing/src/engine/processes/local_vol.rs) は、
各ステップの有効局所分散への感応度を直ちに元格子へ配賦していました。
複数のシミュレーション時刻が同じ元格子区間に入ると、集約結果から各時刻の感応度を
取り出せず、式(11)に必要な時刻と `delta_t` が失われます。

現在は `LocalVolReverseAdjoints::step_adjoints()` に、実際のシミュレーション区間ごとの
[LocalVolStepAdjoint](../../crates/pricing/src/engine/processes/local_vol/step_adjoints.rs) を
時刻順で保持します。元格子の行数とシミュレーションのステップ数を混同しません。
初期区間も保存します。価格評価のpublic APIが使うreverse sweep自体の変更です。

各項目は、有効局所分散への感応度と、SSRN 4107770の式(10)に対応する有効局所Volへの
感応度を持ちます。空間hatへの配賦に必要な情報も保存しますが、密な全時刻×全Strike行列は
パスごとに生成しません。追加保存量はステップ数に比例します。

`accumulate_local_volatility_nodes` は、その1区間の局所Vol感応度を空間hatへ配賦します。
時間補間の重みやhat面積の除算は適用しません。これにより、式(11)へ渡す前の
「各ステップ・各空間節点の積分済み局所Vega」を構成できます。
形状違反や非有限値はエラーとなり、配賦途中で行を半分だけ変更しません。

元格子への分散随伴配賦と状態／乱数への逆伝播は、正常な有限入力に対する計算順を維持します。
入力seed、伝播後の随伴、格子合計がNaN／無限大になる場合は明示的にエラーを返します。

## 局所Vol感応度の2種類の単位

非平坦な格子では、「ステップで実際に使った局所Vol」と「元格子の節点Vol」が異なります。
元格子節点のVol感応度を再構成するには、時間・空間補間に加えて
`節点Vol / ステップの有効Vol` の比率が必要です。
この比率を省いた単純な再集約は、非平坦ケースで価格差分と一致しません。
式と単位は [計算仕様6.1節](../../docs/models/local-vol-vegakt-calculation-specifications.md) に記録しました。

## 独立した価格差分

[参照スクリプト](../../scripts/check_local_vol_step_adjoints.py) は、100桁Decimalで
Log-Eulerパスとペイオフ価格を再計算し、次の4種類の対象だけを個別にバンプします。
nativeのreverseやその微分公式は使っていません。

- 1ステップの有効局所分散。
- 1ステップの有効局所Vol。
- 1ステップ・1空間hatの局所Vol。
- 元格子の1分散節点。

非平坦・経路依存ペイオフ、平坦・二次ペイオフ、左右の空間外挿の4ケースを用意しました。
元格子の時刻は `0, 0.4, 1`、シミュレーション時刻は `0, 0.1, 0.23, 0.4, 0.7, 1` とし、
配当座標を仮定せず、forward normalizerの時刻依存も含めています。
5点差分の幅 `1e-6` と `5e-7` の一致を確認して [参照値](../../fixtures/local-vol/step-adjoints-v0.1.json)
を固定しました。バンプの影響がない箇所は、対称差分を先に作り厳密なゼロを保ちます。

136個の価格微分との比較はすべて相対誤差 `1e-9`、絶対誤差床ゼロで合格しました。
最大相対誤差は `1.021405183e-14` でした。既存のGamma・本体受入の閾値は変更していません。

保存したステップ分散随伴を元の逆時刻順で格子へ戻す検査は、全4ケースでbit単位の一致を確認しました。
符号付きseed、fixing以降のゼロ感応度、初期感応度の保存、不正形状、非有限値、
配賦エラー時の行の保存も検査しています。

## 各ステップからGamma回復への接続

4096パス、seed 41、平坦Vol 20%、`S0=100`、満期ペイオフ `S_T²/2` を使用しました。
各ステップの局所Volだけを `±1e-5` 動かした共通乱数（CRN）価格差分と、
保存した随伴を空間節点に配賦して平均した行を比較しています。

| source time | delta_t | 平均AD | CRN価格差分 |
|---|---:|---:|---:|
| 0 | 0.10 | 147.323436780930 | 147.323436783811 |
| 0.10 | 0.13 | 233.961075453120 | 233.961075455844 |
| 0.23 | 0.17 | 298.931376452607 | 298.931376459398 |
| 0.40 | 約0.30 | 650.986970905556 | 650.986970920140 |
| 0.70 | 約0.30 | 610.864377504005 | 610.864377520114 |

正時刻の4行を、それぞれの実際の時刻で計算した密度と区間長により
`recover_equation11_gamma_row` へ渡し、入力量、回復時刻、有限なGammaを確認しました。
この検査はGammaそのものの厳密精度や、経路数に対する統計的収束の検査ではありません。
初期行は保持したまま、特異な `t=0` 密度による回復を拒否しています。

## 検証手順

[新規Rust検査](../../crates/pricing/tests/local_vol_step_adjoints.rs) 6件が合格しました。
既存のGamma数値5件、Gamma回復7件、論文演算11件、MC LSV 8件も合格しています。
libのLocal Vol 46件、VegaKT 37件も合格（両者に重複あり）。対象libと4 integration testの
Clippyはwarnings deniedで合格しました。既存の式参照67検査も合格です。

```sh
python scripts/check_local_vol_step_adjoints.py
cargo test --offline --locked -p pricing --test local_vol_step_adjoints -- --nocapture
cargo test --offline --locked -p pricing --test local_vol_gamma_numerics --test local_vol_gamma_recovery --test local_vol_vegakt_paper --test mc_lsv
cargo test --offline --locked -p pricing --lib local_vol
cargo test --offline --locked -p pricing --lib vegakt
cargo clippy --offline --locked -p pricing --lib --test local_vol_step_adjoints --test local_vol_gamma_numerics --test local_vol_gamma_recovery --test local_vol_vegakt_paper -- -D warnings
```

現在の作業ツリーを対象にrelease native wheelを作成し、一時ディレクトリからロードして
Python検査22件の合格を確認しました。共有環境のnative拡張は置換していません。
元の独立数値panelは **154件中54件失敗** のままです。前回panelとの比較では、
全観測値と判定結果が一致し、既存の価格・リスク出力に変化がないことを確認しました。

nativeのビルド、Python検査、独立数値panelの結果とhashは
`benchmark-results/vega-kt-step-adjoints/` に保存しています。panelの出力先は
`benchmark-results/vega-kt-numerics/step-adjoints/` です。
remote CIや全workspaceのテストを実行したという記録ではありません。

## 残る本体接続

各ステップの遷移随伴保存は実装済みですが、本体のIVバケットは引き続き
`local_vol_node_allocation_diagnostic_v1` を使用します。次の作業が残っています。

- 本体のパス集約・不確実性計算に各ステップの行を接続し、実際の全正時刻で密度を準備すること。
  後続の[ステップ集約検証](local-vol-step-recovery-v0.1.md)でRust部品を実装しました。
  価格評価のIVバケットへの接続は引き続き残っています。
- 連続バリアのbridgeなど、遷移以外の直接的な分散感応度を対応する回復基底へ含めること。
  今回のstep tapeにこれらを含めたとは扱いません。
- 初期Gammaの回復、非平滑な分散の微分、満期補間・除外領域の監査。
- 式(4)・(9)へ接続した結果を元の独立数値panelで受け入れること。

今回の6検査の合格だけで、診断モード制限の解除やH2/H3合格とは扱いません。
