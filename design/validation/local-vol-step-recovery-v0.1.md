# VegaKTステップ集約・Gamma標準誤差の検証 v0.1

日付: 2026-10-03  
状態: 実際のシミュレーション時刻での集約・正時刻Gamma回復をRust部品として実装。
価格評価のIVバケットへの接続・H2/H3受入は未完了。

## 改善した点

[前段のステップ随伴保存](local-vol-step-adjoints-v0.1.md)を
[VegaKtStepRecoveryPlan](../../crates/pricing/src/engine/risk/vegakt_steps.rs)へ接続しました。
元の分散格子に存在しないイベント時刻も含め、実際の各区間の開始時刻で密度と
有効局所分散を準備し、局所Vegaの平均から式(11)の次時刻Gammaを回復します。
初期区間の局所Vegaは保存し、特異な初期密度によるGamma回復は行いません。

計画と異なる時刻・補間・forwardで生成されたパスや、異なる計画の標本の混在を拒否します。
初期状態は計画の初期forwardと一致する必要があります。呼出側は、入力IV曲面が
その座標とシミュレーション対象モデルに対応することを保証します。
この部品によって曲面と局所分散の較正整合性が証明されるわけではありません。

## 標準誤差の扱い

既存の決定論的executorを使い、固定ブロックのNeumaier加算と二分木の中心化モーメントで
集約します。独立パスとantithetic pairを明示的に区別し、ペアは2パスを先に平均して
1つの独立標本と数えます。反対符号の乱数生成は呼出側の責務です。
1標本では標本分散・標準誤差は `None`、0標本はエラーです。
RQMCの各点を独立パスと数える使い方は対応しません。

節点だけでなく、各時刻の行合計、密度の採用領域・除外領域の合計、全時刻合計を
標本ごとに計算してから集約します。したがって、それぞれの合計の標準誤差には
節点間・時刻間の共分散が含まれます。節点分散を単純に加算する方法は使用しません。
全共分散行列と価格との共分散は今回のAPIには含めていません。

正時刻のGamma標準誤差は、節点の局所Vega標準誤差を式(11)と同じ正の係数で割ります。
これは密度・曲面・採用領域を固定した条件付きMC誤差です。
時間・空間離散化、較正誤差、初期境界の誤差を含む総合精度とは扱いません。

疎なstep tapeとは別に、集約時の各標本にはステップ数×空間節点数の密なベクトルを
一時的に構成します。全パスを保存せず、標本と集約ブロック単位で処理します。

## 独立参照と検査

[参照スクリプト](../../scripts/check_local_vol_step_recovery.py)は100桁Decimalで
10パスの価格を再計算し、5区間×3空間hatを個別にVolバンプします。
5点差分の幅 `1e-6` と `5e-7` の収束を確認し、native reverseを使わず微分を求めます。
5本のprimary pathと5組のantithetic pairについて、独立した2回走査方式で平均・分散・
標準誤差を計算しました。平坦曲面の解析密度と全格子hat面積からGamma参照を計算しています。
価格には複数fixingの二次項、符号付き係数、時刻間の積を含めています。

[固定した参照値](../../fixtures/local-vol/step-recovery-v0.1.json)との比較は
相対許容誤差 `1e-9`、絶対誤差床ゼロです。既存の数値受入閾値は変更していません。

[新規Rust検査](../../crates/pricing/tests/local_vol_step_recovery.rs)8件が合格しました。

- 独立パス・ペアの節点／合計モーメント、Gamma平均・標準誤差のDecimal参照比較。
- worker数1と4でbit単位一致。5標本・ブロック幅2で最後の不完全ブロックも検査。
- 行合計と全時刻合計に含まれる共分散、および標本数0／1の扱い。
- 計画・標本種別の混在、ペアの二重化、異なる時刻／分散／初期状態、不正seedの拒否。
- 1区間だけの計画、初期行だけの場合の閾値検証、有限標本からの分散overflowの拒否。
- 4,096組のantithetic CRN価格バンプから集約・Gamma回復までの接続。

最後の検査は平坦Vol 20%、`S0=100`、seed 41、ペイオフ `S_T²/2` です。
各区間のVolだけを `±1e-5` 動かした価格差分と、随伴の行平均・標準誤差を比較しました。
行平均は解析期待値 `S0² * sigma * delta_t * exp(sigma²*T)` の6標準誤差以内です。

| source time | 平均AD | CRN価格差分 | ペア単位の標準誤差 |
|---|---:|---:|---:|
| 0 | 262.657566010421 | 262.657566014451 | 21.865164178825 |
| 0.10 | 258.034962531757 | 258.034962535862 | 25.003579241885 |
| 0.23 | 355.261624683581 | 355.261624692045 | 29.688616668523 |
| 0.40 | 652.047271856492 | 652.047271871274 | 40.230383561217 |
| 0.70 | 690.499163182724 | 690.499163206795 | 42.745754537993 |

この表は遷移Vegaの価格微分と集約の検査です。回復Gammaの厳密精度や
時間・空間離散化の収束がこの表だけで証明されたとは扱いません。

## 再現手順

```sh
python scripts/check_local_vol_step_recovery.py
cargo test --offline --locked -p pricing --test local_vol_step_recovery -- --nocapture
cargo test --offline --locked -p pricing --test local_vol_step_adjoints --test local_vol_gamma_numerics --test local_vol_gamma_recovery --test local_vol_vegakt_paper
cargo clippy --offline --locked -p pricing --lib --test local_vol_step_recovery --test local_vol_step_adjoints --test local_vol_gamma_numerics --test local_vol_gamma_recovery --test local_vol_vegakt_paper -- -D warnings
```

既存のGamma数値5件、Gamma回復7件、論文演算11件、ステップ随伴6件も合格しました。
libのLocal Vol 46件、VegaKT 37件も合格（フィルタに重複あり）。上記Clippyは
warnings deniedで合格です。既存5種類と新規の独立参照checkerも合格しています。

現在の作業ツリーからrelease native wheelを再ビルドし、一時ディレクトリからロードして
Python検査22件を通しました。元の独立数値panelは **154件中54件失敗** のままです。
前段のstep-adjoints panelと比較し、全24観測値と全154判定が一致しました。
受入ポリシーと診断モード制限は変更していません。

ビルド・テストログとhash記録は `benchmark-results/vega-kt-step-recovery/`、
panelは `benchmark-results/vega-kt-numerics/step-recovery/` に保存しています。
共有環境のnative拡張は置換していません。これは対象部品のローカル検証であり、
remote CIや全workspaceの実行結果ではありません。

## 残る本体接続

この部品は遷移の状態随伴を対象とします。連続バリアbridgeなどの直接的な分散seed、
初期Gammaの回復、非平滑な分散の微分、満期補間、式(4)・(9)への接続が残ります。
後続の[直接分散感応度の検証](local-vol-direct-variance-v0.1.md)で、分散seedの保存と
統計集約を追加しました。直接寄与を含むGamma回復は引き続き未検証です。
本体IVバケットは引き続き `local_vol_node_allocation_diagnostic_v1` です。
今回の検査だけで診断モード制限を解除したり、H2/H3合格と判断したりしません。
