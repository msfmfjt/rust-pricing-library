# VegaKT GammaからIVバケットまでの標本単位接続 v0.1

日付: 2026-10-04  
状態: 空間平坦・forward一定のRust診断APIを接続。一般モデルと本体の数値受入は未完了。

## 実装範囲

[連結計画](../../crates/pricing/src/engine/risk/vegakt_chain.rs)は、条件付き価格からの
初期Gamma、正時刻の式(11)回復、式(4)のヘッジ密度、式(9)のIVバケットを、
1つのパス／antithetic pairごとに計算します。初期行と後続行の平均を別々に射影する
方法では失われる相関を、バケット・ヘッジ数量・合計の標準誤差に含めます。
ラベルは `flat_variance_paper_chain_diagnostic_v1` です。

時刻ごとに異なる分散は許容しますが、空間平坦・forward一定に限定します。
初期計画と正時刻計画のモデル一致、全正時刻と報告満期の一致、各Gamma行の2点以上の
有効Strikeを要求します。密度と報告IVを、実際の各区間の左端分散から積み上げた
離散lognormal分布と相対 `1e-10`・絶対下駄なしで照合します。
この構成検査は既存の数値受入ポリシーを緩和するものではありません。

callbackは、割引済みの状態依存価格とその状態偏微分を返します。
同じcallbackと乱数suffixを初期条件付き再評価でも使います。直接分散依存、bridge、
affine配当再構成、再学習した行使方針は対象外です。callbackにそれらを隠して渡すことや、
価格と一致しない偏微分を渡すことまで自動判定できるAPIではありません。

## 積分と監査

Gammaと報告IVの両方の節点で積分区間を分割し、Strike単位の中点積分を行います。
報告値は各区間 `2q` 点、細分化差は `2q − q` です。初期Gammaの細かい差分幅と
粗い幅の差も、別の標本単位チャネルで集約します。いずれも誤差上界ではありません。

以下は別々に保持します。

- 各報告行の有限Strike区間、回復元の時刻、除外した密度確率。
- 回復から除外した局所Vega。IVバケットやその残差へ足し戻しません。
- 初期Gammaから再構成したVegaとnative初期ADとの差、および相関を含む標準誤差。
- 初期Gamma kernelのtail寄与、次Gammaの定数外挿によるIV寄与。
- 報告Strikeの外側から左右の端バケットへ配賦した符号付きIV感応度。

式(4)は次Gammaの両端を定数外挿します。式(9)は各行の有限区間だけを積分し、
その外側の未知の寄与をゼロ誤差と扱いません。標準誤差は固定した密度・領域・積分の
もとでの標本誤差であり、式(11)の `O(dt)` 誤差、Gamma補間、初期差分、領域打切りの
バイアスを含みません。完全なバケット共分散行列や価格との共分散は未提供です。

## 独立検証

[100桁Decimal参照スクリプト](../../scripts/check_local_vol_vegakt_chain.py)は、
価格差分から局所Vegaを求め、独立のGamma回復・Gaussian級数積分・中点積分・2回走査
モーメントで[参照値](../../fixtures/local-vol/vegakt-chain-v0.1.json)を生成します。
nativeのAD・Gamma回復・射影・統計処理は参照値作成に使いません。
2次項と6次項のペイオフ、3区間、異なる初期／回復／報告Strike格子を用います。

54チャネルについて3パスと3反対変量ペアの平均・標準偏差・標準誤差を照合します。
通常チャネルは相対 `1e-8`、小さな幅差・積分差はそれぞれの単位で絶対 `1e-8` の
丸め誤差予算です。既存の受入閾値は変更していません。

[新規Rust検査](../../crates/pricing/tests/local_vol_vegakt_chain.rs)の内容:

- 独立参照、worker数1／4の再現性、合計分散と周辺分散の和の区別。
- 満期・モデル・密度・報告IVの不一致、1点有効領域、無効な積分数の拒否。
- 計画起源・標本種別の混在、ペア二重化、0標本、不正callbackの拒否。
- 最初のfixingだけの2次ペイオフでの満期局在、符号反転、積分の2次収束。
- nativeの2次ペイオフを全段接続した4096ペアでの統計診断。
- 時変・空間平坦分散と、離散積分分散に整合する密度／IVの受入。

最初のfixingだけのペイオフ `S1²/2` では、最初の満期にだけIVリスクが残ります。
`q=8,16,32` に対する合計Vegaの解析差は、0.464016889、0.116381304、0.029118307。
分割数倍増で誤差がほぼ1/4になります。積分区間 `[10,400]` の解析tailはこの比較の
精度で無視できる大きさですが、一般の有限区間のtail誤差が解決したわけではありません。

| 全段接続例: `ST²/2` | 値 |
|---|---:|
| 条件 | S0=100、Vol=20%、時刻0/0.1/0.2/0.3/0.4、seed=41、4096ペア |
| 合計IV Vega | 832.194852123 |
| 合計の標準誤差 | 40.943564706 |
| 解析値 | 812.902948325 |
| 初期再構成 − native初期AD | −26.503495830 |
| 上の差の標準誤差 | 14.325479410 |
| 合計積分細分化差 | 0.339144886 |

解析差は約2.37%、標準誤差は解析値の約5.04%です。6標準誤差幅の統計整合検査は
満たしますが、これだけで高精度受入や満期／Strikeごとの回復精度を認定しません。
初期再構成とADの差も、以前の保守的な標準誤差の和ではなく、同一標本の差を集約しています。

## 検証実行

新規Rust 6件、既存関連integration 52件、Local Vol lib 46件が合格しました。
対象lib／testのClippyはwarnings deniedで合格、独立参照checker 9本も合格しました。
新しいfixture checkerをCIとCONTRIBUTINGへ追加しています。

作業ツリーからrelease native wheelを作り、一時ディレクトリのnative拡張をロードして
Python検査22件が合格しました。共有環境へはインストールしていません。
本体の数値panelは **154件中54件失敗、exit code 1** でした。前回initial-Gamma panelと
全24観測値・全154判定・受入ポリシーhashが一致しました。新しい連結APIはこの本体panelの
評価経路には入っておらず、Rustの独立参照と統計診断で検証しています。

```sh
python scripts/check_local_vol_vegakt_chain.py
cargo test --offline --locked -p pricing --test local_vol_vegakt_chain -- --nocapture
cargo clippy --offline --locked -p pricing --lib --test local_vol_vegakt_chain --test local_vol_initial_gamma --test local_vol_step_recovery -- -D warnings
```

ログとhash記録は `benchmark-results/vega-kt-chain/`、本体panelは
`benchmark-results/vega-kt-numerics/chain/` に保存しました。
これは対象範囲のローカル検証であり、remote CI／全workspace検査を実行した記録ではありません。

## 本体受入との関係

新APIはRust診断部品です。本体の `local_vol_node_allocation_diagnostic_v1` と
Pythonの `diagnostic_only=True` 条件は維持します。既存54/154失敗を改善したとは
扱いません。非平坦曲面、直接分散／bridgeのGamma分解、報告満期補間、有限領域と
式(11)回復の精度、価格・全バケットの共分散、本体への実装接続が引き続き必要です。
