# VegaKT原論文の全微分・バニラ配賦検証 v0.1

日付: 2026-10-03  
状態: 式(4)、(5)、(9)のRust部品と独立検証を追加。本体の数値受入は未完了。

## 参照資料と仕様の確定

利用者が提供した *VegaKT for the Local Volatility Model: an AD approach*
（J. Adrienほか、Natixis、May 2022、SSRN 4107770）を参照しました。
17ページのPDFの本文・数式を抽出し、式(3)–(5)の3ページ、式(9)の4ページ、
導出の6ページをレンダリングして確認しました。資料は参照データとして扱い、
原本の変更やリポジトリへのPDF複製は行っていません。

PDF SHA-256: `3230782dc76efbad8477b6db268e8c8cbd67a12c6c5372bbab21cabed3377bbf`。
書誌参照: [SSRN 4107770](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4107770)。

確認した重要な差分は次の3点です。

1. 式(5)は `BS(S,K,v(S))` のSpot全微分です。局所分散の一次・二次微分を
   含むため、既存の凍結分散Gamma核では非平坦な面を表せません。
2. 式(4)の `Delta` は相殺ヘッジのバニラ数量密度です。
   「次時刻Gammaを伝播した値 − 現時刻Gamma」で、終端は `−Gamma` です。
3. 式(9)の数量には「積分StrikeのVega / 報告節点のVega」の比率が入ります。
   ポジションのIVリスクはヘッジ数量と逆符号です。

前回の「本文未取得」という制約は解消しました。
詳細な式・単位・境界は [計算仕様8.1・9節](../../docs/models/local-vol-vegakt-calculation-specifications.md)
に反映しています。

## 追加した部品

- [全微分Gamma核](../../crates/pricing/src/engine/risk/vegakt.rs):
  `integrate_local_gamma_total_variance_transition` は区間分散の
  `[v, dv/dS, d²v/dS²]` を明示的に受け取ります。全微分核は負になることがあり、
  確率密度とは扱いません。微分がともにゼロなら従来関数の結果をbit単位で保ちます。
- [バニラ演算](../../crates/pricing/src/engine/risk/vegakt_paper.rs):
  `vanilla_hedge_density_from_next` は隣接Gamma行を接続し、
  `terminal_vanilla_hedge_density` は終端符号を適用します。
- 同ファイルの `project_vanilla_hedge_measure_to_reporting_iv` は
  `Delta*dK` またはDirac質量を受け取り、continuum Vegaを掛けてからIV基底へ配賦します。
  生のIVリスクと、節点Vegaで割ったヘッジ数量を別々に返します。

満期は報告節点との完全一致を要求します。論文の式(9)はその満期でのStrike補間であり、
満期間の配賦を勝手に追加していません。非ゼロリスクをゼロの節点Vegaで割る場合はエラーです。
配賦残差ゼロは渡された有限の測度の配賦だけを意味し、Gamma回復や除外領域の合格ではありません。

## 独立した数値検査

[生成・検査スクリプト](../../scripts/check_local_vol_vegakt_paper.py) は
nativeライブラリや全微分Gamma公式を呼ばず、120桁Decimalの価格差分を使います。
[固定参照値](../../fixtures/local-vol/vegakt-paper-v0.1.json) にはbinary64入力を正確に取り込みます。

Gamma検査は、指定した区分線形Gammaを2回積分して区分3次の満期ペイオフを構成し、
その対数正規分布下の期待値を計算します。Spotを動かす際には分散も
`v(S+h)=v+v_S*h+v_SS*h²/2` と動かし、5点差分で2階微分を求めます。
相対Spot幅 `1e-6` とその半分の一致を `1e-12` で確認します。

| ケース | Rustと独立価格差分の相対誤差 |
|---|---:|
| 二次ペイオフ・分散勾配あり | 0（binary64表示値） |
| 二次ペイオフ・負の分散曲率 | 0（binary64表示値） |
| 広いセル・符号付きGamma | 2.56e-15以下 |
| 中央の狭いhat | 2.23e-16以下 |
| 右裾の狭いhat | 8.44e-15以下 |
| 左裾 | 2.23e-15以下 |
| 小さい区間分散 | 2.23e-15以下 |
| 負のGamma | 4.45e-16以下 |

この新規参照の受入は相対 `1e-9`、絶対誤差床ゼロです。
既存Gamma核の相対 `1e-10` と本体の数値ポリシーは変更していません。
全微分核に関する一様な誤差上界を主張するものではありません。

IV検査は、節点上の単位コール、節点間のコール、ロング／ショート混合、報告範囲外の
4ケースです。補間されたIVから直接Black–Scholes価格を計算し、各報告IVを
`1e-5` と `5e-6` の幅で5点バンプして、生のバケットリスクと比較します。
単位コールのヘッジ数量 `−1`、満期の局在、Vega比率、端点診断を確認しました。

接続検査では、二次ペイオフの既知のGamma行を式(4)から式(9)まで通します。
中間満期のヘッジ密度は `1e-13` 未満となり、リスクは終端満期にのみ配賦されます。
Strike積分を128、256、512区間へ細分化したScalar Vegaの絶対誤差は
`0.2194081542 → 0.05485073775 → 0.01371260313`、収束次数は約2でした。
これは既知Gammaを入力した接続検査であり、初期Gamma回復や本体の検査とは区別します。

## 実行結果と再現

[Rust検査](../../crates/pricing/tests/local_vol_vegakt_paper.rs) は新規11件が合格。
極短ステップで遥か遠いセルの求積分割数が過大にならないことも検査しています。
旧Gamma核5件・Gamma回復7件も合格しました。既存libのVegaKT 37件、
Local Vol 46件も合格（両者は重複を含む）。対象libと3 integration testのClippyは
warnings deniedで合格しました。3種類のDecimal参照と既存の式参照67検査も合格です。
新参照チェックをCIと貢献者向け手順に追加しました。

```sh
python scripts/check_local_vol_vegakt_paper.py
cargo test --offline --locked -p pricing --test local_vol_vegakt_paper -- --nocapture
cargo test --offline --locked -p pricing --test local_vol_gamma_numerics --test local_vol_gamma_recovery
cargo clippy --offline --locked -p pricing --lib --test local_vol_gamma_numerics --test local_vol_gamma_recovery --test local_vol_vegakt_paper -- -D warnings
```

現在の作業ツリーからrelease native wheelを再ビルドし、共有環境を置換せずに
一時ディレクトリからロードしました。native Python検査22件は合格しました。
元の独立数値panelは **154件中54件失敗** で、診断モード制限を維持しています。
panelは `benchmark-results/vega-kt-numerics/paper-operators/`、
検証ログとhashは `benchmark-results/vega-kt-paper-operators/` に保存しています。
remote CIや全workspaceテストを実行したという記録ではありません。

## 本体受入までの残件

1. 式(10)のシミュレーション時点ごとの局所Vol随伴を保存し、回復へ接続すること。
   その後の [step随伴実装](local-vol-step-adjoints-v0.1.md) で本体の保存処理は追加しました。
   パス集約・bridgeの直接分散seed・本体のIV配賦への接続は残っています。
2. 初期境界。`d(0,S)` はDirac測度であり、最初の1個の局所Vegaから
   `Gamma(t1,K)` の関数全体は一意に回復できません。原論文はこの点の
   実装手順を与えていないため、検証可能な境界条件／推定法が必要です。
3. 非平滑な局所分散。log-strike区分線形補間は節点で微分が跳ぶため、
   弱形式の離散化または滑らかな再構成を定義・検証すること。
4. 回復・Strike積分・満期補間・除外領域・配当座標を本体へ接続し、
   パス／時間／Strikeの収束と実際の満期・Strike局在性を元のpanelで受け入れること。

本体の `local_vol_node_allocation_diagnostic_v1` はこれらの部品をまだ使用していません。
今回の部品検査をもってH2/H3合格や、本体の54件の失敗解消とは扱いません。
