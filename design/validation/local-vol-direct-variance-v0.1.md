# VegaKT直接分散感応度・バリア端点の検証 v0.1

日付: 2026-10-04  
状態: nativeバリア評価の分散seedを共通reverseへ接続し、全時刻の直接寄与を保持・集約。
直接寄与を含むGamma回復・本体IVバケットの論文分解は未完了。

## 実装

[共通reverse](../../crates/pricing/src/engine/processes/local_vol/direct_adjoints.rs)に
`reverse_path_with_variance_seeds` を追加しました。状態と有効局所分散それぞれへの
偏微分seedを受け取り、分散の状態依存を1回だけ状態seedへ加えて逆伝播します。
直接分散seedは、空間hatと実際の時刻を持つ別のtapeにも保存します。
入力はN区間に対しN+1時点分であり、初期時点と満期を含みます。

本体のexact／smoothed連続バリアは、両端の分散seedをこのreverseへ渡します。
従来の格子配賦だけで消えていた時刻別の直接寄与が残るようになりました。
状態seedには分散を固定した偏微分を渡します。呼出側で分散からの状態feedbackを
先に加えると二重計上になるため、APIコメントでこの契約を明示しています。

`VegaKtStepRecoveryPlan::sample_from_path_with_variance_seeds` から統計集約へ渡せます。
`direct_variance_rows()` はN+1時点の節点・行合計モーメント、`combined_total()` は
遷移と直接寄与の合計モーメントを返します。後者は各標本で合計してから集約するため、
両者の共分散を含みます。既存の `total()` は遷移分を返すままです。

## Gammaを返さない条件

bridgeの分散には区間両端の平均を使うため、最後の監視時点にも直接寄与があります。
監視終了が満期の場合、その寄与には次のシミュレーション区間がありません。
前区間へ移したり、任意の区間長で式(11)へ代入したりする根拠は未検証です。

非ゼロの直接分散seedが1つでもあれば `has_unrecovered_direct_variance()` をtrueとし、
すべてのGamma平均・標準誤差を `None` とします。これは符号付き平均と別に集約するため、
パス間／antithetic pair内の相殺によって未回復の状態が消えることはありません。
局所Vegaの平均・標準誤差と、満期端を含む直接寄与は取得できます。
直接seedがすべてゼロなら、前段の遷移のみのGamma回復を維持します。

## 独立価格差分

[参照スクリプト](../../scripts/check_local_vol_direct_variance.py)は100桁Decimalで
Log-Eulerパスとbridgeの生存確率積を価格評価し、5点差分を `1e-6` と `5e-7` の
2幅で検証します。nativeのbridge微分、reverse、統計処理を参照計算に使いません。
これは採用中の離散化方式の微分検証であり、連続時間バリア価格への収束保証ではありません。

上側ノックアウト、下側ノックイン、満期より早い監視終了、平坦分散、接触済みの5ケースで、
次を合計215個の独立価格微分と比較しました。リベートと割引も含みます。

- 各遷移・空間hatの局所Vol感応度: 15個／ケース。
- 全N+1時点・空間hatの直接局所Vol感応度: 18個／ケース。
- 元格子の分散節点感応度: 9個／ケース。
- 遷移・直接寄与の両方を動かす平行局所Vol感応度: 1個／ケース。

さらに5本のprimary pathと反対符号の5本について、全体、遷移合計、直接寄与6行の
計80微分を価格差分で計算しました。独立した2回走査方式のモーメントから、
5独立パス／5独立ペアの平均・標本分散・標準誤差を固定しています。

全比較の相対許容誤差は `1e-9`、絶対誤差床はゼロです。
価格差分の参照値は [fixture](../../fixtures/local-vol/direct-variance-v0.1.json) に保存しました。
既存の本体受入閾値は変更していません。

## Rust検査

[新規integration test](../../crates/pricing/tests/local_vol_direct_variance.rs)6件が合格しました。
価格差分に加え、次を検査しています。

- 従来のbridge格子随伴とのbit単位の一致と、状態feedbackを省いた場合の差。
- 初期・満期・監視終了時点の直接寄与、および監視終了後のゼロ寄与。
- 遷移／直接寄与の共分散、antithetic標本数、worker数1／4での再現性。
- 符号付き寄与の完全相殺時にもGammaを返さないこと。
- ゼロ直接seedでの従来結果の保存、不正な配列長、NaN、無限大、overflow、異なる計画の拒否。

局所Volへの変換では `seed * (2 * sqrt(variance))` とし、`2 * seed` の中間overflowを
避けます。例えば分散0.04、seed `1e308` の有限な結果 `4e307` を保持する検査も含みます。

```sh
python scripts/check_local_vol_direct_variance.py
cargo test --offline --locked -p pricing --test local_vol_direct_variance
cargo test --offline --locked -p pricing --lib local_vol
cargo test --offline --locked -p pricing --test local_vol_step_recovery --test local_vol_step_adjoints --test local_vol_gamma_numerics --test local_vol_gamma_recovery --test local_vol_vegakt_paper
cargo clippy --offline --locked -p pricing --lib --test local_vol_direct_variance --test local_vol_step_recovery --test local_vol_step_adjoints -- -D warnings
```

既存の関連integration test 37件、Local Vol lib 46件、MC LSV 8件も合格しました。
Local Volのlib検査にはexact／smoothedバリア、配当イベント、価格バンプとの比較を含みます。
上記Clippyはwarnings deniedで合格です。新規fixture checker、既存の式参照67検査、
ステップ随伴・ステップ集約の独立checkerも合格しています。

最終コードでrelease native wheelを作成し、一時ディレクトリからロードしてPython検査
22件が合格しました。本体の独立数値panelは **154件中54件失敗** のままです。
前段のstep-recovery panelと比べ、全24観測値、全154判定、受入ポリシーhashが一致しました。

ログとhash記録は `benchmark-results/vega-kt-direct-variance/`、panelは
`benchmark-results/vega-kt-numerics/direct-variance/` に保存しています。
共有環境のnative拡張は置換していません。remote CIや全workspace検査を実行したという
記録ではなく、今回変更した部品に関するローカル検証です。

## 残る受入条件

本体の報告IVバケットは `local_vol_node_allocation_diagnostic_v1` のままです。
今回の変更で直接寄与を落とさず監査できるようになりましたが、直接寄与と初期境界の
Gamma分解、非平滑な分散の微分、式(4)・(9)への接続はまだ必要です。
この検証をもって診断制限を解除したり、H2/H3合格と判断したりはしません。
