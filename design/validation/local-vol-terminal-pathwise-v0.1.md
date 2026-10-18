# VegaKTの満期測度を通常評価へ統合 v0.1

日付: 2026-10-04  
状態: 制限付き満期測度を通常の `PricingPlan.evaluate()` に統合。
元の数値受入は **154項目中54件の失敗から6件の失敗** に改善した。
受入ポリシー・許容値は変更していない。全体判定は引き続き `failed`。

## 修正内容と統計量

[前回の満期Gamma測度](local-vol-terminal-measure-v0.1.md)は決定論的な診断APIだった。
今回は同じ符号付きペイオフ点質量を、通常評価の実際の価格パスへ接続した。
商品名による分岐ではなく、コンパイル済みペイオフの区分線形表現を用いる。

```text
f(S) = cash + slope*S + sum_a q_a*(S-K_a)+
V = sum_i sigma_i^2*dt_i
sigma_IV = sqrt(V/T)
D = -V/sigma_IV + sum_i sigma_i*sqrt(dt_i)*Z_i/sigma_IV
atom_sample_a = q_a * 1(S_T>K_a) * S_T * D
bucket_sample_j = sum_a atom_sample_a * IV_hat_j(log(K_a/S0))
```

全区間のvolを `sigma_IV'/sigma_IV` 倍する同一乱数微分に対応する。
中間満期は、同じ満期測度を伝播する半群性からヘッジ測度が相殺され、0となる。
満期の各点質量をその物理StrikeでIVのlog-hatへ投影する。
ゼロキャリーの対象範囲ではaffine部分の期待IVリスクは厳密に0なので、その既知の
期待値を使う。putのIV標本もcall複製部分から作り、価格標本には元のputを使う。
したがってputとcallのIV標本は一致しても、価格との共分散は商品ごとに異なる。

新しい計算方式名は `flat_variance_terminal_measure_pathwise_v1`。
`truncation_order` は `not_applicable`。解析VegaによるMC平均の置換、合計値への
倍率合わせ、残差の配り直し、人工的な分散・共分散は行っていない。
各パスのIV微分を、既存の反対変数ペアとRQMCのスクランブル平均の集計に渡し、
その標本からバケット分散、価格との共分散、指定時の全バケット共分散を計算する。

`pre_projection` は投影前の点質量標本の独立した合計であり、報告バケットの和や
local parallel Vegaから作らない。バケットを意図的に壊す回帰検証でも整合性エラーを
確認した。今回の測度には初期時刻・密度切捨ての除外がないので残差は0。
端点のhatは定数外挿で全Strikeを覆い、範囲外の感応度は端点配賦として別途監査する。

既存のlocal parallel Vegaは元のAD計算を保持する。時間方向にvolが異なる場合や
affine部分を持つ場合、IV parallel Vegaとは一般に異なるので、差を残差へ埋めない。
元の24ケースの価格・Delta・local Vegaとそれぞれの標準誤差は、前回の結果と
binary64で完全一致した。適用対象のplan指紋だけに新方式名を加え、古い計算計画との
取り違えを防止する。要求指紋および対象外のplan指紋は維持する。

## 適用範囲

単一原資産・単一満期の連続区分線形ペイオフ、空間平坦な局所分散、単位割引、
一定forward、恒等配当座標が対象。時間方向のvol変化は扱う。報告満期は実際の全正時刻
シミュレーション節点に一致し、報告IVは左端局所分散の積算と整合する必要がある。
call・put・符号付きspread・線形相殺を同じペイオフ変換で扱う。

非平坦なsmile、多時点観測、非区分線形ペイオフ、配当、smoothing、barrier、
早期行使、Bassモデルは対象外。対象外は従来の明示的な
`local_vol_node_allocation_diagnostic_v1` へ戻る。
対象内での数値不正・計算量超過・連続性の破れはエラーとし、黙って戻さない。
一般のノイズ付きADからのGamma回復を解決した実装ではない。

## 元の受入154項目

| 検証 | 前回の失敗 | 今回の失敗 |
|---|---:|---:|
| バケット合計とnative local parallel Vega | 24 | 0 |
| 投影残差 | 24 | 0 |
| バケット合計と同一乱数の価格差分 | 6 | 0 |
| 格子変更時のヘッジ数量 | 0 | 6 |
| その他 | 0 | 0 |
| 合計 | 54 | 6 |

以前失敗していた54判定は全て通過した。バケット合計とnative local Vegaの最大差は
`3.552713678800501e-15`。ただしこれは一定volのcallでの結果であり、一般に両者を
同一視する根拠にはしない。新しく顕在化した6件を除外せず保持する。

| Seed | Strike節点数 | 17節点に対する数量の相対差 | 許容値 |
|---|---:|---:|---:|
| 7 | 9 | 0.0539562041 | 0.05 |
| 7 | 33 | 0.9807645049 | 0.05 |
| 41 | 9 | 0.0540244838 | 0.05 |
| 41 | 33 | 0.9807246287 | 0.05 |
| 97 | 9 | 0.0539678320 | 0.05 |
| 97 | 33 | 0.9807724549 | 0.05 |

## 残る6件の原因と解消方針

元のヘッジ最適化は「各報告節点のIVが独立に動く」という単位重みの二乗誤差を
最小化する。節点を増やすと市場ショックの集合も変わる。ATM100のtargetに対し、
hedge Strikeは95と105。細かい格子では、それぞれのIVリスクの支配する節点が
分離するため、最適数量が大きく変わる。

nativeを使わず、独立なBlack–Scholes Vegaとlog-hatでリスクを作っても同じ6件が
失敗した。乱数seedに依存しない解析参照の数量は次の通り。

| Strike節点数 | K95数量 | K105数量 | 17節点からの相対差 |
|---|---:|---:|---:|
| 9 | 0.6686539050 | 0.7088506892 | 0.0539031960 |
| 17 | 0.6925535306 | 0.7588043905 | 0 |
| 33 | 0 | 0.0269213442 | 0.9808036416 |

切り分けのため、共通の9節点市場ショックを固定し、細かい格子上でその同じショックを
補間した。細格子IVから共通市場IVへの微分行列をHとすると、共通リスクは `r_fine*H`。
リッジ、数量制約、ソルバー許容値は元の設定を使った。3seed・3格子で、nativeの
共通リスクの最大絶対差は `3.552713678800501e-15`、数量の最大相対差は
`1.0785308569744714e-15` だった。

この比較はヘッジ対象を共通ショックの部分空間に限定した診断である。
細格子の独立ショックをヘッジできる証拠でも、元の6失敗を合格へ変更する根拠でもない。
次の受入仕様では、市場のquote/shock basisと数値報告格子を別に定義し、同じ市場
ショックでの格子安定性と、追加された独立ショックへの残余リスクを別々に評価する。
元のポリシーは保存し、仕様を確定した別バージョンで検証する。

## 検証範囲と再実行

Rust関連123件（integration 76、local-vol library 46、符号付きspread標本1）、
Python VegaKT関連30件、native smoke 2件が成功。call/put、一定vol/時間方向のvol変化、
反対変数の有無、Brownian bridgeの有無、worker 1/4、RQMCの分散を検査した。
独立な同一パス価格差分からバケット平均・分散・価格共分散・全共分散を照合した。
v3 wire往復とJSON Schema検証、既存の非平坦local-vol replayのbitwise照合、Clippy、
Python型定義とnative公開APIの厳密照合も成功した。

決定論的な満期測度の独立参照96判定も失敗0件。過去に記録した「正確な期待値とMC値を
標準誤差なしで3%比較すると1件失敗」の反例はそのまま残す。今回の通常レポートは
同一標本のIVリスクを使うので、その決定論的な反例で通常受入を置き換えていない。

初期時刻の欠落を調べる旧allocation probeは旧方式専用とした。新方式には
`not_applicable`・終了コード2を返し、0判定を合格と見なさない。
バックテストadapterの `diagnostic_only` と `unvalidated` は維持する。

```shell
python -m rust_pricing_backtest.numerics \
  --policy fixtures/backtesting/numerical-policy-v0.1.json \
  --output benchmark-results/vega-kt-numerics/terminal-pathwise
# 元の判定は6件失敗のため終了コード1
python scripts/check_vega_kt_terminal_measure.py \
  --output benchmark-results/vega-kt-terminal-pathwise/deterministic-reference.json
python scripts/analyze_vega_kt_hedge_grid.py \
  --output benchmark-results/vega-kt-terminal-pathwise/hedge-grid.json
python -m unittest discover -s tests/python -p 'test_vega_kt_*.py'
```

元の受入観測・独立比較・ソース指紋・失敗6件は[検証記録](local-vol-terminal-pathwise-v0.1.json)
に保存した。実装は[同一パス満期IV](../../crates/pricing/src/engine/risk/vegakt_terminal_pathwise.rs)、
検証は[価格評価テスト](../../crates/pricing/tests/local_vol_vegakt_pricing.rs)と
[格子比較](../../scripts/analyze_vega_kt_hedge_grid.py)を参照。
