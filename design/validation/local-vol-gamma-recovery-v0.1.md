# VegaKT正時刻Gamma回復・伝播の接続検証 v0.1

日付: 2026-10-03  
状態: Rust部品の接続・独立検証は合格。本体のIV配賦・H2/H3受入は未完了。

## 接続した範囲

[実装](../../crates/pricing/src/engine/risk/vegakt_recovery.rs) の
`recover_equation11_gamma_row` は、1時間区間に対応する局所Volの節点随伴から、
全格子のhat面積による密度化、Strike Jacobian、active domain、式(11)の回復を順に適用します。
戻り値 `Equation11GammaRow` の `propagate_to_source` が、
[修正済みGamma核](local-vol-gamma-numerics-v0.1.md) に回復したGamma行を渡します。

入力は通貨／単位局所Volの積分済み節点随伴です。分散随伴の場合は先に `2*sqrt(variance)` を掛けます。
全格子と一致する長さ、有限の随伴、正の有効分散を検査し、active domainの外側も検査を省略しません。
領域を切り詰める前のhat面積を使用し、除外された局所Vegaを符号付きで別集計します。
入力・残存・除外の3合計を保持し、単位の違うGammaへ残差を足しません。

密度行の元時刻、明示的な `delta_t`、回復先 `source + delta_t` を保持します。
加算がoverflowする場合や、刻みが小さすぎて次時刻を区別できない場合もエラーです。
Gammaの外側は既存仕様どおり端点の定数値を延長しますが、これは除外Vegaの回収ではありません。
active nodeが1点だけなら回復値は参照できますが、伝播は型付きエラーにします。
密度がDirac分布となる時刻0は、正時刻密度行のAPIへ入れられません。

```rust,ignore
use pricing::risk::recover_equation11_gamma_row;

// density_rowと両配列は同じ元格子・同じ正時刻を表す。
let recovered = recover_equation11_gamma_row(
    &density_row, &one_interval_local_vol_adjoints, &effective_variances, delta_t,
)?;
let gamma_at_source = recovered.propagate_to_source(source_strike, source_variance)?;
```

これは新しいRust APIの追加です。既存signature・JSON schema・Python bindingは変更しません。
`SimulationPlan` の公開価格計算は従来の診断用配賦を維持します。
新部品の政策名は既存の `equation_11_first_order_v1`、近似次数は `O(delta_t_k)` です。
論文全体への適合を認定するラベルではありません。

## 独立した価格差分による検査

[生成・検査スクリプト](../../scripts/check_local_vol_gamma_recovery.py) と
[固定参照値](../../fixtures/local-vol/gamma-recovery-v0.1.json) は、
120桁Decimalで条件付きBlack–Scholes価格を直接計算します。
元時刻0.4、満期1.0、区間0.05、Vol 25%、forward 100、payoff Strike 105です。
残り全期間の分散を固定したうえで、対象の1区間だけ局所Volを±0.0001、±0.00002ずらします。
正規CDFにはRustと別の冪級数を使います。

非等間隔の8節点で、条件付き価格差分×周辺密度×Strike Jacobian×hat面積を入力随伴とします。
回復したGammaを独立の解析Gammaと比較し、相対誤差 `1e-7` 以内で合格しました。
ここで比較する解析Gammaは元時刻の値です。式(11)はこれを次時刻の一次近似として解釈するため、
次時刻のGammaとの誤差は別の細分化検査で評価します。次時刻に厳密一致したとは扱いません。
この参照はmass-lumped節点サンプルであり、hat全支持域の厳密積分値ではありません。

## 時間・空間の細分化

[Rust検査](../../crates/pricing/tests/local_vol_gamma_recovery.rs) は、Vol 20%、満期1.0、
元時刻0.4のvanilla callを使用します。時間誤差検査ではlog-moneyness [-1,1]を2,048区間とし、
解析的な条件付き局所Vegaから回復したGammaと次時刻の解析Gammaを比較します。
回復行を伝播した結果は元時刻の解析Gammaと比較します。

| delta_t | Gamma回復の絶対誤差 | 回復後に伝播したGammaの絶対誤差 |
| ---: | ---: | ---: |
| 0.0625 | 1.460140948e-3 | 1.248755460e-3 |
| 0.03125 | 7.000319966e-4 | 6.475549831e-4 |
| 0.015625 | 3.430099474e-4 | 3.299666013e-4 |
| 0.0078125 | 1.698113040e-4 | 1.666089012e-4 |

観測次数は回復1.014–1.061、伝播後0.947–0.986で、事前の許容区間0.85–1.15を満たします。
空間誤差だけの検査は、GBMのGamma半群の解析解（分散の加算）と比較します。
128／256／512区間で誤差はそれぞれ `1.687525342e-5`／`4.221401476e-6`／`1.055512236e-6`、
観測次数は1.9991／1.9998で、許容区間1.8–2.2を満たします。
これは記載した一定Volのケースの証拠であり、任意の非平坦面の収束保証ではありません。

## 実際のnative ADとの接続

`LocalVolLogEulerPlan` の4,096パス、seed 41、時間格子0／0.25／0.5／0.75／1.0、
5 Strike節点、二乗payoffを使いました。時刻0.25の分散随伴を `2*sigma` 倍し、新APIへ渡します。
同じ乱数で、その行の局所Volだけを±1e-5ずらした価格差分と比較しました。

- AD局所Vega: `932.6937386908016`
- CRN価格差分: `932.6937387141342`

回復行からGamma伝播まで実行できることも確認しています。
このケースは元格子とsimulation stepが一致しています。イベント挿入などで格子が一致しない場合、
複数stepから集めた節点随伴を単一 `delta_t` で割ることはできません。
公開経路への接続には、stepごとの随伴保持・再構成も必要です。

## 検証方法と残件

新規7検査が合格し、前回のGamma核5検査、既存VegaKT 37検査、Local Vol 46検査も合格しました。
37件と46件は重複を含みます。Clippyは対象libと両Gamma integration testでwarnings deniedとして合格。
2つのDecimal参照、既存Local Vol式参照67検査も合格しました。新参照チェックをCIへ追加しています。

```shell
python scripts/check_local_vol_gamma_recovery.py
cargo test --offline --locked -p pricing --test local_vol_gamma_recovery -- --nocapture
cargo test --offline --locked -p pricing --test local_vol_gamma_numerics
cargo clippy --offline --locked -p pricing --lib --test local_vol_gamma_numerics --test local_vol_gamma_recovery -- -D warnings
```

release native wheelの再ビルドと、その展開先でのPython連携22検査も合格しました。
既存の数値panelを閾値変更なしで再実行した結果は **154検査中54失敗** です。
panelは `benchmark-results/vega-kt-numerics/gamma-recovery/` にあります。
native binary SHA-256は `f48b72cbc10f45273116e1289ca46cd26af8c914c9203c439e51a95de91856b0`、
wheel SHA-256は `33cd7853d873263313220c1abe3301cca9b36fd95fd79fbbf0abf19ef76003ad` です。
共有環境のnative拡張は置換せず、未コミット変更を含む現在の作業ツリーからビルドしました。

ログ・source/binary hash・再ビルド結果は `benchmark-results/vega-kt-gamma-recovery/` に保存しています。
remote CI・全workspaceのRustテストを実行したという記録ではありません。

この検証時点では原論文の本文未取得が残件でした。その後、利用者から提供されたPDFで
式(3)–(5)・(9)を確認し、[全微分・バニラ配賦部品](local-vol-vegakt-paper-v0.1.md) を追加しました。
非平坦面では式(5)に局所分散のSpot微分が必要であり、ここで検証した
`propagate_to_source` は凍結分散の演算です。初期時刻、stepと元格子の対応、
非平滑な局所分散の微分、実際のIV配賦への接続は残件です。
新部品の合格で既存の54/154件の数値失敗を解決したとは扱わず、診断モード制限を維持します。
