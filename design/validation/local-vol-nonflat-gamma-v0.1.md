# 2区間・非平坦な第2区間のGamma回復 v0.1

2026-10-05、macOS aarch64、release native wheel / Python 3.13。

固定Gaussian第1分布と非平坦な第2区間について、**1,914 / 1,914チェック合格**。
2種類のスマイル×4商品×2 seedの全16設定が、独立した評価標本で品質判定を通過した。
第2区間の回復を追加したもので、非平坦な将来を持つ初期Gamma、一般多区間、
任意の第1満期スマイルを含む全面的なVegaKT受入は未完了である。

[機械可読記録](local-vol-nonflat-gamma-v0.1.json)、
[固定ポリシー](../../fixtures/backtesting/nonflat-gamma-policy-v0.1.json)、
[検証プログラム](../../scripts/check_nonflat_gamma_recovery.py)。
詳細な全判定・全pilot・評価結果・価格差分は
`benchmark-results/vega-kt-nonflat-gamma/report.json` に保存した。

## 修正した座標と作用素

非平坦な局所ボラでは、実効ボラ感応度を局所gridへhat配賦した値を
各節点の `2 sigma_j` で割っても、年率分散節点の微分にはならない。
実際に較正される年率分散節点 `v_j` に対する経路ADを直接入力する。

第1区間は一定年率分散であり、`X1 = log(S1/S0)` は `N(-V1/2,V1)`。
第2区間の `v(X1)` はnativeと同じ線形補間・境界一定延長とする。
終端Gammaの符号付き原子質量 `q_k` に対応する作用素は

```
B[j,k] = E[hat_j(X1) * S1 * phi(d1) * dt / (2 sqrt(v(X1) dt))]
d1     = log(S1/K_k) / sqrt(v(X1) dt) + sqrt(v(X1) dt)/2
b_j    = d(payoff price) / d(annual variance node j)
```

である。終端分布をGaussianへ置き換えない。
第1分布の±12標準偏差を局所gridで分割して32点・64点Gauss–Legendreを実施し、
各作用素要素の最大差と解析的な省略tail上界を個別に `1e-10 * max(B)` 以下に要求する。
32/64点差は収束診断であり、厳密な積分総誤差上界ではない。
独立96点積分との最大差は **4.885e-14**。
極端に狭い遷移や大きな第1分散でこの精度確認を満たさなければ、明示的に失敗させる。

年率分散の較正接線 `U = d(v_j)/d(terminal quote IV)` の転置を方向行列Dとし、
`D B q = D b` を満たす固定の制約付き正則化逆行列を使う。
平坦APIにおける `U.T/(2 sigma)` の換算は適用しない。
制約は指定方向の感応度を保存するもので、経路依存Gamma全体を一意に同定するものではない。

## 適用条件と失敗の扱い

新APIは `evaluate_vega_kt_two_step_smile_gamma_audit`。
実際のシミュレーション時刻と局所分散gridが一致する2区間、空間一定の第1行、
zero carry・unit discount、連続したstate-onlyペイオフに限定する。
指示関数、バリア、早期行使、配当、smoothing、Bass、RQMCは対象外。

- 出力座標は `annual_local_variance_nodes`。
- `variance_conservation_directions` に固定方向を渡す。旧policyのボラ感応度保存オプションは拒否する。
- 正則化候補はpilotで選択し、held-outデータで一度だけ検証する。失敗後の候補変更は行わない。
- `estimate` は合格時の指定方向の感応度。不合格なら `None`。方向なしの場合、方向別配列は空。
- 全空間残差、非ゼロ標本数、方向感応度・差の実測分散を保持する。
- 初期年率分散ADは `unrecovered_initial_variance_sensitivity` として保持し、初期Gammaは `not_implemented`。

初期Gammaの従来の条件付きaffine continuationは、将来の局所分散が状態依存なら
成立しないため呼び出さない。通常価格、既存flat診断、受入済みP0 providerは変更しない。
全面IV受入は `not_established` のままとする。

## 独立検証と固定した判定基準

第1満期64日、終端128日、spot 100。局所grid33点、終端quoteのlog-moneynessは
`[-0.2,-0.1,0,0.1,0.2]`。flat 20%モデルを基準に、以下の終端IV変化へ
非線形再較正した2種類のモデルを使った。第1分布は固定する。

| モデル | 終端IVの変化 |
| --- | --- |
| 上向きskew | `0.015 x + 0.02 x²` |
| 下向きskew | `-0.02 x + 0.03 x²` |

評価は524,288 antitheticペア、seed 7 / 41。
pilotは262,144ペアで、counterは評価範囲の直後 `[524288,786432)`。
候補lambdaは `[0.01,0.003,0.001,0.0003,0.0001]`。
局所**分散**感応度の再構成margin上限3%、Gamma相対SE上限25%、MC margin倍率2を固定した。
前回のflat試験と閾値の数値は同じだが、今回は正しい年率分散座標で品質を測る。
MC marginは同時信頼区間ではない。

価格側は各終端quoteを `±1e-5` / `±5e-6` 変更して毎回再較正し、
seed 4001〜4008、各16,384ペアでnative価格を再計算した。
価格差分側とGamma側のseedは独立で、符号と差分幅には共通乱数を使う。
別実装の96点条件付き価格積分、差分幅の細分化、独立primal KKT解とも照合した。
作用素の方向はペイオフや価格差分を見ずに、基準モデルの較正から固定する。

母集団quote偏りの目標は `0.002 + 1% * abs(独立母集団感応度)`。
MC推定平均の比較は実測SEを用いた6SE条件と絶対許容差0.002で別に判定する。

| 商品 | 選択lambda（2モデル・2 seed共通） | 最大再構成margin | 最大Gamma相対SE |
| --- | ---: | ---: | ---: |
| Call | 0.01 | 0.8268% | 6.3767% |
| Asian | 0.0003 | 1.9151% | 4.8479% |
| 既知fixing付きAsian | 0.01 | 2.9804% | 2.7219% |
| Lookback | 0.01 | 1.0895% | 5.1330% |

全1,914チェック、80件の母集団quote偏り、80件の再較正native価格差分、
80件の独立価格参照との比較がすべて合格した。

- 方向保存誤差（平均絶対値＋6SE）最大：**1.171e-12**。
- 独立primal KKT回復質量との差：**7.512e-8**（許容2e-6）。
- 母集団の方向保存によるquote偏り最大：**3.376e-13**。
- 作用素32/64点差最大：**5.330e-15**。省略tail各要素上界最大：**2.426e-31**。

母集団の極小偏りは制約の数値精度であり、MC推定値が同じ桁で正確という意味ではない。
実際のGamma方向平均と独立価格微分の最大差は約0.645、native価格差分平均との差は
約1.046で、それぞれ所定のMC誤差条件の中に収まった。
最大再構成marginは3%上限に近く、未試験のスマイルやseedへ合格を一般化しない。

## 互換性・配布・残件

旧binaryとの比較で、制約なし／合計保存／方向保存×4商品×2 seedの**24設定**が
通常価格とatom-chain診断（全共分散を含む）のJSONまでbyte単位で一致した。
比較結果のSHA-256は
`df5d3377a1d01d40ac7fef3be6ab11e95b647265772c0500e63e40f861e2ab07`。

Rust関連27テスト、Python VegaKT 89テスト、wheel API契約3テストが合格。
新テストは平坦極限、独立非平坦作用素・KKT、真の分散節点価格差分、rank/shape/
精度拒否、worker再現性、実測分散、初期Gammaへの誤適用拒否、評価失敗時の非昇格を確認する。
既存Gamma policy / 診断APIも含めてwheelの公開API検査に登録し、型定義と実binaryを照合した。
Clippy・整形・依存方向・差分の空白検査も通過。
手動CIへ数値検証と成果物保存を追加した。リモートCIは未実行。

過去のflat基準30件の失敗や標本不足の記録は変更していない。
残件は非平坦な将来を持つ初期Gamma、第1分布の一般化、多区間のGamma回復・較正、
キャリー・配当・バリア・早期行使、および全面IVリスクとヘッジバックテストへの昇格。

再実行（現行native wheelとcompanionを使用）：

```sh
python scripts/check_nonflat_gamma_recovery.py --policy fixtures/backtesting/nonflat-gamma-policy-v0.1.json --output benchmark-results/vega-kt-nonflat-gamma/report.json
```
