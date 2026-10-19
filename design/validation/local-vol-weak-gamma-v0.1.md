# VegaKTの密度積分・Gamma回復の数値改善 v0.1

日付: 2026-10-04  
状態: 制限付きの新しい計算経路で、24ケース・63比較の失敗を32件から0件へ削減。
既存 `evaluate()` の154項目の本体受入とは別の検証。

## 解消した問題

正時刻のADは空間hatに集めた積分感応度です。これを「節点の点密度 × hat面積」で
割ると、短い時間区間に対して格子が粗い場合、hat内部には確率があるのに節点密度は
極小となり、Gammaが発散しました。seed 7・4096ペア・16区間・17節点では、
連結Vegaが −2493.654939、標準誤差が1928.724457になっていました。

[新しい計算経路](../../crates/pricing/src/engine/risk/vegakt_pricing.rs)は、
コンパイル済みのペイオフ・観測時点・乱数・native ADを使い、次の修正を適用します。

- 密度を先に積分し、節点の点密度による割り算を避ける。
- 回復するGammaの時点の密度を用いる、積分形に基づく近似に変更する。
- 初期Gamma格子と回復格子を、最初の分散と差分幅から決めた解像度まで細分化する。
- 異なる計算経路から得たほぼ同じ積分節点でも、微小区間の測度を捨てずに積分する。

受入結果・解析Vega・native Vegaを使った正規化、閾値変更、除外感応度の足し戻しは
行いません。報告用の満期・Strike節点も変えません。

## 密度積分と時間近似

空間平坦・zero carryのモデルで、`X_i = log(S_i/S0)`、積分分散を `v_i` とします。
式(3)をsource状態で積分すると、局所Vegaのhat別期待値は

```text
A_ij = sigma_i * dt_i * integral Gamma_(i+1)(K) * K^2 * d_(i+1)(K)
       * E[h_j(X_i) | X_(i+1) = log(K/S0)] dK
```

となります。条件付きhatを `h_j(X_(i+1))` で近似し、その積分質量

```text
M_j(v) = E[S^2 * h_j(log(S/S0))]
       = S0^2 * exp(v) * E[h_j(Z)],  Z ~ Normal(1.5*v, v)
Gamma_j = A_ij / (sigma_i * dt_i * M_j(v_(i+1)))
```

で回復します。両端のhatはnative ADと同じ定数外挿です。
[密度積分](../../crates/pricing/src/engine/risk/vegakt_density_mass.rs)は正のGaussian積分を
使い、狭い区間ではCDFや一次モーメントの近接差を取らずに計算します。

これは式(11)の点ごとの除算そのものでも、式(3)の厳密な逆作用素でもありません。
条件付きhatの置換、log-hat上の感応度からStrike線形Gammaへの補間、有限領域が近似です。
特にペイオフや基底の折れ点では一様な誤差次数を保証しません。
旧 `VegaKtChainPlan::new` の点密度方式と独立参照は保持し、新方式は
`with_integrated_density_recovery()` と別ラベル
`flat_variance_weak_gamma_chain_diagnostic_v1` で識別します。

最大log刻みは `min(relative_bump/2, sqrt(first_step_variance)/2)` です。
各入力区間を分割し、1025節点を超える要求は明示的に拒否します。入力で指定した初期
Strike、実際の初期Strike、回復格子、最大刻み、source時刻と密度時刻を結果に記録します。
出力を見て刻みを調整する方式ではありません。

## 同条件の比較

[既存ポリシー](../../fixtures/backtesting/numerical-policy-v0.1.json)のseed・標本数・
時間分割・空間節点をそのまま使います。ポリシー指紋は
`fce9be03ab787b19028243dc894960c342d0fcf75ad7043bb222184ca8655b09` です。

| 比較 | 件数 | 修正前の失敗 | 修正後の失敗 |
|---|---:|---:|---:|
| 合計Vegaとnative局所Vega、3% | 24 | 20 | 0 |
| 合計Vegaと解析解、6SE + 3% | 24 | 0 | 0 |
| 合計Vegaと共通乱数による価格差分、3% | 6 | 6 | 0 |
| 空間格子変更時の合計Vega、5% | 9 | 6 | 0 |
| 合計 | 63 | 32 | 0 |

seed 7・4096ペア・17報告節点の例:

| 時間区間 | native局所Vega | 修正前の連結Vega | 修正後の連結Vega |
|---:|---:|---:|---:|
| 2 | 23.375994 | 21.180253 | 23.646807 |
| 8 | 23.578941 | 21.252461 | 23.777008 |
| 16 | 23.371385 | −2493.654939 | 23.588743 |

局所Vegaとの差は全24ケースで最大約1.67%。16分割例の合計標準誤差は約0.326635です。
満期以外のバケットのL1量も結果に保存しますが、この値には標本雑音と回復近似の影響が
残り、合計だけの通過で満期・Strike別の正しさを認定しません。

## 独立検証と使い方

[密度参照](../../scripts/check_local_vol_density_mass.py)は120桁DecimalのGaussian級数と
切断一次モーメントから8ケースを作成します。中心、粗い格子、左右のtail、狭いhat、
短い分散、両端外挿を含み、Rust値を相対 `5e-11` で照合します。
[連結参照](../../scripts/check_local_vol_weak_chain.py)は100桁の価格差分から54チャネルの
平均・標準偏差・標準誤差を独立計算し、3パスと3反対変量ペアを照合します。
参照値の生成にnative AD・Gamma回復・統計処理を使いません。

```python
report = plan.evaluate_vega_kt_chain_diagnostic(
    initial_strikes, relative_bump=0.04, subdivisions=2
)
audit = report.to_dict()
```

```shell
python scripts/check_local_vol_density_mass.py
python scripts/check_local_vol_weak_chain.py
python -m rust_pricing_backtest.chain_numerics \
  --policy fixtures/backtesting/numerical-policy-v0.1.json \
  --output benchmark-results/vega-kt-pricing-chain/final-comparison
```

診断APIはRust/Pythonの両方から利用できます。空間平坦、一定forward、単位割引、
恒等的な配当座標、pseudo-MC、全正時刻と報告満期の一致に限定します。call、Asian、
lookbackのコンパイル済み状態偏微分、反対変量、Brownian bridge、worker再現性を検査し、
indicator、barrier、早期行使、smoothing、RQMC、全共分散要求は拒否します。
出力は通常のpricing-result wireではなく、`diagnostic_only: true` の診断文書です。

## 本体受入との境界

通常の `PricingPlan.evaluate()` は従来の局所節点配賦を維持し、初期行欠落を含む
154項目中54件の失敗が残ります。今回の63比較は、この受入を置き換えません。
原式の完全再現、非平坦分散、直接bridge分散リスク、報告満期補間、満期・Strike別精度、
価格と全バケットの共分散、ヘッジ数量の安定性は引き続き必要です。

修正前後の全観測値・バイナリ指紋・閾値・初期差分・打切り監査は
`benchmark-results/vega-kt-pricing-chain/{before-comparison,final-comparison}/report.json`、
本体再検証は `benchmark-results/vega-kt-numerics/weak-chain/report.json` に保存します。
