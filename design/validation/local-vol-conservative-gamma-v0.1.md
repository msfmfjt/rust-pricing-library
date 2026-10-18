# 局所Vega合計を保存するGamma回復と満期IVの再較正検証

2026-10-05。正則化によって失われていた局所Vegaの積分量を、固定の線形制約で保存する
回復方式を追加した。満期別IVを動かして全区間の分散を再較正する価格差分と照合した。
空間平坦・ゼロキャリーの診断範囲に限る。一般スマイルの全面的なIVバケット受入は未成立。

## 改善した点

従来は `min ||Bq-b||² + alpha²||q||²` を解くため、
source-hat入力 `b` の合計まで縮小し、満期別IV感応度に偏りが残り得た。
新方式は同じ目的関数に **`1'Bq = 1'b`** を追加する。
source hatsが全領域で1へ分割されるため、これは既知の局所Vega積分を保存する制約である。

従来の逆行列を `R=(B'B+alpha²I)^(-1)B'`、`c=B'1`、`u=R1` とすると、

`Rc = R + u * (1' - c'R) / (c'u)`

が制約付き問題の解を与える。行列はペイオフ標本を見る前に決まる。
負のGammaを維持し、標本ごとに質量を割り直す処理や価格参照への調整は行わない。
合計保存で各価格帯の残差が消えるわけではない。`Bq-b` の全成分と実際の残差統計を残す。
新しい行列で雑音増幅指標とresolution traceも再計算する。

## APIと判定

```python
policy = rust_pricing.VegaKtGammaRecoveryPolicy(
    [0.01, 0.003, 0.001, 0.0003, 0.0001],
    pilot_sampling_units=131072,
    reconstruction_relative_limit=0.03,
    gamma_relative_standard_error_limit=0.25,
    mc_margin_multiplier=2.0,
    preserve_local_vega_total=True,
)
result = plan.evaluate_vega_kt_atom_chain_diagnostic(strikes, policy=policy)
```

既存の `evaluate_vega_kt_gamma_recovery_audit` にも同じポリシーを渡せる。
Rustでは `VegaKtGammaRecoveryPolicy::with_local_vega_conservation()`、
単独の逆行列では `VegaKtConvolutionPlan::conserve_local_vega_total()` を使う。

既定値は従来方式のまま。新方式は別のpolicy labelと制約メタデータを持ち、
元のrequest/plan指紋に加えてポリシー全体を保存する。
正則化候補を選ぶ標本と評価標本は分離し、全候補・全時刻で空間再構成3%、
相対Gamma SE 25%、MC余裕2という既存の基準を維持する。
評価後の再選択や基準緩和を行わず、不合格時はIV計算結果を未算出として返す。

## 市場IVからの再較正

新しい `rust_pricing_backtest.flat_iv.recalibrate_flat_market_iv` は、各満期で
ストライク方向に一様なIVシフトを受け取り、累積分散 `V_i=t_i*IV_i²` を作る。
全区間を `local_variance_i=(V_(i+1)-V_i)/dt_i` から再構築する。
最初の満期だけのIV変更でも、その次の区間の局所分散を変更する。
後の満期のIVを勝手に変えない。入力を変更せず、変更前後の指紋と全分散を記録する。
カレンダー分散・floor/cap違反は拒否し、クリップで修復しない。

`maturity_risk_statistics` はバケットを満期別に合算し、全共分散から満期間の
平均推定値の共分散とSEを計算する。共分散がない場合は拒否し、独立とは仮定しない。

## 検証結果

[市場IV検証ポリシー](../../fixtures/backtesting/conservative-gamma-market-policy-v0.1.json)は
4商品（call / Asian / 既知fixing付きAsian / Lookback）×期間ボラ2種×seed 7/41の16設定。
局所・Gamma・表示は17節点。選択・評価はそれぞれ131,072 antitheticペア。
各満期のIVを±0.0001と±0.00005変更して全区間を再較正する。
価格差分は別seed 2001〜2008、各16,384ペアを使用し、各seed内の全符号・幅で共通乱数を使う。
Gamma標本とは独立であるため、双方の平均推定分散を足して比較する。

- **560/560チェック合格**。価格参照は2観測ペイオフの独立一次元積分。
  Gamma回復や正則化した入力から価格参照を作っていない。
- 同じlambdaで比較した満期別Vegaの母集団参照誤差は、従来最大 **0.08036575** に対して
  新方式最大 **1.0783e-6**。残りには有限幅の価格差分誤差を含む。
  これはMC標本平均がその精度になったという主張ではない。
- 局所Vega合計の保存誤差（平均絶対値＋6SE）は最大 **4.07e-14**。
- 評価標本の空間再構成判定値は最大 **2.1126%** で、3%基準を維持。
- [33節点の再検証](../../fixtures/backtesting/conservative-gamma-refinement-policy-v0.1.json)も
  **88/88チェック合格**。Asian / 既知fixing付きAsian × seed 7/41の4設定を、
  元の評価131,072ペア、選択262,144ペアで実行した。
  Asianはlambda=0.0003、既知fixing付きAsianは0.003を選び、全設定で評価に合格。
- 旧版バイナリとの比較では、従来方式の診断出力が4商品×2 seedの8設定で完全一致。
  過去の固定lambda失敗記録と受入記録を維持する。
- 関連Rust **21テスト**、Python VegaKT **75テスト**、新しい再較正・共分散処理の
  **4テスト**が合格。Clippy、整形、依存方向も確認。リモートCIは未実行。

[機械可読の検証記録](local-vol-conservative-gamma-v0.1.json)に、全判定、設定要約、
元レポート・ソース・実行バイナリのハッシュを保存する。

## 残るスマイル較正誤差

[スマイルの診断](../../fixtures/backtesting/discrete-smile-probe-policy-v0.1.json)では、
平坦な基準点から市場IVを `epsilon*log(K/S0)` だけ変形し、解析的なDupire局所ボラを
両符号で作り直した。等間隔N区間のEuler過程では、この例のquote価格微分は
連続時間の市場価格微分の **`1-1/N` 倍**になる。
これはGammaの保存制約で修復できる誤差ではない。

| 区間数 | 市場quote応答の解析上の不足 |
| --- | --- |
| 2 | 50% |
| 4 | 25% |
| 8 | 12.5% |
| 16 | 6.25% |
| 32 | 3.125% |

実際の両符号の価格差分は、5設定ともこの離散過程の参照とMC誤差内で一致した。
この **5件は較正不足を確認する診断の合格**であり、スマイル受入の合格ではない。
有限時間刻みで市場スマイルに合わせる再較正、非平坦Gamma回復、ストライク別IVバケット、
キャリー・配当・バリア・早期行使の全面的な受入は残る。
通常のリスク計算と受入済みP0 providerを拡張せず、`iv_bucket_acceptance='not_established'` を維持する。
