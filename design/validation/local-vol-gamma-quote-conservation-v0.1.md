# 較正方向を保存するGamma回復とquote別精度の再検証

2026-10-05。較正された市場IV節点の感応度を、Gamma回復で保存する固定線形制約を
追加した。[前回の検証](local-vol-discrete-smile-calibration-v0.1.md)で残った
quote別の正則化偏りを対象とする。適用範囲は平坦な基準モデルからの2区間・
第1満期固定の較正方向であり、一般スマイルの全面受入ではない。

評価標本数を増やした再検証で **1,828/1,828チェック合格**。
前回の40比較（未達30）を含む、拡張した **80/80 quote偏り比較が合格**した。
母集団参照に対する最大偏りは **0.817302 → 3.003e-13未満**になった。
これは正則化の偏りについての結果であり、MC平均自体の誤差がその大きさになったという意味ではない。

## 変更内容

従来の `1'Bq=1'b` という合計保存だけでは、較正Jacobianの右逆行列 `U` に対する
`U'(Bq-b)/(2*sigma1)` が残った。新方式は、ペイオフ標本を観測する前に
`W=U'/(2*sigma1)` を固定し、各行について **`W Bq=W b`** も制約に加える。
市場と局所分散モデルから決まる較正方向を使い、商品の参照価格・MC標本・
回復した質量から方向を調整しない。

制約の列を並べた行列を `D` とし、合計保存を指定した場合は1の列も加える。
元のTikhonov目的関数に `D'Bq=D'b` を課し、

```text
H = B'B + alpha² I
R = H^(-1) B'
Q = R D,  C = B'D,  G = C'Q
R_D = R + Q G^(-1) (D' - C'R)
```

という固定逆行列を構築する。すべての制約を同時に解く。
方向を正規化し、`G` の対角をそろえてからCholesky分解する。
正規化後のpivotが `1e-10` 以下、または行列全列での制約残差が `1e-9` を超える場合は
拒否する。重複方向、合計と重複する方向、ゼロ・非有限係数、shape不一致も拒否する。
失敗時に制約を捨てたり、質量を標本ごとに割り直したりしない。

負のGamma、全source節点での空間残差、逆行列の雑音増幅、resolution traceを維持する。
空間残差3%、Gamma相対SE25%、MC余裕2の選択・評価条件を変えない。
pilotで最も強い合格lambdaを固定し、独立の評価標本で一度だけ判定する。
評価失敗後の再選択は行わない。

## APIと統計

Pythonでは既存policyに
`local_vega_conservation_directions[回復時刻][方向][source節点]` を渡す。
1〜64回復時刻、各時刻で共通の1〜8方向・2〜129節点、合計16,384係数以内とする。
実際のシミュレーション時間格子・source格子との一致と、各行列のランクを標本生成前に確認する。

```python
U, _ = calibrator.terminal_iv_tangent()  # 平坦な2区間モデル、第一満期固定
directions = [(U.T / (2 * sigma1)).tolist()]
policy = rust_pricing.VegaKtGammaRecoveryPolicy(
    [0.01, 0.003, 0.001, 0.0003, 0.0001],
    pilot_sampling_units=262144,
    reconstruction_relative_limit=0.03,
    gamma_relative_standard_error_limit=0.25,
    mc_margin_multiplier=2.0,
    preserve_local_vega_total=True,
    local_vega_conservation_directions=directions,
)
audit = plan.evaluate_vega_kt_gamma_recovery_audit(strikes, policy=policy)
```

今回の再検証では `plan` の評価標本数を524,288とする。
Rustではpolicyの `with_local_vega_direction_conservation`、単独の逆問題では
`VegaKtConvolutionPlan::conserve_local_vega_directions(directions, preserve_total)` を使う。

各回復行に次のmomentを追加した。

- `local_vega_directions`: 元の局所Vegaの方向別射影。
- `reconstructed_directions`: 回復Gammaから再構成した方向別射影。
- `direction_differences`: 同じ標本内の両者の差。

各pathで射影してからantithetic pairを平均し、固定blockで統計を集計する。
したがって、方向別SEにはsource節点間の共分散が入る。周辺SEからの上限ではなく、
実際の射影標本のSEである。ただし方向間の全共分散は今回出力していない。
atom-chain側でも同じ制約付き逆行列を再構築し、追加momentを記憶域上限に含める。

`reconstructed_directions` は、指定した**較正方向**のリスクである。
既存のatom-chain `raw_buckets` と同一の市場パラメータ化とは限らない。
それらの表示IVバケットをこの検証の合格だけで昇格させない。

## 元の標本数で残った失敗

[元条件のポリシー](../../fixtures/backtesting/gamma-quote-conservation-policy-v0.1.json)では、
4商品×seed 7/41、評価131,072ペア、pilot262,144ペアを維持した。
追加制約により、通過した7設定では35/35のquote偏り比較が合格した。
ただしCallのseed 7はGamma相対SEが **25.2447%** となり、25%上限に未達だった。
空間残差の判定値は1.5068%で、失敗理由はGammaのMC誤差である。

この試験は **909チェック中1件失敗、終了コード1** のまま
`benchmark-results/vega-kt-quote-conservation/report.json` に保存する。
Gammaの合計保存やquote偏りが小さいことを理由に、失敗した評価を合格扱いしない。

## 標本数を増やした再検証

[細分化ポリシー](../../fixtures/backtesting/gamma-quote-conservation-refinement-policy-v0.1.json)は、
評価標本を4倍の524,288ペアとし、pilotは262,144ペアのままにした。
seedは7/41に101/509を加え、4商品×4 seedを検証する。
各実験のpilotはその評価範囲の直後のcounterを使い、評価標本と重ならない。
全正則化候補、空間・MC・quote精度の閾値は元のままとする。

価格側は同じ5個の終端quoteを `±1e-5` と `±5e-6` で非線形再較正し、
各商品をnativeで計算し直す。価格差分はseed 4001〜4008、各16,384ペアで、
Gamma評価から独立させる。符号と幅の間には共通乱数を使う。
さらに、別実装の96点条件付きペイオフ積分、変更幅の細分化、局所ADとの一致を確認する。

母集団quote偏りの精度目標は前回と同じ
**`0.002 + 1% * abs(独立母集団quote感応度)`**。
MC誤差に関係なく、正則化した母集団の偏りを別に検査する。
nativeの方向別Gamma平均は、再較正したnative価格差分・独立価格参照の両方と
実際のSEを使って比較する。全候補について独立なprimal KKT解、全空間残差、
制約の標本ごとの保存、雑音増幅・trace、従来の選択条件も照合する。

最終結果・最大誤差・全判定は[機械可読記録](local-vol-gamma-quote-conservation-v0.1.json)と
`benchmark-results/vega-kt-quote-conservation/refined-report.json` に保存する。

| 商品 | 選択lambda（4 seed共通） | 最大母集団quote偏り | 最大Gamma相対SE |
| --- | ---: | ---: | ---: |
| Call | 0.01 | 7.11e-14 | 13.7165% |
| Asian | 0.0003 | 3.01e-13 | 4.9909% |
| 既知fixing付きAsian | 0.01 | 3.20e-14 | 3.7792% |
| Lookback | 0.01 | 7.82e-14 | 10.7857% |

全16設定がheld-out判定を通過し、空間残差の判定値は最大 **2.1686%**。
全候補・評価での方向保存誤差（平均絶対値＋6SE）は最大 **2.418e-12未満**。
独立primal KKT解との回復質量の最大差は **1.548e-7未満**で、固定許容値2e-6以内だった。
価格差分との80比較、独立価格参照との80比較も、所定のMC誤差条件で通過した。
母集団quote偏りと、標本平均の価格照合は別々に判定している。

## 互換性と残る範囲

追加オプションを指定しない場合は従来の逆行列とJSON形状を維持する。
旧バイナリとの比較で、制約なし／合計保存のみ×4商品×2 seedの**16設定**について、
通常価格とatom-chain診断（全共分散を含む）のJSONがbyte単位で完全一致した。
Rustの関連24テスト、Python VegaKTの81テストが合格した。
新しいテストは独立KKT、符号・単位・方向順序への不変性、全入力基底での制約、
rank/shape拒否、4区間・Brownian bridge・antithetic有無、並列再現性、notional比例、
評価失敗時の非昇格を含む。Clippy、整形、依存方向も確認した。

手動CIには細分化検証と成果物保存を追加した。リモートCIは未実行。
過去の30/40未達と元標本数の1件失敗は書き換えずに残す。
この制約は指定された局所Vegaの線形汎関数を保存するものであり、Gamma全体の一意性や
非平坦な基準モデルの回復を保証しない。非平坦Gamma、多区間の一般スマイル再較正、
第1満期スマイル、キャリー・配当・バリア・早期行使の全面受入は残る。
通常リスク・受入済みP0 providerの範囲は変えず、`iv_bucket_acceptance='not_established'` を維持する。
