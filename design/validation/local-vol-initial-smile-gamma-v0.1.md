# 非平坦な継続価格の初期Gamma回復 v0.1

2026-10-05、macOS aarch64、release native wheel / Python 3.13。

**2区間・一定第1分散の範囲で、非平坦な将来を持つ初期Gammaの近似回復を追加した。**
2種類のスマイル×4商品×2 seedの全16設定、**256 / 256チェック合格**。
回復した初期年率分散感応度と独立な参照微分の差は最大 **3.365e-7未満**だった。

これは有限領域のGamma近似と、局所分散モデルの初期年率分散感応度の受入である。
第2満期quoteを固定して再較正する第1市場IV bucket、一般多区間、全面的なVegaKTの
市場IV受入までを完了したものではない。

[機械可読記録](local-vol-initial-smile-gamma-v0.1.json)、
[固定ポリシー](../../fixtures/backtesting/initial-smile-gamma-policy-v0.1.json)、
[独立検証プログラム](../../scripts/check_initial_smile_gamma.py)。
詳細は `benchmark-results/vega-kt-initial-smile/report.json`。

## 回復するGamma

前回のflat初期Gammaは、将来のshockを固定したペイオフが第1状態の区分線形関数に
なることを使っていた。第2局所分散が状態依存ならその仮定は成立しない。
今回は、第1状態sを固定し、実際のcompiled payoffを**終端状態S2**の連続区分線形関数へ
変換する。第2遷移の条件付き分布は、nativeと同じ局所分散の補間値を使うlognormalであり、
条件付き期待ペイオフG(s)をBlack積分で計算できる。既知fixingもcompiled graphから保持する。

G(s)を物理spotに関して区分線形補間し、隣接区間の傾きの差を符号付きGamma原子として
保持する。局所分散節点の折れ曲がり、lookbackの履歴切替による折れ曲がり、滑らかな
曲率を含む。正の質量への切り詰めや、価格感応度へ合わせる再正規化は行わない。

第1log状態の±12標準偏差に、基礎格子32,768分割、局所分散節点、報告bin境界を置く。
次に**追加した境界の近くも含め、すべての既存区間を二等分**する。
細分化後の点数は `2 * coarse.interpolation_nodes - 1`。
Gammaはbin質量として返し、各原子の重み付き積分は元の位置で計算する。
bin中心への移動は行わない。

binは `[0,b0),[b0,b1),...,[bn,infinity)` と有限log領域の共通部分を意味する。
領域外のGamma質量自体は未認定で、ゼロリスクとみなさない。
価格復元には定数・線形項を含む区分線形補間の両端の線形延長を使う。

## 初期分散感応度とtailの検査

第1年率分散をv、第1期間をt、V=v tとする。第2分散gridは固定する。
第1区間のBlack年率分散感応度でGamma原子を積分した値を、以下の別計算と照合する。

```
S1 = S0 exp(-V/2 + sqrt(V) Z)
d E[G(S1)] / d v
   = E[G(S1) * t * ((Z²-1)/(2V) - Z/(2 sqrt(V)))]
```

このscore積分は、回復Gammaやnative ADを使わない。
局所分散節点・bin境界で分割した32点／64点Gauss–Legendreで収束を確認する。
これらの積分差・細分化差は数値的な収束診断であり、厳密な全近似誤差の証明ではない。

compiled graphの定数・線形演算とmin/maxから、`|G(s)| <= A+B s` を認証する。
変数同士の乗算や変数による除算など、この成長上界を保証できない演算は拒否する。
Gaussianの尾確率と指数tilt後の2次モーメントを用い、領域外の価格と年率分散微分の
解析的上界を別に計算する。今回の初期分散微分のtail上界は最大 **2.795e-27未満**。
Gamma質量のtail上界とは異なる。

`relative_tolerance=1e-8` に対して、各価格・感応度の精度判定は
`relative_tolerance * (A+B S0)`。
bin質量は `sqrt(relative_tolerance) * (A/S0+B)`、bin別分散感応度はこの質量許容値に
第1期間のBlack年率分散kernelの最大値を掛けた値で判定する。
合計感応度と区間別Gammaの精度を分けて開示する。

## APIと失敗時の扱い

```python
audit = plan.evaluate_vega_kt_initial_smile_gamma_audit(
    [60., 80., 90., 100., 110., 120., 140.],
    subdivisions=32768,
    relative_tolerance=1e-8,
)
```

対象は、native simulation gridと局所分散gridが一致する2区間、一定の第1分散、
正の分散、zero carry・unit discount、連続区分線形ペイオフ。
配当・バリア・早期行使・smoothing・Bass・RQMCは対象外。
演算量・標本集計メモリにも上限を置く。

- `coarse` / `fine`：決定的なGamma bin質量・重み付き感応度・復元価格。
- `quadrature_*`：別のscore積分による価格・感応度と32/64点差。
- `native_*` / `*_difference`：同じnative経路の価格・初期分散ADと差の実測MC統計。
- `recovery_passed`：細分化、score照合、積分精度、tail、bin精度の判定。
- `validation_passed`：native価格・ADとの差が6SE＋機能別許容値以内かの判定。
- `estimate`：両判定が通った場合だけ初期年率分散感応度を返す。それ以外は `None`。

決定的なGammaに架空のMC標準誤差を付けない。
失敗後に格子・許容値・seedを自動変更することもない。
旧flat初期Gamma APIと、前回の第2区間診断のJSONは維持する。
第2区間診断にある `initial_gamma_recovery='not_implemented'` は、そのAPI内では
初期Gammaを計算しないという意味であり、今回の別APIは明示的に呼び出す。

## 独立検証

第1満期64日、終端128日、spot 100、局所grid33点。
第1分布を固定し、終端IV変化 `0.015 x+0.02 x²` と `-0.02 x+0.03 x²` に
非線形再較正した2モデルを使った。商品はCall、Asian、既知fixing付きAsian、Lookback。

独立参照は商品ごとの条件付き価格の**解析的な1階微分**と部分積分を使う。
分散補間の傾きの跳び、ペイオフの履歴切替における左右微分も含め、
nativeの区分線形Gamma近似とは異なる方法でbin質量・重み付き感応度を求めた。
96点と192点積分で参照値の収束を確認した。

初期年率分散のみを `±1e-5` / `±5e-6` 変更し、第2分散を固定した価格差分も実施した。
独立価格積分に加え、native価格差分はseed 4001〜4008、各16,384 antitheticペアを使用。
符号と差分幅には共通乱数を使う。Gamma側のnative AD検証はseed 7 / 41、各131,072ペア。

| 商品 | 感応度の細分化差（最大） | 感応度の許容値 | bin質量の細分化差（最大） |
| --- | ---: | ---: | ---: |
| Call | 5.020e-7 | 2.000e-6 | 1.545e-5 |
| Asian | 1.010e-6 | 2.000e-6 | 2.495e-5 |
| 既知fixing付きAsian | 6.083e-7 | 1.995e-6 | 1.388e-5 |
| Lookback | 2.612e-7 | 3.000e-6 | 1.545e-5 |

全16設定で回復・native照合が合格し、256チェックすべて通過した。

- 初期感応度と独立な部分積分参照との差：最大 **3.365e-7未満**。
- 復元価格と独立条件付き価格積分との差：最大 **3.340e-8未満**。
- score積分と独立な部分積分の差：最大 **2.843e-14未満**。
- bin質量の独立参照との差：最大 **2.495e-5未満**。
- bin別重み付き感応度の独立参照との差：最大 **0.001042未満**。合計感応度の精度とは異なる。

native価格差分平均との差は最大約0.1515で、実測MC誤差を用いた所定の条件内だった。
独立な決定的参照との3.365e-7という差を、MC価格差分の精度と混同しない。

## 失敗例・互換性・残件

`controls.json` には、64分割による精度不足、1e-12の過度に厳しい許容値、
2標本・seed 4でのnative照合失敗を保存した。すべて `estimate=None` を維持する。
分散の傾きが100で下がるモデルでは、その直右のbinに約 **-0.013583** のGamma質量が
生じ、負の原子を保存して独立参照と一致することを確認した。
背景格子だけを二倍にした中間試験も `background-grid-report.json` に保持し、
最終判定は全小区間を二等分した `report.json` を用いる。

旧binaryとの**40設定**の比較で、通常価格、flat atom-chain全共分散、flat初期Gamma、
非平坦な第2区間Gammaの出力がbyte単位で完全一致した。
SHA-256：`5aeb7a5eb2c5a4841a6a4c9bec1eb5774b5bf3b0d72f9470292719dac22f5f84`。
Rust関連30テスト、Python VegaKT 97テスト、wheel API契約4テストが合格。
Clippy・整形・依存方向・差分の空白検査・実wheelと型定義の照合も通過した。
手動CIとsource archive対象へ追加。リモートCIは未実行。

一般の第1分布、多区間、第1市場IVの較正接線と満期間の同時リスク、
キャリー・配当・バリア・早期行使、通常ヘッジバックテストproviderへの全面昇格は残る。

```sh
python scripts/check_initial_smile_gamma.py --policy fixtures/backtesting/initial-smile-gamma-policy-v0.1.json --output benchmark-results/vega-kt-initial-smile/report.json
```
