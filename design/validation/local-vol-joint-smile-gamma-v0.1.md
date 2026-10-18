# 第1満期と終端スマイルの共同IV Gamma検証

2026-10-05。第1満期が平坦な2区間モデルで、初期Gammaと非平坦な第2区間Gammaを
市場較正接線で結合した。**636 / 636 checks合格、失敗0**。
第1parallel IVを動かしたときに終端quoteを保つ再較正を含む。
通常バックテストproviderおよび一般スマイルの全面IV受入は未変更・未確立。
機械可読記録は [JSON](local-vol-joint-smile-gamma-v0.1.json)。

## 実装と座標

- `TwoStepSmileCalibrator.calibrate_joint(first_iv_shift, terminal_iv_shifts)`：
  第1flat IVを動かした後、既存minimum-norm annual variance Newtonで終端quoteを較正。
  元requestや固定第1分布のメソッドを変更しない。価格精度・rank・分散境界を満たさない要求は拒否する。
- `joint_iv_tangent()`：第1平行IV 1方向＋終端5quote IVの34×6接線。
  第1年率分散の係数は `2 sigma0`。終端callに対する初期分散の応答をscore積分で計算し、
  終端quoteを相殺する第2分散変化を求める。再較正の有限差分と照合する。
- `PricingPlan.evaluate_vega_kt_joint_gamma_audit`：初期・第2Gamma、全方向のnative AD、
  回復感応度、同一経路の差、全18×18標本共分散を返す。native単独は市場較正を認証しない。
- `rust_pricing_backtest.joint_smile.evaluate_joint_market_gamma`：較正済みpayloadを検証し、
  接線作成と同じpayloadのcompileを行う。小数表示の絶対IV単位で結果を返す。

終端接線をU、第1IVを補償する第2分散接線をu0とすると、u0はUの列空間に属する。
したがって保存制約は独立なU.Tの5本のみ。第1補償方向を6本目の制約に追加してrankを落とさない。
従属する方向の読み出しと共分散は許可する。初期回復・第2回復が成功した場合でも、
共同のnative照合が失敗すれば `estimate=None` としraw統計を残す。

## 変更していない判定基準

- 第2Gamma：pilot 262,144、held-out 524,288 antithetic pairs、seed 7 / 41。
  正則化候補 `[.01,.003,.001,.0003,.0001]`。空間再構成3%、Gamma相対SE25%、MC margin 2。
  pilotで最強の合格候補を凍結し、held-out失敗で再選択しない。
- 初期Gamma：32,768分割から全区間を二分。相対許容値1e-8。
  native MCと独立score積分・tail・格子細分化の判定を維持する。
- 独立価格：192点積分、IV bump 1e-5 / 5e-6。native価格は別seed 4001..4008、各16,384 pairs。
  bump収束は0.002、価格照合は6 SE＋0.002。母集団のGamma射影bias目標は0.002＋1%。

共同推定のSEは、antithetic平均後の同一経路共分散を使い、満期間・quote間の相関を含む。
deterministic初期Gammaの近似誤差はSEへ混ぜず別allowanceで扱う。
pilot選択を条件としたMC診断であり、全方向の同時信頼区間ではない。

## 結果

2種類の較正済みスマイル（正・負skew）×4商品×2 seedsの16条件が全て合格した。

| 検査 | 最大値 |
|---|---:|
| 共同接線の終端quote応答 `[0,I]` からの残差 | 4.45e-15 |
| bumped modelのquote価格誤差上界 | 1.08e-13 |
| 母集団の共同感応度と独立再較正価格差分の差 | 7.92e-5 |
| 2つのbump幅による独立価格微分の差 | 5.94e-5 |
| 母集団でのGamma射影による方向別bias | 2.99e-13 |
| 共分散の対角と実測標本分散の差 | 0 |

母集団の小さい射影biasは保存制約の検証値であり、有限標本の市場感応度がこの精度で
求まることを意味しない。実測Gammaとnative価格差分の最大差は1.047で、独立標本のSEを
合成した基準内だった。全方向の符号付き値・SE・共分散はraw reportに保持する。

第1parallel IVを1単位変えたときの母集団感応度（終端quote固定）は次の通り。

| 商品 | 正skew | 負skew |
|---|---:|---:|
| ATM call | 約0 | 約0 |
| Asian | 11.15527 | 11.14707 |
| 既知平均付きAsian | 5.76830 | 5.78926 |
| Lookback | 0.22546 | -0.16497 |

ATM callの初期成分だけの感応度は非ゼロだが、終端再較正による補償を合計すると約0になる。
これは第2分散固定の初期IV感応度をそのまま市場IV感応度と解釈していないことを確認する検査である。

失敗制御は、粗すぎる初期格子、第2回復の不合格、保存されていない方向の共同照合不合格を検証した。
全て最上位 `estimate=None`。共同照合だけの失敗ではraw差・共分散と選択済み正則化を保持した。
旧40ケースの出力は前回native binaryの結果とbyte単位で一致した。

## 再実行

```sh
python scripts/check_joint_smile_gamma.py \
  --policy fixtures/backtesting/joint-smile-gamma-policy-v0.1.json \
  --output benchmark-results/vega-kt-joint-smile/report.json
```

現行native wheelとbacktesting companionが必要。手動numerical CIにも同じコマンドを追加した。
raw evidenceは `benchmark-results/vega-kt-joint-smile/`、policy・source・binaryのhashはJSON記録に保持する。

関連Rust 33件、Python VegaKT 105件、companion較正11件、wheel contract 4件を確認。
worker再現、方向の加法・符号付き倍数、同一経路の価格差分、従属制約の拒否、共分散予算も検証した。

## 残る範囲

第1満期そのものの非平坦スマイルは単一Gaussian遷移で表せないため未対応。
一般多区間、carry・配当・バリア・早期行使、全IV座標と通常providerの受入は今後の対象。
初期Gammaは有限領域の区分線形近似であり、領域外の無加重Gamma質量は未認定。
初期Gammaと作用素の積分細分化は数値収束の診断であり、厳密な総誤差保証とは区別する。
