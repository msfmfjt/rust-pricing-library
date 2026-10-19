# VegaKT 一般スマイル数値受入 v0.3

日付: 2026-10-04  
状態: **P0 European満期リスクの数値受入に合格**。1,077判定中、失敗0。
旧v0.1の6失敗は原記録を保持して、同一市場ショックの合格判定へ紐づけて解消。
P1/P2の経路依存Gamma回復、実市場でのヘッジ有効性、全プラットフォーム受入は含まない。

## 旧基準の6件

旧v0.1は報告格子を変えると独立quoteショックの集合も変わるため、最適ヘッジ数量が
一致しない。独立解析解でも同じ失敗が再現する。ここへ数量の一致を強制すると、
別のリスクをヘッジする目的関数を同一視してしまう。

v0.3は[固定市場ショックv0.2](vega-kt-market-shocks-v0.2.md)を正式な比較基準に採用した。
元の154判定・24観測は直前の保存結果と完全一致し、6件の `passed=false` も保持する。
各失敗にseed・報告節点数が一致する合格した置換判定を付け、
`superseded_comparison` と記録した。未解決は0件。
未知の失敗、置換判定の欠落・失敗は `unresolved` となり、全体判定を失敗させる。
元の許容値を緩めたり、失敗を合格へ書き換えたりはしていない。

一般スマイルの格子比較では、市場ショックに加えて基準IVの補間関数も固定する。
曲面を別々に粗い格子へ標本化すると基準価格まで変わるため、17節点の基準IVを
33・65節点へ線形補間で延長する。固定した市場リスク、ヘッジ枚数、独立した
正規方程式解、ショック集合に含まれない方向の価格変化をそれぞれ検証した。

## 本体の計算

原資料は、ユーザー提供の *Vega KT for the Local Volatility Model: An AD Approach*
（SSRN 4107770、17ページ）の式(1)・式(9)。満期の連続区分線形payoffは

```text
g(S_T) = a + b*S_T + sum_j q_j*(S_T-K_j)+
```

とパスごとに恒等的に複製できる。定数と線形項のIVリスクは0なので、任意の
スマイルに対する市場IVリスクは、各符号付きcall質量 `q_j` に連続Strike位置の
vanilla Vegaを掛け、IV補間の転置を適用すれば求まる。
割引とforwardは価格計算と同じ曲線から取る。**Vegaを掛けてから射影する**順序は
論文式(9)と一致し、一般のノイズを含むAD Gamma逆問題を必要としない。

[本体実装](../../crates/pricing/src/engine/risk/vegakt_terminal_replication.rs)は
コンパイル済みpayoffの満期Gamma測度を使用し、商品名だけによる特別処理をしない。
新しい方式名は `terminal_payoff_replication_iv_v1`。
満期が報告節点に厳密に存在すれば、局所時間格子と報告格子を分離でき、
契約満期後の報告節点も保持できる。静的複製の射影残差は0で、密度による切捨ては行わない。
空間平坦な入力では、満期までの全報告行を実際のEuler累積分散と照合する。
既存の平坦・zero-carry経路は変更せず、従来の標本統計を維持する。

厳密な市場IVリスクにMC標本を作らない。新方式のバケット標本分散・価格共分散は
未適用、要求された全共分散行列の要素は `unavailable`、バックテストのバケットSEは
`None` となる。価格・Delta・local Vegaは従来のMC評価を続ける。
市場IVリスクと、未校正の任意の局所分散グリッドの価格微分は同じものではない。

## 数値結果

| 検証 | 判定数 | 結果 |
|---|---:|---|
| 既存固定市場ショック（145個の旧判定を含む） | 307 | 全合格・前回と同一 |
| 5スマイル×2金利×3Strike×call/put、7判定 | 420 | 全合格 |
| 時間32/64/128、報告17/33/65、seed 7/41/97の細分化 | 270 | 全合格 |
| 一般スマイルの固定市場ヘッジ・補集合監査 | 60 | 全合格 |
| 全SSVI/eSSVI再構築と価格差分・連鎖則 | 20 | 全合格 |
| 合計 | **1,077** | **失敗0** |

スマイルは左スキュー、右スキュー、強い曲率、Heston型SSVI、eSSVI。ATM分散の
時間構造も変える。金利0%/3%、Strike 90/100/110、満期128日、Spot 100。
quote別価格差分との最大絶対誤差は `9.0691e-9`（許容 `2e-6`）。
価格・Deltaは従来の `6 SE + 0.002` を維持する。

quote別差分だけでLV価格の微分を受入しない。別にサーフェス全体を
`1±0.0001` と `1±0.0002` で摂動してDupire局所分散を再構築し、共通乱数の価格差分を
求めた。独立32バッチ・各512反対変数ペアの差分からSEを計算する。
上下価格の周辺SEを加算して共分散を無視することはない。
投影リスクとの比較は `3% + 6 paired SE`、市場価格との連鎖則は別判定である。
有限の受入パネルであり、あらゆるパラメータ・尾部・粗い格子の精度を保証するものではない。

## バックテストへの接続

```python
provider = RustPricingProvider(numerical_scope="p0_terminal_replication")
```

CLIは `--validated-p0`。既定の無指定構築は引き続き拒否し、診断用は
`diagnostic_only=True` / `--diagnostic-only` を明示する。両モードの同時指定は拒否する。
受入済みモードはゼロ配当European、受入方式名、厳密な報告満期を要求し、**各live契約の
native価格と市場IVによる独立Black価格を `6 price SE + 0.002` で照合する**。
失敗は `native_market_calibration_failure` として停止し、台帳の途中状態を破壊しない。
検証状態は `numerically_validated_p0`、モードはprovider/再開の識別情報に入る。

eSSVIの11営業日・4契約（2満期）で、無ヘッジ・Delta・scalar Vega・VegaKTの
4戦略を実行した。通常の報告座標と固定市場ショックの両方で完走し、後者の全戦略で
5日目からの再開と一括実行が完全一致した。P&L日次照合の最大誤差は `2.624e-13`。
input/result/checkpointのSchema検証とCLIによる再開も合格。
合成市場と同じモデルの価格を約定quoteに使う統合試験であり、戦略優位性の根拠ではない。
Vegaを含む完全なDelta/IV損益分解も、この受入には含めない。

## 回帰・再現性

- Rust価格統合10件、Local Vol関連46件、native Python VegaKT関連44件、
  companion212件が成功。新Python6件はSchema依存を省いたwheel相当環境でも成功。
- RQMCでも厳密リスクの標本統計を捏造せず、価格・Delta・local VegaはVegaKTなしの
  同一乱数評価と一致する。worker 1/4、seed変更、call/put、notional換算、v3 wire往復を検査。
- 経路依存Asianは受入方式へ昇格せず、診断経路を保持。不整合な市場IVは受入モードで拒否。
- 再構築したnativeバイナリで全数値判定を実行。Clippyの警告は0。
- macOS arm64 replayは新fixtureと完全一致。変更前fixtureを丸ごと保存し、
  [218項目の差分監査](../../fixtures/replay/local-vol-terminal-replication-change-audit-v3.json)を追加。
  212項目はVegaKT、2項目は計算方式を含むplan指紋、4項目は既存スカラー統計経路への
  切替に伴うpseudo-MC Delta/Gamma SEの1 ULP差。価格・各Greekの点推定・入力・
  price-onlyケースは不変。連続配当2%を含むreplayのquote差分誤差は `6.804e-10`。
- Windowsの現在のアルゴリズムによるcaptureは未取得。既存の古いfixtureを
  macOSの結果で置換せず、そのプラットフォームの厳密再現受入は未完了とする。
- 既存wire/バックテストSchemaや旧数値ポリシーを変更していない。nativeバイナリと
  companionソースの識別が変わるため、旧版checkpointは流用できない。

機械可読の判定内訳・6件の処置・ソース/バイナリ指紋は
[JSON記録](vega-kt-smile-acceptance-v0.3.json)。詳細は
`benchmark-results/vega-kt-smile-v0.3-final/report.json` と
`benchmark-results/vega-kt-smile-v0.3-daily/integration-audit.json` に保存した。

```shell
python -m rust_pricing_backtest.smile_numerics \
  --policy fixtures/backtesting/smile-policy-v0.3.json \
  --output benchmark-results/vega-kt-smile-v0.3
python examples/python/vega_kt_backtest.py --paths 2048 --validated-p0
cargo test --locked -p pricing --test local_vol_vegakt_pricing
cargo test --locked -p pricing --lib local_vol
python -m unittest discover -s tests/python -p 'test_vega_kt*.py'
python -m unittest discover -s backtesting/tests
```

Python実行には今回のソースから再構築したnativeライブラリとcompanionが必要。
