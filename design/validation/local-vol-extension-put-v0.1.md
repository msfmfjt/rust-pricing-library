# 全区間較正のPut受入と混合ヘッジ台帳

日付: 2026-10-10。[集約結果](local-vol-extension-put-v0.1.json)。
[全区間較正](local-vol-variance-extension-v0.1.md)と[Call側provider](local-vol-extension-provider-v0.1.md)を基礎に、
Put側の独立した価格差分を検査し、明示的なPut対応scopeへ接続する。

## API

```python
provider = RustPricingProvider(
    numerical_scope="multi_step_extension_put_gamma",
    variance_extension="all_rows",  # または linear_time
    worker_threads=4,
)
result = run_backtest(config, history, contracts, portfolio, provider)
```

```sh
python -m rust_pricing_backtest input.json --output result \
  --validated-put-extension linear_time
```

新scopeはEuropean・算術平均Asian・固定行使価格LookbackのCall/Putを扱う。
既存 `--validated-extension` はCall専用を維持する。方式・scope・ソース・nativeバイナリの識別子が
異なればcheckpointを再開できない。今回のソース更新後、旧checkpointは保存入力から再実行する。

4/6区間、33分散節点、2区間ごとの市場満期に各5 quotes、単位割引、carry・配当なし、満期払いに限定する。
4区間4,096組以上・6区間2,048組以上のantithetic標本を要求し、既存native計算量上限を維持する。
価格・Deltaは通常評価、Vegaは全区間の条件付きGammaと市場IV方向の接線から求める。
精度不合格時はRiskとcacheを確定せず、当該snapshotの発注へ進まない。

## Putを直接評価する独立基準

最終遷移を解析積分し、それ以前の経路を8 scrambled Sobol組・各2^16点で積分する。
Putの負側正規確率から直接価格を求め、Callの感応度を流用しない。
最終shockの損益を直接数値積分する別実装で、Asianの実効行使価格が負になる場合も照合した。

Lookback Putでは最大値でなく最小値を使う。既知最安値Hに対して
`(K - min(H,S))⁺ = (K-H)⁺ + (min(K,H)-S)⁺` を用いる。
未観測、過去の最安値98、過去98と当日90の固定観測、の3ケースを独立に検査する。

各市場quoteを±1e-5、±5e-6動かした再較正曲面で価格差分を計算する。
較正は商品のpayoffに依存しないため、前回のパラメータ増分から曲面を復元し、
リクエストhash・較正ソース・nativeバイナリ・独立quote価格の合格証跡を照合して再利用する。
Putの価格と価格差分は新規計算し、native AD/Gammaや較正Jacobianを参照しない。

両方式、4/6区間、6種類のPut履歴、2 seedsを対象とする。
価格・市場方向Gammaと独立価格差分を6標準誤差＋0.002で比較し、差分幅の収束も検査する。
European Putは市場quoteに対応するBlack Vegaにも照合する。
同一標本で合成した方向間共分散、各rowの回復と条件付きAD照合を保持する。

## 既知の折れ曲がりを積分区切りへ含める

初回には、4区間・過去最安値98のLookback Putで初期rowの回復精度検査が失敗した。
両方式・両seedの4パネルで8検査が失敗し、価格差分・条件付きADとの照合は合格していた。
固定の市場bin境界だけでは、既知の折れ曲がり98をまたぐ補間誤差が残っていた。
新scopeではPut Lookbackに限り、行使価格・過去最安値・当日固定値の最小値を積分区切りへ加える。
nativeではbin境界を回復曲面の区切りにも使うため、追加された報告binも監査に明示される。

標本数、分割数512、相対許容差1e-6、相対SE上限25%を変えずに検査する。
Rust本体・全区間較正の数値実装は変更していない。初回の失敗と独立価格計算を保存し、
過去最安値ケースの両方式・両区間数・両seedを再計算する。
他の合格パネルはソースとバイナリを照合して引き継ぎ、想定外の失敗があれば引き継がない。

48パネル・3,076検査が合格した。40パネルを同一数値ソースから引き継ぎ、
既知最安値付きの8パネルを新規再計算して、初回の8失敗を解消した。
4区間・all_rows・seed 7の初期rowでは、分散方向の細分化誤差が約7.34e-4から2.39e-4へ縮小した。

通常providerも同じ区切り規則を使う。provider経由の価格・Deltaと通常評価を比較し、
市場方向Gammaの全計算値を独立受入済みnative監査へ照合する。
request/plan識別子は比較対象から除き、bin入力ラベルの浮動小数点丸めだけを1 ULP以内で別検査する。
計算値にこの許容差を適用しない。

## 台帳と配布物の検証

PutターゲットとCallヘッジを組み合わせ、4区間では当日観測を含む4評価日、6区間では2評価日から満期まで進める。
両方式・Asian/Lookback・全4戦略で、売買、観測値からの直接決済、schema、途中保存・再開の完全一致を検査する。
固定値の欠落・未来の公表・採用済み観測の改訂を拒否し、失敗snapshot以前の状態を保持して修正後に再開する。
異なる較正方式やCall専用scopeでの再開拒否も検査する。

providerの24パネルと混合ヘッジ32戦略が合格し、48件の契約評価監査を保存した。
24パネルの価格・Gamma・共分散などの計算値は、独立数値受入済みの監査と完全一致した。
補助ライブラリ255件、既存native回帰13件、新provider5件、独立Put式2件の計275テストが合格した。

配布wheelをインストールした環境で、Lookback PutとCallヘッジの満期決済までを再実行した。
ソース実行の結果・checkpointと完全一致し、途中再開・異方式の拒否・実CLIの起動も合格した。
Pythonとschemaの54ファイルが配布物と一致し、旧Call版との差分はadapter・provider・CLIの3ファイルに限定される。
生成schemaの整合性を確認し、CIと配布ソースの必須ファイルにも今回の検証を追加した。

生の結果・初回失敗・実行ログは `benchmark-results/vega-kt-extension-put/` に保存し、
集約JSONにソース・数値結果・配布物のSHA-256を保持した。

任意custom基底、他の区間数/節点配置、carry・配当、異商品の合同共分散、実市場のヘッジ効果は未受入。
`full_market_iv_bucket_acceptance=not_established` を継続する。

## 再実行

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_extension_put_refinement.py \
  --policy fixtures/backtesting/variance-extension-put-refinement-policy-v0.1.json \
  --baseline-report benchmark-results/vega-kt-extension-put/numerical-report.json \
  --output benchmark-results/vega-kt-extension-put/refined-numerical-report.json
OPENBLAS_NUM_THREADS=1 python scripts/check_extension_put_provider.py \
  --reference-report benchmark-results/vega-kt-extension-put/refined-numerical-report.json \
  --output benchmark-results/vega-kt-extension-put/provider-report.json
```

Putの初期reportがなければ、前段のCall側較正受入reportを使い初期行列から生成する。
