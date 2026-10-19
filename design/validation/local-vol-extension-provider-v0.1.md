# 全区間較正を使うヘッジバックテスト

日付: 2026-10-09。[集約結果](local-vol-extension-provider-v0.1.json)。
[全区間較正の独立数値受入](local-vol-variance-extension-v0.1.md)で合格した2つの延長規則を、
明示的なprovider設定として台帳へ接続する。

## 選択と識別

```python
provider = RustPricingProvider(
    numerical_scope="multi_step_extension_gamma",
    variance_extension="all_rows",  # または linear_time
    worker_threads=4,
)
result = run_backtest(config, history, contracts, portfolio, provider)
audit = provider.gamma_audit(snapshot, contract)
```

```sh
python -m rust_pricing_backtest input.json --output result \
  --validated-extension linear_time
```

延長規則の省略、custom基底、他のnumerical scopeとの併用を拒否する。
providerの識別子には規則名・較正ソース・nativeバイナリを含め、cache・リスク由来情報・
manifest・checkpointへ引き継ぐ。異なる規則で保存状態を再開できない。
公開プロパティの `variance_extension` は読取り専用。

既存 `--validated-path` / `--validated-multi-step` は従来の選択row較正を使う。
今回のソース更新に伴い実装識別子は変わるので、旧checkpointは保存入力から再実行する。

## 採用前の検査

入力曲面は市場quotesに較正済みであることが必要。4または6区間、33分散節点、
2遷移ごとの報告満期に5 quotesを要求し、密度積分で価格誤差1e-10以下を確認する。
Call側のEuropean、算術平均Asian、固定行使価格Lookbackに限定する。
観測履歴は日付・公表時刻・初回採用値を照合し、当日観測を含めたnative payoffと一致させる。
満期払い、単位割引、carry・配当なしを継続する。

標本数は4区間4,096組以上、6区間2,048組以上のantithetic擬似乱数を要求する。
各入力は既存native計算量上限にも従う。6区間Asianの4,096組はその上限を超えるため拒否される。
入力の標本数を自動削減したり、精度不合格後に別方式へ切り替えたりしない。

価格とfrozen-grid Deltaは通常の評価を使い、全遷移の条件付きGammaから得た市場IV方向のVegaを
ヘッジに使う。同じ標本単位の方向間共分散と標準誤差を保持し、価格分布の一致・市場座標順序・
各rowの回復・条件付きAD照合がすべて合格した後だけRiskとcacheを確定する。
積分分割512、相対許容差1e-6、相対SE上限25%は変更しない。

`gamma_audit` 内の低水準計算は引き続き診断情報を含む。台帳に採用できる範囲は
外側のprovider検査が定め、`diagnostic_only=False` と対象商品・較正方式を記録する。
異商品の合同共分散は未受入で、同一商品の共分散を合算して代用しない。

## 検証

4/6区間 × 2方式 × 4商品の16パネルを実評価し、通常価格・Deltaと共分散を確認する。
前回の独立価格差分とGammaの証跡は、数値計算ソースとnativeバイナリが一致するときだけ参照する。
通常評価で不要な旧VegaKT要求を外すためrequest/plan fingerprintは比較から除き、
数値監査の計算値は前回と厳密一致を要求する。
bin境界の入力ラベルだけは、`100 * 1.1` と定数 `110` の表現差を1 ULP以内で別検査する。
価格・Gamma・標準誤差・共分散にはこの許容差を適用しない。

Asian・Lookbackの4区間では当日観測を含む4評価日から満期まで、6区間では2評価日から満期までを
全4戦略で評価する。売買・日付付き観測による決済・schema・保存再開の完全一致・方式変更の再開拒否を検査する。
native失敗時の無発注とcache非汚染、同一入力の再試行、実際の計算量超過の拒否も確認する。

結果: **16価格・Gammaパネル、32戦略リプレイが合格**。関連テスト**273件合格、失敗0**。
台帳上の48契約評価監査を記録した。同一snapshotでのcache再利用を含むため、独立標本数とは区別する。
新しい16パネルのGamma数値監査は、独立価格差分に照合済みの前回結果と全計算値が一致した。
要求・plan fingerprintを除外し、bin入力ラベルは最大1 ULPの丸め差を別検査している。
初回の比較ではこのラベル差だけで16パネルが不一致となったため、そのreportを保存した。
checkerだけを修正し、native・provider・テストソースと参照reportのhashを照合して、合格済み実計算を再検査した。
価格・Gamma・標準誤差・共分散に1 ULPでも差があれば拒否する4件の回帰テストも追加した。
Rust本体および全区間較正の数値実装は変更していない。

新しいwheelを導入し、別プロセスでlinear_timeのLookback観測日リプレイを再現した。
全結果・途中checkpoint・再開結果がソース側の受入結果と完全一致した。
他方式・従来scopeのcheckpoint再開は拒否された。schema生成検査も合格した。
導入済みCLIの `--validated-extension linear_time` でも1評価日の実入力を完走し、結果schemaを検査した。
旧wheelとの比較では入口・provider・CLIの3ファイルだけが変わり、他のPythonとschema 51ファイルはbyte同一である。


今回も `full_market_iv_bucket_acceptance=not_established`。
Put側、任意custom基底、他の区間数/節点配置、carry・配当、実市場のヘッジ効果は未受入。

## 再実行

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_extension_provider.py \
  --reference-report benchmark-results/vega-kt-extension/refined-report.json \
  --output benchmark-results/vega-kt-extension-provider/report.json
```

参照reportは同じnativeバイナリで前段の全区間較正検査を実行して作る。
集約結果には固定したソースhash、参照report hash、全監査値、台帳結果、再現用入力を保存する。
