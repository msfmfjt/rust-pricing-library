# 全区間の分散較正と市場方向Gamma

日付: 2026-10-09。集約結果は[JSON](local-vol-variance-extension-v0.1.json)。
各報告満期直前の分散rowだけを動かす既存方式に加え、初期遷移を含む全区間を同時に動かす
較正と、その市場IV方向のGamma計算を診断APIに追加する。

## 延長規則を明示する

`MultiStepExtensionCalibrator` は、全区間の分散増分を `B @ theta` とする。
配列順序は遷移row、log moneyness節点。模擬遷移に使わない終端grid rowは固定する。

- `all_rows`: Bは単位行列。全ての遷移row・分散節点の増分を独立に選ぶ。
- `linear_time`: 報告満期ごとの分散増分を区間の中点へ線形補間する。両端は定数外挿。
  ストライク方向は元の33節点を保持する。
- `custom`: 利用者が有限の2次元Bを明示する。市場quotesを張れない基底、形状不正、数値分解の失敗を拒否する。

同じ市場quotesを再現しても、延長規則が異なればAsian・Lookbackの感応度は異なり得る。
基底の列の尺度もパラメータの距離を変えるため、較正規則の一部である。
監査情報に規則名、基底の形状とhash、全市場quoteの順序を保存する。

各Newton反復で、市場Vegaで規格化した価格JacobianをBへ写し、全quotesを同時に満たす
最小Euclideanパラメータ増分を選ぶ。これは反復ごとの最小ノルムであり、
最終曲面が大域的に最も近いという主張ではない。
分散floor/cap、quoteの正値・上下限・strike方向凸性・満期間の価格順序を検査し、
範囲内で残差を減らすbacktrackingだけを採用する。クリッピングや失敗時の代替曲面は使わない。

## 初期遷移を含む価格微分

vanilla callの密度積分と、各lognormal遷移kernelのscoreを使い、全rowの価格Jacobianを計算する。
初期状態の分散も動くため、最初のGaussian密度の微分を追加した。
初期積分分散をv、標準正規座標をz、期間をdtとすると、使用するscoreは
`dt * ((z*z-1)/(2*v) - z/(2*sqrt(v)))`。
初期spotのlog moneyness=0での補間weightsを掛け、節点感応度へ配賦する。

coarse/fineの積分で市場IV応答の単位行列誤差を1e-9以下とする。
全rowの接線を既存native条件付きGammaへ渡し、同一sampling unitで各rowを合算する。
方向間の共分散はその同じ標本から計算し、独立rowとみなして標準誤差を足さない。
較正失敗は例外、native精度失敗は診断を残して `estimate=None` とし、市場共分散の採用値も返さない。

## APIと適用範囲

```python
from rust_pricing_backtest.multi_step_extension import (
    MultiStepExtensionCalibrator, evaluate_extended_market_gamma,
)
calibrator = MultiStepExtensionCalibrator(payload, extension="linear_time")
fitted_payload, calibration_audit = calibrator.calibrate_all_maturities(iv_shifts)
audit = evaluate_extended_market_gamma(
    payload, [80., 90., 100., 110., 120.], extension="linear_time",
    subdivisions=512, relative_tolerance=1e-6,
    gamma_relative_standard_error_limit=.25, worker_threads=4,
)
```

今回の数値受入は4・6区間、33分散節点、2遷移ごとの満期に各5 quotes、
非平坦な2市場プロファイル、Call・Asian・既知観測付きAsian・LookbackのCall側を対象とする。
2つのpresetを両方検証する。任意のcustom基底すべての受入ではない。
単位割引、carry・配当なし、元の離散時間遷移の条件を継続する。

既存 `RustPricingProvider` と `--validated-path` の較正規則は変更していない。
新APIは `diagnostic_only=True` を返す。新しい延長規則の通常台帳への接続、Put側への数値行列拡張、
他の時間/ストライク配置、carry・配当、異なる商品の合同共分散、実市場のヘッジ効果は未受入。
`full_market_iv_bucket_acceptance=not_established` を継続する。

## 独立した価格差分

native AD・Gamma・較正Jacobianを使わないSobol経路積分で、最後の遷移を解析積分する。
全市場quotesを一つずつ±1e-5、±5e-6だけ動かして毎回再較正し、各商品の価格差分を求める。
8つの独立scramble、各2^16点を使い、差分幅間は同じ乱数で照合する。
別実装の密度積分でも、再較正後の全quote価格が1e-10以内で一致することを要求する。
第1満期の±0.003平行IV移動について、後続満期のquote価格が保たれることも検査する。

Gammaは各rowの回復・条件付きAD照合、方向の共分散、独立した再較正価格差分に照合する。
European Callは市場quoteと一致する行使価格なので、終端ATM quote以外の感応度0と
終端ATMのBlack Vegaにも比較する。

## 最初の失敗と精細化

最初の診断は全条件を2,048 antithetic組で評価した。
4区間Asian系・seed 7で中間rowの条件付きAD照合が失敗し、その推定値を公開しなかった。
4区間について全商品・両方式・両seedを4,096組で再評価する。
6区間は全rowの計算量を既存native上限に収める2,048組とする。
6区間Asianを4,096組で要求した場合の上限超過拒否もテストで維持する。
精細化は最初の失敗結果を保存し、合格した独立価格証跡と6区間Gammaを、
較正・価格・nativeソースとバイナリの一致を確認して引き継ぐ。

較正接線と有限差分の直接比較では、幅1e-5の打切り誤差も検出した。
追加の±1e-6再較正では元の絶対許容差2e-5を維持する。
元の2幅から求めるRichardson外挿は、より小さい絶対許容差2e-6で照合する。
大きい差分幅での失敗を削除したり、許容差を広げたりしない。
Gammaは積分分割512、相対許容差1e-6、相対SE上限25%、独立価格比較は
6標準誤差＋0.002を維持する。Rust本体には変更を加えない。

結果: **32パネル・2,546検査が合格、最終失敗0**。関連テスト**265件合格**。
4区間は全16パネルを4,096組で再計算し、6区間の16パネルは2,048組の合格証跡を保持した。
市場方向は4区間で10、6区間で15。2方式 × 4商品 × 2 seedsを検査した。
元の41不合格検査（接線の有限差分33、4パネルのGamma採否・中間row照合8）を初期reportに残している。
独立の再較正価格差分そのものには初期から不合格はない。

追加した50方向の接線検査で、最大誤差は±1e-6の差分で2.42273e-06、
Richardson外挿で7.09528e-09。許容差はそれぞれ2e-5、2e-6である。
最終受入には、小さい差分幅での独立quote価格の再確認と、増標本後のGammaと独立価格差分の照合を含める。

新しいcompanion wheelを導入し、別プロセスで4区間Asianの全区間Gamma監査が完全一致した。
以前のLookback観測日バックテストも保存入力から再実行し、価格・リスク・売買・決済・状態が完全一致した。
新モジュールを含む実装hashと、それに由来するmanifest/checkpointのidentity hashだけが更新された。
旧wheelと新しい導入済みソースから各hashを再計算し、差がこの3項目だけであることを確認した。
旧checkpointは既存の整合性検査で拒否され、新版での途中保存・再開は新版の一括実行と全結果が完全一致した。
旧wheelのPython・schemaファイル53個はbyte同一で、新しいモジュールだけを追加している。
Rustソースとnativeバイナリは変更していない。通常providerの較正规則も引き続き従来どおりである。


## 再実行

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_variance_extension_refinement.py \
  --policy fixtures/backtesting/variance-extension-refinement-policy-v0.1.json \
  --baseline-report benchmark-results/vega-kt-extension/report.json \
  --output benchmark-results/vega-kt-extension/refined-report.json
```

baselineがない場合は初期診断から実行する。初期診断の失敗を保存し、想定外の失敗がある場合は
引き継ぎを拒否する。再利用した証跡と新規計算を最終reportで区別する。
