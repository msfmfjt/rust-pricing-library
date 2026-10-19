# 時間・マネーネスを結び付けた基底の台帳接続

日付: 2026-10-12。[集約記録](local-vol-coupled-basis-provider-v0.1.json)。
[raw行列の独立数値受入](local-vol-raw-basis-v0.1.md)で検証した4/6区間の行列を、
`coupled_row_hats` の明示選択でCall/Putのヘッジ台帳へ接続する。

## 検証結果

20評価パネルで基底・較正接線・native Gamma・共分散が前回の独立数値基準と完全一致した。
32戦略の決済・再開、48件の契約評価監査、16件の失敗・修復検査が合格した。
補助ライブラリ266件、raw入力・尺度4件、既存CLI3件、接続の5テスト×2区間、
計283テスト実行が合格した。精度基準や標本数は変更していない。

配布wheelから4/6区間のLookback Put＋Callヘッジを再実行した。
4区間は同じ4 workersで結果全体が完全一致した。6区間は1 workerで実行し、
全計算値・売買・決済・損益と4件のnative監査が4 workersのソース版と完全一致した。
実行設定に依存する識別子16箇所だけを明示的に区別し、保存状態・損益表・リスクのhashを再計算して検証した。
4区間の8件を合わせ、配布版のnative監査は計12件が完全一致した。
両区間で別providerからの途中再開が一致し、6区間では実CLIからの再開も一致した。

数値ソース424ファイル（Rust 360ファイルを含む）とnativeバイナリを維持した。
配布Python/schema 56ファイルはソースと一致し、前版との差分は接続部分の4ファイルだけである。
既存2基底の行列・構成監査・較正接線も4/6区間で前回と完全一致した。
今回の新しい単体テストは既存CIの補助ライブラリ検査に含まれる。数値受入は保存したローカル実行の証跡である。

## 選択方法と行列の固定

```python
provider = RustPricingProvider(
    numerical_scope="multi_step_custom_basis_gamma",
    variance_basis_policy="coupled_row_hats",
    worker_threads=4,
)
```

```sh
python -m rust_pricing_backtest input.json --output result \
  --validated-custom-basis coupled_row_hats
```

`variance_extension` は指定しない。従来の `row_hats` / `weighted_time_hats` も選択できる。
基底を外部ファイルから自動採用する機能ではなく、実装した生成規則から行列を作り、
次の受入済みhashと完全一致した場合だけ較正へ渡す。異なる節点配置などで行列が変わると拒否する。

| 区間数 | 行列形状 | 受入済み基底hash |
| --- | --- | --- |
| 4 | 132×68 | `8d957c3f233540f07698515b1a985d7d2760b0b28d99e9359dddb65c8f4293e7` |
| 6 | 198×102 | `821589f0d43f22d8593f009dafe268cd473a591c4f4fef6630ab807e7c775f33` |

行列は `row_hats` に固定のパラメータ混合行列を右から掛けて作る。
後続時間rowの同じマネーネス節点と隣の節点へ影響し、隣接節点への負の寄与も含む。
列を正規化せず、Euclideanパラメータノルムで較正する。これは元のrow_hatsと異なる較正規則である。
`basis_construction` には、混合前の節点・列尺度と基底hash、混合規則・係数・混合行列hash、
最終基底hashを保存する。較正・Gammaの監査が同じ基底hashを示すことも要求する。

## 受入範囲と保存状態

European・算術平均Asian・固定行使価格LookbackのCall/Put、4/6区間、33分散節点、
2区間ごとの市場満期と各5 quotes、carry・配当なし、満期払いを対象とする。
価格とDeltaは通常評価、Vegaは全区間Gammaと市場IV較正接線から求める。
既知のPut Lookback最安値・当日確定値を積分区切りに反映する。

4区間4,096組以上・6区間2,048組以上、antithetic、分割数512、相対許容差1e-6、
相対SE上限25%を維持する。商品・市場較正・精度検査に失敗したsnapshotでは、
Riskとcacheを確定せず、発注を停止する。観測値の欠落・未来公表・採用済み値の改訂も拒否する。

基底名と実装を保存状態へ結び付ける。同じ実装・基底での途中再開は一括実行と一致する。
今回の更新前のcheckpoint、および異なる基底のcheckpointは拒否する。
旧版の保存入力は保持し、新版では入力から再実行する。

合格は検証した行列・商品履歴・市場条件と精度方針に限定する。
任意の行列、任意市場の全面受入、異商品の合同共分散、実市場のヘッジ効果は未受入。
低水準のGamma・raw行列APIは診断用を維持し、`full_market_iv_bucket_acceptance=not_established` とする。

## 検証方法

独立価格差分を含む前回の40パネル・2,770検査の結果とnativeバイナリを固定する。
接続部分4ファイルを除く較正・Gamma・価格基準・方針のソースを維持する。
今回の20パネル（4/6区間×10商品履歴×seed 7）で、基底・較正接線・Gamma・共分散を
前回の独立受入済み監査へ照合する。計算値には丸め許容差を加えない。
request/plan識別子と積分区切りの入力ラベルは分けて扱い、ラベルの差は1 ULP以内を確認する。

Call/PutのAsian/LookbackとCallヘッジについて、各4戦略の決済、schema、途中保存・再開を検証する。
支払額は観測値から直接求める。精度失敗・基底不一致を注入し、発注とcacheが残らず、
修正後に同じ結果へ戻ることを確認する。基底行列の変更も単体テストで拒否を確認する。

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=backtesting python scripts/check_coupled_basis_provider.py \
  --intervals 4 --output benchmark-results/vega-kt-coupled-basis-provider/4.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=backtesting python scripts/check_coupled_basis_provider.py \
  --intervals 6 --output benchmark-results/vega-kt-coupled-basis-provider/6.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=backtesting python scripts/check_coupled_basis_provider.py \
  --aggregate benchmark-results/vega-kt-coupled-basis-provider/4.json \
              benchmark-results/vega-kt-coupled-basis-provider/6.json \
  --output benchmark-results/vega-kt-coupled-basis-provider/report.json
```

`--recheck-report` は同じソース・native・参照結果・区間数による合格済みruntime結果の再照合に使う。
4/6区間とも、この再照合の出力は元の完了結果とバイト単位で一致した。
前回のrawチェッカーは前回版のソース・バイナリに固定されているため、現行の台帳接続の再検証には上記のスクリプトを使う。
生の結果は `benchmark-results/vega-kt-coupled-basis-provider/` に保存する。
