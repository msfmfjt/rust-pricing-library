# 複数満期Gammaのバックテスト接続

日付: 2026-10-07。集約結果は[JSON](local-vol-multi-step-provider-v0.1.json)。
[条件付きnative ADの受入](local-vol-conditional-ad-v0.1.md)を終えたGammaを、
既存の欧州型バックテスト台帳に接続する。一般の経路依存商品台帳の受入とは区別する。

## 利用範囲と入口

```python
provider = RustPricingProvider(numerical_scope="multi_step_gamma", worker_threads=4)
risk = provider.evaluate(snapshot, contract)
audit = provider.gamma_audit(snapshot, contract)
```

CLIでは `--validated-multi-step` を指定する。従来のP0と診断モードは継続する。
新モードの結果には `numerically_validated_multi_step_european` を記録する。

- 欧州型Call/Put、単位notional、最後の報告満期で満期・同日払い。
- 4・6遷移、局所分散33節点、2遷移ごとに5市場quotesを報告する2・3満期。
- 割引係数1、carry・配当なし。非平坦スマイルを許容する。
- 4,096以上の独立pseudo-MC標本、集計block 256。
- 各報告満期直前の分散rowを較正し、報告されないrowを固定する。

市場quotesの価格一致（精細化込みで1e-10）、較正接線の精細化、Gamma回復、
条件付きnative AD照合、方向ベクトルの相対SE上限25%をすべて要求する。
Gamma積分分割数512、相対許容差1e-6と6 SEの基準を維持する。
不合格時にP0や旧診断値へ切り替える処理はなく、リスクを返さずバックテストを停止する。

各snapshotの絶対日付から計算したACT/365満期と、モデルの報告満期・節点順を照合する。
同じsnapshotの各商品は市場・IVに加えて局所分散モデル全体も共有する。
数量と契約倍率は既存台帳で適用する。

## 値・不確実性・再開

価格とfrozen-grid Deltaには従来の通常評価を用い、その分布の価格がGamma監査と一致することを確認する。
旧VegaKT診断の同時計算を避けるため、通常評価のrequestからVegaKTだけを外す。
受入済みの市場Gammaを `Risk.buckets`、同一標本の標準誤差を `bucket_standard_errors` に渡す。
報告する全市場quote方向を直接評価するため、追加の射影はなく、射影残差は0である。

`gamma_audit` は較正接線、節点照合、標本共分散と平均共分散を含む独立コピーを返す。
共分散は同一商品の方向間に限られ、異なる商品のポートフォリオ共分散とは扱わない。
既存のヘッジ最適化は点推定を使い、共分散を使った最適化には変更しない。

request、snapshot、契約、Gamma監査をリスクのprovenanceに結びつける。
provider identityにはnativeバイナリに加え、Pythonの較正・接続コードのhashを含める。
モード・実行設定・実装が変わった場合、以前のcheckpointからの再開を拒否する。
失敗した評価は成功キャッシュにも共有面の登録にも残さない。

## 検証

4・6区間、2 seeds、Call/Putの8条件を検査する。対象strikeは直接較正した終端quoteであり、
独立な市場Vega参照はBlackの解析式とPut–Call parityから求める。
終端の該当quote以外の市場偏微分は0となる。この参照との6 SE＋0.002照合を全10・15座標で行う。
価格とDeltaがVegaKTなしの通常評価と完全一致することも検査する。

日付とspotを進めた3つの合成較正面、および満期snapshotで、
unhedged・delta・scalar_vega・vega_ktの4戦略を既存台帳に通す。
注文、翌snapshotでの約定、損益、満期決済、入力/結果/checkpointスキーマ、
中断再開の完全一致を確認する。合成価格による接続試験であり、実市場のヘッジ効果の受入ではない。

6区間、strike 200、4,096標本では実際にGamma精度判定が不合格となる。
市場推定値・共分散を抑止し、注文・約定を作らないことを確認する。
これとは別にnative判定失敗を注入し、キャッシュ非汚染と再試行を検査する。
不整合IV、carry、報告座標の不一致、局所分散モデルの混在、低標本、経路依存商品、
遅延払い、未受入の集計blockも拒否する。

結果: **8条件・市場Vega 100座標・4戦略が合格**。最大照合比は0.432944（許容上限1）。
新規7テスト、既存native連携32テスト、companion 244テスト、計**283テスト合格・失敗0**。
再実行の監査・台帳出力は完全一致した。現行入力・結果・checkpointスキーマを維持する。

Rust Gammaのソースとnativeバイナリは前回受入と同一。前回5,904検査の全行列を
今回再計算したとは扱わない。今回は接続に必要なCall/Putと台帳・拒否条件を検査した。
更新したcompanion wheelを導入し、新しいプロセスでVegaKT戦略全体を実行した結果も完全一致した。

| 戦略 | 約定数 | 決済イベント数 | 再開 |
|---|---:|---:|---|
| unhedged | 0 | 2 | 完全一致 |
| delta | 2 | 2 | 完全一致 |
| scalar_vega | 3 | 4 | 完全一致 |
| vega_kt | 3 | 4 | 完全一致 |

## 再実行

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_multi_step_provider.py \
  --output benchmark-results/vega-kt-multi-step-provider/report.json
python -m rust_pricing_backtest \
  benchmark-results/vega-kt-multi-step-provider/input.json \
  --output benchmark-results/vega-kt-multi-step-provider/replay \
  --validated-multi-step
```

受入スクリプトのレポートに監査と台帳結果を保存し、CLIで再実行できる入力を併せて出力する。

## 未受入の範囲

Asian・Lookbackのnative Gammaは前回までの数値受入を維持するが、
既存 `Contract` のpayoff・満期決済は欧州型専用である。
そのため今回のproviderはこれらの商品を引き続き拒否する。
観測履歴・既知fixingの保持と経路依存決済、満期構成が変わる長期ロール、
一般の較正延長、異なる商品の合同共分散、carry・配当・barrier・早期行使は未受入。
`full_market_iv_bucket_acceptance=not_established` を維持する。
