# 現行版での基底行列別検証と安全な再開

日付: 2026-10-13。[集約記録](local-vol-current-raw-basis-v0.1.json)。
[結合基底の台帳接続](local-vol-coupled-basis-provider-v0.1.md)後の実装で、
行列ごとの独立数値検証を実行できる入口を追加する。

## 検証結果

4/6区間の2行列を先頭から再計算し、40パネル・2,770検査がすべて合格した。
独立価格、104件の較正・ストレス監査、接線、Gamma、共分散と全検査結果は、
前回の結果と完全一致した。来歴・保存プロトコル・内容hashの4項目だけを数値比較から区別する。

| 区間数 | 全数値パネル | 検査数 | 途中再開時の保持／再計算パネル |
| --- | --- | --- | --- |
| 4 | 20 | 1,125 | 18／2 |
| 6 | 20 | 1,645 | 18／2 |

両区間で独立価格と18パネルを保持し、残り2パネルから再開すると、一括実行の出力とバイト単位で一致した。
旧版結果・事前確認結果・破損結果・基底変更後の再開、入力・再開元への上書きのCLI拒否6件も合格した。
新しい9件の単体テストは、nativeモジュールを読み込めない条件での再実行でも全件合格した。
この単体検査は来歴取得を模擬しており、実際のnativeバイナリの数値受入は上記40パネルで別に実施した。

元の行列を一部変更した例でも事前確認できたが、状態は `ready_for_validation` のまま保持した。
その行列の数値合格や、元の行列の結果からの再開は認めていない。

## 変更理由

旧 `check_raw_variance_basis.py` は台帳接続前のソースへ固定されており、
接続部分が更新されると `basis_reference` で拒否する。
新入口は現行の受入記録へ結び付け、Rust本体・配布Python/schema・価格基準・方針と
旧チェッカーのソースを検査する。旧チェッカーは変更せず、独立したモジュールとして読み込む。
そのインスタンスの来歴取得と結果保存だけを差し替え、数値計算・検査・許容差・標本数はそのまま実行する。
旧チェックのモジュールや過去の受入記録は変更しない。

## 計算前の確認

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_current_variance_basis.py \
  --intervals 4 \
  --basis-file fixtures/backtesting/raw-coupled-variance-basis-4-v0.1.json \
  --preflight --output benchmark-results/vega-kt-current-raw-basis/4-preflight.json
```

市場・格子・行列の結び付け、有限値・形状、現行のソースとnativeバイナリ、
全quoteを動かせる較正Jacobianの階数と接線を確認する。
成功状態は `ready_for_validation`、`numerical_acceptance=not_evaluated`。
事前確認だけでは独立価格・Gammaの数値受入は成立せず、台帳への採用も行わない。
6区間には `--intervals 6` と対応する6区間の行列ファイルを使う。

## 全数値検証

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_current_variance_basis.py \
  --intervals 4 \
  --basis-file fixtures/backtesting/raw-coupled-variance-basis-4-v0.1.json \
  --output benchmark-results/vega-kt-current-raw-basis/4.json
```

行列JSONの形式と市場fixtureは[従来の行列別検証](local-vol-raw-basis-v0.1.md)と同じ。
別の行列も指定できるが、その行列について全検証を新しく実行する必要がある。
評価市場は4/6区間の固定fixture。任意の市場を入力できる入口ではない。
全市場quoteの±1e-6・±5e-7再較正、±0.001の第1満期ストレス、独立価格差分、
10商品履歴・2 seedsのGammaと共分散、行ごとの回復・AD照合を検査する。

数値受入の合格は入力行列・市場・商品履歴・seed・精度方針に限定する。
`diagnostic_only=true`、`provider_promotion=false`、
`full_market_iv_bucket_acceptance=not_established` を維持する。
通常台帳の3つの名前付き基底の選択方法と受入範囲は変わらない。

## 保存結果の検査と再開

partial結果の `phase` が `native` になった段階で、独立価格の全計算が保存される。
この段階以降は未完了のnativeパネルから再開できる。

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_current_variance_basis.py \
  --intervals 4 \
  --basis-file fixtures/backtesting/raw-coupled-variance-basis-4-v0.1.json \
  --resume benchmark-results/vega-kt-current-raw-basis/4.partial.json \
  --output benchmark-results/vega-kt-current-raw-basis/4-resumed.json
```

結果には `runtime_protocol=current_variance_basis_validation_v1` と
内容の `report_content_sha256` を保存する。再開前に内容の整合性を検査し、
さらに行列・市場・ソース・nativeバイナリ・方針・較正接線が一致することを要求する。
内容hashは保存事故や取り違えの検出用であり、発行者の署名や第三者による認証ではない。

旧版の結果、破損した結果、事前確認だけの結果、独立価格の途中結果は再開対象にしない。
異なる行列やソースでの再開も拒否する。旧版の数値結果は保存し、新版では先頭から実行する。
出力先は入力行列・再開元・受入済みソースと別に指定する。
最終結果・partial・一時保存の名前と、シンボリックリンク・ハードリンクによる同じファイルへの上書きも検査する。
出力先が衝突する場合はエラー結果で元ファイルを上書きせず、非ゼロ終了する。

## 互換性と証跡

ライブラリ本体・配布Python/schema 56ファイルとnativeバイナリは変更しない。
前回の結合基底台帳の途中保存から決済まで再開でき、前回の受入結果の再照合もバイト単位で一致する。
新しい9件の入口・整合性テストは通常CIへ追加する。数値受入の実行記録はローカル実行の証跡とする。
生の結果とログは `benchmark-results/vega-kt-current-raw-basis/` に保存する。
