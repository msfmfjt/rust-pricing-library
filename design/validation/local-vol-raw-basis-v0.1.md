# 基底行列ごとの独立数値検証

日付: 2026-10-12。[集約記録](local-vol-raw-basis-v0.1.json)。
[名前付き基底の台帳接続](local-vol-custom-basis-provider-v0.1.md)を維持し、
入力したraw行列を個別に検証する診断用の入口を追加する。

## 受入結果

時間とマネーネスを結び付ける2行列について、全40パネル・2,770検査が合格した。
許容差を変えずに、100件の符号付きquote再較正と4件の有限幅ストレスを検証した。

| 区間数 | 行列形状 | 商品・seedパネル | 検査数 | 再較正曲面と接線の最大差 |
| --- | --- | --- | --- | --- |
| 4 | 132×68 | 20 | 1,125 | 5.789e-6 |
| 6 | 198×102 | 20 | 1,645 | 5.881e-6 |

接線の許容差は2e-5、独立価格によるquote誤差は全再較正・ストレスで最大9.238e-14
（許容差1e-10）。入力とパラメータ尺度の単体テスト4件、基底変更後の再開・市場の不一致・
階数不足のCLI拒否3件も合格した。4/6区間の完了結果から再開すると出力がバイト単位で一致する。
今回の4件の単体テストは通常のCIへ追加した。数値40パネルの結果は、この記録に保存した
ローカル実行の証跡であり、他のOSでの数値受入を意味しない。

## 入力と実行

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_raw_variance_basis.py \
  --intervals 4 \
  --basis-file fixtures/backtesting/raw-coupled-variance-basis-4-v0.1.json \
  --output benchmark-results/vega-kt-raw-basis/4.json
```

6区間には `--intervals 6` と `raw-coupled-variance-basis-6-v0.1.json` を使う。
行列JSONの `values` を変更すれば、その行列について新しく検証できる。
評価市場は従来の非平坦スマイルの4/6区間fixtureに固定する。任意の市場入力を受け付ける入口ではない。

JSONは `variance_basis_matrix` v1。`source_request_hash`、時間・マネーネス格子、
行の順序、パラメータの尺度の意味を市場fixtureへ結び付ける。
行は遷移row→マネーネス節点、列は較正パラメータで、形状は `(区間数×33, パラメータ数)`。
行列には有限な正負の数を指定できる。列の正規化は行わず、Newton反復ごとに
パラメータ増分のEuclideanノルムを最小化する。入力はコピーして保持する。

別市場・別格子・別尺度、重複JSONキー、非有限値、文字列や真偽値の行列要素、形状違いを拒否する。
全市場quoteを動かせるかは、較正Jacobianの階数と精細化を既存の基準で検査する。
不適合時は非ゼロ終了し、結果に `status=rejected` と理由を保存する。

独立価格の全検証が終わると、partial結果の `phase` が `native` になる。
この段階以降は、保存結果を指定して未完了のnativeパネルから続けられる。

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_raw_variance_basis.py \
  --intervals 4 \
  --basis-file fixtures/backtesting/raw-coupled-variance-basis-4-v0.1.json \
  --resume benchmark-results/vega-kt-raw-basis/4.partial.json \
  --output benchmark-results/vega-kt-raw-basis/4-resumed.json
```

入力ファイル・基底hash・市場・較正接線・方針・ソース・nativeバイナリが同一の場合だけ再開する。
独立価格の計算途中で中断した場合は先頭から実行する。

## 検査する数値

全市場quoteを±1e-6、±5e-7動かして同時再較正する。
別実装の価格積分でquote誤差1e-10以下、実際の再較正曲面の差分と接線の差2e-5以下を確認する。
第1満期のIVを±0.001動かし、後続満期のquotesが維持されることも検査する。

Call/PutのEuropean・Asian・既知観測付きAsian・Lookbackを検査する。
Put Lookbackは未観測、過去最安値98、過去98と当日90の3履歴を区別する。
独立価格は最終遷移を解析積分し、それ以前を8 scrambled Sobol組・各2^16点で積分する。
較正Jacobian、native価格、AD、Gammaを価格差分の計算に使用しない。

各商品の全市場方向Gammaを価格差分へ6標準誤差＋0.002で照合し、差分幅の収束も検査する。
4区間4,096組・6区間2,048組、antithetic、seeds 7/41、分割数512、
相対許容差1e-6、相対SE上限25%を維持する。
全rowの回復・条件付きAD照合、方向間共分散、Europeanの解析Vegaも確認する。

## 時間とマネーネスを結び付ける検証用行列

既存のrow_hatsを出発点とし、あるパラメータが次の時間rowの同じ節点と隣の節点にも影響するようにする。
結合係数を時間rowとマネーネスの両方で変化させ、隣接節点への負の寄与も含める。
張る空間は元と同じだが、パラメータのノルムが定める較正規則は変わる。
4/6区間の行列は132×68、198×102。

`B Bᵀ` の並べ替え行列の階数を調べ、時間とマネーネスの1つの積へ分離できないことを確認する。
市場接線は元のrow_hatsから変化する。一方、列を直交変換した場合には接線と有限幅の較正曲面が一致する。
同じ市場quotesへの適合だけで較正規則の同一性を判断しない。

## 適用範囲

合格は入力した行列、市場fixture、商品・観測履歴、seed、精度方針に限定する。
`diagnostic_only=true`、`provider_promotion=false`、
`full_market_iv_bucket_acceptance=not_established` を維持する。
任意行列すべての受入、通常台帳への自動採用、異商品の合同共分散、実市場のヘッジ効果は含めない。

既存の較正・Gamma・売買providerと配布Python/schema 56ファイルは変更していない。
名前付きproviderは引き続き2種類だけを受け付け、前回の保存状態から同じ結果へ再開できる。
生の数値結果とログは `benchmark-results/vega-kt-raw-basis/` に保存する。
