# 経路依存Gammaの全市場方向共分散

日付: 2026-10-06。集約結果は[JSON](local-vol-path-covariance-v0.1.json)。
前提は[条件付きGamma連鎖](local-vol-path-gamma-chain-v0.1.md)と同じ。

## 対応した内容

4区間2満期・6区間3満期の全10・15市場方向について、同じ標本から方向間共分散を求める。
これにより、複数のquoteリスクを合成したときの標準誤差を、方向間の相殺を含めて計算できる。
較正规則、条件付きGammaの回復方法、数値許容差、価格差分の参照方法は変更しない。

native `evaluate_vega_kt_multi_step_gamma_chain_audit` の出力に以下を追加した。

| 出力 | 内容 |
|---|---|
| `joint_sample_covariance` | 行優先の `(3D) × (3D)` 標本共分散 |
| `covariance_dimension` | `3D`。Dは読み出し方向数 |
| `covariance_layout` | native AD、回復Gamma、回復Gamma−native ADの順 |
| `covariance_unit` | 独立標本単位の共分散。平均の共分散はNで割る |
| `covariance_scope` | MC標本誤差。総数値誤差は含まない |

対角ブロックに加え、nativeとGamma間の交差共分散も保持する。
antithetic使用時は反対符号の2経路を平均してから集計し、Nはペア数とする。
使用しない場合は1経路を1標本とする。積の平均から大きな平均同士を引く計算を避け、
中心化した共分散を固定順序のブロックで還元する。スレッド数を変えても再現できる。
上三角だけを集計し、出力時に対称行列へ展開する。保存量上限に共分散成分も含める。

companion `evaluate_multi_step_market_gamma` は、受入成功時に
`market_sample_covariance` と `market_mean_covariance` をD×Dの配列で返す。
座標順は `coordinate_order` と同じ。失敗時は両方と `estimate` を `None` とし、
native診断内のraw共分散を調査用に保持する。

市場方向の係数ベクトルw、Gamma平均ベクトルg、標本共分散Cに対して、
合成平均は `wᵀg`、その標準誤差は `sqrt(wᵀCw/N)`。
各方向のSEの単純加算や、方向間の独立性を仮定した平方和ではない。
数値的な相殺でごく小さい負の分散が出た場合は丸め誤差の範囲を確認する必要がある。

## 受入方法

前回の全16条件を再実行し、既存のすべての数値出力と受入判定の完全一致を要求する。
新しい共分散と説明用メタデータはこの比較から除く。
価格差分を生成する実装・fixtureのhashの一致も要求する。
両条件を満たした場合だけ、前回計算した2種類の再較正価格差分の標本を参照する。
価格差分自体を新規計算したとは扱わない。

各条件で、共分散の対称性・半正定値性、対角成分と既存の分散・SEの一致、
`Cov(G−A,Y)=Cov(G,Y)−Cov(A,Y)` を検査する。
さらに、全方向の平行合成、正負交互の合成、最初と最後のquoteの差、ゼロ方向を
nativeに直接入力して再評価し、行列から合成した平均・全共分散と照合する。

合成Gammaと再較正価格差分の比較には6 combined SEを使う。
絶対許容差は元の1方向0.002から `0.002 × sum(abs(w))` とする。
行列の代数的照合は1e-10〜1e-9のscale付き丸め誤差許容幅で行い、
誤差を隠すための対角化、正定値化、固有値clippingは行わない。

**共分散304検査合格。既存の全16条件は数値出力・判定が完全一致。追加の合成方向は14/16条件が受入、2条件は未解消。**

前回の2,704検査を裏付けるGamma出力の一致を確認し、価格差分の参照標本を引き継いだ。

| 新規検査 | 最大値 | 最大許容幅比 |
|---|---:|---:|
| 対角成分と既存分散の差 | 0 | 0 |
| 差の交差共分散恒等式 | 5.82077e-11 | 1.03636e-06 |
| 直接合成した全共分散との差 | 8.73115e-11 | 4.97151e-07 |
| 合成Gamma対native価格差分 | 5.31428 | 0.409469 |
| 合成Gamma対独立価格差分 | 6.20637 | 0.446898 |

合成方向の直接計算は当初10/16条件のみ受入だった。残る6条件を同じ乱数列の8,192 pairs・N=256で再評価し、4区間Asian/既知平均付きAsianの4条件を解消した。相対SE上限25%と汎関数許容差1e-6は変更していない。

6区間Asianと既知平均付きAsianのseed 7は、row 3・node 11のnative AD照合が失敗する。方向の相対SEは約17.8%・17.6%と合格範囲だが、native ADの観測平均・分散がともに0の節点でGamma感応度が非ゼロとなるため、推定値はNoneを維持した。既知平均付きAsianは16,384 pairs・N=128でも同じ節点が失敗した。

増標本6条件のGammaとnative/独立再較正価格差分の追加48比較はすべて合格したが、節点照合を迂回して合格とは扱わない。次の課題は中間区間のまれなpayoffに対するnative ADの条件付き不確実性評価。

Rust 5件、Python経路関連8件、companion較正28件、wheel契約5件、合計46テスト合格。
clippy、format、依存方向、差分、実行時wheel契約も合格。
検証済みnative・companionを作業用 `.venv` に反映し、binary hashと成功・失敗時の共分散出力を確認。
ソース配布物は必須524件・計903ファイルを検査し合格。文書のローカルリンク検査も合格。


## 再実行

```sh
python scripts/check_path_gamma_covariance.py \
  --previous-report benchmark-results/vega-kt-path-chain/report.json \
  --output benchmark-results/vega-kt-path-covariance/report.json
```

合成方向の追加精度試験（残る失敗は非ゼロ終了コードで報告する）:

```sh
python scripts/check_path_gamma_composed_precision.py --covariance-report benchmark-results/vega-kt-path-covariance/report.json --output benchmark-results/vega-kt-path-covariance/precision-report.json
```

前回レポートがない場合は、先に `check_multi_step_gamma_chain.py` を実行する。
CIはGamma連鎖→共分散の順で実行し、結果をartifactへ保存する。合成方向の追加精度試験は上記コマンドで再現する。

## 残る範囲

共分散は同一商品・同一経路サンプル上の市場方向に関するMC標本共分散である。
異なる商品の合同共分散、回復の離散化誤差、較正誤差、モデル誤差を含まない。
個別方向で受入したことは、任意の相殺ポートフォリオの相対精度を保証しない。
合成方向の精度判定が必要な場合は、その方向をnative APIに直接渡して再評価する。

報告されない区間の分散は固定する。任意の較正延長、carry・配当・barrier・早期行使、
通常バックテストproviderへの接続は未対応。
`full_market_iv_bucket_acceptance=not_established` を維持する。
