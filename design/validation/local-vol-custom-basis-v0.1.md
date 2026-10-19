# 明示したカスタム較正基底の数値受入

日付: 2026-10-11。[集約結果](local-vol-custom-basis-v0.1.json)。
[Call/Putの全区間Gamma受入](local-vol-extension-put-v0.1.md)を基礎として、
時間とマネーネスを粗く表現する2種類の基底を検査する。

## APIと較正の意味

```python
from rust_pricing_backtest.variance_basis import tensor_variance_basis
from rust_pricing_backtest.multi_step_extension import evaluate_extended_market_gamma
from rust_pricing_backtest.multi_step_provider import gamma_boundaries

grid = request["model"]["local_variance_grid"]
times = grid["time_nodes"]
basis, basis_audit = tensor_variance_basis(
    grid["time_nodes"], grid["log_forward_moneyness_nodes"],
    time_knots=[(left + right) / 2 for left, right in zip(times[:-1], times[1:])],
    moneyness_knots=grid["log_forward_moneyness_nodes"][::2],
)
audit = evaluate_extended_market_gamma(
    request, gamma_boundaries(request, request["market"]["spot"], put_knots=True),
    extension="custom", variance_basis=basis, worker_threads=4,
)
# basis_auditとauditを一緒に保存し、basis_hashの一致を確認する。
```

各遷移の中点と局所分散格子で線形hat関数を評価し、時間とマネーネスの積から行列を作る。
端点の外側では値を一定とする。行は遷移row→マネーネス節点、列は時間節点→マネーネス節点の順序。
`parameter_scales` は時間節点数×マネーネス節点数の正の行列で、各列の尺度を明示する。
省略時は1であり、列の正規化や直交化を自動適用しない。
節点の重複・未支持のパラメータ・形状違い・非有限値・非正の尺度は拒否する。
戻る基底は入力からコピーされ、通常の要素代入を禁止する。監査には節点・尺度・行列のhashを記録する。

較正は各Newton反復でパラメータ増分のEuclideanノルムを最小化する。
同じ市場価格に一致しても、基底や尺度が違えば経路依存商品の市場感応度は異なる。
基底の直交変換では曲面と接線が一致し、尺度変更では市場quoteへの応答を保って曲面の応答が変わることを検査する。
既存のall_rows/linear_time基底も同じ生成規則で完全再現できる。

## 検証する2種類

| 基底 | 時間方向 | マネーネス方向 | パラメータ数（4/6区間） |
|---|---|---|---|
| row_hats | 各遷移の中点 | 元の33節点から1つおきの17節点 | 68 / 102 |
| weighted_time_hats | 市場報告満期 | 同じ17節点 | 34 / 51 |

後者は、時間方向の0.8〜1.3の係数とマネーネス方向の`1+0.3*k`の積を列の尺度に使う。
マネーネスの節点を減らすことで、複数の局所分散節点の変化を同じパラメータへ結び付ける。
検証対象の行列は監査の節点・尺度から再構成できる。

両基底、4/6区間、Call4履歴・Put6履歴、2 seedsで80パネルを検査する。
European、算術平均Asian、既知観測を含むAsian、固定行使価格Lookbackを含み、
Put Lookbackでは未観測・過去最安値98・過去98と当日90を区別する。

各quoteを上下へ動かして新しく同時再較正し、別実装のquote価格で誤差1e-10以下を確認する。
最終遷移を解析積分、それ以前を8組のscrambled Sobol・各2^16点で積分したCall/Put価格差分とGammaを照合する。
この価格基準はnativeのAD/Gammaや較正Jacobianを使わない。
第1満期のIVを±0.001動かして後続満期の全quotesが維持されることも検査する。

## 差分幅の精細化

初回の差分幅1e-5/5e-6では、一部の曲面接線照合に打切り誤差が残った。
初回の記録を保存し、1e-6/5e-7で全市場quoteの再較正と独立価格差分を再計算する。
合格したnative監査は、基底・接線・ソース・バイナリが同一の場合だけ引き継ぐ。
想定外のnative失敗や価格差分の失敗があれば引き継がない。

接線誤差2e-5、quote価格誤差1e-10、価格差分との比較6標準誤差＋0.002、
分割数512、相対許容差1e-6、相対SE上限25%は維持する。
標本数は4区間4,096組・6区間2,048組、antithetic、seeds 7/41。
Put Lookbackの実効固定行使価格を積分区切りに含める前回の規則も維持する。

80パネル・5,540検査が合格し、初回の接線照合34失敗を解消した。
上下へのquote変更計200回について、再較正と独立価格差分を細かい幅で再計算した。
第1満期の有限幅ストレス8条件、全80パネルのnative回復・AD照合・共分散監査を維持した。

補助ライブラリ259件（新しい基底の4件を含む）と既存の公開Gamma APIの3件、計262テストが合格した。
配布wheelから4区間・6区間の両基底を実行した4パネルで、基底・接線・native計算値が
独立受入の監査と完全一致した。配布版とソースのPython/schema 55ファイルも一致した。
カスタム基底の売買providerへの投入拒否と、旧実装のcheckpointの再開拒否を確認した。

生の結果・初回失敗・実行ログは `benchmark-results/vega-kt-custom-basis/` に保存した。
集約JSONにソース・結果・配布物のSHA-256を記録する。

## 範囲

新しいAPIは診断用。数値受入は上記2基底と宣言した格子・商品・履歴に限定する。
任意のraw行列、他の基底設定や格子、carry・配当、異商品の合同共分散、実市場のヘッジ効果は未受入。
カスタム基底を通常売買台帳へ接続するproviderは今回追加しない。
既存のCall/Put providerは2つのpresetだけを受け付ける。
`diagnostic_only=true` と `full_market_iv_bucket_acceptance=not_established` を維持する。

既存の較正計算・Rust本体・売買providerは変更していない。
ただし補助モジュール追加により台帳の実装識別子は変わるため、旧checkpointは保存入力から再実行する。

## 再実行

初回の失敗も監査対象として残す。初回プログラムの非ゼロ終了だけを最終受入と解釈しない。
前回受入と同じnativeバイナリ・既存数値ソースを使用する。まとめて実行するには次を使う。

```sh
OPENBLAS_NUM_THREADS=1 python scripts/check_custom_basis_acceptance.py \
  --output-directory benchmark-results/vega-kt-custom-basis
```

既存結果がある場合は再利用前に、全ソース・バイナリ・方針・80パネルの網羅性を照合する。
`check_custom_variance_basis.py` は `--intervals` / `--basis` で4分割して実行できる。
`--aggregate` で4結果のソース・バイナリ・全80パネルの網羅を確認する。
`check_custom_variance_basis_refinement.py` で各分割の価格差分を再計算後、
`--baseline-report` / `--price-reports` を指定して最終照合を行う。
