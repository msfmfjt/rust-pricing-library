# VegaKT初期Gammaの条件付き再評価 v0.1

日付: 2026-10-04  
状態: 限定したモデルで初期境界を追加の条件付き価格から推定するRust診断部品を実装。
本体のIVバケット置換・一般モデルの初期境界・H2/H3受入は未完了。

## 初期境界の扱い

`t=0` の密度はDiracなので、初期Vegaを点ごとの密度で割ってGamma曲線を作ることは
できません。今回の[初期Gamma計画](../../crates/pricing/src/engine/risk/vegakt_initial.rs)は、
追加の条件付き価格を計算します。最初の正時刻の状態をStrikeごとに固定し、
その後のパスを再評価して、条件付き価格の2階微分を推定します。
初期Vegaや既存の残差を使った正規化・配り直しは行いません。

[Native continuation](../../crates/pricing/src/engine/processes/local_vol/continuation.rs)は
`[S0, 固定したS1, S2, ...]` を返し、fixingの元の添字を保ちます。
初期区間の乱数を逆算したり、仮の初期遷移を作ったりせず、残りの区間だけを進めます。
元パスのS1と後続乱数を渡す場合、その後の状態がbit単位で再現されることを確認しました。

この実装は論文に明記された初期境界アルゴリズムの再現ではありません。
論文の式(3)へ整合する初期Gammaを、追加情報から推定する別ラベルの診断です。
ラベルは `initial_gamma_conditional_fd_diagnostic_v1` です。

## 適用範囲と誤差

現時点では、forward一定、空間方向に平坦な局所分散、2区間以上の計画に限定します。
分散の時刻依存は許容します。非平滑な空間分散格子、時変forward、1区間だけの
終端Dirac、無効なStrikeや表現不能なバンプは拒否します。
ペイオフcallbackは、割引済みの状態依存価格を決定論的に返す必要があります。
直接分散依存、バリアbridge、affine配当再構成、再学習した行使方針は対象外です。

Strikeごとに `h = Strike * relative_bump` と `h/2` の5点2階差分を取り、
7つの価格を共有します。全Strike・バンプで後続乱数を共通化します。
パス／antithetic pairの平均、標本分散、標準誤差を返し、1ペアを1独立標本と数えます。
1標本の標準誤差は `None`、0標本はエラーです。RQMC点を独立標本とは扱いません。

細かい幅と粗い幅の差は、標本ごとに計算してから集約します。
その標準誤差はバンプ間の共分散を含みます。ただし、この差はバンプ依存性の診断であり、
厳密なバイアス上界ではありません。バンプを小さくすると丸め誤差や標本分散が増える
場合があります。バニラのkinkは条件付き期待値として検証し、個々の非平滑パスの
古典的Gammaを計算できたとは扱いません。

## 初期Vegaへの戻しと論文演算への接続

推定Gammaから、初期区間のVol感応度を次の式で再計算します。

```text
初期Vega = ∫ Gamma1(K) * BS_Vega(S0, K, sigma0, dt0) dK
         = S0² * sigma0 * dt0 * frozen_Gamma_transition(Gamma1)
```

積分は標本ごとに行い、Strike間の相関を含む標準誤差を計算します。
既存の区分線形Gamma積分を用い、両端の外側は定数外挿します。
正規化したGamma kernelの左右のtail massと、fine幅での符号付きtail Vegaも返します。
これは状態密度の除外確率ではなく、未知のGamma tail誤差の上界でもありません。

推定した最初のGamma行から、隣接する式(11)行との式(4)計算、式(9)のIV射影へ
つなぐ検査も追加しました。この検査は演算の接続を確認するもので、後続のGamma回復精度や
ヘッジ全体の標準誤差を証明するものではありません。両推定間の共分散は別途必要です。

## 独立検証

[参照スクリプト](../../scripts/check_local_vol_initial_gamma.py)は100桁Decimalで
条件付き価格と両幅の差分を計算し、独立した2回走査のモーメントとGaussian級数積分から
[参照値](../../fixtures/local-vol/initial-gamma-v0.1.json)を作成します。
nativeのcontinuation、差分実装、統計処理は参照値作成に使っていません。
時間依存分散、複数fixingの積、2次項・6次項、割引、線形項・定数を含みます。

Gamma／再構成Vegaの平均・分散・標準誤差は相対誤差 `1e-8` で比較します。
小さなfine-minus-coarseの平均・標準偏差・標準誤差には、それぞれの単位で `1e-9` の
絶対丸め誤差予算を宣言しました。差の分散は平方根にして同じ単位で比較します。
既存の本体受入・Gamma回復などの閾値は変更していません。

[新規Rust検査](../../crates/pricing/tests/local_vol_initial_gamma.rs)9件が合格しました。

- Native continuationの状態再現、固定したS1の厳密保持、無効乱数・形状の拒否。
- 3パス／3ペアの両幅Gamma、差、初期Vega、tail寄与の独立参照比較。
- worker数1／4での再現性と、計画・標本種別の混在やペア二重化の拒否。
- 6次ペイオフで、5点2階差分のバイアスがバンプ幅の4乗で減ること。
- 2次ペイオフでの解析値、native初期AD、完全なStrike間相関を含む標準誤差の検証。
- 最初のGamma行から式(4)・(9)への接続、時刻不一致・形状違反の拒否。
- 狭いStrike領域で大きなtail寄与を隠さないこと。
- 適用外モデル・不正なバンプ・NaN価格・0標本の拒否。
- バニラの条件付きGammaと初期Vegaを解析式と比較。

| 初期Vegaの検証例 | 独立ペア数 | 再構成値 | 再構成の標準誤差 | 解析値 |
|---|---:|---:|---:|---:|
| `S_T²/2`、seed 41 | 4,096 | 203.180608880 | 0.108611732 | 203.225737081 |
| ATM call、seed 97 | 8,192 | 6.281205515 | 0.066144477 | 6.295228250 |

共通条件は `S0=100`、Vol 20%、時刻 `0, 0.1, 0.4`、金利等によるforward変動なしです。
2次ペイオフのnative初期ADは205.955974484、標準誤差14.158109956でした。
再構成値との比較は相関を決め打ちせず、両標準誤差の和による保守的な6倍幅で確認しました。
ATM callの再構成は解析値との差約0.223%で、事前に設定した2%条件と6標準誤差条件を満たします。
fine-minus-coarseの初期Vega差は0.004282425です。
この表は限定した初期境界の証拠であり、一般モデルやヘッジ全体の受入結果ではありません。

```sh
python scripts/check_local_vol_initial_gamma.py
cargo test --offline --locked -p pricing --test local_vol_initial_gamma -- --nocapture
cargo clippy --offline --locked -p pricing --lib --test local_vol_initial_gamma --test local_vol_direct_variance --test local_vol_step_recovery -- -D warnings
```

既存の関連integration test 43件、Local Vol lib 46件も合格しました。
上記Clippyはwarnings deniedで合格です。新規の独立fixture checkerと、前段の
ステップ集約・直接分散感応度checkerも合格しました。

作業ツリーからrelease native wheelを再ビルドし、一時ディレクトリからロードして
Python検査22件が合格しました。本体の独立数値panelは **154件中54件失敗** のままです。
前段のdirect-variance panelと比較し、全24観測値、全154判定、受入ポリシーhashが一致しました。
共有環境のnative拡張は置換していません。

ログとhash記録は `benchmark-results/vega-kt-initial-gamma/`、panelは
`benchmark-results/vega-kt-numerics/initial-gamma/` に保存しました。
これは対象範囲のローカル検証であり、remote CIや全workspace検査の実行記録ではありません。

## 残る本体接続

初期行の追加情報を得る経路ができましたが、価格評価のIVバケットは引き続き
`local_vol_node_allocation_diagnostic_v1` です。非平滑な分散の扱い、直接バリア感応度の
Gamma分解、後続行の精度・共分散、満期補間・全体の射影監査が残ります。
本体の診断制限を解除するためには、これらをつないだ結果を既存の独立panelと
満期／Strike局在・非平坦曲面・細分化検査で受け入れる必要があります。
