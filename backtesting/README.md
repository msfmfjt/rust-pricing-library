# VegaKT hedging backtest companion

This separate Python package runs daily, cash-settled European-option replay
using the public `rust_pricing` engine. It adds a self-financing ledger,
next-snapshot execution, bounded VegaKT optimization and four-strategy comparison.

Install the native library following the repository development guide, then run
`python -m pip install ./backtesting` from the repository root. NumPy is the only
additional runtime dependency; the native adapter also requires `rust_pricing`.

The full guide is in the repository's `docs/library/vega-kt-backtesting.md`.
P0 European terminal risk now passes the general-smile numerical gate. Use
`RustPricingProvider(numerical_scope="p0_terminal_replication")` or CLI
`--validated-p0`; every live contract must pass a market calibration check.
Outputs record `numerically_validated_p0`. The default still requires an explicit
mode, and `diagnostic_only=True` / `--diagnostic-only` retains `unvalidated` status.
General path-dependent Gamma recovery and historical effectiveness remain open.
Run `python -m rust_pricing_backtest.smile_numerics --policy
fixtures/backtesting/smile-policy-v0.3.json --output benchmark-results/vega-kt-smile`
from the repository root for the combined 1,077-check gate. Its report preserves
the six superseded legacy failures without changing their original results.
Expected replay failures retain the completed ledger and a resumable checkpoint
in `BacktestRunError.result`, or return them with `on_error="return"`. The CLI
saves failure output and exits 1. Incomplete snapshot changes are rolled back.
Contracts support explicit sourced expiry fixings, with availability/date
checks and audited settlement cashflows. The snapshot-spot compatibility rule
remains explicit in outputs. Delayed option payments use separately discounted
claims until payment; funding remains on actual cash. Daily/cumulative P&L
components reconcile instrument marks and cashflows against ledger equity;
unreconciled snapshots fail and roll back. `pnl_components.csv` is the eighth
output table. An explicit shared money-market account enables discounted equity,
P&L and terminal hedging error, with initial capital converted at its starting
account value. Set `numeraire_id` and supply dated `NumeraireObservation` values;
disabled mode leaves these metrics null. `ExternalCashflow` records investor
deposits/withdrawals at snapshots, after carry and before settlement/fills.
Capital changes cash but is excluded from nominal/discounted P&L; replay guards
against duplicate flow IDs. Moving-average execution cost tracks realized and
unrealized nominal P&L in `cost_basis.csv`, the ninth output table. Fees are
expensed separately; fixed payoffs realize before payment, with unpaid claims
carried at face cost and marked at present value. Both daily changes and totals
reconcile to the capital-adjusted ledger. `greek_pnl.csv`, the tenth table, explains
asset P&L using previous holdings' Delta and optional Gamma; missing Gamma stays
null and unexplained P&L remains explicit. `iv_alignment.csv`, the eleventh table,
audits IV moves at exact absolute maturities and opening strikes, with linear
interpolation inside the current moneyness domain and explicit missing risk.
Covered bucket dot products remain diagnostic: full time-roll and Delta/IV
P&L attribution is outside the P0 numerical acceptance, so they do not enter Greek P&L.
The native adapter preserves requested Gamma and bump metadata; it
does not enable Gamma implicitly. Native Gamma bump settings also affect Delta.
Default requests are version 6, results version 10 and checkpoints version 8. Result
versions 1–9 remain readable without inventing missing report history. Request versions
1–4 have no external capital movements.
Request versions 1–3 retain disabled numeraire reporting; versions 1/2 load with payment
on expiry. Delayed native pricing requires the rebuilt `DiscountCurve.discount`
API and remains diagnostic only.
`rust_pricing_backtest.ensemble` compares terminal errors on the intersection of
four completed, fully settled strategy runs across a fixed-book/schedule cohort.
It records failed/partial/missing coverage and preserves native risk-validation
status. Mean, sample standard deviation, RMSE, signed-loss quantiles and fractional
tail ES are available in nominal or discounted units. Declared independent outer
simulations support paired bootstrap intervals; dependent/unknown samples do not.
Sparse tails stay reference-only, and near-zero baseline variance yields null
improvement. The ensemble request remains version 1; reports now use version 2 and preserve
the cohort’s constraint/backend conventions in shared conditions. See the guide for the CLI and synthetic example.
The wheel bundles sixteen Draft 2020-12 schema resources covering current request,
result, checkpoint and ensemble formats, plus frozen v5/v9/v7/v1 contracts. Install `rust-pricing-backtest[validation]`
to use `python -m rust_pricing_backtest.validation FILE.json` for offline structural
checks, or pass `--export-schemas DIR` to copy the schemas for external tools.
This opt-in check preserves legacy readers and never substitutes for replay,
hash/P&L reconciliation or numerical acceptance.
Historical data onboarding and acceptance beyond the tested P0 numerical scope are open.

Set `market_shock_log_moneyness` to explicit nested log-forward-moneyness nodes
to optimize the VegaKT strategy in a fixed market shock basis. Every daily
reporting grid must contain these knots and share the endpoints. Maturities
match that day's reporting maturities exactly. All four strategies retain held
and proposed residual-risk audits, including directions outside the chosen
shocks, in `risks` and `decisions`. This opt-in mode uses request/result/checkpoint/
ensemble versions 7/11/9/3, with five new schemas; older schemas remain frozen.
Changing the configured nodes prevents checkpoint resume or mixed-condition
strategy aggregation. Native validation metadata retains the explicitly selected numerical scope.

Install `rust-pricing-backtest[constraints]` for optional SciPy cash/gross optimization.
Set `constraint_mode="decision_cash_gross"` and a finite `gross_notional_limit`.
The continuous solver jointly chooses options and Delta-neutral stock using current
bid/ask, slippage and fees. It checks lot-rounded cash/gross again, then checks actual
execution at the next snapshot. The default `execution_cash` mode remains NumPy-only.
The cap is spot times gross underlying units of targets, hedge options and stock;
it is not an option-premium or margin cap. See the guide for failure policies.

## Checked multi-step Gamma replay

Use `RustPricingProvider(numerical_scope="multi_step_gamma")` or the CLI flag
`--validated-multi-step` for terminal European Call/Put contracts on the accepted
four/six-interval, zero-carry calibration extension. Market quote calibration and
Gamma/AD precision checks must pass before risk is released to the ledger.
`provider.gamma_audit(snapshot, contract)` returns the full calibration and
same-contract covariance record. See the
[scope and validation](../design/validation/local-vol-multi-step-provider-v0.1.md).

## Dated Asian and Lookback replay

`PathContract` and `PathObservation` bind scheduled observations to sourced
`SettlementFixing` values. Use `numerical_scope="multi_step_path_gamma"` or
`--validated-path`; see [accepted scope and replay](../design/validation/local-vol-path-ledger-v0.1.md).
Available path fixings persist across checkpoints, including observations for
candidates that have not yet been quoted. Missing observations fail the snapshot
when a contract is valued or settled; an unused candidate does not require a
complete history. Recorded fixings remain immutable.
[Lookback fixing-day valuation](../design/validation/local-vol-lookback-fixing-day-v0.1.md)
uses the sourced current-day observation as a fixed value, separate from spot.
Expiry settles directly from the recorded history.

The separate diagnostic [variance-extension API](../design/validation/local-vol-variance-extension-v0.1.md) calibrates every transition row, including the initial law, with explicit all-row, linear-time or custom bases. Existing ledger providers keep their previously validated calibration rule.
Adding the module updates the package implementation identity, so checkpoints from the previous package must be replayed from their saved inputs.

Use `--validated-extension all_rows` or `--validated-extension linear_time` for the
explicit [all-transition provider](../design/validation/local-vol-extension-provider-v0.1.md).
It accepts checked Call-side European, Asian and Lookback contracts and binds the chosen extension to checkpoints.

For Put targets and mixed Call/Put hedges, use `--validated-put-extension all_rows`
or `--validated-put-extension linear_time`, or select
`numerical_scope="multi_step_extension_put_gamma"` with an explicit `variance_extension`.
The [Put acceptance record](../design/validation/local-vol-extension-put-v0.1.md)
covers independent Put price differences and known-minimum Lookback integration knots,
with the same precision gates and dated settlement rules.

Use `rust_pricing_backtest.variance_basis.tensor_variance_basis` to construct a
diagnostic custom basis from time and moneyness knots, with explicit parameter scales.
See [custom-basis acceptance](../design/validation/local-vol-custom-basis-v0.1.md)
for the two checked configurations and the distinction between the parameter metric
and market quote fit. To replay with either accepted named construction, use
`numerical_scope="multi_step_custom_basis_gamma"` and
`variance_basis_policy="row_hats"` or `"weighted_time_hats"`, or the CLI flag
`--validated-custom-basis row_hats`. The [named-basis provider](../design/validation/local-vol-custom-basis-provider-v0.1.md)
binds the construction to risk audits and checkpoints. Arbitrary raw matrices remain diagnostic-only.

The [raw-matrix checker](../design/validation/local-vol-raw-basis-v0.1.md) accepts a versioned matrix file
and checks independent recalibrated Call/Put prices, Gamma and covariance on the frozen four/six-step markets.
Its result binds the exact matrix, market, policy and numerical implementation. It does not add a ledger policy.

The fixed matrices from that independent check now have a separate, explicit
[ledger policy](../design/validation/local-vol-coupled-basis-provider-v0.1.md):
`--validated-custom-basis coupled_row_hats`. The generated matrix must match one of
the two accepted hashes. Older implementation checkpoints must be replayed from saved inputs.

For per-matrix diagnostics on the current accepted runtime, use
[`check_current_variance_basis.py`](../design/validation/local-vol-current-raw-basis-v0.1.md).
`--preflight` checks compatibility without claiming numerical acceptance. Full runs keep
the existing independent numerical checks and save integrity hashes for safe continuation.
The historical raw checker and its recorded evidence remain unchanged.

## 商品間の推定誤差とVegaKT損益説明

[通常バックテストへの接続](../design/validation/vega-kt-joint-replay-v0.1.md)では、
`BacktestConfig(max_joint_bucket_standard_error=...)` を指定し、保有・発注予定の合同精度検査と
保存済み期首リスクによる日次IV損益説明を有効にできる。新形式での保存・再開と通常CLIに対応する。

[市場拡張・合同共分散・損益説明の実装記録](../design/validation/vega-kt-portfolio-market-v0.1.md)を参照。
`portfolio_market_risk.evaluate_portfolio` で共通経路の異商品共分散を計算し、
`explain_backtest` で保存済み台帳に対応する期首リスクから追加の損益説明を作成する。
固定strikeのIV変化にDeltaを合わせ、期限切れ・欠落・外挿が必要な区間は未説明とする。
既存の台帳と旧受入記録は保持する。

## 明示的なRQMCによる通常バックテスト

[RQMC接続・追加検証](../design/validation/vega-kt-rqmc-provider-v0.1.md)では、
`RustPricingProvider(sampling_policy="rqmc_path_16_scrambles", ...)` を指定できる。
4/6区間の既存範囲、`all_rows` または `coupled_row_hats` に限定する。
入力は16スクランブル、反対変量あり、Brownian bridgeなし。
6区間Asianは128点、他は256点を明示する。`rqmc_256x16` は全商品256点。
点数・seedの自動変更は行わず、個別・合同精度検査を維持する。

CLIでは校正方式の指定に `--rqmc-path-16-scrambles`（または `--rqmc-256x16`）を追加する。
`--worker-threads 4` で並列数も指定できる。再開時は元の並列数とsampling policyを一致させる。
検証状態は `runtime_checked_multi_step_rqmc`。数値受入の実測範囲はリンク先に記録する。

## 日次の市場クォートによる誤差削減

欧州コールの6区間・`all_rows` 校正では、明示的に
`sampling_policy="rqmc_quote_control_512x24"` を指定すると、その日の満期ATMクォートと
同じ行使価格の参照コールを生成する。入力は512点×24スクランブル、反対変量あり、
Brownian bridgeなし、金利・配当ゼロに限定する。

```python
provider = RustPricingProvider(
    numerical_scope="multi_step_extension_gamma",
    variance_extension="all_rows",
    sampling_policy="rqmc_quote_control_512x24",
    worker_threads=4,
)
```

CLIでは `--validated-extension all_rows --rqmc-quote-control --worker-threads 4` を指定する。
入力のエンジン指定は自動変換しない。

各スクランブルで、各商品のVegaKTから参照コールのVegaKTを引き、校正クォートの
Black Vegaを足す。係数は常に1で、標本から推定しない。価格・凍結グリッドDelta・Gammaは
元の推定値を使い、元のGamma回復・AD検証も必須のままにする。
参照コールは候補銘柄・注文・残高に含めない。実商品は最大31件。

個別評価と合同共分散は同じ標本を使う。数値計算のキャッシュは当該スナップショットに
限定し、チェックポイントからの再開では次の日を新しく評価する。保持ポジションと
ロット丸め後の予定ポジションに対する標準誤差上限は既存設定を維持する。
検証状態は `runtime_checked_market_quote_control`。
[数値検証と適用範囲](../design/validation/vega-kt-quote-control-provider-v0.1.md)を参照。
年間252営業日の検証と、校正・離散化誤差を含む全面的な数値受入は別途必要。
