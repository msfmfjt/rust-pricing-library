# Design and development records

This directory contains requirements, architecture, decisions, implementation
plans and acceptance evidence. The [library guide](../docs/library/README.md),
[product reference](../docs/products/README.md) and
[model reference](../docs/models/README.md) describe the usable APIs, product
contracts and calculation specifications.

## Baselines and change process

- [Frozen MVP requirements](requirements-v1.0.md)
- [Architecture](architecture-v0.1.md)
- [Requirements change template](requirements-change-template.md)
- [Accepted VegaKT hedging backtest requirements v0.1 (Japanese)](vega-kt-hedging-backtest-requirements-v0.1.md)
- [Path-dependent Gamma recovery component (Japanese)](validation/local-vol-path-gamma-recovery-v0.1.md)
- [P0 VegaKT general-smile numerical acceptance v0.3 (Japanese)](validation/vega-kt-smile-acceptance-v0.3.md)
- [ADR template](adr/0000-template.md)
- [Release readiness](release-readiness-v0.1.md)
- [Contributor workflow and checks](../CONTRIBUTING.md)

Requirements and ADRs retain their recorded scope and chronology. Older records
may describe earlier crate names, defaults or extension boundaries. Follow the
model contracts and their explicit superseding decisions for current behavior;
the [three-crate migration guide](../docs/library/three-crate-migration.md)
documents the public API migration.

## Architecture decisions

- [ADR 0038: Immutable backtest provider configuration](adr/0038-immutable-provider-configuration.md)
- [ADR 0039: Native provider API compatibility](adr/0039-native-provider-compatibility.md)

- [ADR 0037: Connect backtests to common tasks and reports](adr/0037-backtest-task-report-bridge.md)

- [ADR 0036: Bound backtest provider cache lifetime](adr/0036-backtest-provider-cache.md)

- [ADR 0035: Calibration diagnostics in common reports](adr/0035-calibration-report-diagnostics.md)

- [ADR 0034: Reuse compiled pricing tasks across operations](adr/0034-compiled-task-reuse.md)

- [ADR 0033: Modular wheel gates and installed API signatures](adr/0033-wheel-api-contracts.md)

- [ADR 0032: Typed Python source construction](adr/0032-typed-python-source-construction.md)

- [0030: Complete tasks, capability queries and common reports](adr/0030-pricing-task-capability-report.md),
- [ADR 0031: Typed risk coordinates and structured Python failures](adr/0031-typed-risk-parameters-python-errors.md)
  with a [usage and extension guide](../docs/library/pricing-tasks.md).

- [0028: Complete pricing source specifications](adr/0028-pricing-specification.md),
  with a [usage and extension guide](../docs/library/pricing-specification.md).
- [0029: Generic product graph source](adr/0029-product-graph-source.md),
  with a [usage and extension guide](../docs/library/product-graphs.md).

- [0027: Andersen–Broadie bounds](adr/0027-american-dual.md), with
  [validation evidence](validation/american-dual.md).

| ADR | Decision |
| --- | --- |
| [0001](adr/0001-hull-white-equity-hybrid.md) | Equity/Hull–White hybrid |
| [0002](adr/0002-hull-white-cash-dividends.md) | Hull–White cash dividends |
| [0003](adr/0003-hull-white-aad.md) | Hull–White AAD |
| [0004](adr/0004-hull-white-vegakt.md) | Hull–White market-IV VegaKT |
| [0005](adr/0005-rough-bergomi.md) | Rough Bergomi |
| [0006](adr/0006-integrate-completed-baseline.md) | Integration of completed baselines |
| [0007](adr/0007-affine-paid-cash-dividends.md) | Historical paid-cash default (superseded by 0009) |
| [0008](adr/0008-deterministic-rate-rough-lsv.md) | Deterministic-rate rough-LSV |
| [0009](adr/0009-escrowed-simulation-and-iv-conversion.md) | Escrowed simulation (IV conversion superseded by 0010) |
| [0010](adr/0010-remove-paid-cash.md) | Remove paid-cash support; defer Bos–Vandermark |
| [0011](adr/0011-static-model-boundaries.md) | Compiled static boundaries for model extensions |
| [0012](adr/0012-pure-stochastic-volatility.md) | Pure SV independently of particle calibration |
| [0013](adr/0013-stochastic-cash-dividends.md) | Stochastic discrete cash dividends, staged integration |

| [0014](adr/0014-bergomi-stochastic-dividends.md) | Pure 1F/2F Bergomi with stochastic cash dividends |
| [0015](adr/0015-stochastic-dividend-risk.md) | Buehler first-order reverse and timestep refinement |

| [0016](adr/0016-stochastic-dividend-bergomi-risk.md) | Optional Bergomi parameter reverse with stochastic dividends |
| [0017](adr/0017-stochastic-dividend-correlation-risk.md) | Optional raw Brownian-correlation reverse with stochastic dividends |

| [0018](adr/0018-stochastic-dividend-gamma.md) | Paired AAD-Delta Gamma and bump diagnostics |
| [0019](adr/0019-rough-stochastic-dividends.md) | Rough Bergomi with stochastic cash dividends, price-only |

## Implementation roadmaps and acceptance

- [Rust mutation-testing rollout, 2026-10-22](validation/mutation-testing-2026-10-22.md)
- [Numerical mutation-testing follow-up, 2026-10-22](validation/mutation-numerics-2026-10-22.md)
- [Numerical mutation survivor triage, 2026-10-22](validation/mutation-triage-2026-10-22.md)
- [Mutation evidence integrity, 2026-10-22](validation/mutation-evidence-integrity-2026-10-22.md)
- [Date and calendar mutation coverage, 2026-10-22](validation/mutation-calendar-2026-10-22.md)
- [Mutation execution evidence and changed-line gate, 2026-10-22](validation/mutation-reporting-2026-10-22.md)
- [Product Graph / Pricing Task architecture review, 2026-10-22](validation/product-graph-task-review-2026-10-22.md)

The [continuous stochastic-dividend price approximation](validation/stochastic-dividend-continuous-barrier-approximation.md)
adds a separate price-only rough-LSV plan, physical log-Spot bridge with cash-jump
checks, independent NumPy comparisons and coupled fixed-surface refinement.
It does not claim an exact nonlinear crossing law or continuous Barrier Greeks.

The [hard Barrier Spot-risk method](validation/stochastic-dividend-hard-barrier-spot-risk.md)
adds production survival-conditioned price and Delta for discrete rough-LSV
up/down knock-in/out calls and puts, including
[fixed cash rebates](validation/stochastic-dividend-hard-barrier-rebates.md),
with independent-reference and sampling-error controls. The
[Python example](../examples/python/rough_dividend_hard_barrier.py) uses the
dedicated hard-payoff method on a price-only rough-LSV plan.

The [hard Barrier refinement panel](validation/stochastic-dividend-hard-barrier-refinement.md)
checks price and analytic Spot Delta on 16/32/64/128-step grids while keeping
the calibrated surface and four observation dates fixed.

The [multi-step survival reference](validation/stochastic-dividend-survival-barrier-reference.md)
checks hard Barrier price and Spot Delta on 4/8-step grids with 2/4 monitoring
dates, including independent reference sampling uncertainty.

The [conditional hard Barrier reference](validation/stochastic-dividend-conditional-barrier-reference.md)
adds independent price and Spot Delta for two-step rough LSV with nonzero
vol-of-vol, dividend mean reversion, and a nonflat frozen leverage surface.

The [hard two-date Barrier reference](validation/stochastic-dividend-hard-barrier-reference.md)
checks stochastic-dividend price and Spot Delta in an exact lognormal limit
using independent Gaussian integration, including both monitoring boundaries.

The [rough diffusion-cache record](validation/rough-diffusion-cache.md) isolates
arithmetic-preserving path acceleration from numerical-scheme changes.

The [2026-10-03 local validation record](validation/library-full-check-2026-10-03.md)
covers the full library gates, packaging repairs and the reviewed Local Volatility
replay correction, with experimental numerical acceptance kept separate.

The [2026-10-04 portability record](validation/library-cross-platform-2026-10-04.md)
checks that retained snapshot on Linux ARM64 and documents native Windows
capture/review preparation without promoting a new Windows golden file.

The [VegaKT hedging backtest record](roadmaps/vega-kt-hedging-backtest-roadmap-v0.1.md)
tracks the first executable Python/native slice, validation and remaining H0–H4 gates.
The [independent VegaKT numerical record](validation/vega-kt-backtest-numerics-v0.1.md)
documents the corrected grid-scaling defect and the still-failing paper-decomposition gate.
The [Gamma numerical record](validation/local-vol-gamma-numerics-v0.1.md) covers
tail-mass preservation, short-step/narrow-strike integration and independent Decimal references.
The [Gamma recovery bridge](validation/local-vol-gamma-recovery-v0.1.md) connects
positive-time node-adjoint recovery and propagation, with independent price bumps
and time/strike refinement checks.
The [paper operator record](validation/local-vol-vegakt-paper-v0.1.md) adds the
nonflat total Gamma derivative, vanilla hedge sign, and equation-(9) Vega ratios,
checked against independent price differences. Pricing-path integration remains open.
The [step-adjoint record](validation/local-vol-step-adjoints-v0.1.md) adds native
retention before time/space aggregation, 136 independent price derivatives,
and event-inserted step rows feeding positive-time Gamma recovery.
The [step-recovery record](validation/local-vol-step-recovery-v0.1.md) adds
compiled actual-time density rows, deterministic path/pair statistics, and
conditional Gamma standard errors, checked with independent price bumps.
The [direct-variance record](validation/local-vol-direct-variance-v0.1.md) retains
bridge endpoint risk through native reverse and path/pair statistics, including
the terminal node, while withholding Gamma for its unresolved decomposition.
The [initial-Gamma record](validation/local-vol-initial-gamma-v0.1.md) adds
conditional continuation repricing, correlated bump diagnostics and an initial
Vega reconstruction without singular density division, for a restricted model class.
The [joint-chain record](validation/local-vol-vegakt-chain-v0.1.md) connects
conditional initial Gamma and positive-time recovery per sampling unit to
paper hedge densities and IV buckets, retaining covariance in total errors
and separate quadrature, bump, excluded-local-Vega and tail/edge diagnostics.
The [allocation audit](validation/vega-kt-backtest-allocation-audit-v0.1.md)
records independent residual accounting and corrected native method metadata.
The [failure-recovery record](validation/vega-kt-backtest-failure-recovery-v0.1.md)
documents failed-run artifacts, atomic snapshot rollback and exact retry tests.
The [settlement record](validation/vega-kt-backtest-settlement-v0.1.md) covers
explicit fixing provenance, availability checks and independent payoff tests.
The [payment record](validation/vega-kt-backtest-payments-v0.1.md) covers
discounted unpaid claims, native payment-date scaling and payment recovery.
The [P&L attribution record](validation/vega-kt-backtest-pnl-attribution-v0.1.md)
covers exact component/equity reconciliation, signed costs and mismatch recovery.
The [numeraire record](validation/vega-kt-backtest-numeraire-v0.1.md) covers
shared-account discounted P&L, initial-capital conversion and input/recovery checks.
The [capital-flow record](validation/vega-kt-backtest-capital-v0.1.md) covers
investor deposits/withdrawals, flow-adjusted P&L, duplicate IDs and exact retry.
The [cost-basis record](validation/vega-kt-backtest-cost-basis-v0.1.md) covers
moving-average realized/unrealized P&L, fixing/payment conventions and reconciliation.
The [Greek explanation record](validation/vega-kt-backtest-greek-pnl-v0.1.md) covers
lagged Delta/optional Gamma, missing-risk coverage and explicit unexplained P&L.
The [IV alignment record](validation/vega-kt-backtest-iv-alignment-v0.1.md) covers
fixed-strike IV mapping, absolute-date matching and diagnostic missing-risk coverage.
The [scenario-statistics record](validation/vega-kt-backtest-ensemble-v0.1.md) covers
paired terminal-error cohorts, failure exclusions, empirical tails and conditional bootstrap intervals.
The [schema validation record](validation/vega-kt-backtest-schemas-v0.1.md) covers
offline current-format schemas, malformed input rejection and wheel-bundled resources.
The [budget-constraint record](validation/vega-kt-backtest-budget-v0.1.md) covers
joint option/stock cash and gross constraints, KKT checks, rounding and execution rejection.

The [Gamma validation protocol](validation/stochastic-dividend-gamma.md) covers
common-noise Delta bumps and their uncertainty.

The [Bergomi parameter-risk record](validation/stochastic-dividend-bergomi-risk.md)
tracks fixed-correlation model-risk differentiation and its numerical domain.

The [stochastic-dividend record](validation/stochastic-dividends.md) distinguishes
finite-split and limiting-price tests from the deferred hybrid/calibration scope.
The [Bergomi coupling record](validation/bergomi-stochastic-dividends.md) adds
exact joint OU covariance and independent finite-step price references.

The [model-boundary extension/comparison record](validation/model-boundary-extension.md)
defines the test-only extension exercise and paired native timing, heap and RSS
evidence for stages S1–S6 of the [static-model roadmap](roadmaps/static-model-boundaries.md).
The [Gaussian HW reference correction](validation/hw-escrowed-gaussian-reference.md)
tracks the separate funded-strike reporting mismatch found in extended CI.
The [local-correlation scaling record](validation/local-correlation-scaling.md)
starts S7 with independent workload axes and cold/warm AAD allocation attribution.
The first measured candidate is [reverse-moment buffer reuse](validation/local-correlation-moment-workspace.md),
with exact outputs, lower first-AAD allocation churn and an explicit peak-heap follow-up.
The [repeated-AAD heap study](validation/local-correlation-heap-followup.md)
checks the full peak-stack difference and fresh-process reproducibility.

| Stage | Roadmap | Acceptance evidence |
| --- | --- | --- |
| European Black–Scholes | [G0–G8](roadmaps/european-bs-roadmap-v0.1.md) | [Conformance report](validation/european-bs-conformance-v0.1.md) |
| Local Volatility / VegaKT | [L0–L8](roadmaps/local-vol-vegakt-roadmap-v0.1.md) | [Conformance report](validation/local-vol-vegakt-conformance-v0.1.md) |
| Path dependence | [P0–P8](roadmaps/path-dependence-roadmap-v0.1.md) | [Conformance report](validation/path-dependence-conformance-v0.1.md) |
| Early exercise | [E0–E8](roadmaps/early-exercise-roadmap-v0.1.md) | [Conformance report](validation/early-exercise-conformance-v0.1.md) |
| Bergomi LSV | [Implementation and remaining acceptance](roadmaps/lsv-roadmap-v0.1.md) | Experimental status and evidence recorded in the roadmap |
| Hull–White | [Implementation and acceptance](roadmaps/hull-white-roadmap-v0.1.md) | Experimental status and evidence recorded in the roadmap |
| Extended-model accuracy | [Panel and remaining scope](validation/extended-model-accuracy.md) | Independent repricing, multiple calibration seeds and refinement gates |
| Extended-model refinement and stress | [Cases and measurements](validation/extended-model-refinement-stress.md) | Original 73/74 result and joint 2F sampling-error failure; subsequent correction linked |
| Multi-asset Sobol sampling | [Attribution and correction](validation/multi-asset-sobol-attribution.md) | Bridge coordinate correction; original 2F stress gate passes locally, native CI pending |
| Extended-model AAD and VegaKT | [Sensitivity acceptance](validation/extended-model-risk-accuracy.md) | Multi-seed, multi-bump comparison with full recalibration through public plans |
| Static model boundaries | [S0–S7](roadmaps/static-model-boundaries.md) | Driver metadata, typed volatility/rate inputs and explicit reverse capabilities; native CI and later stages tracked in the roadmap |

The [three-crate validation report](validation/three-crate-validation.md) links
the preserved [raw evidence](validation/three-crate/). Captured JSON and the
compressed run archive retain their original bytes, including paths and command
output from the revisions they validated. Those historical paths are not a
directory map for the current checkout.

## Where new documentation belongs

| Content | Location |
| --- | --- |
| Usage, serialization, migration and benchmarking | `docs/library/` |
| Product definitions, payoff behavior and event conventions | `docs/products/` |
| Model equations, calibration, simulation, risk and model diagnostics | `docs/models/` |
| Requirements, architecture and release planning | `design/` |
| Architecture decisions | `design/adr/` |
| Implementation plans and remaining work | `design/roadmaps/` |
| Conformance reports and captured validation evidence | `design/validation/` |

Keep calculation specifications linked from the library or model index, even when a
design decision introduces them. Update the relevant index when adding a page.
Repository-local links in both `docs/` and `design/` are checked by
`python3 scripts/check_markdown_links.py`.

Escrowed simulation and IV conversion: [validation record](validation/escrowed-dividends.md).

- [Stochastic-dividend risk and refinement](validation/stochastic-dividend-risk.md)

The [correlation-risk validation protocol](validation/stochastic-dividend-correlation-risk.md)
separates raw-entry derivatives from the instantaneous/integrated numerical domain.

Rough/dividend joint-law and price checks: [validation protocol](validation/rough-stochastic-dividends.md).

Rough/dividend AAD and Gamma: [decision](adr/0020-rough-stochastic-dividend-risk.md),
[validation protocol](validation/rough-stochastic-dividend-risk.md).

Rough/dividend correlation risk: [decision](adr/0021-rough-stochastic-dividend-correlation-risk.md),
[validation protocol](validation/rough-stochastic-dividend-correlation-risk.md).

## Stochastic-dividend Hull–White extension

- [Decision 0022](adr/0022-stochastic-dividend-hull-white.md)
- [Validation protocol](validation/stochastic-dividend-hull-white.md)

HW stochastic-dividend basic AAD: [decision](adr/0023-stochastic-dividend-hull-white-risk.md),
[validation protocol](validation/stochastic-dividend-hull-white-risk.md).

## Rough-volatility extension validation

- [Six-family definitions, boundary/API checks and two-step prices](validation/rough-volatility-families.md)
- [Coupled multistep prices and sampled-lag lift-kernel refinement](validation/rough-volatility-refinement.md)

- [Heston IV grid-refinement validation](validation/heston-iv-refinement.md)

Bass local volatility: [decision](adr/0027-bass-local-volatility.md),
[validation protocol](validation/bass-local-volatility.md).

- [Mixed rough Bergomi eta/rho MC and LSV risk](validation/mixed-bergomi-parameter-risk.md)
- [Quadratic rough Heston parameter reverse validation](validation/quadratic-heston-parameter-risk.md).

The [weak Gamma numerical correction](validation/local-vol-weak-gamma-v0.1.md) integrates density before inversion,
uses next-time weak recovery and resolves the auxiliary grid independently of
reporting nodes. The explicit compiled-payoff diagnostic reduces failures from
32/63 to 0/63 under unchanged thresholds; ordinary native acceptance remains
54/154 failures. Bucket localization and full paper conformance remain open.

- [Conditional initial Gamma measure and residual-bias analysis](validation/local-vol-initial-gamma-measure-v0.1.md)

- [Split-sample Gamma selection and independent validation](validation/local-vol-gamma-selection-v0.1.md)

- [Signed-atom path Gamma IV chain and joint covariance](validation/local-vol-atom-chain-v0.1.md)

- [Conservative Gamma and recalibrated maturity-IV validation](validation/local-vol-conservative-gamma-v0.1.md)

- [Finite-step smile quote calibration and remaining Gamma quote bias](validation/local-vol-discrete-smile-calibration-v0.1.md)

- [Calibration-direction Gamma conservation and quote-bias retest](validation/local-vol-gamma-quote-conservation-v0.1.md)

- [2区間・非平坦な第2区間のGamma回復](validation/local-vol-nonflat-gamma-v0.1.md): 年率分散作用素、較正方向保存、独立価格差分。第1分布は固定。

- [非平坦な継続価格の初期Gamma](validation/local-vol-initial-smile-gamma-v0.1.md): 2区間の符号付きGamma近似、tail上界、初期分散の独立価格差分。

- [第1満期と終端スマイルの共同IV Gamma](validation/local-vol-joint-smile-gamma-v0.1.md): 初期Gamma・第2区間回復・再較正接線を結合し、満期間・節点間の共分散を保持。

- [3区間の終端Gammaと非正規source分布](validation/local-vol-three-step-gamma-v0.1.md): 実際の2遷移から分布を積分し、第3区間の回復・終端スマイル再較正価格を照合。

- [3区間の中間Gammaと中間・終端IV連鎖](validation/local-vol-middle-smile-gamma-v0.1.md): 履歴条件付き回復、終端再較正の補償、10市場座標の独立価格差分。

- [3区間の初期Gammaと全満期の較正連鎖](validation/local-vol-initial-three-gamma-v0.1.md): 第1平行IV＋中間・終端各5点、履歴を動かした継続価格、独立部分積分と再較正価格差分。
- [可変区間数の終端Gamma](validation/local-vol-multi-step-gamma-v0.1.md): 4・6区間、第1市場満期の非平坦スマイル、全履歴を条件とした終端積分と再較正価格差分。
- [可変区間数のGammaと全報告満期の接続](validation/local-vol-path-gamma-chain-v0.1.md): 各満期直前の分散rowを選択し、非平坦な第1市場満期を含む10・15方向を再較正価格で検証。

- [経路依存Gammaの全市場方向共分散](validation/local-vol-path-covariance-v0.1.md)

- [経路依存Gammaの中間区間AD照合](validation/local-vol-conditional-ad-v0.1.md)

- [複数満期Gammaのバックテスト接続](validation/local-vol-multi-step-provider-v0.1.md)

- [Asian・Lookbackの観測履歴と決済](validation/local-vol-path-ledger-v0.1.md)

- [Lookback fixing-day valuation and replay](validation/local-vol-lookback-fixing-day-v0.1.md): immutable current-day observations, native Greeks and conditional Gamma, settlement and resume.

- [All-transition variance extensions](validation/local-vol-variance-extension-v0.1.md): initial-law score, explicit parameter metric, simultaneous market quote calibration and independent path-price differences.
- [Variance-extension backtest provider](validation/local-vol-extension-provider-v0.1.md): explicit all-row or linear-time rules for checked Call-side ledgers, same-product covariance, dated fixings and checkpoint isolation.
- [Put extension acceptance](validation/local-vol-extension-put-v0.1.md): independent Put price differences, known-minimum Lookback recovery and Put targets with Call hedges under both preset rules.
- [Explicit custom variance bases](validation/local-vol-custom-basis-v0.1.md): tensor interpolation, declared parameter scales and independent Call/Put price differences for two coarse bases.
- [Named-basis backtest provider](validation/local-vol-custom-basis-provider-v0.1.md): the two accepted constructions in Call/Put ledgers, audited basis identity, dated settlement and checkpoint isolation.
- [Per-matrix numerical validation](validation/local-vol-raw-basis-v0.1.md): bound raw matrices, nonseparable calibration metrics, independent Call/Put price differences and resumable native diagnostics.
- [Coupled-basis backtest provider](validation/local-vol-coupled-basis-provider-v0.1.md): explicitly selected fixed matrices, matching calibration/Gamma hashes, dated Call/Put ledgers and checkpoint isolation.
- [Current per-matrix runtime](validation/local-vol-current-raw-basis-v0.1.md): current accepted-source binding, unchanged numerical checks, preflight without numerical acceptance and integrity-checked continuation.

## 商品間の推定誤差とVegaKT損益説明

[最新mainとの統合検証](validation/vega-kt-integration-2026-10-19.md)に、
競合解消、履歴確定済みBarrierのVegaKT回帰修正、過去の数値受入の保存範囲を記録する。

[通常バックテストへの接続](validation/vega-kt-joint-replay-v0.1.md)では、
`BacktestConfig(max_joint_bucket_standard_error=...)` を指定し、保有・発注予定の合同精度検査と
保存済み期首リスクによる日次IV損益説明を有効にできる。新形式での保存・再開と通常CLIに対応する。

[市場拡張・合同共分散・損益説明の実装記録](validation/vega-kt-portfolio-market-v0.1.md)を参照。
`portfolio_market_risk.evaluate_portfolio` で共通経路の異商品共分散を計算し、
`explain_backtest` で保存済み台帳に対応する期首リスクから追加の損益説明を作成する。
固定strikeのIV変化にDeltaを合わせ、期限切れ・欠落・外挿が必要な区間は未説明とする。
既存の台帳と旧受入記録は保持する。

[未使用seedの安定性検証](validation/vega-kt-holdout-v0.1.md)では、元の精度で120パネル中112件が合格し、
4/6区間の8件に照合失敗が残った。RQMCによる失敗例の再検証結果と、受入精度での年間計測の
事前実行・数量最適化の反復上限の検証を分けて記録している。

- [RQMCの通常バックテスト接続と追加受入](validation/vega-kt-rqmc-provider-v0.1.md):
  4/6区間の未使用seed40ケース、混在点数の経路依存商品、保存・再開、年間用ポートフォリオの精度検査。
- [年間初日の市場クォートによる誤差削減](validation/vega-kt-quote-control-v0.1.md):
  未使用2seedで保有・丸め後発注予定のSE上限10を通過。60銘柄、2段階の独立価格参照。
  保存パネルによる初日再生を確認し、通常providerへの採用と翌日以降は別の受入範囲とする。

- [VegaKT 市場クォート制御変量の通常provider接続](validation/vega-kt-quote-control-provider-v0.1.md)
- [VegaKT クォート制御変量の日次継続と圧縮保存](validation/vega-kt-quote-control-continuation-v0.1.md):
  3営業日目の追加受入、元のチェックポイントを保持する圧縮保存、年間入力の事前チェック。
