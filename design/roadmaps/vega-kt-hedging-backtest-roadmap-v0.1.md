# VegaKT backtest implementation record v0.1

Date: 2026-09-29  
Updated: 2026-10-13  
Status: Explicit four/six-step European, Asian and fixed Lookback Call/Put
scopes include calibrated market Gamma, named variance bases and ledger replay.
The six legacy grid comparisons retain their original evidence and audited
superseding checks. See the [current extension record](../validation/vega-kt-portfolio-market-v0.1.md)
for broader declared markets, cross-instrument estimation covariance and
fixed-strike Delta/VegaKT P&L. Arbitrary-market acceptance, formal full H0–H4
evidence and historical performance remain open.
The [joint replay integration](../validation/vega-kt-joint-replay-v0.1.md) adds
opt-in held/proposed precision gates, saved-opening-risk attribution and versioned
checkpoint replay to the ordinary four/six-step extension providers.
The chronological entries below retain earlier states; their failure counts
and “not yet” statements describe those historical implementations.

The [independent numerical panel](../validation/vega-kt-backtest-numerics-v0.1.md)
found an inverse-grid-spacing defect and an incomplete paper gamma pipeline.
The scaling defect is corrected; the remaining failure is explicit. Native
adapter use requires diagnostic mode, with unvalidated status in every output.

The [2026-10-02 allocation audit](../validation/vega-kt-backtest-allocation-audit-v0.1.md)
adds actual exclusion accounting, detects corrupted buckets, and labels native
LV output as diagnostic local-node allocation. A first-local-row CRN probe
explains the initial-time loss; the original numerical gate still fails.

The [Gamma numerical correction](../validation/local-vol-gamma-numerics-v0.1.md)
preserves tiny tail mass and stabilizes short-step/narrow-strike integration in
the native transition primitive. Eight independent Decimal cases pass. The public
path still does not use that primitive; the rebuilt native panel retains 54/154
failures, and the diagnostic guard remains in place.

The [positive-time Gamma recovery bridge](../validation/local-vol-gamma-recovery-v0.1.md)
now connects full-grid adjoint normalization, equation (11), and the transition
primitive. Conditional price bumps, native AD/CRN row checks, and first-order
time refinement pass. This Rust component is not yet the missing vanilla-weight
construction or a replacement for the diagnostic allocation in pricing results.

The [paper operator record](../validation/local-vol-vegakt-paper-v0.1.md), based
on the supplied original PDF, now provides the total spot derivative for nonflat
variance, equation-(4) hedge densities and equation-(9) Vega-ratio projection.
Eleven new Rust tests pass, including independent price differences and a composed
quadratic-payoff check. The rebuilt production panel still fails 54/154 checks;
step-local adjoints, the initial boundary and nonsmooth variance derivatives
remain prerequisites for connecting these operators to pricing results.

The [step-adjoint record](../validation/local-vol-step-adjoints-v0.1.md) now
implements retention of each simulation interval's effective variance/volatility
adjoints in the native reverse sweep. Original grid adjoints reconstruct bitwise;
136 independent price derivatives and event-inserted recovery rows pass. The
bucket path still needs step-row aggregation, direct bridge variance seeds and
the remaining boundary/variance conventions before paper acceptance.

The [step-recovery record](../validation/local-vol-step-recovery-v0.1.md) adds
the compiled Rust aggregation component: actual-time densities, path or
antithetic-pair moments, covariance-preserving total uncertainties and
positive-time equation-(11) Gamma means/standard errors. Independent Decimal
price-bump statistics and 4096-pair CRN checks pass. Connecting this component
to production buckets, direct bridge risk and the remaining boundary/variance
conventions is still required; the native diagnostic guard is unchanged.

The [direct-variance record](../validation/local-vol-direct-variance-v0.1.md)
connects exact/smoothed barrier variance seeds to a shared native reverse and
adds N+1 direct-node risk rows to the statistical component. State feedback is
applied once, terminal and early-monitoring endpoints are retained, and combined
uncertainty includes transition/direct covariance. Independent Decimal prices
validate 295 derivatives. Nonzero direct seeds explicitly withhold Gamma even
when signed risks cancel; their paper decomposition still requires validation.

The [initial-Gamma record](../validation/local-vol-initial-gamma-v0.1.md) now
estimates the first positive-time Gamma row from additional conditional prices,
with common remaining shocks, two bump widths and path/pair uncertainty. The
initial Vega reconstruction and equations-(4)/(9) connection pass restricted
flat-in-space, constant-forward checks, including a vanilla payoff kink in
expectation. This diagnostic does not replace production IV bucket allocation;
nonflat variance, direct bridge risk and full numerical acceptance remain open.

The [joint-chain record](../validation/local-vol-vegakt-chain-v0.1.md) composes
the initial and positive-time Gamma rows per draw into equations (4) and (9),
with joint path/pair bucket and scalar uncertainties. The restricted flat-space
model law, density and reporting IV must agree. Finite-strike quadrature,
initial bump, omitted local Vega and tail/edge effects remain separate audits.
This Rust diagnostic still does not replace production bucket allocation.

The [2026-10-03 recovery record](../validation/vega-kt-backtest-failure-recovery-v0.1.md)
adds failed-run reports and atomic snapshot recovery. Fault injection and native
CLI replay confirm that retrying does not duplicate carry, fills or entry cash.

The [settlement record](../validation/vega-kt-backtest-settlement-v0.1.md) adds
explicit fixing inputs, source/revision and availability checks, audited expiry
cashflows, and version 2 requests with version 1 compatibility.

The [payment record](../validation/vega-kt-backtest-payments-v0.1.md) adds
discounted unpaid claims, payment-date cashflows, native deterministic-curve
payment adjustment and request/result version 3. Earlier requests still load.

The [P&L attribution record](../validation/vega-kt-backtest-pnl-attribution-v0.1.md)
adds exact cashflow/mark-change components, separate cost/carry contributions,
daily/cumulative equity reconciliation and rollback on mismatches. It introduced
result version 4 and checkpoint version 3.

The [numeraire record](../validation/vega-kt-backtest-numeraire-v0.1.md) adds
explicit shared money-market observations, discounted equity/P&L/terminal error,
initial-capital conversion and reconciliation including numeraire carry. It
introduced request version 4 and result version 5.

The [capital-flow record](../validation/vega-kt-backtest-capital-v0.1.md) adds
snapshot investor deposits/withdrawals, P&L adjustments at each flow's own
numeraire, global ID deduplication and atomic replay. It introduced request
version 5, result version 6 and checkpoint version 4.

The [cost-basis record](../validation/vega-kt-backtest-cost-basis-v0.1.md) adds
moving-average execution cost, realized/unrealized nominal P&L, payoff-fixing
realization and face-cost unpaid claims, with inventory and P&L reconciliation.
It introduced result version 7 and checkpoint version 5.

The [Greek explanation record](../validation/vega-kt-backtest-greek-pnl-v0.1.md)
adds opening-holdings Delta and optional Gamma explanations, explicit missing
risk/residual reporting and native Gamma metadata. It introduced result version 8
and checkpoint version 6.

The [IV alignment record](../validation/vega-kt-backtest-iv-alignment-v0.1.md) adds
absolute-maturity/fixed-strike IV diagnostics, explicit uncovered instrument risk
and non-adopted bucket dot products. The LV shock interpretation and consistency
with frozen-grid Delta remain unvalidated; full PNL-04 is open. That slice
introduced result version 9 and checkpoint version 7.

The [budget-constraint record](../validation/vega-kt-backtest-budget-v0.1.md) adds
optional decision-time cash/gross optimization with Delta-neutral stock, KKT
checks, post-lot budget checks and actual execution rejection. Current requests
are version 6, results version 10, checkpoints version 8 and ensemble reports
version 2. Historical v5/v9/v7/v1 schemas remain bundled unchanged.

The [accepted requirements](../vega-kt-hedging-backtest-requirements-v0.1.md)
define the complete scope. This record distinguishes executable functionality
from numerical acceptance and historical-market evidence.

## H0 decisions materialized in the first slice

| Decision | Implementation and reason |
| --- | --- |
| Orchestration | Separate installable Python companion under `backtesting/`; public Rust pricing APIs remain the numerical engine. This preserves the existing native wheel/API contract. |
| Time and execution | One explicit timestamp per valuation date; decisions fill on the next supplied snapshot, using bid/ask plus costs. |
| Product state | European cash-settled options with explicit payment dates and unpaid claims, zero-dividend native pricing. Historical fixtures can exercise same-day cash dividend accounting independently. |
| Risk meaning | Requested LV paper decomposition, canonical f-IV, raw units; existing frozen-local-grid Delta. Native paper conformance is not established; output is diagnostic only. Other coordinates and risk methods are rejected. |
| Shared basis | Native probing found that every positive local time row needs a matching reporting maturity and the log axes must match. The adapter validates this; the example uses calendar-day rows and a common basis across expiries. |
| Optimizer | Deterministic cyclic coordinate descent for the convex quadratic/L1 objective, box constraints, explicit lot quantization, SVD diagnostics and projected-gradient stopping. The default checks cash atomically at fill time; optional SciPy LP/QP solves include cash, gross and Delta-neutral stock jointly, followed by independent KKT and lot checks. |
| Reproducibility | Complete market/request inputs, source/binary hashes, effective settings, deterministic order, checkpoint state and processed-prefix hashes. |
| Failure policy | Typed errors stop invalid market/risk runs. Infeasible cash-funded rebalance baskets are rejected without partial fills. No stale-risk fallback. |

The package is a companion rather than a new Rust crate. It depends on NumPy;
the native adapter additionally uses the installed `rust_pricing` binary. The
binary hash is recorded because a checkout revision alone does not prove which
native wheel was executed.

## Implemented functionality

- Daily replay, entry premiums, cash, stock and option holdings, deterministic
  expiry settlement and a precommitted terminal hedge liquidation.
- Lending/borrowing accrual, stock borrow costs, synthetic bid/ask execution,
  slippage, per-unit fees and cash limits.
- Unhedged, Delta, scalar Vega and VegaKT strategies with consistent conventions.
- Full instrument-to-bucket matrices; fixed exposure for held instruments that
  are no longer eligible; explicit scalar-option rolls.
- Per-instrument price/bucket sampling errors, projection residual and edge
  diagnostics, with configurable quality gates.
- JSON/CSV outputs, input replay, checkpoint resume and a command-line entry.
- Expected-failure reports with coverage, last completed time, remaining
  positions and pending orders; rollback of incomplete snapshots before resume.
- Contract-specific point fixings, independent of tradable spot quotes, with
  dated provenance and no fallback when a required fixing is missing.
- Delayed option payment, dated discount factors, separate target/hedge claims,
  non-cash fixing events and payment-day cashflows with exact checkpoint resume.
- Signed target, option-hedge, stock, dividend, funding, borrow and execution-cost
  contributions, including unpaid-claim mark changes, with independent equity reconciliation.
- An explicit common numeraire, separate from strategy funding and claim discount
  factors, with causal input checks and reconciled discounted hedging error.
- Explicit investor capital at snapshots, excluded from nominal/discounted P&L,
  with provenance, duplicate-ID rejection and recovery after booking/fill failures.
- Moving-average execution cost, partial-close/reversal realization and separate
  target/hedge claim cost, with daily/cumulative nominal P&L reconciliation.
- Previous-snapshot Delta/optional Gamma explanations with separate exact cash/cost
  components, explicit residuals, missing-Gamma coverage and no unaligned IV products.
- A fixed-strike IV alignment audit with exact absolute dates, bounded linear
  interpolation, gross missing-risk coverage and diagnostic-only bucket products.
- Fixed-book/schedule terminal-error aggregation on a common four-strategy cohort,
  with explicit failed/partial/missing exclusions, empirical loss quantiles/ES,
  sparse-tail flags and paired bootstrap intervals for declared independent simulations.
- Bundled Draft 2020-12 schemas for current request/result/checkpoint/ensemble
  formats, offline structural validation, frozen wire examples and CI drift checks.
- Optional cash/gross constrained option-and-stock decisions with audited cost inputs,
  feasibility/KKT gates, rounded Delta residuals, and atomic next-price rejection.
- A synthetic 11-session, four-instrument, two-expiry eSSVI example using native
  price/Delta/VegaKT rather than precomputed stand-ins.

## Validation completed locally

| Check | Result |
| --- | --- |
| Companion ledger/optimizer/oracle/schema tests | 193 passed with validation/constraints extras (2026-10-03) |
| Native adapter/integration tests | 22 passed with retained diagnostic binary and validation/constraints extras (2026-10-03) |
| Packaging | Editable installation and `py3-none-any` wheel build passed |
| Markdown links and diff whitespace | Passed |
| Synthetic four-strategy execution | All four completed; targets settled and hedge positions closed |
| Outer-scenario integration | 24 scenarios × 4 strategies completed; saved-result reaggregation identical; sparse tails reference-only |
| Native JSON replay | All four strategy inputs replayed exactly in the integration suite |
| Checkpoint replay | Partial run plus resume equals uninterrupted result; altered prefix/state rejected |

Tests cover AC-01–10, AC-12 and AC-14–16 in focused fixtures. This is **partial
coverage**, not acceptance of all conditions under those IDs. AC-07 validates
units, shared basis and native output identity; it does not provide new
independent financial validation of the existing VegaKT decomposition. AC-11
and AC-13 now have a systematic constant-volatility/seed/grid panel; numerical
acceptance fails. Complete non-flat-surface and hedge-error convergence remain open.

The dedicated backtesting workflow tests the companion on Linux, macOS and
Windows. Native integration is also part of the repository's existing Python
smoke-test discovery. Remote CI has not been run as part of this local work.

## Gate status and next work

| Gate | Status | Remaining acceptance work |
| --- | --- | --- |
| H0 | Architecture and input contracts implemented | Select real market/data and its fixing/calendar conventions, numerical tolerances and performance budget |
| H1 | Core ledger/replay, fixings, unpaid claims, investor capital, cost-basis reporting, nominal/discounted P&L reconciliation and expected-failure recovery implemented and fixture-tested | Fixing corrections, delayed dividends, intraday capital timing, broader calendar cases, abrupt-process recovery |
| H2 | Matrix/solver implemented; native numerical gate fails | Wire paper gamma recovery/propagation into the native path; initial-time handling; all-maturity projection-domain diagnostics; risk convergence; portfolio uncertainty; canonical tie-break for nonunique zero-ridge constrained solutions; collateral/margin and integer optimization; calibration provenance |
| H3 | Synthetic replay, paired outer-scenario statistics, current-format JSON Schemas and diagnostic 252-session workload measured | Resolve failed numerical panel; dependent-history inference, legacy-schema migrations, full acceptance panel and an accuracy-qualified performance budget |
| H4 | Not started | Point-in-time option/spot/surface data and out-of-sample historical study |

The initial configuration values are explicit research defaults, not accepted
market tolerances. Large default error limits keep the example inspectable;
production-quality thresholds must be established by the remaining numerical
studies. Independent necessary-condition thresholds are now frozen in
`fixtures/backtesting/numerical-policy-v0.1.json`. The synthetic data and
model-generated quotes cannot establish real-market hedge effectiveness.
Existing P&L comparisons additionally used defective native VegaKT and must not
be interpreted as strategy evidence. Numerical corrections change native
hashes and therefore invalidate prior checkpoint identities.

See the [usage guide](../../docs/library/vega-kt-backtesting.md) for commands,
input conventions and the complete list of remaining capabilities.

The [weak Gamma numerical correction](../validation/local-vol-weak-gamma-v0.1.md) integrates density before inversion,
uses next-time weak recovery and resolves the auxiliary grid independently of
reporting nodes. The explicit compiled-payoff diagnostic reduces failures from
32/63 to 0/63 under unchanged thresholds; ordinary native acceptance remains
54/154 failures. Bucket localization and full paper conformance remain open.

The [remaining-failure investigation](../validation/vega-kt-remaining-failures-v0.1.md)
finds a persistent terminal Strike-allocation bias despite the weak chain's
passing scalar comparisons. Independent conditional-Gaussian expectations show
about 32% ATM-bucket bias at eight steps. Preserve Gamma point masses and test
the conditional recovery operator before integrating this estimator into normal
pricing. An exact-value counterexample also requires a separately versioned
statistical comparison specification; the current acceptance policy is unchanged.

The [terminal Gamma measure implementation](../validation/local-vol-terminal-measure-v0.1.md)
now preserves signed payoff kinks, propagates them with the exact flat kernel,
and projects the terminal hedge measure through the public Rust/Python pricing
plan diagnostic. All 24 terminal bucket references and source-hat expectations
pass (96 diagnostic checks). This resolves the terminal allocation bias in that
restricted path; ordinary acceptance remains 54/154 failures. General noisy-AD
recovery, covariance and ordinary-report integration remain separate work.

The subsequent [ordinary terminal-measure integration](../validation/local-vol-terminal-pathwise-v0.1.md)
resolves all of the previous 54 failures and exposes 6 hedge-grid failures under
the unchanged 154-check policy. Eligible flat terminal payoffs now retain actual
per-path bucket variance, price covariance and optional full covariance, including
antithetic and RQMC reductions. General noisy-AD recovery remains open. Exact
reference risk reproduces all 6 remaining failures because independent quote
shocks change with reporting-grid refinement. A fixed coarse market-shock basis
gives invariant quantities to rounding error in a separate investigation.
The next acceptance specification must freeze market shocks independently of
the numerical reporting grid, preserve all original failed observations, and
separately test independent fine-grid risk. The adapter remains unvalidated.

The [fixed-market-shock v0.2 gate](../validation/vega-kt-market-shocks-v0.2.md)
now implements that comparison: 307 checks pass across three explicitly fixed
market bases, with transformed full sample covariance and independent price
differences. The original 145 non-grid-quantity checks are retained unchanged;
all nine original grid-quantity comparisons, including the six failures, remain
in the report. The optimizer API always returns both the chosen-market residual
and the omitted-direction audit. A 33-node counterexample retains substantial
risk outside the nine-node market shock family, so this scoped result does not
establish hedge effectiveness. General smile acceptance and explicit daily
market-shock configuration remain open; the native adapter guard stays in place.

The [daily fixed-market-shock integration](../validation/vega-kt-market-shock-replay-v0.1.md)
adds opt-in shock nodes to replay, candidate selection and both optimizer
backends. It records held, continuous and rounded-proposed residuals, binds the
definition to resume/ensemble identities, and preserves atomic failure recovery.
Enabled runs use request/result/checkpoint/ensemble versions 7/11/9/3; the eleven
older schema resources and default formats remain unchanged. Native four-strategy
replay and exact resume pass. General-smile numerical acceptance, empirical
market-shock selection and hedge effectiveness remain open.

The [reporting-maturity correction](../validation/vega-kt-reporting-maturities-v0.1.md)
now decouples reporting dates from simulation times within the restricted flat
terminal model. Sparse dates and dates inside simulation intervals retain the
same terminal risk samples and covariance. Each quoted IV row is still checked
against actual cumulative Euler variance, and contract expiry must be the final
reporting date. The prior 307-check scoped gate and all original 154-check
observations remain unchanged, including the six original failures.


The [general-smile numerical acceptance v0.3](../validation/vega-kt-smile-acceptance-v0.3.md)
closes the P0 terminal-smile numerical blocker with exact signed static vanilla
replication, independent quote differences, whole-surface SSVI/eSSVI CRN price
rebuilds, and fixed-market risk/quantity refinement. All 770 new and 307 existing
checks pass. The original 154 observations/checks remain unchanged, including
six failed incomparable-grid comparisons. Each is linked to a passing
fixed-market replacement. Explicit `numerical_scope="p0_terminal_replication"`
enables accepted-P0 replay with per-contract model/market calibration checks;
the default guard and diagnostic mode remain. General noisy-AD nonterminal
Gamma inversion is not included in this acceptance.


The [positive-time path Gamma component](../validation/local-vol-path-gamma-recovery-v0.1.md)
now solves the full integrated source-hat convolution with explicit signed
Tikhonov recovery. Asian, fixed-history Asian and Fixed Lookback use their actual
compiled-payoff AD. Independent quadrature/SVD and CRN price differences pass
372 high-sample checks; one low-sample tail failure and three regularization
settings above the exploratory reconstruction threshold remain visible. Initial
Gamma, nonflat recovery and complete path-dependent IV-hedge acceptance remain
open. Ordinary P0 results and guards are unchanged.


The [conditional initial Gamma measure](../validation/local-vol-initial-gamma-measure-v0.1.md)
now connects the first boundary for flat-space continuous piecewise-affine path
payoffs. Future-shock conditioning preserves history and integrates the first
transition without density division or bumps. The former scalar-identification
blocker is resolved within this scope by using additional payoff information.
The three positive-time reconstruction quality failures remain; deterministic
input analysis attributes the dominant error to regularization rather than
sampling alone. Full path-dependent IV reconstruction, nonflat models and
barriers remain open.


The [split-sample Gamma audit](../validation/local-vol-gamma-selection-v0.1.md)
adds strongest-feasible regularization selection on disjoint pilot counters,
followed by one held-out validation and the initial conditional measure.
Reconstruction and Gamma sampling error are checked together. The original
fixed-lambda failures remain part of the historical record; validation failure
never triggers another selection on the same held-out data. Full IV-hedge
construction and acceptance remain separate work.

The [signed-atom IV chain](../validation/local-vol-atom-chain-v0.1.md) connects
conditional initial Gamma and pilot-selected positive-time Gamma to equations
(4)/(9) with analytic Gaussian product projection and same-sample cross-maturity
covariance. Sixteen flat calibrated configurations pass 168 component checks;
constant-vol parallel price differences are also checked. Full independently
recalibrated IV bucket acceptance, nonflat path recovery and inverse-bias/grid
convergence remain open; the accepted P0 provider is unchanged.

The [conservative Gamma extension](../validation/local-vol-conservative-gamma-v0.1.md)
adds a fixed equality constraint preserving integrated local Vega. Recalibrated
flat maturity-IV price comparisons pass 560 checks and difficult 33-node
conditions pass 88 checks without relaxing reconstruction limits. An independent
smile probe also identifies a remaining finite-Euler quote-response error:
sampling a continuous Dupire surface is not exact discrete smile recalibration.
That gap, nonflat path Gamma and complete strike-bucket acceptance remain open.

The [two-step discrete calibration](../validation/local-vol-discrete-smile-calibration-v0.1.md)
now fits listed terminal smile quotes with a fixed first marginal, including a
nonflat second-step base. Calibration/price/consistency checks pass 222/222.
Projecting Gamma through the same calibration tangent reveals 30/40 quote-bias
comparisons outside a separate 1% plus 0.002 target, despite passing the old
3% spatial gate. These failures remain explicit. Quote-sensitive recovery,
nonflat Gamma and general multi-interval calibration are still open.

The [calibration-direction constraints](../validation/local-vol-gamma-quote-conservation-v0.1.md)
preserve those quote functionals in the Gamma inverse. The original sampling
count retains one held-out Gamma-noise failure (25.2447% against 25%). With
fourfold validation sampling and unchanged thresholds, 1,828 checks pass across
four products and four seeds, including all 80 quote-bias comparisons. The
previous 30 failed quote comparisons are covered by this passing scope; old
evidence remains unchanged. General nonflat/multi-interval market-IV acceptance
remains open.


### 2026-10-05: 非平坦な第2区間のGamma回復

固定Gaussian第1分布・2区間限定で、状態依存の年率局所分散を直接使う作用素と
経路ADを追加した。初期Gammaを平坦式で代用せず、較正方向の感応度と実測SEを
別APIで報告する。[検証と残件](../validation/local-vol-nonflat-gamma-v0.1.md)。
次の対象は非平坦な将来を持つ初期Gamma、一般の第1分布・多区間への拡張である。


### 2026-10-05: 非平坦な継続価格の初期Gamma

2区間・一定第1分散に限定し、条件付き継続価格の符号付きGammaを細分化で回復する
別APIを追加した。初期年率分散感応度をscore積分・native AD・価格差分で照合する。
[検証記録](../validation/local-vol-initial-smile-gamma-v0.1.md)。
任意の第1分布、一般多区間、終端quoteを固定した第1市場IV bucketへの変換、
全IVリスクの同時受入は残る。有限領域外のGamma質量も未認定である。

### 2026-10-05: 第1parallel IVと終端スマイルの共同Gamma

上記2区間の初期Gammaと第2Gammaを較正接線で結合する診断を追加。
第1IV変更に伴う第2分散の再較正を含め、満期間・quote間の共分散を同一経路から計算する。
第1満期flat parallel 1方向と終端5quoteの範囲で、独立な再較正価格差分による検証を行う。
[検証記録](../validation/local-vol-joint-smile-gamma-v0.1.md)。
次の残件は第1満期自体の非平坦スマイル、一般多区間、全IV座標の受入と通常provider接続。
有限領域Gammaの制限およびpilot選択を条件とした不確実性の範囲は引き続き適用する。

### 2026-10-06: 3区間の非Gaussian sourceと終端Gamma

第1・第2のlog-Euler遷移から第2時点の密度を積分し、第3区間のGamma回復と終端IV再較正を追加。
2プロファイル×4商品×2 seedsで984件合格。旧40ケースは出力完全一致。
[検証記録](../validation/local-vol-three-step-gamma-v0.1.md)。
第3rowのみの受入で、先行2rowは固定する。次は初期・中間Gammaおよび満期間接線の結合、
その後に任意区間数と非平坦な第1市場満期へ拡張する。一般IV全体の受入は未確立のまま保持する。


The [2026-10-06 middle/terminal Gamma record](../validation/local-vol-middle-smile-gamma-v0.1.md)
adds history-conditional middle Gamma, signed boundary-aware refinement, and
joint middle/terminal quote recalibration. All 1,344 checks pass for four
payoffs, two nonflat profiles and two seeds. The first marginal remains fixed.
Native MC validation uses weighted first-state cell sweeps; insufficient sampling
still withholds estimates. The old 40 diagnostics and 8 three-step terminal cases
remain byte-identical. Three-step initial Gamma, the first-maturity calibration
chain, arbitrary step counts and production full-IV acceptance remain open.


### 2026-10-06: 3区間の初期Gammaと第1満期の較正連鎖

[初期Gammaの受入記録](../validation/local-vol-initial-three-gamma-v0.1.md)で、最初の観測値と
後続2遷移をともに動かしたGammaを回復。第1平行IV＋中間5点＋終端5点を接続した。
全16条件、1,480検査合格、失敗0。第1方向の独立再較正価格差分を新規検査し、
従来10方向の価格差分証跡は、較正requestと中間・終端Gammaの完全一致を条件に引き継いだ。
関連192テスト合格、旧48ケースはbyte同一。第1満期の個別スマイル点、任意区間数、
全方向共分散、carry・配当・barrier・早期行使と通常providerでの全面受入は引き続き対象外。


### 2026-10-06: 可変区間数の終端Gammaと非平坦な第1市場満期

[可変区間の受入記録](../validation/local-vol-multi-step-gamma-v0.1.md)で4・6区間の終端Gammaと終端5 quoteを検証。
全先行遷移を密度積分し、全観測履歴を固定した終端payoff積分でAsianの標本誤差を低減した。
第1市場満期は2遷移後の非平坦スマイル。16条件、1,402検査合格、失敗0。
204テスト合格、旧48ケースはbyte同一。入力上限は2..16区間。
先行Gammaと先行市場quotesの変更、全区間の較正連鎖、通常providerへの全面接続は残る課題。


### 2026-10-06: 可変区間の先行Gammaと全報告満期の接続

[条件付きGamma連鎖の受入](../validation/local-vol-path-gamma-chain-v0.1.md)で、4区間2満期・6区間3満期の
各5 quotesを再較正し、先行Gammaと終端Gammaを同一パスで合算した。第1満期は非平坦。
2,704検査・215テスト合格、失敗0。従来64ケースの出力はbyte同一。
各報告満期直前の分散rowを動かす延長規則を採用し、報告されない区間のrowは固定する。
異なる較正延長、全方向共分散、一般商品の通常provider接続は残る課題。


### 2026-10-06: 可変区間Gammaの全市場方向共分散

[全方向共分散の受入](../validation/local-vol-path-covariance-v0.1.md)で、native AD・回復Gamma・
両者の差を同じsampling unit上で集計し、方向間の相殺を保持した標準誤差を追加。
共分散304検査・46テスト合格、前回全16条件の数値出力・判定は完全一致。
追加合成方向の6件の標本精度不足は増標本で解消したが、うち6区間Asian系2件は
中間rowのまれなpayoffに対するnative AD照合が未解消。合成方向の最終受入は14/16条件。
較正规則と数値受入閾値は変更しない。異なる商品の合同共分散、較正延長の一般化、
通常バックテストproviderへの接続は残る。


### 2026-10-06: 中間区間のまれなpayoffに対するAD照合

[条件付きnative ADの受入](../validation/local-vol-conditional-ad-v0.1.md)で、最終乱数の解析積分と
条件付き分散を用い、残っていた6区間Asian系2条件を解消した。合成方向は16/16条件合格。
全市場方向と合わせて32パネル・5,904検査、関連206テストが合格。
Gamma・raw AD・全方向共分散は完全一致し、許容差は維持。従来の直接照合結果も診断に残す。
一般の較正延長、異なる商品の合同共分散、通常バックテストproviderへの接続は引き続き未受入。


### 2026-10-07: 複数満期Gammaの欧州型バックテスト接続

[複数満期providerの受入](../validation/local-vol-multi-step-provider-v0.1.md)で、
4・6区間の条件付きGammaと較正接線を既存の欧州型台帳へ接続した。
Call/Put・2 seedsの8条件を解析的な市場Vegaに照合し、4戦略の約定・満期決済・再開を検証。
実際の尾部精度不足ではリスクを抑止し、注文・約定を作らない。
受入範囲は各報告満期直前の分散rowを動かす延長に限定する。
Asian/Lookbackの観測履歴と決済、一般較正延長、異なる商品の合同共分散は引き続き未受入。


### 2026-10-07: Asian・Lookbackの観測履歴と決済接続

[経路依存台帳の受入](../validation/local-vol-path-ledger-v0.1.md)で、
日付・出所付きfixingの保持、native payoffとの照合、全観測からの満期決済を追加した。
20評価条件、16通りの満期バックテスト、関連290テストが合格。
使用済みfixingの変更は拒否し、欠損等の失敗ではsnapshot全体を取り消して修復再開できる。
request v8 / result v12 / checkpoint v10 / ensemble v4を追加し、旧16スキーマはbyte同一。
Lookback確定日当日のnative評価、一般の較正延長、異なる商品の合同共分散は未受入。


### 2026-10-07: Lookback観測日当日の評価

[当日確定値の受入](../validation/local-vol-lookback-fixing-day-v0.1.md)で、
前回残した観測日当日のnative評価を追加する。
当日値を過去の極値と分けて保持し、価格と全感応度で定数として扱う。
確定前の同日snapshot、一般の較正延長、実市場のヘッジ効果は範囲外を維持する。


### 2026-10-09: 全区間の較正延長

[全区間の分散較正](../validation/local-vol-variance-extension-v0.1.md)で、初期rowを含む同時較正と
時間方向の線形補間を診断APIに追加。各延長規則で再較正した独立価格差分をGammaへ照合する。
最初の差分幅・標本不足の失敗は保存し、差分の精細化と4区間の増標本で再検査した。
4・6区間、2方式、4商品のCall側、2 seedsの32パネル・2,546検査と関連265テストが合格した。
通常台帳への新方式の接続、Put側・任意の基底の全面的な受入、実市場のヘッジ効果は引き続き残る。

### 2026-10-09: 全区間較正と通常台帳の接続

[全区間較正provider](../validation/local-vol-extension-provider-v0.1.md)に、all_rows / linear_timeの明示選択を追加。
較正方式をprovider・監査・保存状態の識別子に含め、異なる方式での再開を拒否する。
Call側European・Asian・Lookbackの4/6区間を対象に、価格分布、Gamma精度、観測履歴、売買、決済、途中再開を検証する。
Put側・任意custom基底、異商品の合同共分散、実市場のヘッジ効果は引き続き未受入。

### 2026-10-10: Put側の全区間Gammaと混合ヘッジ

[Put対応scope](../validation/local-vol-extension-put-v0.1.md)で、両presetのEuropean・Asian・Lookbackを
独立Put価格差分へ照合し、PutターゲットとCallヘッジの決済・再開を検査する。
既知最安値98のLookbackで残った回復誤差には、実効行使価格を積分区切りに含める規則を追加した。
初回失敗を保存し、標本数と許容誤差を維持して再検証する。
48数値パネル・3,076検査、providerの24パネル・混合ヘッジ32戦略、関連275テストが合格した。
任意custom基底、異商品の合同共分散、実市場のヘッジ効果は未受入のままとする。

### 2026-10-11: 明示したカスタム較正基底の数値受入

[時間・マネーネスの積で作る基底](../validation/local-vol-custom-basis-v0.1.md)と、列ごとの尺度を記録する診断APIを追加。
2種類の粗い基底について、4/6区間・Call/Putの10履歴・2 seedsを独立価格差分へ照合した。
初回34件の接線差分誤差を、許容差を保った差分幅の精細化で解消した。
80パネル・5,540検査、関連262テスト、配布版の公開API 4パネルが合格した。
任意の基底設定の全面受入、カスタム基底の通常台帳への接続、異商品の合同共分散、実市場のヘッジ効果は残る。

### 2026-10-12: 名前付き基底と通常台帳の接続

[名前付き基底provider](../validation/local-vol-custom-basis-provider-v0.1.md)で、受入済みの
row_hats / weighted_time_hatsを明示選択し、4/6区間のCall/Put台帳へ接続する。
生成規則・尺度・基底hashを監査し、異なる基底や実装での保存状態の再開を拒否する。
40数値パネルの完全一致、64戦略、96契約評価、285テスト実行と配布版・CLIの照合が合格した。
任意基底の全面受入、異商品の合同共分散、実市場のヘッジ効果は未受入を維持する。

### 2026-10-12: raw行列ごとの独立数値検証

[基底行列の検証入口](../validation/local-vol-raw-basis-v0.1.md)を追加。
行列を市場・格子・尺度の意味に結び付け、独立再較正価格とnative Gammaを照合する。
時間とマネーネスを結び付ける行列を対象に、4/6区間・Call/Putの全市場方向を検査する。
40パネル・2,770検査、単体テスト4件、CLI拒否3件、完了結果からの再開2件が合格した。
配布Python/schema 56ファイルを維持し、前回の台帳の保存状態から同じ結果へ再開できた。
既存providerと数値実装を維持し、検証結果から通常台帳への自動採用は行わない。

### 2026-10-12: 受入済み結合行列と通常台帳の接続

[結合基底provider](../validation/local-vol-coupled-basis-provider-v0.1.md)で、
`coupled_row_hats` を明示選択する。4/6区間の生成行列を独立受入済みhashへ完全一致で制限し、
Call/Putの経路依存台帳へ接続する。基底の変更や旧実装の保存状態からの再開は拒否する。
20パネルの独立基準との完全一致、32戦略、48件の契約評価、16件の失敗・修復検査と283テスト実行が合格した。
配布版の両区間でも売買・決済・損益が一致し、別provider・実CLIによる再開を確認した。
任意の行列・市場の全面受入、異商品の合同共分散、実市場のヘッジ効果は引き続き未受入。

### 2026-10-13: 現行版の行列別検証と保存結果の整合性

[現行版チェッカー](../validation/local-vol-current-raw-basis-v0.1.md)を追加。
旧版に固定されていた検証入口を、現行の受入記録と変更のない数値計算へ接続する。
事前確認では数値合格を主張せず、保存内容・行列・実装が一致する場合だけ未完了パネルから再開する。
40パネル・2,770検査を再計算して以前の数値結果と完全一致し、両区間の途中再開もバイト単位で一致した。
入口の単体テスト9件とCLI拒否6件が合格。nativeを読み込まない条件の単体検査も9件が合格した。
ライブラリ本体と既存の台帳保存状態を維持し、raw行列の通常台帳への自動採用は行わない。
