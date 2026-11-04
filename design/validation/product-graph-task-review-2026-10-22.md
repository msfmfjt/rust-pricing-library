# Product Graph and Pricing Task review — 2026-10-22

## Scope and source revisions

This continues the main-branch architecture review at `45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`.
Detailed review resumed with Product Graph (#158, `9fc6b46`) and Pricing Task
(#160, `f62aa72`), comparing their implementations with the current stacked tip
#169 (`aab90e58ecf2a446ea32ca559fdefae384940d3d`). Fixes are based on that tip so
the later typed risk, Python source, compiled-task reuse and report changes are
preserved. This record does not certify an exhaustive review of every later PR.

The open dependency chain is linear; none was merged as part of this review:

| PR | Head branch | Target |
| --- | --- | --- |
| #155 | `feat/unified-pricing-specification` | `main` |
| #158 | `feat/generic-product-graph` | #155 |
| #160 | `feat/pricing-capabilities-replay` | #158 |
| #161 | `feat/typed-risk-errors` | #160 |
| #162 | `feat/typed-python-sources` | #161 |
| #163 | `refactor/wheel-api-contracts` | #162 |
| #164 | `feat/reuse-compiled-pricing-tasks` | #163 |
| #165 | `feat/calibration-report-diagnostics` | #164 |
| #166 | `fix/backtest-provider-cache-lifetime` | #165 |
| #167 | `feat/backtest-task-report-bridge` | #166 |
| #168 | `fix/immutable-provider-configuration` | #167 |
| #169 | `fix/native-provider-compatibility` | #168 |

## Findings and fixes

### P1: Windows CI cannot discover model workflow guards

`ArchiveIntegrationTests.sources` used `str(relative_path)` as a tar-member key.
On Windows this produces backslashes, while archive validation and the workflow
filter require `.github/workflows/`. The real Windows job for #169 failed at
`test_model_ci_still_triggers_on_shared_checker_changes` because its workflow
list was empty. The same Rust/Windows gate is red on #163–#168.

Use `Path.as_posix()` for archive member names. Preserve the existing mutation
checks, including rejection when a model workflow drops the shared checker
path filter. No workflow, timeout or numerical assertion is removed.

### P2: Compiled capability accepts an unusable basket Gamma bump

The native preflight checked the relative bump range and down-Spot domain, but
the representability check lived inside the evaluators. Thus a tiny positive
bump produced `supported`, succeeded in task compilation/reselection, and then
failed after pricing work began. A new regression fails on the unmodified tip:
`multi_asset_bs` reports `Supported` for the smallest positive `f64` bump.

Share Gamma bump resolution between native preflight, ordinary pathwise Gamma
and recalibrated Gamma. Require a positive finite displacement, a finite up
Spot, and distinct up/down Spots. Validate the actual down Spot `S - b*S`, the
same arithmetic used by the evaluators. Task capability, fresh compilation,
reselection and direct native AAD all use the same native guard.

Rust and Python regressions cover MC/BS and RQMC/joint-calibration sources,
unrepresentable positive bumps, equal fresh/reused errors, and continued exact
usability of the original plan after a failed selection. Existing tests retain
exact price/risk/replay comparisons for ordinary Gamma bumps.

## Responsibility boundaries assessed

| Boundary | Assessment |
| --- | --- |
| Product Graph | Owns immutable payoff nodes, ordered underlyings and payment dates. Validates DAGs, resource limits and each output's observation dependencies. Market-dependent discount ratios remain compiled state. |
| Pricing source / Task | Retains source model, market, calibration, numerical policy and selected operation. Serialized documents do not contain execution workspaces or calibrated caches. |
| Rust engine | Private adapters select existing native plans. Preflight and execution share guards; corrected Gamma validation closes one demonstrated gap. |
| Python | Frozen source/task wrappers delegate validation and numerical work to Rust. Operation selection preserves the structured compile-stage error contract. |
| Risk reports | Construction remains above numerical loops; native estimates, optional uncertainty and native results are retained. Later typed-coordinate and calibration-report changes are preserved. |
| Numerical engine | The three-crate dependency direction remains `pricing-python -> pricing -> pricing-numerics`. Financial composition remains in `pricing`; numerical primitives remain independent. |

Graph review covered constant/shared outputs, per-output observation/payment
ordering, dead nodes, hard-indicator rejection, deterministic multiple-payment
lowering, the existing curve-discount AAD correction, and nonzero-HW-volatility
guards. The targeted tests did not reveal another blocker in those paths. This
is not a claim that arbitrary new primitives or model combinations are supported.

## Numerical, API and replay impact

There are no signature, schema, fixture, RNG, calibration, reduction-order or
tolerance changes. Valid Gamma evaluations use the same displacement and the
same numerical kernels. Source/task/plan fingerprints remain unchanged because
valid numerical semantics are unchanged. Invalid Gamma inputs fail earlier;
Python reports their failure at compilation/reselection rather than evaluation.

## Verification

The initial debug-symbol build exhausted the transient filesystem; its failed
run is not treated as a test pass. The successful repeat uses environment-only
`CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`
and two build jobs, without changing repository profiles or acceptance policy.

- The unmodified tip passed 23 targeted Graph/Task/reuse tests. The new Gamma
  regression then failed on that implementation before the fix.
- The corrected Graph/Task/reuse tests passed: 24 tests, including the new
  early-rejection test and existing analytic fixed-cash-flow curve AAD/replay.
- Rust workspace/all-features suite excluding the Python binding crate:
  1,064 passed, 0 failed, 91 ignored. The ignored panels remain explicit
  acceptance gates; five of them were run separately as recorded below.
- Native Python-binding Rust tests: 5 passed. The host Python distribution had
  a stale library-directory setting and a broken linker symlink; a temporary
  linker path to its existing shared library resolved the initial link failure.
  No repository build configuration or runtime distribution file was changed.
- Explicit statistical, path-dependent and early-exercise acceptance: 5 passed.
- Companion backtesting suite: 330 passed.
- Fresh Linux CPython 3.12 release wheel passed package/archive checks, metadata,
  checksums, dependency SBOM, ABI, type stub, clean installation and runtime
  contract validation of 341 callables.
- Installed-wheel tests: 53 passed across Graph, Task, task reuse, specification,
  BS/LSV/rough multi-asset APIs, wheel contracts and native provider compatibility.
  The two public Graph/Task examples also passed. A mistyped module selector
  initially selected zero tests and failed the runner; the correctly named
  contract modules were then executed successfully. No failed test was skipped.
- Python discovery matched 104 modules / 589 tests. These are inventory counts,
  not a claim that the full Python suite ran locally.
- Wheel validation script tests: 26 passed. Schema gates passed, including 50
  specification rejections, 38 graph rejections, and 42 task/report rejections.
- Formatting, workspace/all-target check, all-target/all-feature Clippy with
  warnings denied, public Rust API docs, dependency direction, Markdown links
  and the staged source archive passed.

## Outstanding merge gates and follow-up

At the initial check on 2026-10-22, main had 31 successful checks and 4 skips;
#158 had 102 successful checks and 4 skips. #155 and #160 each had one cancelled
gate; #161/#162 also had cancellations. #163–#169 had the Windows Rust failure
described above, with some later platform/wheel jobs still queued or running.
Check-run pagination was included; absence from the PR-only workflow-run listing
was not interpreted as absence of CI.

#164/#165 also reported failed Windows extended-model accuracy jobs. Their risk
steps passed, but the accuracy step remained marked in progress and GitHub log
retrieval returned `BlobNotFound`. Their numerical outcome cannot be established
from those records; these failures remain unresolved, not classified as benign.

Require current-head cross-platform Rust, full installed-wheel and independent
numerical acceptance checks before merge. The full Python suite and all ignored
heavy model/refinement panels are separate hosted gates. Continue detailed
review of the later typed Python, reuse, report and backtest PRs without treating
their presence on the branch as review approval. All merges require the user's
explicit approval.
