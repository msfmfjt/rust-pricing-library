# Codex handoff: rust-pricing-library

**Snapshot:** 2026-10-29 (Europe/London)  
**Repository:** [msfmfjt/rust-pricing-library](https://github.com/msfmfjt/rust-pricing-library)  
**Default branch:** `main` at `615f648c2673debafd685fe32d1722bbf921a07d`  
**Latest merged PR:** [#155](https://github.com/msfmfjt/rust-pricing-library/pull/155), merged 2026-10-29.

## Start here

This repository is a three-crate Rust 2024 workspace with a typed PyO3 Python interface:

- `crates/pricing-numerics`: numerical primitives.
- `crates/pricing`: pricing APIs and private execution engines.
- `crates/pricing-python`: Python bindings.
- `docs/`: user-facing library, products and model contracts.
- `design/`: requirements, architecture, ADRs, roadmaps and validation evidence.
- `tests/python`, `scripts/`, `.github/workflows/`: Python tests, conformance gates and CI.

Read [CONTRIBUTING.md](../CONTRIBUTING.md), [design index](README.md), and [pricing specification ADR 0028](adr/0028-pricing-specification.md) before making changes. Current contracts live in `docs/`; older ADRs and validation reports are historical records and can be superseded. Preserve the documented scope, numerical evidence and replay compatibility.

## Repository state

PR #155 added complete `PricingSpecification` source composition and replay adapters. Its current head was `158155ce90ca1b18cd8c38a497c949fde4e91891`. It was merged to `main` with merge commit `615f648c2673debafd685fe32d1722bbf921a07d`. All 106 checks on that head were successful or explicitly skipped after the previously cancelled macOS stochastic-dividend risk/refinement job was rerun successfully. No review submissions were present when checked.

At the snapshot there are 32 open PRs, all marked Draft. They form a large stacked sequence; the default base of most PRs is another feature branch, not `main`. Do not merge, retarget, close, or delete the stack as a batch. Refresh every PR head, base, checks, and branch state before taking action.

### Open PR stack

The main line continues from #155:

`#155 → #158 → #160 → #161 → #162 → #163 → #164 → #165 → #166 → #167 → #168 → #169 → #170`

Two branches start at #170: #171 continues the main development line; #172 is the separate mutation-testing branch.

`#170 → #171 → #173 → #174 → #175 → #176 → #177 → #178 → #179 → #180 → #181 → #182 → #183 → #184 → #185 → #186 → #187 → #188 → #189 → #190`

| PR | Head branch | Work |
| --- | --- | --- |
| [#190](https://github.com/msfmfjt/rust-pricing-library/pull/190) | `fix/compact-smoothing-numerical-range` | Compact smoothing values and adjoints at extreme scales |
| [#189](https://github.com/msfmfjt/rust-pricing-library/pull/189) | `fix/provider-ledger-case-shards` | Split long provider ledgers without dropping numerical coverage |
| [#188](https://github.com/msfmfjt/rust-pricing-library/pull/188) | `fix/graph-division-adjoint-range` | Preserve finite graph division adjoints at extreme scales |
| [#187](https://github.com/msfmfjt/rust-pricing-library/pull/187) | `fix/windows-ci-failure-propagation` | Expose Windows CI failures and correct report replay checks |
| [#186](https://github.com/msfmfjt/rust-pricing-library/pull/186) | `fix/macos-debugger-venv-fixtures` | Canonicalize macOS debugger virtual-environment fixtures |
| [#185](https://github.com/msfmfjt/rust-pricing-library/pull/185) | `fix/executor-vector-allocation-errors` | Typed errors for dynamic executor allocation failures |
| [#184](https://github.com/msfmfjt/rust-pricing-library/pull/184) | `fix/lldb-native-launch-arguments` | LLDB literal arguments and Python startup handling |
| [#183](https://github.com/msfmfjt/rust-pricing-library/pull/183) | `fix/brownian-bridge-coefficient-validation` | Reject degenerate Brownian bridge coefficients |
| [#182](https://github.com/msfmfjt/rust-pricing-library/pull/182) | `fix/macos-native-wheel-backtraces` | Capture native macOS wheel failures with LLDB |
| [#181](https://github.com/msfmfjt/rust-pricing-library/pull/181) | `fix/time-grid-allocation-errors` | Typed errors for impossible time-grid allocations |
| [#180](https://github.com/msfmfjt/rust-pricing-library/pull/180) | `fix/portable-wheel-crash-diagnostics` | Portable wheel crash checks and Windows refinement timing |
| [#179](https://github.com/msfmfjt/rust-pricing-library/pull/179) | `fix/legacy-wire-error-schema` | Preserve legacy wire input versions in validation errors |
| [#178](https://github.com/msfmfjt/rust-pricing-library/pull/178) | `fix/wheel-provider-validation-shards` | Partition heavy provider wheel validation |
| [#177](https://github.com/msfmfjt/rust-pricing-library/pull/177) | `fix/report-validation-schema-version` | Preserve report input versions in validation errors |
| [#176](https://github.com/msfmfjt/rust-pricing-library/pull/176) | `fix/valuation-report-error-pointers` | Correct validation error locations in valuation reports |
| [#175](https://github.com/msfmfjt/rust-pricing-library/pull/175) | `fix/multi-asset-error-sources` | Preserve typed numerical causes across multi-asset APIs |
| [#174](https://github.com/msfmfjt/rust-pricing-library/pull/174) | `fix/executor-allocation-errors` | Typed errors for oversized reduction block tables |
| [#173](https://github.com/msfmfjt/rust-pricing-library/pull/173) | `fix/hw-market-iv-boundary-preflight` | Reject HW market-IV risk at exact variance bounds |
| [#172](https://github.com/msfmfjt/rust-pricing-library/pull/172) | `test/mutation-testing` | Scoped mutation testing and stronger pricing contracts |
| [#171](https://github.com/msfmfjt/rust-pricing-library/pull/171) | `fix/task-validation-pointers` | Fix nested task validation error locations |
| [#170](https://github.com/msfmfjt/rust-pricing-library/pull/170) | `fix/product-graph-task-review` | Gamma preflight and Windows workflow validation |
| [#169](https://github.com/msfmfjt/rust-pricing-library/pull/169) | `fix/native-provider-compatibility` | Check native API compatibility before provider startup |
| [#168](https://github.com/msfmfjt/rust-pricing-library/pull/168) | `fix/immutable-provider-configuration` | Freeze provider configuration and prevent stale acceptance |
| [#167](https://github.com/msfmfjt/rust-pricing-library/pull/167) | `feat/backtest-task-report-bridge` | Connect backtests to pricing tasks and valuation reports |
| [#166](https://github.com/msfmfjt/rust-pricing-library/pull/166) | `fix/backtest-provider-cache-lifetime` | Bound backtest caches by snapshot and capacity |
| [#165](https://github.com/msfmfjt/rust-pricing-library/pull/165) | `feat/calibration-report-diagnostics` | Expose calibration diagnostics in valuation reports |
| [#164](https://github.com/msfmfjt/rust-pricing-library/pull/164) | `feat/reuse-compiled-pricing-tasks` | Reuse compiled pricing tasks across operations |
| [#163](https://github.com/msfmfjt/rust-pricing-library/pull/163) | `refactor/wheel-api-contracts` | Modularize wheel validation and runtime signatures |
| [#162](https://github.com/msfmfjt/rust-pricing-library/pull/162) | `feat/typed-python-sources` | Typed Python builders for complete pricing sources |
| [#161](https://github.com/msfmfjt/rust-pricing-library/pull/161) | `feat/typed-risk-errors` | Typed risk coordinates and structured Python failures |
| [#160](https://github.com/msfmfjt/rust-pricing-library/pull/160) | `feat/pricing-capabilities-replay` | Capability preflight, risk reports and complete replay |
| [#158](https://github.com/msfmfjt/rust-pricing-library/pull/158) | `feat/generic-product-graph` | Generic product graphs across Rust, Python and JSON |

## Immediate continuation

Start by refreshing the current state of #190 and its ancestors (#189 through #158): read the PR descriptions and comments, inspect current CI on each latest head, and compare their base SHAs against the branch stack above. The PR bodies contain detailed validation protocols and known open gates, but many status snapshots are historical. Do not report an old run as current validation.

Known risks from the latest available PR records:

- #190 described 94 queued and four skipped checks on 2026-10-28; this is stale and must be refreshed. It said the full supported-platform replay/benchmark gates, extended price/risk panel, and 111 wheel selections were still required.
- #189 described 111 generated numerical selections queued, with unresolved intermittent macOS invalid-free failures from #186 and previous Windows timeouts. The prior local Put lookback ledger had no completion record and was explicitly incomplete.
- #187 addresses a Windows CI failure-propagation issue. Verify the workflow no longer masks earlier command failures.
- #186/#182 investigate intermittent native macOS crashes. A successful run is not by itself proof the underlying intermittent issue is resolved.
- PR authors explicitly state that no merges were requested/performed for these Drafts. Treat each PR's stated scope, frozen fixtures, sample counts, tolerances and release gates as intentional.

## Build and validation

The workspace pins Rust via `rust-toolchain.toml`; `Cargo.toml` declares edition 2024 and minimum Rust 1.98. Python packaging is configured in `pyproject.toml` and uses maturin/PyO3 0.30.

Begin with the focused tests for the touched area, then use the applicable checks in [CONTRIBUTING.md](../CONTRIBUTING.md). Common baseline checks:

```bash
cargo fmt --all --check
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features --exclude pricing-python
cargo test --locked -p pricing-python
cargo test --locked -p pricing --no-default-features
cargo test --locked -p pricing-numerics
python3 scripts/check_schemas.py
python3 scripts/check_markdown_links.py
cargo metadata --locked --format-version 1 --no-deps | python3 scripts/check_dependency_direction.py
```

The ignored release-mode statistical, refinement, extended-model and extended-risk panels are substantive gates, not optional cleanup. Follow the relevant CI job and design validation protocol. For wheel work, use the same CPython ABI as the wheel; CI builds each platform wheel once and validates the same artifact in the `custom-basis`, `providers`, and `remaining` groups. See [CONTRIBUTING.md](../CONTRIBUTING.md) for complete commands.

## Codex workspace note

During preparation of this handoff, the current Codex task had no attached worktree or repository artifact. A local clone attempt failed because this environment could not resolve `github.com`. GitHub's connected tools were used to inspect the repository and merge #155; this handoff is being prepared on its own branch. To work on source code in a local Codex checkout, clone the repository in an environment with GitHub network access, check out `main`, then choose one active PR branch after reviewing its stack and current CI. Do not start by rebasing all stacked PRs together.

## Current documented base

PR #155 adds the common `PricingSpecification` source API, schema, fingerprint, replay fixtures and compiler shared by Rust/Python. The Rust workspace and Python interfaces remain three-crate and typed. See [ADR 0028](adr/0028-pricing-specification.md), [pricing specification guide](../docs/library/pricing-specification.md), and the [design index](README.md).
