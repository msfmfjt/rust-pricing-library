# Agent instructions

## Project map

- Rust workspace: `crates/pricing-numerics`, `crates/pricing`, and `crates/pricing-python`.
- User-facing documentation: `docs/library`, `docs/products`, and `docs/models`.
- Requirements, ADRs, roadmaps, and validation evidence: `design`.
- Python tests: `tests/python`. Shared checks and wheel tooling: `scripts`.
- Contributor workflow and required commands: [CONTRIBUTING.md](CONTRIBUTING.md).
- Current repository state and stacked PR handoff: [design/CODEX_HANDOFF_2026-10-29.md](design/CODEX_HANDOFF_2026-10-29.md).

## Before changing behavior

1. Read the applicable current contract in `docs/`, then the governing requirement, ADR, and validation protocol in `design/`.
2. Keep numerical methods, sample counts, seeds, tolerances, fixtures, serialization, and fingerprints unchanged unless the task explicitly calls for changing them and the change has a documented numerical/reproducibility rationale.
3. Keep implementation slices narrow. Add regression coverage for observable behavior and failure cases. Do not claim an acceptance panel passed based on a focused test or a stale CI run.
4. Preserve the Rust/Python/wire API contract and source archive checks. Put execution logic in private engine modules; keep public domain/source types separate from execution workspaces.

## Validation

Use the focused test first, then the applicable checks in `CONTRIBUTING.md`. At minimum for Rust source changes, run:

```bash
cargo fmt --all --check
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
```

Do not omit ignored statistical/refinement/release acceptance panels when they are relevant. For Python binding or packaging changes, build/install the wheel and run its ABI-matched validation groups. See `CONTRIBUTING.md` for the complete matrix.

## Branch and PR handling

The repository currently has a large stack of Draft PRs; see the handoff document. Refresh the head/base and current checks before working. Preserve stacked ancestry and do not bulk rebase, merge, close, or delete branches. Make changes on the selected active branch and report any required ancestor/base updates clearly.
