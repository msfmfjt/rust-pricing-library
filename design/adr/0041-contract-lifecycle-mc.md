# ADR 0041: Contract semantics lowered to the existing Monte Carlo graph

Date: 2026-11-05. Status: accepted for implementation.

## Context and baseline

Contract DSL / Product Graph requirements v1.1 require Phase 0–4, retaining three
crates and existing MC simulation, discounting, calibration and risk rules.
Inspected main: `75902772ea615b7cf551f8f849db7cb7636ef87c` (PR #203).
PR #155 introduced pricing specifications; #158 introduced GraphProduct and
multi-payment discount pullbacks; #160 introduced PricingTask, shared capability
checks and risk reports. The latter PRs remain open as historical stacked
branches; their implementation landed through #203.

Main [CI](https://github.com/msfmfjt/rust-pricing-library/actions/runs/41882223740)
and [Backtesting](https://github.com/msfmfjt/rust-pricing-library/actions/runs/41882223610)
passed. Earlier runs on that commit were cancelled/failed and are not passes.
Open #205 changes executor reuse; #206 changes shared validation/aggregation and
has separate native-memory diagnostics. Neither is a dependency of this work.

The library is not exclusively MC: it contains analytical closed-form utilities
and Heston Fourier calculations. This work adds only an MC contract integration;
existing analytical compatibility APIs remain intact.

## Decision

Add a model-independent canonical Contract source in `pricing::contract`.
Typed numeric, Boolean and date expressions describe contractual decisions,
observations and cash flows. Contract combinators are source semantics. Lower
them into existing SourceGraph / GraphProduct / CompiledPayoff. Never persist a
second independently editable payoff graph alongside the contract.

Observation identity includes underlying, type, date and fixing source. Intraday
or unsupported model observations fail explicitly until a native simulation
mapping exists. Historical fixings require exact identity; no spot fallback.
Payment currency, cash-flow identity, determination and settlement are retained.
Different currencies remain separate valuation groups, never implicitly added.

Lifecycle reduction is pure and independent of simulation. Events have stable
IDs; identical duplicates are idempotent and conflicting IDs fail. Substitute
known observations, eliminate resolved branches, retain determined unpaid
amounts. Settlement is explicit, not inferred from passage of a payment date.
Retain event provenance for incremental replay; do not apply future events.

Discrete barrier boundaries are inclusive. Autocall is sequential conditional
composition: redemption cancels later contingent cash flows. Choice/exercise and
continuous monitoring are distinct unsupported features in this delivery; no
max-of-future-payoffs or unrequested bridge approximation is substituted.

Reuse native graph compilation, grids, paths, pathwise discounting, aggregation
and AAD. Add graph primitives only with matching forward, constant-fold, reverse,
fingerprint and wire behavior. Preserve legacy numerical results and opcode
identities. Scope compiler optimizations to newly lowered contracts so legacy
plan identities stay stable. Hard discontinuities retain native pathwise-risk
rejection. MC finite differences use common random numbers and explicit bumps.

## Compatibility and migration

Contract JSON is independently versioned, strictly decoded and validated with
stable IDs. Existing request/result v1–v3, graph v1, specification and task
documents retain their semantics. New operations must not silently extend
frozen schemas. Never serialize compiled tapes or model workspaces. Python uses
the same Rust validation and native execution.

Migration is opt-in: existing named products remain available; contract builders
provide equivalent sources and identical-path regression checks. Reduce saved
contracts at valuation before entering existing pricing specifications/tasks.
Capability checks remain within the existing MC integration. Unsupported rate
observations and multi-payment stochastic-rate adapter combinations fail rather
than substitute deterministic approximations.

## Delivery and verification

Separate reviewable changes: design; typed source/analysis/serialization;
lifecycle/replay; graph/MC integration; exotic builders/Python/performance.
Test behavior and rejection, deterministic path equivalence, event replay,
residual future cash-flow equivalence, same-seed MC PV/Greeks, native AAD,
Python/JSON round trips, properties and mutations. Benchmark compilation,
evaluation, memory and complete MC runtime against existing paths.
Report hosted CI for submitted heads; pending checks never count as passes.
No direct main push or merge is part of this work.

## Implemented graph extension

The native graph adds Select, Compare, Log and Exp with appended opcode tags;
legacy opcode tags and nonbranching instruction order remain unchanged. Lazy
selection uses reusable iterative traversal state, avoiding recursion in path
loops and avoiding inactive branch arithmetic/observations. Contract lowering
interns exact operations and drops known branches before native compilation.
Graph v2 / specification v3 / task v2 are selected only when these operations
occur. Legacy writers and fingerprint domains remain at v1 / v2 / v1. Current
schemas include conditional version guards; old schema files remain frozen.
Lifecycle partial evaluation retains arithmetic with an unresolved branch rather
than raising its inactive error; selected invalid determined amounts still fail.

The first performance pass exposed allocation overhead for new lazy tapes.
Their existing `evaluate` entrypoint now reuses thread-local scratch arrays,
retaining at most 1 MiB per worker and clearing values/visited state on each path.
Reentrant observation callbacks use an owned fallback. This cache contains no
contract/model inputs, does not affect fingerprints, and is never serialized.
Legacy nonbranching evaluation and AAD keep their original allocation/order.
Callers that own workspaces can use `evaluate_reusing` without allocating an
owned output vector. Existing model simulation buffers are unchanged.
