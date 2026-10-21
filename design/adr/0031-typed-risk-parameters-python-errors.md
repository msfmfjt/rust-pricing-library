# ADR 0031: Typed risk coordinates and structured Python failures

Date: 2026-10-21. Status: implemented.

## Context and decision

ADR 0030 introduced common risk units, but labeled native AAD results still
required the report adapter to infer units from label substrings. The Python
boundary also converted native compilation/evaluation failures to strings,
discarding useful error variants.

Native Hull–White and stochastic-dividend AAD producers now supply ordered
`RiskParameterKind` metadata alongside labels and derivatives. The enum maps
exhaustively to `SensitivityUnit`. All basic, Bergomi, rough, rate-parameter and
correlation scopes populate it explicitly. Labels remain display identifiers.
The common report checks lengths and uses kinds, without parsing labels or
falling back based on unknown text. Python exposes owned `parameter_kinds` and
`parameter_units` snapshots. Cash-mean and model-parameter sensitivities retain
the existing currency-per-unit-parameter reporting convention.

A shared Python adapter maps typed Rust errors to stable codes and attaches
stage, field, operation, capability reason and retained cause messages. Stage
is supplied by the caller, not inferred from a message. Unsupported source
operations retain their typed capability reason through Rust validation.
Existing exception classes/messages and ValidationIssue fields, equality and
dictionaries remain unchanged. Unknown native errors use an explicit generic
code instead of message parsing. The engine remains private; no new crate or
dependency is introduced.

## Compatibility and reproducibility

- No request, task, report, specification or graph schema changes. Existing
  label/order/unit strings, canonical JSON and fingerprint domains are unchanged.
- Python API changes are additive. Exception-level codes refine legacy issue
  codes without changing those issue codes or their exception inheritance.
- Rust adds `RiskParameterKind`, `SensitivityUnit::as_str()`, a compiled-task
  operation accessor and a variant of the already non-exhaustive validation
  error. `StochasticDividendAadRisk` gains a public `parameter_kinds` field:
  callers constructing this output struct with literals must supply that field.
  Native factory/evaluation calls and existing field access remain unchanged.
  `HullWhiteAadRisk` already prevented external struct literals with private state.
- No path evolution, adjoint arithmetic, calibration, random stream, reduction
  order, acceptance threshold or uncertainty convention changes. Metadata is
  prepared once per risk calculation, outside path loops.

## Validation and gates

Regressions deliberately rename Spot to `volatility` and volatility to `spot`:
units and estimates must remain unchanged. Mismatched label/kind/value/error
lengths must fail before indexing. Installed-wheel tests cover metadata order
for BS, 1F/2F/rough Bergomi, HW rate parameters and correlation scopes, snapshots,
and report/native equality. Error tests exercise parse limits, schema failures,
source preflight reasons, native compile rejection and actual payoff overflow
through both specification and task evaluation. Rust tests cover nested
calibration causes and message-independent error codes.

Extend the existing independent wheel/stub and source-archive contracts for
the additive API. Keep all numerical, statistical, refinement and cross-platform
gates required by CONTRIBUTING; this change does not relax them.

User guidance: [risk reports](../../docs/library/pricing-tasks.md) and
[Python errors](../../docs/library/python-errors.md).
