# Python error metadata

Library-generated `PricingError` remains a `RuntimeError`, and
`ValidationError` remains a `ValueError`. Existing messages, `args` and
`ValidationError.issues` are preserved. Both exceptions now carry:

| Attribute | Meaning |
| --- | --- |
| `code` | Stable machine-readable failure code; never inferred from the message. |
| `stage` | `parse`, `validation`, `compile`, `calibrate`, `evaluate` or `serialize`. |
| `field` | Native field identifier or JSON pointer, when supplied by the underlying error; otherwise `None`. |
| `operation` | Selected common-task operation where known; otherwise `None`. |
| `reason` | Capability reason for an unsupported task operation; otherwise `None`. |
| `causes` | Ordered messages from the Rust error source chain, outermost first. An empty list means no nested source was retained. |

`stage` identifies the boundary where the failure occurred. Particle calibration
performed inside `compile()` reports `stage == "compile"`; an unsupported
calibration node has `code == "calibration_error"`. An explicit calibration
entry point uses `calibrate`. Ordinary argument conversion errors raised by
Python/PyO3 retain their built-in exception contracts. Do not assume these
attributes exist on manually constructed exceptions or arbitrary `ValueError`s.

For domain failures in `PricingTask.from_json`, JSON pointers are relative to
the complete task, including source containers and array indices. For example,
an invalid second-asset Spot points to `/source/markets/1/spot`; an embedded
specification's execution-policy error points to
`/source/specification/numerics`. A composition error can identify the enclosing
source object when several inputs are involved. Standalone specification errors
remain relative to the standalone specification document.

For `ValuationReport.from_json`, domain-error pointers are relative to the
report document itself. Invalid scalar/replay metadata identifies its field,
such as `/standard_error` or `/plan_fingerprint`. Invalid or duplicate risk
coordinates identify the affected block, such as `/risks/1`, because shape and
identity checks can involve several fields. There is no `/report` container in
either report v1 or v2. The exception's `field` and issue's `pointer` agree.

Report issues retain the input's declared `schema_version` after bounded JSON
validation and envelope decoding succeed. An invalid v1 report therefore reports
version 1, even though successful reads are normalized to the current v2 writer.
Unsupported versions also retain the supplied version. If syntax, duplicate
members, resource limits or an invalid envelope prevent identifying the version,
the integer issue field keeps its existing current-report fallback of 2; this
fallback is not evidence that the rejected input declared v2.

```python
import rust_pricing as rp

def value(task):
    try:
        return task.compile().evaluate().report
    except rp.PricingError as error:
        # Record structured fields in the batch result; preserve the failure.
        print(error.code, error.stage, error.field, error.operation, error.reason)
        raise
```

Selected codes supported by the shared adapters:

| Code | Meaning |
| --- | --- |
| `unsupported_operation` | Source preflight rejected the requested operation; inspect `reason` (for example `reverse_trace_required` or `invalid_gamma_bump`). |
| `unsupported_risk_for_model` | Native risk preflight rejected the model/payoff combination. |
| `invalid_configuration`, `invalid_execution_policy` | Invalid source or execution inputs; inspect `field`. |
| `invalid_number` | A scalar violates its native finite/positive/nonnegative constraint; `field` retains the Rust identifier. |
| `invalid_core_input` | Another typed core input error; preserve the original message. |
| `invalid_multi_asset_input`, `invalid_correlation` | Invalid basket input or correlation structure. |
| `resource_limit` | JSON, grid, reduction or RQMC capacity limit; not a retry instruction. |
| `calibration_error` | A calibration/optimizer failure. |
| `non_finite_state`, `non_finite_total_variance`, `graph_error` | A numerical path, variance or payoff-graph failure. |
| `invalid_json`, `unsupported_schema_version`, `invalid_domain_value` | Parsing/schema/domain validation failures. |
| `pricing_error` | A native failure without a more specific mapping. Preserve the message and causes. |
| `multi_asset_numerical_error` | A basket numerical diagnostic with no recognized typed cause. Preserve the message and causes. |

Codes are extensible; handle unknown codes by retaining the failed calculation.
Messages and cause strings are diagnostics, not stable dispatch keys. Historical
`ValidationIssue.code` values remain unchanged: for example, an existing
`plan_compile_error` issue can accompany the more specific exception-level
`invalid_execution_policy` code. Issue equality, dictionaries and schema metadata
keep their existing contracts.

An execution request whose reduction-block table cannot be reserved fails during
evaluation before any sampling callback runs. Single-asset, multi-asset and
stochastic-dividend task adapters retain `code == "resource_limit"` and
`field == "reduction_blocks"`. This
allocation check does not impose a new sampling-count or block-size cap, nor
guarantee that all subsequent allocations will fit in memory.

Basket numerical wrappers preserve their native sources: RQMC dimension limits
report `resource_limit` / `effective_dimension`, failed LSV calibration nodes
report `calibration_error`, and runtime payoff failures report `graph_error` /
`product`. Evaluation causes retain the failed sampling-unit context. Legacy
text-only diagnostics and unrecognized numerical causes retain the generic
`multi_asset_numerical_error` code; messages are never parsed to choose a code.

`MultiAssetPlan.compile()` keeps its historical `ValidationError` class and
`issues` payload. Failures returned by the native compiler now carry
`stage == "compile"` and the same cause classification as `PricingTask.compile()`.
Argument validation before the native compiler still reports `validation`.
The common-task API continues to raise `PricingError` for native compile failures.

See [complete pricing tasks](pricing-tasks.md) for capability queries before
compilation and [the type stub](../../rust_pricing.pyi) for attribute types.
