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
| `invalid_multi_asset_input`, `invalid_correlation` | Invalid basket input or correlation structure. |
| `resource_limit` | JSON, grid, reduction or RQMC capacity limit; not a retry instruction. |
| `calibration_error` | A calibration/optimizer failure. |
| `non_finite_state`, `non_finite_total_variance`, `graph_error` | A numerical path, variance or payoff-graph failure. |
| `invalid_json`, `unsupported_schema_version`, `invalid_domain_value` | Parsing/schema/domain validation failures. |
| `pricing_error` | A native failure without a more specific mapping. Preserve the message and causes. |

Codes are extensible; handle unknown codes by retaining the failed calculation.
Messages and cause strings are diagnostics, not stable dispatch keys. Historical
`ValidationIssue.code` values remain unchanged: for example, an existing
`plan_compile_error` issue can accompany the more specific exception-level
`invalid_execution_policy` code. Issue equality, dictionaries and schema metadata
keep their existing contracts.

See [complete pricing tasks](pricing-tasks.md) for capability queries before
compilation and [the type stub](../../rust_pricing.pyi) for attribute types.
