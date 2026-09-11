# Wire-schema compatibility

Status: Frozen for library `0.x`

| Library major | Current schema | Accepted request versions | Accepted result versions | Writer output |
|---|---:|---:|---:|---:|
| `0` | `2` | `1, 2` | `1, 2` | `2` |

Schema v2 adds the optional `risk.payoff_smoothing_width_ladder` field. The
forward-only v1-to-v2 migration preserves all v1 financial and execution
meaning and leaves that new field absent. Version 1 remains readable, while
zero, missing, and versions newer than 2 are rejected before domain
construction. A v1 document that includes the v2 field is rejected rather than
interpreted as an extension.

Successful reads retain migration provenance separately from normalized
financial identity. Compiled plans and results expose the original and current
schema versions, ordered stable migration identifiers, and BLAKE3-256
fingerprints before and after request migration. Version 2 result replay data
serializes the same immutable provenance; v1 result migration is labelled
`pricing_result/v1-to-v2` and preserves its only available request fingerprint
at both endpoints because the historical result document does not contain the
request body needed to recompute a v2 request fingerprint.

The committed files under `schemas/v1/` and `schemas/v2/` are the Draft 2020-12
interoperability contracts. The corresponding `fixtures/v1/` and
`fixtures/v2/` files freeze deterministic compact output, including field
order, tagged-enum representation, numeric spelling, UTF-8/LF policy, and the
final newline. Pretty JSON is an inspection view and normalizes back to the
same typed-data fingerprint.

Optional result fields use omission only. In particular, a VegaKT result with
`covariance_layout.type = "full_bucket_matrix_row_major"` must include
`full_bucket_covariance`, while `covariance_layout.type =
"price_and_bucket_variance_only"` must omit `full_bucket_covariance`. A present
`null`, a missing full matrix under the full-matrix layout, or an unexpected
matrix under the compact layout is invalid schema/domain input.

The canonical fingerprint is separate from JSON. It hashes a domain-separated,
versioned, self-delimiting typed-data stream with BLAKE3-256 and renders as
`blake3-256:` followed by 64 lowercase hexadecimal digits.
