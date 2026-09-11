# Wire-schema compatibility

Status: Frozen for library `0.x`

| Library major | Current schema | Accepted request versions | Accepted result versions | Writer output |
|---|---:|---:|---:|---:|
| `0` | `3` | `1, 2, 3` | `1, 2, 3` | `3` |

Schema v2 adds the optional `risk.payoff_smoothing_width_ladder` field. Schema
v3 adds the `american_vanilla` product and its required top-level `lsm`
configuration. The configuration retains the independent training engine,
ordered state variables, polynomial basis, ITM tolerance, CPQR tolerances, and
regression matrix resource limit needed to reproduce the LSM configuration
fingerprint. The field is required for American products and forbidden for all
other products.

The forward-only v1-to-v2-to-v3 migration preserves all v1 financial and
execution meaning, leaving newer optional fields absent. The v2-to-v3 step adds
no financial defaults. Versions 1 and 2 remain readable, while zero, missing,
and versions newer than 3 are rejected before domain construction. Historical
documents containing fields from a later schema are rejected rather than
interpreted as extensions.

Successful reads retain migration provenance separately from normalized
financial identity. Compiled plans and results expose the original and current
schema versions, ordered stable migration identifiers, and BLAKE3-256
fingerprints before and after request migration. Version 3 result replay data
serializes the same immutable provenance. Migration identifiers form an ordered
adjacent chain, using `pricing_request/v2-to-v3` or
`pricing_result/v2-to-v3` for the new step. Historical result migration
preserves its only available request fingerprint where the result document does
not contain the request body needed to recompute a current request fingerprint.

The committed files under `schemas/v1/`, `schemas/v2/`, and `schemas/v3/` are
the Draft 2020-12 interoperability contracts. Their corresponding fixture
directories freeze deterministic compact output, including field order,
tagged-enum representation, numeric spelling, UTF-8/LF policy, and the final
newline. Pretty JSON is an inspection view and normalizes back to the same
typed-data fingerprint.

Optional result fields use omission only. In particular, a VegaKT result with
`covariance_layout.type = "full_bucket_matrix_row_major"` must include
`full_bucket_covariance`, while `covariance_layout.type =
"price_and_bucket_variance_only"` must omit `full_bucket_covariance`. A present
`null`, a missing full matrix under the full-matrix layout, or an unexpected
matrix under the compact layout is invalid schema/domain input.

The canonical fingerprint is separate from JSON. It hashes a domain-separated,
versioned, self-delimiting typed-data stream with BLAKE3-256 and renders as
`blake3-256:` followed by 64 lowercase hexadecimal digits.
