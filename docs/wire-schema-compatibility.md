# Wire-schema compatibility

Status: Frozen for library `0.x`

| Library major | Current schema | Accepted request versions | Accepted result versions | Writer output |
|---|---:|---:|---:|---:|
| `0` | `1` | `1` | `1` | `1` |

Schema v1 is the first public wire contract, so its forward-only migration
registry is intentionally empty. The registry is nevertheless exercised by
every read: version 1 is accepted and zero, missing, or future versions are
rejected before domain construction. When schema v2 is introduced, v1-to-v2
migration must preserve the complete financial and execution meaning.

The committed files under `schemas/v1/` are the Draft 2020-12 interoperability
contract. The committed files under `fixtures/v1/` freeze the deterministic
compact writer, including field order, tagged-enum representation, numeric
spelling, UTF-8/LF policy, and the final newline. Pretty JSON is an inspection
view and normalizes back to the same typed-data fingerprint.

The canonical fingerprint is separate from JSON. It hashes a domain-separated,
versioned, self-delimiting typed-data stream with BLAKE3-256 and renders as
`blake3-256:` followed by 64 lowercase hexadecimal digits.
