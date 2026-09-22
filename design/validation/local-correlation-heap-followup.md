# S7: repeated-AAD peak-heap follow-up

Date: 2026-09-22. Follows the [reverse-buffer measurement](local-correlation-moment-workspace.md)
on PR #81.

## Question

The base two-AAD whole-process DHAT pair in CI run 289 reported a 5,146-byte
peak increase (372,619 to 377,765 bytes), while the one-AAD pair increased by
49 bytes. The twelve largest peak entries were unchanged in the two-AAD pair.
The complete lower entries and fresh repetitions are needed to attribute the
remaining difference. Do not infer peak-memory neutrality from fewer allocations.

## Measurement protocol

The [focused script](../../scripts/profile_model_heap_followup.py) reads the
original complete baseline/candidate profiles from run 35708535272. It checks
their SHA-256 hashes and totals against the captured scaling report and each
DHAT log, including end-of-process heap. Every live-at-global-peak point is
retained; there is no top-N cutoff.

Display frames omit load addresses and executable directories but retain
symbols and source locations. For comparison, call stacks are grouped after
removing source-line offsets and compiler-generated implementation numbers.
These are grouped stacks, not unique allocation-site identities. Original
point indices and source frames remain available to inspect any grouping.
Both bytes and block deltas must sum exactly to the process-peak difference.

The same pinned baseline and unchanged public-API benchmark are built in clean
release directories with thin LTO, one codegen unit and debug level 1. Five
fresh-process pairs alternate baseline/candidate execution order. Each process
uses one worker, eight steps, 128 particles, 512 independent pricing units and
two assets, constructs one plan, then runs AAD twice without warmup. Output
checksums must match the original pair exactly. This is a DHAT heap experiment,
with no native elapsed-time inference.

The [separate workflow](../../.github/workflows/model-heap-followup.yml) leaves
the parent's full price run active. It has read-only repository/actions access
to recover the original artifact. Once the resulting record is committed, its
complete original peak points make that analysis independent of artifact expiry.
Fresh repetitions continue to retain complete profiles and logs as CI artifacts.

The accounting tests exercise a thirteenth peak point, corrupted end-heap
totals, regrouped call stacks and signed byte/block conservation. No production
Rust, numerical threshold, seed, benchmark request, schema or replay fixture
changes are part of this follow-up.

## Status

Native repetitions and full original-stack attribution are pending. Keep the
peak-heap acceptance question open until the results are recorded. The broader
price gate and the inherited 2F ensemble-SE issue remain separate.
