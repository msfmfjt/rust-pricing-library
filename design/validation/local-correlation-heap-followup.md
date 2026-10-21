# S7: historical repeated-AAD peak-heap replay

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

The original baseline `ea759bf5d73b06e4c480665e59c7daa37e6667be` and candidate
`5886ac6834676ea6c10cd89aa087a58c81625ae6` are built in clean release directories
with the unchanged public-API benchmark, thin LTO, one codegen unit and debug
level 1. Both source commits are checked, and their `Cargo.lock` and Rust
toolchain files must match. Five fresh-process pairs alternate baseline/candidate
execution order. Each process
uses one worker, eight steps, 128 particles, 512 independent pricing units and
two assets, constructs one plan, then runs AAD twice without warmup. Output
checksums must match the original pair exactly. This is a DHAT heap experiment,
with no native elapsed-time inference.

The [separate workflow](../../.github/workflows/model-heap-followup.yml) runs
the accounting tests and measurement script from a current-head tooling checkout,
alongside separate checkouts of the two original source commits. It records the
tooling commit, tree and script SHA-256 in the report. This is a historical S7
replay; it does not measure current-head production performance. The parent's
full price run remains separate.

The workflow reads the [retained record](local-correlation-heap-followup-2026-10-21.json)
from the current tooling checkout. It contains all original peak points and
their frames, so the original comparison no longer depends on artifact expiry.
The read-only artifact download remains a fallback when that record is absent.
Fresh repetitions retain complete profiles and logs as CI artifacts.

The accounting tests exercise a thirteenth peak point, corrupted end-heap
totals, regrouped call stacks, signed byte/block conservation, rejected source
commits/build-input mismatches, current-tooling retained evidence and the three
checkout roles. No production Rust, numerical threshold, seed, benchmark request,
schema or replay fixture
changes are part of this follow-up.

The 2026-10-21 maintenance change fixes the candidate to the original S7 commit
so that later dependency updates cannot change the historical comparison. The
equal-lockfile, equal-toolchain and exact output-checksum requirements remain in
force; this replay supplies no acceptance result for later production changes.

## Recorded result: 2026-10-21

[CI run 39522704971](https://github.com/msfmfjt/rust-pricing-library/actions/runs/39522704971/job/118970083964)
completed all five fresh-process pairs with exact original output checksums.
The retained JSON is the complete structured report from that job: all peak
points, shared frames, original profile hashes, paired observations, build
metadata and tooling provenance. It records tooling commit
`8fd547ead0726245af8018720c2dbf477a4c6414`, Rust 1.98.1 and Valgrind 3.22.0
on Linux; the measured source pair remains the two historical commits above.

| Observation | Baseline peak bytes | Candidate peak bytes | Delta | Extra peak blocks |
|---|---:|---:|---:|---:|
| Original pair, run 35708535272 | 372,619 | 377,765 | +5,146 | +21 |
| Each of the five fresh pairs | 372,619 | 372,668 | +49 | +3 |

The complete original grouped-stack differences account for every byte:

| Allocation stacks | Original byte delta | Fresh byte delta |
|---|---:|---:|
| Three moment-workspace vectors in `joint_reverse.rs` | +48 | +48 |
| Process argument storage, `std::env::args_os` | +1 | +1 |
| Rayon/Crossbeam and thread-runtime stacks, net | +5,097 | 0 |
| Total | +5,146 | +49 |

The runtime subtotal includes +7,169 bytes in thread-pool, worker, queue and
thread-local stacks and a -2,072-byte Crossbeam epoch-cleanup entry. These are
live allocations at the global process peak, not changes to the numerical
history buffers. Their different overlap at the original peak is consistent
with runtime lifecycle timing; the profiles do not establish an exact causal
schedule. In the fresh runs the runtime peak groups match, leaving the same
48-byte workspace and one-byte argument-storage difference in every pair.

This resolves the attribution of the recorded historical peak difference.
The fresh candidate still has a 49-byte whole-process increase, so the result
does not establish strict peak-memory neutrality. It also supplies no
acceptance result for current `main`, broader model workloads or native
execution speed. The broader price gate and inherited 2F ensemble-SE issue
remain separate.
