# S7: coupled local-correlation scaling and allocation attribution

Date: 2026-09-22. Follows the [S6 record](model-boundary-extension.md).

The small S6 local-correlation/HW AAD case allocated 977,715 blocks in a
whole-process DHAT run. Its repeated native AAD timing excludes the first
evaluation, which initializes cached market-risk transpose weights. These
different scopes require a first-evaluation measurement before choosing an
optimization. This stage initially changes the benchmark and measurement
harness only; numerical implementation changes need their own measured PR.

## Protocol

The [scaling runner](../../scripts/profile_model_scaling.py) overlays the same
[public-API example](../../crates/pricing/examples/benchmark_model_boundaries.rs)
on a pinned baseline and candidate. Baseline
`ea759bf5d73b06e4c480665e59c7daa37e6667be` contains the S6 implementation and the
separate Gaussian-reference correction; production numerical code matches S6.

| Axis | Base | Variants (other settings fixed) |
| --- | --- | --- |
| Workers | 1 | 2, 4 |
| Time steps | 8 | 16, 32 |
| Calibration particles | 128 | 256, 512 |
| Antithetic MC units | 512 | 2,048, 8,192 |
| Assets | 2 | 3, 4 |

One asset uses 1F LSV and the others BS, with shared HW and joint local
correlation. Basket weights remain equal and total one as assets are added.
The synthetic basket target and endpoint correlations are fixed. This is a
workload study, not evidence for new model combinations or arbitrary markets.

Each of the 11 workloads measures construction, price, first AAD and repeated
AAD, using seven fresh-process paired rounds with alternating version order,
one warmup and three retained samples per process. `aad_cold` constructs a new
plan before each evaluation but excludes construction from its timer. `aad`
reuses one plan and excludes its first evaluation as warmup. Both must return
identical price/risk checksums. Every baseline/candidate pair must also match
exactly; an initial median slowdown above 10% requests seven confirmation pairs.
Timing noise is not a CI pass/fail criterion.

Builds use the same lock/toolchain, release/thin LTO/single codegen unit and
line-level debug information (`CARGO_PROFILE_RELEASE_DEBUG=1`). Native process
peak RSS is recorded separately from DHAT. Build size/time are descriptive and
not directly comparable with the S6 binaries built without debug information.

At the base workload, separate DHAT processes run construction, price, one
AAD and two AAD evaluations on one plan. Larger step/particle/asset variants
profile one AAD. All include input/plan setup and result formatting; none use
DHAT times as native timings. The one-versus-two evaluation allocation totals
help distinguish recurring work from first-evaluation setup, but process/runtime
differences mean their subtraction is not an exact isolated allocation count.

DHAT summaries retain the 12 largest allocation-count and global-peak-live-byte
stacks, profile hashes and whole-process totals. The parser checks the sums
against DHAT's log. Full profiles remain CI artifacts. These are allocation
stacks, not CPU hot stacks; elapsed-time attribution additionally needs the
cold/warm and independent-axis comparisons. See the
[DHAT manual](https://valgrind.org/docs/manual/dh-manual.html) for measurement scope.

## Acceptance and remaining work

The first measurement revision provides a control comparison with identical
production code. Record its source-linked results before selecting a change.
Potential sites include repeatedly allocated reverse buffers and construction
of the cached market transpose. Preserve the exact forward/reverse arithmetic,
RNG coordinates, reduction order, public APIs and all numerical budgets.

This is the first S7 workload family. Broader 2F/rough/mixed-model scaling,
rough convolution, 2F flattening, the isolated S6 1F peak-lifetime difference
and cross-platform performance measurements remain later work. The inherited
2F ensemble-SE acceptance issue remains separately tracked.
