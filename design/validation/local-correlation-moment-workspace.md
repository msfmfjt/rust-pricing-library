# S7: reuse reverse-moment buffers

Date: 2026-09-22. Follows the measured [scaling control](local-correlation-scaling.md).

## Measured problem and change

The base first-AAD profile identifies 154,056 allocations at the sigma,
weighted-exposure and endpoint cross-loading vector sites inside
`reverse_joint_moments`, from calls made while constructing cached market-risk
transpose weights. These three sites account for 15.76% of allocations in the
complete first-AAD process; the same routine is also used in the final target
pullback. The control provides call stacks and raw samples, not a CPU-time
percentage attributed to allocations.

The change allocates three asset-sized buffers once per calibration pullback
and overwrites every element before use. The cross-loading buffer is reused
for the two endpoints. Workspace ownership is local to a pullback, so concurrent
pricing/calibration requests do not share mutable scratch state.

Sigma evaluation still completes in asset order before weighted exposures are
formed. Endpoint order, covariance entries, sums and adjoint accumulations are
unchanged. The primal, random streams, reverse traces, market-transpose cache,
support indicators, error policy, public interfaces and numerical budgets are
unchanged. No covariance/moment is approximated or cached across changing states.

## Validation

Run the same 11-workload/four-operation paired scaling panel against baseline
`ea759bf5d73b06e4c480665e59c7daa37e6667be`. The identical benchmark is overlaid
on both versions; only the candidate changes production reverse storage.
Exact baseline/candidate and cold/repeated AAD checksums are required. Separate
DHAT pairs compare allocations and global peak heap. Record native results and
scopes before claiming a speed or memory improvement.

Existing normal and extended-risk tests cover deterministic/HW local correlation,
1F/2F/rough marginals, finite differences, multiple seeds, dividends, MC/RQMC and
worker replay. The independent Gaussian reference and unchanged full price gate
continue to run. The inherited 2F ensemble-SE issue remains separate.

## Native observations

[CI run 289](https://github.com/msfmfjt/rust-pricing-library/actions/runs/35708535272),
profiling job 106683587115, completed successfully. The
[raw observations](local-correlation-moment-workspace-observations-2026-09-22.json)
retain all paired samples, checksums, compiler/CPU/build metadata and DHAT
allocation/peak leaders with full call stacks.

- Baseline source: `ea759bf5d73b06e4c480665e59c7daa37e6667be`,
  tree `c31f67b850424fcf80cc392f5a9ee0e3947030e2`.
- Candidate source: `5886ac6834676ea6c10cd89aa087a58c81625ae6`,
  tree `b3579bb091235b2fc02363bc65ccdaaf1d40d9a9`.
- Linux x86-64, AMD EPYC 7763, four exposed CPUs, rustc 1.98.1/LLVM 22.1.8.
  Both builds use release, thin LTO, one codegen unit and debug level 1 for
  allocation-site attribution; identical toolchain, Cargo.lock and benchmark.
- All 44 native combinations and seven DHAT pairs have exact baseline/candidate
  output checksums. Initial/repeated AAD checksums also match. Each native
  combination has seven alternating-order pairs, with one excluded warmup and
  three timed operations per process.
- No median exceeded the protocol's 1.10 timing-ratio trigger, so no additional
  confirmation pairs were required.

The base workload is one worker, eight steps, 128 calibration particles,
512 independent pricing units and two assets. Each other row changes only its
named setting. AAD cold timing excludes plan construction but includes creation
of the market-transpose cache. Warm AAD excludes its initial evaluation.

| Workload | Baseline cold AAD, ms | Candidate cold AAD, ms | Paired cold ratio | Paired warm ratio |
| --- | ---: | ---: | ---: | ---: |
| base | 141.979 | 138.144 | 0.9735 | 0.9982 |
| workers-2 | 134.230 | 130.612 | 0.9732 | 0.9860 |
| workers-4 | 132.534 | 128.924 | 0.9691 | 0.9881 |
| steps-16 | 498.228 | 483.319 | 0.9713 | 0.9985 |
| steps-32 | 1871.568 | 1809.188 | 0.9683 | 0.9915 |
| particles-256 | 266.615 | 258.150 | 0.9686 | 0.9891 |
| particles-512 | 516.292 | 500.000 | 0.9703 | 0.9799 |
| units-2048 | 191.382 | 188.492 | 0.9832 | 1.0006 |
| units-8192 | 388.450 | 385.762 | 0.9951 | 1.0092 |
| assets-3 | 166.208 | 162.409 | 0.9779 | 0.9889 |
| assets-4 | 190.213 | 186.687 | 0.9820 | 0.9960 |

Times are medians of process medians. Ratios are medians of matched
candidate/baseline ratios, so they need not equal ratios of the displayed
time medians. Compare versions within this run: the earlier control run's
absolute times are not a valid speedup denominator.

Cold AAD paired medians decrease by 0.49–3.17%, including 2.65% at the base
and 3.17% at 32 steps. The benefit is smaller when pricing paths dominate.
Warm AAD ratios span 0.9799–1.0092; the 8,192-unit workload is 0.92% slower
in this panel, with all seven pairs above one. This is not evidence of a
general warm-AAD speedup, and the small slowdown is retained rather than
rounded away. Compile and price ratios span 0.9896–1.0011 and
0.9885–1.0053, respectively.

The symbol-bearing executable decreases from 33,605,752 to 33,569,904 bytes.
Single clean builds take 75.66 and 73.24 seconds; one build per version does
not establish a compile-time improvement.

## Allocation reduction and peak-memory limitation

DHAT measures the whole process, including inputs, one plan, evaluation(s),
cache construction, runtime and output, with no warmup. The third component
of each row label is the evaluation count; the two-evaluation row is not a
standalone warm-operation allocation profile. Native elapsed time is measured
separately from DHAT.

| Workload/operation/count | Allocation blocks, baseline → candidate | Reduction | Allocated bytes, baseline → candidate | Global peak heap bytes, baseline → candidate |
| --- | ---: | ---: | ---: | ---: |
| base/compile/1 | 13,239 → 13,239 | 0.00% | 1,274,743 → 1,274,744 | 272,810 → 272,811 |
| base/price/1 | 108,547 → 108,547 | 0.00% | 10,074,337 → 10,074,338 | 272,806 → 272,807 |
| base/aad/1 | 977,717 → 812,625 | 16.89% | 42,106,571 → 38,804,396 | 367,811 → 367,860 |
| base/aad/2 | 1,215,586 → 1,039,376 | 14.50% | 58,748,115 → 55,221,516 | 372,619 → 377,765 |
| steps-32/aad/1 | 11,942,434 → 9,527,302 | 20.22% | 411,834,336 → 363,530,497 | 1,255,740 → 1,255,789 |
| particles-512/aad/1 | 3,276,518 → 2,616,886 | 20.13% | 119,764,763 → 106,571,788 | 1,166,915 → 1,166,964 |
| assets-4/aad/1 | 1,205,403 → 1,041,807 | 13.57% | 55,381,475 → 50,146,404 | 471,523 → 471,620 |

Base first-AAD allocations fall by 165,092 (16.89%) and allocated bytes by
7.84%. The reductions are 20.22%/11.73% at 32 steps, 20.13%/11.02% at 512
particles, and 13.57%/9.45% at four assets. Savings include the final target
pullback as well as the market-transpose construction used to select this
change. Compile and price allocation counts are unchanged.

This is an allocation-churn improvement, not a peak-memory reduction.
The three reused buffers remain live throughout one calibration pullback:
their payload is `3 * assets * size_of::<f64>()`, or 48 bytes at two assets
and 96 bytes at four. The first-AAD global peaks rise by 49/97 bytes with
three additional blocks, consistent with that lifetime and the one-byte
whole-process difference already present in the control.

The two-evaluation profile rises by 5,146 bytes (1.38%) and 21 blocks.
Its twelve largest global-peak entries have identical byte/block totals
between versions, including calibration histories, adjoint arrays,
market-transpose weights and the execution statistics. The extra 5,097 bytes
beyond the 49-byte first-AAD difference are outside those retained leaders.
This single profile does not establish whether the remainder is reproducible
runtime overlap or a persistent increase. Repeat this exact pair and inspect
the complete smaller allocation sites before declaring peak-memory neutrality;
keep that acceptance question open on the draft.
The [focused follow-up](local-correlation-heap-followup.md) records the complete
peak-stack comparison and repeated-pair protocol.

Native process-peak RSS median differences range from −160 to +132 KiB,
with both signs across workloads. They do not show a consistent RSS reduction.
No numerical or performance threshold was widened to remove these observations.

## Regression gates and remaining work

On the measured production source, CI run 289 passed formatting, Clippy,
default/minimal/all-feature tests, statistical acceptance, documentation and
the extended finite-difference/model-risk panel on Linux, Windows and macOS.
Both dedicated native wheel/Python/replay jobs and all three focused Gaussian
escrow-reference jobs passed. Fixture/schema/source-archive checks passed.

At evidence capture, the full extended price jobs were still running. This
record does not claim those jobs passed; the inherited 2F constituent timestep
ensemble-SE result (4.012373 bp against a 4 bp budget) remains separately
deferred. Keep the PR draft pending that broader acceptance and the
two-evaluation peak-heap follow-up.

The final evidence-only update also carries the corrected exact 64-bit pricing
seed exports from the parent PRs. Those JSON exports do not change the measured
production code or any price/risk value. The raw performance report contains
no integer beyond the exact JSON/JavaScript integer range.

The wider S7 model matrix, persistent memory attribution and further
optimizations remain separate work. The measured cold-AAD benefit and allocation
reduction justify reviewing this narrowly scoped candidate, not closing S7.
