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
