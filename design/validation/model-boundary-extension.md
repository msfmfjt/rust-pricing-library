# Model-boundary extension and comparison evidence

S6 of [the static-model roadmap](../roadmaps/static-model-boundaries.md), under
[ADR 0011](../adr/0011-static-model-boundaries.md). Numerical production kernels,
public APIs, schemas and frozen fixtures remain unchanged in this stage.

## Extension exercise

The test-only [LogCoordinate registration](../../crates/pricing/src/models/bergomi_dynamics/extension_probe.rs)
stores `Y = nu X`, with its own state struct, transition struct, typed innovations
and pullback. It reuses the existing validated OU covariance calculation but
implements evolution and reverse in the new coordinate:

\[
dY_t=-kY_t\,dt+\nu\,dW_t,\qquad a_t=\exp(Y_t).
\]

Only a `cfg(test)` module registration and the alternative kernel/adapters are
added. The existing generic particle calibration, payoff execution, MC/RQMC
pricing and recalibrated reverse algorithms require no edits. Tests compare
calibration, prices and every target-grid adjoint with the existing 1F model,
then independently finite-difference a recalibrated target node and both kinds
of innovation pullback. They exercise one/two workers and MC/RQMC. The coordinate
change reassociates arithmetic, so comparisons between these two different test
implementations use tight tolerances; production baseline/candidate replay in
the performance panel requires exact equality.

This demonstrates an internal extension using associated state/transition types,
not a new production model or a general external plug-in API. The sealed public
Bergomi compatibility methods still have two-slot legacy arrays. Additional
factor counts, non-Gaussian rates, and new HW/rough covariance compositions need
their own registrations and numerical capabilities; this exercise does not
certify those combinations.

## Paired release protocol

The [runner](../../scripts/compare_model_boundaries.py) builds the identical
[public-API example](../../crates/pricing/examples/benchmark_model_boundaries.rs)
against baseline PR #72, commit
`738c020180b343a229ee20aa5acc236dbcd11d88`, and the candidate. The baseline already
contains the escrowed default and no paid-cash model. It precedes S1–S5.
The runner checks equal toolchain files and Cargo.lock digests and overlays only
the benchmark source onto the baseline; production baseline code is untouched.

Both versions run on the same Linux machine with release/thin-LTO/single-codegen
unit builds. Dependency downloads precede timed clean builds. Build time and
executable size are recorded, but one clean build per version is descriptive,
not evidence of a compile-time regression by itself. The report records source
commit/tree, executable/example/lock digests, compiler, CPU, flags and profiler.

| Case | Measured operations |
| --- | --- |
| Deterministic-rate 1F, 2F, rough LSV | Plan construction, price, recalibrated target-variance AAD |
| HW with 1F, 2F, rough LSV | Plan construction, price, paired variance/density AAD, AAD with market-IV VegaKT |
| Three-asset shared HW, mixed 1F/2F/rough | Plan construction, basket price, recalibrated multi-asset AAD |
| Two-asset local correlation with 1F-LSV/HW and BS | Joint plan construction, price, joint recalibrated AAD |

There are 27 case/operation combinations at each of two sizes: one worker,
8 steps, 128 particles and 512 antithetic MC units; and two workers, 16 steps,
256 particles and 1,024 units. Seeds, reduction blocks, affine cash/proportional
dividends and quote arrays are identical within each pair. Single-asset HW cases
also include a future-only dividend. The larger size changes several settings
together; it is not an isolated scaling attribution. An independent parameter
sweep and broader platform/workload coverage remain S7.

Each of seven paired rounds starts fresh processes, alternates baseline/candidate
order, performs one warmup and retains three operation timings. Result formatting,
hashing and destruction are outside the timer. Preparation/calibration is timed
as plan construction, while repeated valuation reuses one plan. Single-asset
target preprocessing is outside that construction timer; multi-asset construction
includes its public configuration assembly. Cases must be compared with their
own baseline, not interpreted as identical work across models. HW VegaKT timing
includes the required path/calibration AAD, not an isolated quote pullback.

The report retains every sample, paired ratio, median/MAD/range and native
process peak RSS. An initial median slowdown above 10% triggers another seven
pairs with reversed starting order for investigation; it is not a speed-based
CI rejection threshold. Public pricing/risk debug representations and compiled
fingerprints are hashed outside timing. Unequal checksums or non-reproducible
checksums fail the job, without a Monte Carlo tolerance or fixture update.

## Memory scope and limitations

GNU time records peak RSS for each complete native child process, including
input construction, plan preparation and warmup. It is neither the operation's
incremental RSS nor Rust heap alone. The separate
[Valgrind DHAT](https://valgrind.org/docs/manual/dh-manual.html) run profiles each
small case with one operation and no warmup, recording total allocated bytes and
blocks, and simultaneously live bytes/blocks at the global heap maximum.
These counts include setup, result formatting and allocator/runtime activity;
they must not be labelled per-path allocation counts. DHAT's reallocation
accounting follows its documented semantics. Stack memory is outside its heap
totals. Native and instrumented elapsed times are never compared.

The baseline/candidate DHAT pair also checks exact output equality, independently
of the native pair. Instrumentation can change CPU-feature selection, so the
two measurement modes are not asserted to have identical checksums. RSS noise,
hosted-runner scheduling, short operations and shared libraries limit timing
and memory attribution. Repeated slowdowns need inspection before performance
neutrality is claimed; no optimization or numerical-policy change is included.

## Execution and evidence

On a Linux host with the pinned Rust toolchain, GNU time and Valgrind:

```bash
python scripts/compare_model_boundaries.py /path/to/baseline /path/to/candidate /path/to/new-results
```

Both checkouts must be disposable workspaces; the baseline receives the identical
example and output must be a new directory. The CI job runs on the S6 branch and
manual workflow dispatch, separately from long statistical/accuracy gates.
It retains `comparison.json` plus DHAT profiles/logs for 30 days and emits the
complete comparison JSON into the job log. The PR records the measured source,
paired summaries and remaining caveats; CI artifacts alone are not a durable
acceptance record. A measurement result does not waive the inherited extended
price/refinement issue or its unchanged 4 bp threshold.

Initial harness validation found that the pre-S1 deterministic LV/LSV compiler
inserts the entire dividend schedule into its execution grid: a dividend beyond
expiry then queries outside the target grid. The unchanged candidate does the
same. Future-only dividends are therefore exercised in the HW cases here; this
existing deterministic-grid limitation is separate follow-up work, not a
performance regression or a numerical fix in S6.
