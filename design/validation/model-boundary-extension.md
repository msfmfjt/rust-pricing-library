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

## Recorded results: 2026-09-21

[CI run 282](https://github.com/msfmfjt/rust-pricing-library/actions/runs/35612297777)
measured candidate `35f076beb99379010fd06cd38804f1395162fb29`, tree
`a2d34ed842ff2e4456cc116cdab8ab9674db31e1`, against the pre-extraction baseline
above. The complete [measurements](model-boundary-measurements-2026-09-21.json)
retain every paired sample and the source/environment metadata. Subsequent
evidence-only commits do not change this measured source.

All 54 native case/operation/size combinations and all 27 separate DHAT pairs
matched their baseline output/fingerprint checksums exactly. The two extension
tests passed on Linux, macOS and Windows. Normal/default/minimal/all-feature,
statistical, documentation and extended-risk checks passed on all three;
both macOS/Windows native wheel, Python and frozen-replay jobs passed.

The host was Linux x86-64, AMD EPYC 7763 with four visible logical CPUs, Rust
1.98.1 / LLVM 22.1.8, and Valgrind 3.22.0. The following entries give the
paired median candidate/baseline time ratios at the smaller / larger size.
They compare each operation only with its own baseline; they are not scaling
ratios between the two sizes.

| Case | Construction | Price | AAD | AAD + VegaKT |
| --- | --- | --- | --- | --- |
| 1F | 0.9751 / 0.9982 | 1.0021 / 1.0073 | 1.0000 / 0.9977 | — |
| 2F | 1.0125 / 1.0063 | 0.9937 / 0.9893 | 1.0077 / 0.9891 | — |
| Rough | 0.9884 / 0.9986 | 0.9898 / 0.9961 | 0.9996 / 1.0041 | — |
| HW + 1F | 0.9963 / 1.0006 | 0.9892 / 1.0361 | 0.9918 / 1.0009 | 0.9937 / 1.0036 |
| HW + 2F | 1.0169 / 1.0029 | 1.0069 / 0.9934 | 1.0016 / 1.0040 | 0.9940 / 1.0025 |
| HW + rough | 1.0100 / 0.9982 | 0.9972 / 1.0023 | 0.9960 / 0.9992 | 0.9991 / 1.0026 |
| Mixed shared HW | 0.9959 / 0.9987 | 0.9834 / 0.9938 | 0.9979 / 0.9941 | — |
| Local correlation + HW | 0.9974 / 0.9925 | 0.9923 / 0.9952 | 1.0056 / 1.0170 | — |

No paired median exceeded the 1.10 investigation trigger. These small workloads
show no large timing regression; they do not establish performance neutrality
at larger particle/path populations or on other CPUs. The executable grew from
6,401,440 to 6,559,776 bytes (+2.47%). Clean build times were 69.21 and 68.19
seconds, one observation per version, without a compile-time speed claim.

Median native RSS differences ranged from -60 to +220 KiB (ratios 0.9885–1.0437).
Whole-process DHAT allocation counts differed by at most one block in each pair.
HW/mixed/local-correlation peak heap decreased in this panel. The small 1F
price case instead increased from 48,614 to 50,071 live bytes (+1,457 bytes,
3.0%) and from 148 to 163 live blocks at the peak, while total allocation was
13,231 blocks in both versions (1,097,576 versus 1,097,577 bytes). A single
instrumented run cannot establish whether that lifetime difference is persistent;
its stack/lifetime attribution remains open for S7. It is not evidence of
increased allocation churn. Native RSS and DHAT heap are different quantities.

S7 should first profile coupled local-correlation AAD: the larger case spent
55.42 ms per evaluation, and the smaller complete DHAT process allocated 977,715
blocks / 42,106,518 bytes. Those totals include calibration and formatting, so
attributing them to a path loop requires call-stack evidence and independent
particle/step/path/worker sweeps before an optimization is selected.

### Separate accuracy follow-ups

The extended price gate completed with 25 passed and two failed tests on each
platform. The [captured failed observations](model-boundary-accuracy-followups-2026-09-21.json)
preserve the exact cases, budgets, prices and seeds. Both failures also occur in
the parent Linux [run 278](https://github.com/msfmfjt/rust-pricing-library/actions/runs/35605479836):

- The deferred one-year 2F constituent timestep ensemble SE is 4.012373 bp
  against its unchanged 4 bp budget.
- The independent BS-HW reference with an expiry cash/proportional dividend,
  zero mean reversion and correlation -0.4 reports up to 38.9220 bp IV error
  against its unchanged 2 bp budget. The request uses the funded terminal
  forward, whereas this test's reporting call still supplies the pre-cash
  forward. The separate [reference correction](hw-escrowed-gaussian-reference.md)
  tracks its fix and native rerun without altering the captured S6 observations.

The full accuracy gate and PR stack are therefore not accepted by this S6
measurement record. No threshold, production formula or frozen replay changes
are made here.
