# macOS native crash triage — 2026-10-29

## Refreshed evidence

This review starts from PR #190 at
`30f98bc0823e2955e5822016289af8fe76cbbd21`. Its description's queued-check
snapshot is historical. On 2026-10-29, all 111 wheel selections in
[the current-head run](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40604344572)
were terminal: **103 success, two failure, six cancelled**. The
[snapshot](native-crash-triage-2026-10-29.json) retains each selection's check ID,
URL and conclusion, cancellation annotations and both failing-thread excerpts.
These results validate the parent head, not the diagnostic changes below.

The six cancelled selections all report that no hosted runner acquired the job,
with macOS arm64 capacity notices. They are incomplete numerical evidence;
they did not hit the configured execution timeout. Changing provider partitioning
or numerical workloads is not justified by these cancellations.

Both failed processes report an invalid free and stop with `EXC_BREAKPOINT`:

- [custom-basis-row_hats-4-call-fixed_lookback](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40604344572/job/122680597669)
- [extension-all_rows-4](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40604344572/job/122680602354)

Their failing stacks share `_dispatch_cache_cleanup` in `libdispatch`, followed
by pthread thread-specific-data cleanup and workqueue-thread exit. The reported
invalid pointers differ. Active pricing-worker stacks are distinct from the
stopped workqueue thread. This identifies the detection site, not the allocator,
library or operation that first corrupted memory. The native fault remains
unresolved; these failures are consistent with the earlier
[retained observations](provider-ledger-case-sharding-2026-10-27.md).

## Diagnostic change

Previously the JSON process record retained thread identities and stop reasons,
while function frames and image identities existed only in a long text log.
The runner now records frame metadata through LLDB's
[SBFrame API](https://lldb.llvm.org/python_api/lldb.SBFrame.html): ordered frame
indices, validity, function/symbol names, module path/UUID, load and file
addresses, and available source file/line/column. File addresses and module UUIDs
allow comparison without relying only on ASLR-dependent load addresses.

Each thread retains at most 128 frames and reports its observed frame count and
whether truncation occurred. Missing metadata stays null; invalid frames remain
explicit entries. An unwind exception is retained as `frames_error` without
discarding the stop reason or suppressing the original text diagnostics.
The first terminal observation, literal argv, no-resume behavior and failure
exit rules are preserved. Fields are additive to diagnostic schema version 1;
no pricing, wire or report schema changes.

No target expressions are evaluated. Numerical code, Python bindings, dependency
versions, test selection, seeds, sample counts, workers and tolerances are unchanged.
This is an investigation aid, not a native-crash repair or a waiver of a gate.

## Local validation

- Before modification, all 14 debugger tests passed on physical macOS outside
  the sandbox. The sandboxed run could not launch the debuggee and was not a pass.
- The added structured-stack regression failed on the unchanged implementation.
- All 15 debugger tests pass after modification, including real SIGABRT/SIGTRAP,
  literal arguments, virtual environments and startup exec handoffs. Crash tests
  require nonempty JSON frames with symbols and module UUIDs and verify the child
  never reaches its post-signal marker.
- Mocked cases cover absent source/module information, symbol fallback, invalid
  frames/addresses, explicit truncation, unwind failure, and preservation of the
  first terminal observation.
- All 44 related smoke-runner, provider-shard, CI failure-propagation and wheel
  validation checker tests pass. The initial local run lacked the `python`
  command expected by CI-shell fixtures; rerunning in a virtual environment
  resolved that environment prerequisite without changing the tests.
- Markdown links, diff whitespace and the candidate source-archive check pass.

The macOS integration host is macOS 27.0.1, Apple LLDB 2103.0.34.103 and Python
3.14.2. It is not the hosted macOS 15 / CPython 3.12 wheel environment from the
failed jobs. No failed numerical provider workload was rerun locally, and no
candidate wheel, extended numerical panel or supported-platform gate is claimed
by that debugger-test checkpoint.

## Installed parent-wheel follow-up

The subsequent local investigation uses the original macOS CPython 3.12 wheel
from CI run `40604344572`, artifact `12797943668`, produced from `30f98bc`.
Its SHA-256 is
`fe349cbac2f245cedef6fe5883ed80f211bbc2da4d6c18313b80becc964accf3`.
The host is macOS 27.0.1 arm64 and MacPorts CPython 3.12.11; NumPy 2.5.4,
SciPy 1.18.1 and jsonschema 4.26.0 match the failed CI installations. The failed
hosted jobs used macOS 15.7.9 and CPython 3.12.10, so this is not an equivalent
operating-system/interpreter environment.

- The installed parent wheel passed all 341 callable contracts and the existing
  package checks. Discovery verified the 112-module / 616-test multiset and
  36-shard / 179-case multiset. Discovery is not numerical execution.
- Under LLDB, 54 tests passed across `test_graph_smoothing_range`,
  `test_graph_division`, `test_product_graph`, `test_pricing_task`,
  `test_pricing_task_reuse`, `test_typed_pricing_sources`,
  `test_calibration_report`, `test_structured_errors`, `test_legacy_wire_context`
  and `test_valuation_report_validation`. Both the debuggee and LLDB exited zero.
- A second LLDB process passed all 34 tests in `test_backtest_native_contract`
  and `test_vega_kt_backtest`, including actual task/report replay, cache eviction,
  delayed-payment scaling and checkpoint resume. It and LLDB exited zero.
  The 30 separate synthetic tests in `test_provider_cache`, `test_task_adapter`
  and `test_native_contract` also passed; these validate boundary and ownership
  behavior, not numerical accuracy.
- A separate process without `rust_pricing` installed evaluated the existing
  `MultiStepExtensionCalibrator.all_maturity_iv_tangent` for all three named
  bases at four and six intervals, eight cycles each: 48 completed evaluations.
  All tangents were finite and the process exited zero. It asserted that
  `rust_pricing` was never imported and idled five seconds between cycles.
  This 64-second negative reproduction does not rule out a native dependency,
  operating-system, lifetime or concurrency fault.

The original `custom-basis-row_hats-4-call-fixed_lookback` selection completed
successfully at 10:15:35 UTC in 1,133.565 seconds, including all four strategies,
checkpoint replays and failure repairs. Samples, seeds, workers, tolerances and
assertions were unchanged. Both the process and LLDB exited zero. This local
pass on a different macOS version does not establish a native-crash repair.

At the 10:17 UTC checkpoint, the other original failed selection,
`extension-all_rows-4`, is running with the same installed wheel and dependency
environment. Its running record is incomplete evidence until the process exits.
The parent CI run also has a single capacity-cancelled
`custom-basis-coupled_row_hats-6` selection restarted as job `123377959795`;
the other cancellations and both failed selections are not repaired by that rerun.

## Memory diagnostic follow-up

The capacity-cancelled parent selection restarted as job `123377959795`
completed successfully at 11:10:22 UTC. This resolves that one incomplete parent
selection; five other cancellations and the two original failures remain.

The original `extension-all_rows-4` selection completed at 10:47:15 UTC in
1,831.092 seconds on the local macOS 27 host. All four price/covariance panels
and both dated ledgers passed, including four strategies and exact checkpoint
resume. The child and LLDB exited zero. Neither original CI failure reproduced
in these local runs; the macOS 15 failures remain unresolved.

An AddressSanitizer wheel built with pinned Rust 1.98.1 passed 18 installed API
preflight tests. Six instrumented Rust integration binaries passed 38 tests
covering graphs, graph division, tasks, operation reuse, calibration reports and
validation pointers. A standalone normal/fault control verified that the runtime
accepts valid execution and detects a deliberate heap use-after-free. At 11:05
UTC the full instrumented fixed-lookback selection is still running, not passed.
These are diagnostic results with a different allocator and operating system;
they do not establish that the original corruption is absent.

The [targeted workflow](../../.github/workflows/native-memory-diagnostic.yml)
runs calibration isolation and both instrumented original selections on hosted
`macos-15`, with CPython 3.12.10 and the observed numerical dependency versions.
It records the actual host/compiler and installed extension hash, verifies the
extension is instrumented, checks the detector's positive/negative controls and
preserves the original workload once. The existing wheel inventory checks and
process failure records remain in use. Diagnostic wheels and debug symbols are
retained separately from ordinary wheel artifacts; all production gates remain.

The helper's eight tests cover runtime selection, framework launchers, rejection
of uninstrumented wheels, genuine detection versus unrelated aborts, compiler and
import failures, literal executable paths, and unchanged shard/failure propagation.
Local integration verifies the helper can launch the actual instrumented wheel,
rejects the actual normal wheel, and completes all 48 calibration-only evaluations
without importing Rust. The staged source archive, workflow YAML and Markdown
links pass. Hosted diagnostic completion is still required.

## Current candidate failure and detector coverage

Candidate `bd5aaf2` reproduced heap corruption in
[coupled-basis-coupled_row_hats-6](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40863601572/job/123463485675).
All ten native panels completed. The first dated ledger, with parameters
`["coupled_row_hats", 6, "call", "arithmetic_asian"]`, was active when the
allocator reported corruption. The job used macOS **15.8.2 / 24H232**, runner
image `macos-15-arm64 20261028.0489.1`; it is a different OS build from the
original 15.7.9 failures.

Artifact `12920020213` was downloaded and verified against SHA-256
`e5f291e8f1bffd8ef5f98c1311bbc37a82c92ef0a4680c0e924b99a113ff3e31`.
The structured record contains all eight threads without truncation. Worker 1
was reporting an invalid free in `Piecewise::into_payoff`; workers 2 and 3 were
reporting damaged nanov2 free-list guards while growing vectors. The active
pricing frames include `Piecewise::combine`, `VegaKtGammaAtoms::new` and
`evaluate_vega_kt_multi_step_gamma_chain_with_samples`. These are detection
sites; they do not identify the corrupting write. A common cause with the prior
libdispatch cleanup failures has not been established.

The first SIGABRT stop was retained, with child exit code null and gate exit 1.
The debugger did not continue the process or retry the workload. No OS crash
report was collected. This failure remains unresolved even though other
candidate selections have passed.

The hosted ASan fixed-lookback selection completed in 17,508.491 seconds on the
original 15.7.9 environment. The ASan extension selection reached the configured
330-minute limit: its four panels completed, its arithmetic-Asian ledger was
interrupted, and its fixed-lookback ledger was not reached. Its running process
record has no exit code and cannot count as a pass. Neither diagnostic proves
a repair of the production-wheel failures.

The diagnostic helper now also verifies allocation/free interception inside
the actual Python interpreter after importing the installed pricing extension.
Separate children exercise normal allocation/free and a deliberate double-free.
The latter must abort with an ASan double-free report containing both allocation
and free stacks. Missing startup markers, unrelated failures, incomplete stacks
and timed-out controls prevent the numerical workload from starting. Partial
logs and control outcomes are retained.

All eleven helper tests pass locally. Actual local integration passes the Rust
normal/UAF controls, wheel import and Python normal/double-free controls. The
Python fault child exits with SIGABRT and retains both history stacks. Local
symbolizer warnings limit source-line detail; module/offset frames are present.
The new Python controls have not yet run in hosted CI.

## Next investigation

Keep both failed artifacts and original combined-shard reproductions. Use the
structured records to compare subsequent failures by stopped thread, module UUID
and file address. Reproduce the provider workload in the matching hosted wheel
environment before changing numerical execution or native dependencies. The
current candidate's pricing-worker corruption and the earlier cleanup failures
remain unresolved; passing reruns do not establish a repair. The five remaining
capacity cancellations in the parent snapshot and outstanding candidate
selections still require completed evidence. The PR stack remains unmerged.
