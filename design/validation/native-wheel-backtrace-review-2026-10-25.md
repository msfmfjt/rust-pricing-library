# Native wheel backtrace review — 2026-10-25

## Scope and evidence

This continues from PR #181, commit
`d249c937a28b2cf031d61669fa489f44b6f8b13d`, on
`fix/macos-native-wheel-backtraces`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The separate #172 mutation branch
is excluded. No merge is authorized.

The remaining highest-priority blocker is the unexplained macOS native crash.
The new evidence does not establish its cause:

- [#179's Put/all-rows/four-interval job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40146118573/job/121070913888)
  passed its native panels, then aborted during the arithmetic-Asian mixed
  ledger. Its process record reports SIGABRT / exit -6 after 1,079.627 seconds.
  Python's fault handler locates the call to
  `evaluate_vega_kt_multi_step_gamma_chain_audit` in
  `backtesting/rust_pricing_backtest/multi_step_extension.py`; it supplies no
  native frames identifying the failing C/Rust function.
- Diagnostic artifact `12602547880` matches SHA-256
  `d39fbc5ce7c2fcd0aeb0cf6cfd5e9d37545c48d3787818150375c0246b015666`.
  It contains the failed process record and complete inventory, but
  `native-crashes/collection.json` has an empty file list. The same Linux and
  Windows selections passed. Neither those passes nor an absent OS report
  determines the macOS cause.
- [#179's macOS remaining job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40146118573/job/121070913851)
  is marked failed after approximately 139 minutes, while its stored step list
  still shows validation in progress and subsequent upload steps pending.
  Downloading its log returns `BlobNotFound`, and the run has no corresponding
  diagnostic artifact. This is incomplete evidence, not a proven signal,
  numerical failure, timeout or passing calculation.

The earlier row-hats SIGTRAP and SIGABRT observations remain in the
[previous crash checkpoint](portable-wheel-crash-review-2026-10-25.md).
At this review's start, #180's new macOS installed-wheel selections were queued;
there was no new Python SIGTRAP traceback to analyze.

## Correction to the diagnostic boundary

macOS installed-wheel selections now use the explicit `--native-debugger lldb`
option. Each example, module and provider shard retains one fresh Python
process and its original arguments, class state, cases and order. LLDB launches
it once with shell argument expansion disabled and the original interpreter
argument retained. The debugger's own embedded Python supplies the LLDB API;
the installed wheel does not gain a debugger dependency.

After normal exit or a native stop, the callback writes a versioned process
record with a per-invocation identifier. On a stop it also records thread stop
descriptions/data and attempts native backtraces for all threads, registers and
loaded-image identities. The LLDB transcript and JSON record are retained in
the existing job artifact under `native-debugger/`. The callback never resumes
the stopped process. Batch quit terminates it; there is no retry or fallback to
an uninstrumented execution.

A debugger returning zero is insufficient to pass. The parent requires its
current invocation's record, a valid process ID, normal child termination with
exit zero, and a successful debugger exit. Missing, malformed, stale, stopped,
crashed or incomplete records fail. Positive child exit codes remain visible;
a native stop with no normal exit has gate status 1 and its actual native stop
state, rather than an invented Unix signal number. This also handles Darwin
Mach exceptions that LLDB does not expose as POSIX signal stops.

Examples now use the same running/completed process records as modules and
shards, retaining their existing fail-fast order. The archive gate still
requires the three rough-dividend examples and now verifies their shared
execution/failure path. Its negative controls reject a missing example or
removed execution/failure gate.

Before expensive validation, every macOS wheel runner must pass real debugger
probes for normal/nonzero exits, a symlink-backed venv, literal arguments,
working directory/environment inheritance, SIGABRT and SIGTRAP. The crash
probes require a native backtrace and verify that post-signal code never runs.
The ordinary Rust CI also runs this checker suite. Missing LLDB on macOS fails
these probes rather than skipping them. Other platforms retain direct Python
execution; their two macOS integration methods are explicitly skipped.

This adds diagnostics, not a native crash fix. Running under a debugger can
affect timing. Complete passing numerical selections remain required, and a
new passing run alone does not explain earlier intermittent crashes. Runner
loss can still prevent artifact upload. No Rust/library/API source, wheel
binary, schema, seed, worker/block count, sampling rule, calibration/AAD policy,
numerical assertion or tolerance is changed.

## Validation

- Checker suites: **55 passed**, with **2 macOS-only integration methods not
  executed locally**. A negative control that trusts only the debugger's exit
  is rejected by the failure-propagation regression. Actual LLDB behavior
  remains subject to the mandatory macOS CI probes.
- All **42 original examples** pass with the new process records, followed by
  **220 tests in 34 installed-wheel modules**. There are 42 and 34 complete
  passing process records respectively. A comparison against the parent
  confirms every example and its execution order is retained.
- The #181 Linux release wheel is reused because native sources are unchanged.
  Its verified SHA-256 is
  `3ba7426072d1f1bfd26c5848be8a4628c4093babf91009616a93913fac91bd3a`;
  all **341 runtime callable signatures** pass. The preceding
  [time-grid review](time-grid-resource-review-2026-10-25.md) records its native
  and exact numerical replay validation.
- Import-only discovery still finds **110 modules / 610 methods** and exactly
  **20 provider shards / 179 cases**. This is coverage evidence, not a claim
  that all expensive provider bodies ran locally.
- All **18 reference/schema/link scripts** pass. Parsed workflow comparison
  confirms that only the debugger checks and macOS diagnostic option change;
  all matrices, numerical commands, time budgets and failure policies remain.
- Formatting, final documentation links and staged source-archive checks pass
  before publication. No native rebuild is needed for this harness-only change.

#180's Windows Rust and stochastic-dividend/refinement jobs now pass, confirming
the earlier Windows fixes. #181 has no failed checks at this checkpoint, but
its macOS wheel work is unfinished. Native SIGABRT/SIGTRAP and incomplete
macOS validation remain merge blockers.

LLDB behavior is based on its [batch-command interface](https://lldb.llvm.org/man/lldb.html),
[process API](https://lldb.llvm.org/python_api/lldb.SBProcess.html),
[thread API](https://lldb.llvm.org/python_api/lldb.SBThread.html) and
[settings](https://lldb.llvm.org/use/settings.html).
