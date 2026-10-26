# LLDB launch and native-crash follow-up — 2026-10-26

## Scope and demonstrated failures

This follows PR #183, commit `192859ddd39b0fe4d77b18c58a06ab7ae2749ea5`,
on `fix/lldb-native-launch-arguments`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The separate #172 mutation
branch is excluded. No merge is authorized.

P1 for CI: #182's diagnostic runner prevents macOS validation from reaching
the numerical tests. The same failure is inherited by
[#183's macOS Rust job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40245545681/job/121326212524)
and #182's
[row-hats job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40241180802/job/121363493029)
and
[Put/linear-time job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40241180802/job/121363493094):

1. LLDB converts the trailing executable/argv into debugger commands. A
   backtick expression in a literal Python argument is evaluated while setting
   `target.run-args`, before `process launch --shell-expand-args false` can
   apply. Both normal-exit and nonzero-exit probes fail without running the
   requested child. Disabling shell expansion alone is insufficient.
2. Framework Python performs an `exec` handoff at startup. LLDB's default
   `target.process.stop-on-exec = true` stops at `dyld` before the Python probe
   executes. The old crash test accepted any stopped process plus a backtrace
   and absent post-signal marker, so it reported success for `stop reason = exec`.
   Those earlier apparent SIGABRT/SIGTRAP probe passes are not evidence that
   the requested signals were reached or captured.

These are diagnostic-harness failures, not new numerical failures caused by
the Brownian-bridge guard.

## Correction

The debugger's embedded Python now constructs the target and `SBLaunchInfo`
directly from the original argument strings. Only a hex-encoded Python
bootstrap crosses the LLDB command parser, so quotes, backticks, dollar signs,
empty arguments, newlines and Unicode remain data. `SetExecutableFile` retains
the symlink-backed venv path and argv0 explicitly; target creation otherwise
resolves that symlink. Working directory and inherited environment are retained.

The launch is synchronous, with shell expansion, stop-at-entry, ASLR disabling
and detach-on-error disabled. `target.process.stop-on-exec` is explicitly false
for normal interpreter handoffs. Signal and exception stops are still observed
and captured. No explicit continue/resume, retry, fallback, or ignored crash
is introduced. The first terminal process observation is retained, and only
the current valid child's normal exit zero plus a successful debugger can pass.
Nonzero child exits and absent, stale or incomplete records still fail.

The real macOS tests now require pre-signal execution evidence and the requested
signal/Mach exception in the captured stop description. They also verify that
post-signal code is not executed. A separate real `exec` probe must reach a
child exit of 3. Normal/nonzero probes check literal arguments, venv identity,
working directory, environment, stdout/stderr, and a once-only execution marker.

Implementation references: LLDB's
[launch-info API](https://lldb.llvm.org/python_api/lldb.SBLaunchInfo.html),
[executable-path contract](https://lldb.llvm.org/cpp_reference/classlldb_1_1SBLaunchInfo.html),
and [stop-on-exec setting](https://lldb.llvm.org/use/settings.html).

## New native-crash evidence

Separately, [#180's macOS remaining job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40193992079/job/121248413956)
failed in `test_vega_kt_path_ledger.py` with exit -5 / SIGTRAP after
1,485.827594 seconds. Unlike the earlier artifact with an empty OS-report
collection, this job retained a native `.ips` report. Artifact `12623534697`
was downloaded and verified against SHA-256
`9c35fbafa2be76011206b44e6dde2e99c449985e2bcb96239c4ad15dbe9ea100`.

The [retained evidence](macos-path-ledger-crash-2026-10-26.json) preserves its
process record and both original IPS JSON documents as parsed objects, including
all threads and image identities. The original IPS member SHA-256 is
`b5569dd2924f61711553363ac3a005bb62d4f44191a15af92719f070710a0628`.

On macOS 15.7.9 / CPython 3.12.10, the faulting thread is a system workqueue
thread. Its frames run through `_pthread_tsd_cleanup`, `_dispatch_cache_cleanup`,
`___BUG_IN_CLIENT_OF_LIBMALLOC_POINTER_BEING_FREED_WAS_NOT_ALLOCATED`,
`malloc_report`, and `abort` / `__abort`. The report labels the exception
`EXC_BREAKPOINT` / SIGTRAP and includes `abort() called`.

At that moment the main Python thread is waiting in Rayon during
`evaluate_vega_kt_multi_step_gamma_chain_with_samples`; pricing workers are
executing that calculation. This establishes an allocator invalid-free
detection in libdispatch thread cleanup. It does **not** establish where the
invalid pointer originated, whether an earlier write corrupted memory, or
whether the library, a dependency, or the OS is responsible. There is no
evidence here for reducing sample counts, relaxing tolerances, changing
numerical formulae, or attributing the failure to memory exhaustion.

The existing native SIGABRT/SIGTRAP remains a merge blocker. A future clean
run under LLDB would not, on its own, explain this intermittent failure.

## Validation and remaining gates

- **57 checker tests pass** across the native debugger, wheel smoke runner,
  provider partition, wheel contracts and replay audit. The native suite has
  **3 macOS-only integration methods**, explicitly skipped on this Linux host.
- All **18 reference/schema/link scripts** pass. The archive gate still checks
  the unchanged full provider matrix and required native-debugger probes.
- LLDB 18.1.3 reproduces the parent's backtick-expression error before launch.
  An actual `SBLaunchInfo` roundtrip also verifies the exact venv path, literal
  argv (including empty/newline/Unicode values), and disabled shell expansion
  and detach flags without launching an inferior.
- Attempts to run the real child-process probes locally are blocked by
  `ptrace: Operation not permitted`. The corrected runner fails the gate for
  these missing process results even though LLDB itself can return zero.
  These attempts are **not passing integration evidence**. Normal/nonzero
  exits, startup handoff and real signal stops require the mandatory macOS CI
  run on this new revision.
- No Rust/library source, native binding, public API/schema, numerical test
  body, fixture, sample count, seed, worker policy or tolerance changes.
  Workflow definitions and numerical commands are unchanged. No fresh native
  rebuild or new numerical-replay result is claimed for this harness correction.

At the pre-publication checkpoint, #183 has 91 successful checks, one macOS
Rust failure described above, one running check, one queued and four skipped.
#181's Windows Put/linear-time job failed while downloading `actions/checkout`
because its runner reported a missing device, before repository checkout or
tests; that infrastructure failure is not evidence of a pricing regression.
Full cross-platform numerical CI and the original native-crash diagnosis remain
required before merging.
