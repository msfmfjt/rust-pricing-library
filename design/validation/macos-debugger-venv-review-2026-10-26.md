# macOS debugger venv fixture follow-up — 2026-10-26

This follows [PR #185](https://github.com/msfmfjt/rust-pricing-library/pull/185),
commit `cf323ac5839e168b16cca31a0585ad391578ce44`. It changes test setup and
diagnostics only. The native runner, Rust library, Python binding, numerical
inputs, tolerances and acceptance gates are unchanged.

## Newly completed macOS CI evidence

The main macOS job for #184,
[121502320648](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40302138642/job/121502320648),
completed on 2026-10-26 at 14:39 UTC. It reached the real debugger tests:

- SIGABRT and SIGTRAP probes passed their pre-signal, native-stop/backtrace and
  never-continue assertions.
- The explicit `exec` handoff reached the requested exit status 3.
- Both literal-argument/venv probes failed: requested exit 0 and 7 both became
  a normal Python exit 1 after a child `AssertionError`.
- LLDB returned zero, but the runner correctly failed both probes from the
  recorded child exit. The remaining Rust steps were skipped after this failure.

The child's assertions were on one line and had no failure messages. The log
does not itself identify which child precondition failed. It therefore cannot
establish that argument transport, venv selection or working-directory
inheritance is broken, nor does this failure reproduce the numerical native
invalid-free crash.

## Canonical fixture directories and diagnostic assertions

The failing launch command used a temporary venv under `/var/folders/...` and
compared `sys.prefix` to that literal spelling. The retained report paths show
the corresponding `/private/var/folders/...` spelling. The
[CPython 3.12.10 framework launcher](https://github.com/python/cpython/blob/v3.12.10/Mac/Tools/pythonw.c)
canonicalizes the executable's **parent directory** before setting the launcher
path used for venv discovery. It deliberately preserves the final executable
symlink. The
[path initialization code](https://github.com/python/cpython/blob/v3.12.10/Modules/getpath.py)
uses that launcher value when locating the environment.

Consequently the fixture's literal-prefix assertion is not valid for an
aliased temporary directory. This is a source-based explanation of a definite
test precondition problem; the old CI log's unlabeled assertion is insufficient
to rule out an additional launch problem.

The fixture now resolves its temporary root before creating the venv and uses
that canonical directory consistently. It does **not** resolve `bin/python`,
which would select the base interpreter. A direct invocation first verifies
that this executable discovers exactly the expected venv. The debugger child
keeps exact comparisons for every argument, `sys.prefix`, working directory and
the test environment value. Its assertions now have separate lines and named
diagnostics. Exit 0/7, once-only execution, stdout and stderr requirements remain.

An additional test exercises the installed LLDB's real `SBFileSpec` and
`SBLaunchInfo` objects through the production launch helper. Target/process
handles are test doubles, so it needs no ptrace permission.
It checks the actual executable and the entire argv vector, including argv0,
an empty argument and newline/Unicode data. This closes the previous test
gap where a mock file-spec constructor could hide binding behavior.

## Validation and remaining gates

On this Linux host with the isolated LLDB 18 installation, the real launch-info
test and ten native-runner unit tests pass. The three physical macOS process
integration tests remain skipped locally. This is real LLDB **API** evidence,
not a claim of a successful native child launch on Linux or macOS.

All 18 reference/schema/link checks and 58 checker tests pass, including the
real LLDB API test. Python compilation and the staged-tree source-archive check
also pass. The three skipped physical macOS tests are excluded from that count.

The production runner and numerical source trees are byte-for-byte unchanged
from #185. Its
[executor and numerical validation record](executor-vector-resource-review-2026-10-26.md)
remains applicable. No test is removed, weakened or retried after failure.

The new macOS CI run must confirm that the canonical fixture reaches both
requested exit statuses while still capturing fatal signals. The original
[native invalid-free investigation](lldb-launch-review-2026-10-26.md) remains
open. Main and the stacked PRs remain unmerged; explicit user approval is still
required before any merge.
