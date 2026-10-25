# Cross-platform wheel crash and CI follow-up — 2026-10-25

## Scope and observed failures

This continues the review from PR #179, commit
`f8230c8bae5cefc3ddf7a299e35001dfec6989a9`, on
`fix/portable-wheel-crash-diagnostics`. Main remains
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. No merge is authorized and the
separate mutation-testing branch in #172 is not included.

Three CI failures require different responses:

1. **Windows runner regression.** The new shard-crash test hard-coded exit
   `-6` while expecting `SIGABRT`. Windows exposes `SIGABRT` as 22, so the
   runner correctly reported `signal=6` and the test failed. Both
   [#178's Windows job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40144409095/job/121022577208)
   and [#179's Windows job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40146118573/job/121027709668)
   failed before the later Rust checks. The original test also fails locally
   when the Windows signal mapping is supplied.
2. **Unresolved native macOS crash.** In
   [#179's row-hats/four-interval shard](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40146118573/job/121070913704),
   all ten native panels and the first three dated ledgers completed. The Put
   fixed-lookback ledger then terminated the process with `SIGTRAP` (exit -5),
   after 7,835.326 seconds in the shard. The diagnostic ZIP contains its failed
   process record, complete inventory, and `native-crashes/collection.json`
   with an empty file list. Its SHA-256 is
   `060db45729e2e21004508384d697f62f034cc73f3e0323eecbc7eac18aba5df5`.
   There is no Python stack or OS crash report identifying the cause.
   [The same #179 Linux shard passed](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40146118573/job/121070913675).
   This does not resolve the macOS failure or the earlier SIGABRT.
3. **Windows refinement timeout.** In
   [#179's stochastic-dividend job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40146118573/job/121027709824),
   all earlier steps passed, but the final Asian/Barrier refinement was
   cancelled at the 20-minute job limit. The two largest cold compilations
   took 4m19s and 5m54s; recalibration/particle tests took 296.81s.
   [#178's successful Windows job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40144409095/job/121022577070)
   took 13m56s overall, with corresponding times 3m00s, 3m53s and 191.69s;
   its final path-refinement tests took another 74.21s. The failed run is
   incomplete evidence, not a numerical failure or a passing result.

## Changes and boundaries

The shard-crash regression now uses the platform signal constant. Its test
explicitly exercises mappings 6 and 22 on every host, retaining the exact
single-child invocation and immediate failure requirements.

Provider shard CLI processes on POSIX register a `SIGTRAP` traceback handler
with `chain=True` before loading any provider suite. Python's
[standard fault handler](https://docs.python.org/3.12/library/faulthandler.html)
does not cover `SIGTRAP`. Chaining retains the previous fatal action after
printing Python stacks; it does not retry, suppress or recover native crashes.
Windows retains `-X faulthandler` and its built-in exception handling; the
unavailable user-signal registration is not called there. Matrix generation
does not install process signal handlers. This records Python stacks, not a
native C/Rust backtrace, and is **not a fix for the macOS native crash**.

Only the stochastic-dividend risk/refinement job's Windows time allowance
changes from 20 to 30 minutes, based on the cold-build and test timings above.
Linux and macOS remain at 20 minutes. All commands, matrices, evidence uploads
and failure policies are identical. Provider selections retain their existing
330-minute limit.

No Rust, Python product API, model, engine, calibration/AAD kernel, provider
calculation, serialized data, seed, sample count, worker/block setting,
numerical assertion or tolerance changes. The same 63 installed-wheel
selections remain required across the three platforms, with 20 provider
shards and 179 expanded cases. The remaining group and original full-module
commands are unchanged.

## Validation

- Runner/shard/API/audit checker tests: **47 passed** (15 + 9 + 15 + 8).
  The real-process POSIX regression invokes the actual shard CLI with a
  synthetic provider, sends SIGTRAP only to that child with core files disabled,
  and checks the traceback, absent post-trap execution, negative exit status,
  failed JSON record and nonzero parent status. A Windows-path test checks
  that unavailable registration is avoided and failures still propagate.
  Two negative controls confirm that omitting the traceback handler or
  swallowing SIGTRAP makes the real-process regression fail.
- Eighteen numerical-fixture/schema/documentation checks and Rust formatting
  pass. A parsed YAML comparison confirms the Windows time allowance is the
  only workflow behavior change.
- The parent #179 CI Linux wheel is reused because all native/library sources
  are unchanged. Artifact 12582419998 belongs to head `f8230c8`; the downloaded
  archive matches its declared digest
  `446b729c93e4aef99c5d688db638da0675a50e6d7d4e4eb2ceeaa040dfcb526e`.
  The wheel SHA-256 is
  `e5a440a393002f3e5815a2e5e047b97ead37db478d16ea3a05c59b1c8c3b34fa`.
  Package/ABI/SBOM/static/runtime checks pass, including 341 callables.
- Installed-wheel API tests: **38 passed** across Specification, Product Graph,
  calibration/report validation, Task/reuse and legacy wire context.
- All four provider-contract shards pass through the updated CLI: **11 test
  methods** (3 custom basis, 3 coupled basis, 3 extension, 2 Put extension).
  Their process durations are 122.582s, 46.588s, 50.926s and 40.978s. These
  retain the original native calculations, atomic failure/retry checks and
  cache/order-release assertions.
  Discovery still finds 110 modules / 608 tests;
  both the test-ID and provider-case multisets match full discovery. Discovery
  does not claim all expensive Python test bodies ran locally.

The complete source archive and local documentation links are checked before
publication. Native workspace tests are not rebuilt for this harness-only
change; the fresh three-platform CI retains all existing Rust and numerical
gates, and the preceding review records the native-source validation.

The macOS native cause remains the highest-priority unresolved issue. No
numerical relaxation or speculative native fix is made without diagnostic
evidence. Fresh Windows/macOS CI must confirm the harness change and full
numerical jobs; queued, cancelled or skipped jobs are not passing evidence.
