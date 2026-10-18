# Local library validation, 2026-10-03

This record covers the current working tree on Apple Silicon macOS with Rust
1.98.1 and CPython 3.13.1. It separates software/replay gates from experimental
model acceptance. Remote Linux/Windows CI was not run by this local task.

## Repairs

- Move `average_local_vol_pathwise` from the process implementation into
  `engine::risk::local_vol_allocation`. Its arithmetic and error handling are
  unchanged. All three risk callers now import it from the risk layer; the
  dependency-direction guard passes without an exception or relaxed rule.
- Escape a formula in the Hull–White reserve documentation. Rustdoc now builds
  with `-D warnings` instead of emitting two broken-link warnings.
- Register the 16 Bass public classes, their methods/properties, the common
  model builder, plan diagnostics and the existing `DiscountCurve.discount`
  method in the strict wheel API inventory. The inventory remains explicit;
  it is not inferred from the stub at validation time. Mutation tests reject
  missing classes/methods, extra methods and incorrect decorators.
- Pass the companion source directory explicitly to the native backtest CLI
  subprocess test, matching the parent process's source import. Run the CLI
  from an unrelated temporary directory. The clean native-wheel test no longer
  depends on an already installed companion package.

## Reviewed Local Volatility replay update

The macOS Local Volatility replay was stale after the earlier
[integrated-adjoint correction](vega-kt-backtest-numerics-v0.1.md) and
[independent exclusion audit](vega-kt-backtest-allocation-audit-v0.1.md).
Those changes were already in the working tree before this validation task.
No allocation formula or numerical tolerance was changed during this task.

The [46-field audit](../../fixtures/replay/local-vol-node-allocation-change-audit-v1.json)
shows that all differences are restricted to `result.risks.vega_kt` in the two
full-risk cases. All other fields are bit-for-bit identical, including price,
Delta, Gamma, scalar local Vega, input requests, fingerprints and RNG metadata.
An independent initial-row price-bump probe passes all 48 declared checks.
The [previous macOS capture](../../fixtures/replay/historical/local_volatility-macos-aarch64-before-node-allocation-audit.json)
is retained unchanged; the current fixture records the corrected diagnostic
allocation and its `local_vol_node_allocation_diagnostic_v1` label.

| Case | Local parallel Vega | Reported bucket sum | Independently accumulated residual |
| --- | ---: | ---: | ---: |
| Pseudo-MC | 38.399608286143724 | 16.221137213812383 | 22.17847107233134 |
| RQMC | 38.403900431392294 | 16.352923233563907 | 22.05097719782839 |

These residuals are not a completed paper decomposition. Re-running the existing
experimental Local Volatility paper-VegaKT numerical gate against the clean
release wheel gives **54 failures out of 154 checks**. It remains unaccepted,
and the companion's diagnostic-only guard stays in place. The numerical
acceptance thresholds were not weakened to make replay pass. Bass's separately
identified recalibrated market-IV sensitivities are not this local-node
allocation algorithm.

The 54 failures comprise 24 bucket-sum versus local-parallel-Vega checks,
24 projection-residual checks and six bucket-sum versus common-random-number
bumps. The other 100 checks pass. The full observations, predeclared thresholds,
policy hash and native binary hash are retained in
`local-vol-paper-numerics-latest/report.json` under the raw-evidence directory.

The Windows Local Volatility fixture still predates that correction. It needs
a native Windows capture and numerical review before current Windows replay
can pass. This task does not substitute macOS numbers for Windows evidence.

## Local runtime setup

The uv-managed CPython shared library has an absolute `/install/lib` load name.
For embedded Python Rust tests and newly created virtual environments, supply
the actual directory returned by `sysconfig.get_config_var("LIBDIR")` through
`DYLD_FALLBACK_LIBRARY_PATH`. This resolves the loader failure without editing
the library or bypassing a test. The retry keeps the original failure logs.

Wheel smoke uses an empty `.wheel-smoke-venv`, the locally built release wheel,
and downloaded NumPy/pip wheels. `PIP_NO_INDEX`, `PIP_FIND_LINKS`, `UV_OFFLINE`
and `UV_FIND_LINKS` allow that environment to be created and tested offline.
Optional companion validation/constraint dependencies are installed in the
development environment for the separate companion suite.

Raw command logs, initial failures and successful retries are retained under
`benchmark-results/full-check-20261003T125142Z/`. Previous benchmark outputs
were copied to its `previous-benchmarks/` directory before regeneration.

Performance reports and four exact replay comparisons pass. The owned extended
model test process was temporarily stopped during benchmark measurement and
resumed afterward; pause/resume records are retained. The initial benchmark
attempt reached the Python stage before encountering the same shared-library
loader issue. The successful retry supplies the library directory. These are
local measurements, not an isolated-host latency guarantee.

The long extended-model gate was subsequently split after six completed tests.
The other 21 named tests run in at most four independent processes, each with
`--exact --ignored --nocapture --test-threads=1`, using a retained copy of the
same release test binary. Each test's calibration populations, seeds, pricing
worker count, numerical tolerances and refinement checks are unchanged. The
original runner's interrupted status is preserved, together with the complete
27-test inventory, binary hash and per-case logs. Completion is determined from
that inventory, not from the interrupted original process's exit status.

## Verification results

The [machine-readable inventory](library-full-check-2026-10-03.json) records the
successful final runs, release-wheel hash and known failures separately.

| Gate | Result |
| --- | --- |
| Final Rust all-features workspace, excluding Python crate | 711 passed; 50 ignored cases executed separately |
| Rust Python crate | 3 passed |
| Extra ignored numerical/stress tests | 50 passed, including all 27 extended-model cases |
| Extended-model accuracy panels | 74 panels / 222 quote points passed |
| Clean release wheel | 199 tests passed; two optional-dependency skips in the minimal environment |
| Same release wheel with optional dependencies | 199 passed, zero skips |
| Companion package with the final native wheel | 193 passed, zero skips |
| Four platform-applicable exact replay checks | Passed |
| Benchmark reports and memory metadata | Passed |
| Clippy, formatting, all-target compilation and strict Rustdoc | Passed |
| Dependency direction, schemas and independent reference fixtures | Passed |
| Source archive, Markdown links and whitespace | Passed |
| Separate experimental Local Volatility paper VegaKT | 100 passed / 54 failed; unaccepted |

Earlier default-workspace and no-default-feature runs passed 706 and 692 tests
respectively. Eight step-recovery tests were added during the check; the final
all-features suite includes them, and the latest version of that test file also
passes separately with no default features. Production Rust sources did not
change during the final full Rust/wheel verification. Concurrent documentation
and test updates are captured in the input/output manifests; the final source
archive includes uncommitted and untracked source files from the working tree.

The 50 extra tests are 27 model, 11 risk, one statistical, two path-dependence,
two early-exercise, five library stress and two dividend-refinement cases.
The interrupted initial model invocation is superseded only by the explicit
complete per-case inventory. Initial environment/harness failures remain in
the raw logs; no failing numerical gate was relabelled as passed.

This closes the local software, distribution and applicable replay checks.
It does not establish acceptance of the unfinished Local Volatility paper
decomposition or validate the Windows/Linux platforms described above.
