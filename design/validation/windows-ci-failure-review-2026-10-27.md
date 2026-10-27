# Windows CI failure propagation review — 2026-10-27

This continues the unmerged review stack on #186,
`78ea2cfd3ba684b7b98c5686a0d06aff81d627ac`. Main is still
`45a662e57f48f35f4ebb6c62e8c9ce0f2533fc14`. The separate #172 mutation-testing
branch is unchanged. No merge is authorized by this review.

## P1: native test failures were reported as successful Windows jobs

The [#186 Windows Rust job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40346566438/job/121654942966)
has a successful GitHub conclusion, but its log contains three failed Cargo
invocations. Default-feature, minimal-feature and all-feature tests each stop at
`calibration_report::v1_migration_preserves_unknown_diagnostics_and_frozen_numerical_fields`.
Subsequent successful `pricing-numerics` or `pricing-python` tests overwrite the
last native exit status. The same failure is present in
[#185's Windows Rust job](https://github.com/msfmfjt/rust-pricing-library/actions/runs/40343033248/job/121642506718).
The successful check badges must not be treated as complete test passes.

The affected jobs used Windows' implicit PowerShell shell. GitHub's
[documented shell behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstepsshell)
appends a final `$LASTEXITCODE` check; it does not check every native command in
the block. Select explicit `shell: bash` for the main Rust and supported-wheel
jobs. Actions invokes this with `-e -o pipefail`, so an earlier failed test,
schema check, build or replay checker fails its step. Existing commands, test
selections, inputs, timeouts and acceptance thresholds are retained.

An inventory of the current workflows found seven multiline command blocks
without an explicit shell in Windows-capable jobs, all in these two jobs. The
new regression suite executes those actual blocks with command probes: each of
the 22 command positions returns a real native exit 23 in turn, and later
commands must not execute. Seven all-success runs must complete every command.
Three dependency-direction pipeline probes check both failure ends and the
all-success path. The source-archive checker also rejects a missing Bash
default, an unsafe step override or `continue-on-error` in these two jobs.
This is a guard for these retained workflow blocks, not a general shell linter.

## Hidden migration-test mismatch

The failed assertion compared the newly compiled Windows plan fingerprint
`17c08190d3b9d40a197b0f8b1fd8c610f4781fa63450947db165ae47f653c713`
against the frozen Linux report's
`65aea19d77eef0e36eb76b4354fbda4cf5786357a4d98eaf92fbbf234130614a`.
The source/task fingerprint comparison immediately before it passes.

`MultiAssetPricingPlan::make_fingerprint` hashes calibrated leverage values,
correlation factors and other realized floating-point quantities. The existing
[task replay contract](../../docs/library/pricing-tasks.md#json-versions-and-replay-identity)
does not promise identical compiled numerical state across platforms. This
review does not identify a particular floating-point operation as the cause of
the two different hashes and does not change their construction.

Move the compiled-hash assertion into the already existing same-platform
frozen numerical comparison. Keep the source/task hash assertion unconditional.
The frozen v1 input still migrates with exact equality of every field on every
platform, and the frozen numerical comparison remains exact on its recorded
platform. Add a separate test using six freshly evaluated native task families:
strip only the v2 diagnostics field, migrate through v1 and back to v2, then
compare the complete serialized report bytes, source/plan identities and a
second parse. No price, Greek, covariance, seed, tolerance, model or calibration
field is excluded from this native migration check.

## Validation and evidence

[Selected original CI evidence](windows-ci-failure-evidence-2026-10-27.json)
contains the failing Windows lines, misleading step conclusions, successful
macOS debugger probes and successful supported-platform numerical gates. It is
an extract with job URLs/IDs, not a replacement for the complete job logs.

- The new failure-propagation suite passes all three tests, including 32 real
  Bash subprocess probes and eight negative archive mutations.
- The related Rust targets pass 29 tests: report migration 5, Pricing Task 9,
  compiled task reuse 6 and Product Graph 9, using pinned Rust 1.98.1.
- The all-feature Rust workspace suite excluding `pricing-python` passes
  1,103 tests, with zero failures and the same 91 opt-in/ignored tests as #185.
- All 18 reference/schema/link checks pass. The six Python checker suites run
  64 tests: 60 pass, three physical macOS debugger tests are skipped on Linux,
  and the real LLDB API probe is skipped because LLDB is unavailable locally.
- The same 29 focused tests pass without default features. Seven native Python
  binding tests, formatting, all-target workspace check, all-target/all-feature
  Clippy with warnings denied, Rust API docs and dependency-direction checks pass.
- The staged source archive passes its checker. New branch CI is recorded in
  the pull request after publication; a pending job is not a pass.
- No files under `crates/*/src`, numerical fixtures, Cargo manifests/lockfile or
  public API definitions change. The installed library implementation remains
  identical to #186/#185.

## Follow-up of #185 and #186

The temporary local extended-price process from the previous session could not
be recovered after the session restart; it is not counted as a completed run.
Independent #186 Linux, macOS and Windows CI each completed all 11 extended-risk
tests and all 27 extended-price acceptance tests with no failures. These jobs
already use explicit Bash and their logs contain successful test summaries.
Supported-wheel jobs on
macOS and Windows passed all four existing frozen replay checkers and benchmark
report validation; their logs contain no hidden native-test failure markers.
This supplies platform CI evidence for the preceding local benchmark/RSS and
unsupported-Linux replay limitations without altering those checkers.

#186's macOS Rust job passed all 14 debugger tests, including real normal and
nonzero venv exits, literal arguments, explicit exec handoff, SIGABRT and SIGTRAP.
Its Linux/macOS Rust job logs have no failed Cargo tests. Windows Rust requires
the corrections above and a fresh complete run. Installed-wheel shards were
still queued/running at this checkpoint. The original intermittent
[native invalid-free crash](lldb-launch-review-2026-10-26.md) remains unresolved;
passing launcher probes do not demonstrate that it is fixed.
