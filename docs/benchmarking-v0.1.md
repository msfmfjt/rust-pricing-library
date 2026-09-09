# European Black–Scholes benchmark baseline v0.1

Status: baseline harness; measured artifacts are produced by CI

The benchmark suite is a regression and optimization baseline, not a latency
service-level agreement. Pull-request CI runs it on the two MVP targets:
Apple Silicon macOS and Windows x86-64. Each run retains `rust.json`,
`local-volatility-rust.json`, `python.json`, `replay.json`, and
`metadata.json` as a short-lived workflow artifact. Supported platforms also
retain `local-volatility-replay.json`.

## Workload

- European equity Call, Spot and Strike 100, one-year expiry;
- continuously compounded rate 5%, continuous dividend yield 2%, volatility 20%;
- counter-based Pseudo-MC with 16,384 independent units and antithetic pairing;
- 32,768 evaluated Paths, two Rayon workers, Reduction block 256;
- full risk requests use Delta, Gamma with a 1% relative Spot bump, and Vega.

Rust records compile and evaluate timings for Price-only, full-risk, and
standalone common-random-number bump Plans. Python records the corresponding
full-risk and standalone-bump timings through the installed wheel plus the
`PricingResult.value` getter overhead. The standalone bump case evaluates five
Price-only Plans: base, Spot-down/up, and volatility-down/up. Every case is
warmed up before the recorded samples, and reports minimum, median, maximum,
sample count, and kernel Path evaluations per second where applicable.

## Current measurement boundary

The v0.1 full-risk executor computes AAD Delta/Vega, central-bumped AAD Delta
for Gamma, and common-random-number validation bumps in one kernel invocation.
The harness therefore labels the integrated measurement
`evaluate_aad_with_crn_bump_validation` and explicitly records that AAD and
Bump timing inside that kernel are not separable. It does not infer one cost by
subtracting noisy wall times. Instead, it reports an independently executed
Price-only CRN bump case, labelled
`evaluate_crn_bump_validation_price_only`. Pure AAD-only timing would require a
new execution mode and remains a documented limitation rather than silently
changing the frozen request/result contract.

## Local Volatility/VegaKT workload

The suite also emits `local-volatility-rust.json` as the accepted L8 benchmark
baseline for the Local Volatility/VegaKT slice. It records Rust compile and
evaluate timings for Price-only, AAD Local Vega, VegaKT decomposition with full
bucket covariance, and selected common-random-number bump workloads. The
standalone bump case evaluates five Price-only Plans: base, Spot-down/up, and
uniform Local-variance-down/up. It uses the same two-worker, Reduction block
256 execution policy as the replay fixture and records grid sizes, checkpoint
policy, AAD tile capacity, and covariance layout.

On supported platforms, the same CI step also generates
`local-volatility-replay.json`. If a matching frozen fixture exists, CI compares
it byte-for-byte. If the platform fixture is not frozen yet, the generated JSON
is retained as an artifact so it can be reviewed and promoted in a follow-up
change.

## Host and resource metadata

`metadata.json` records the commit, runner OS/architecture, platform and CPU
strings, Python version and ABI tag, full `rustc -vV`/`cargo -V` output, target
triple, enabled feature sets for Rust and wheel measurements, the SHA-256
digest of `Cargo.lock`, and observed peak RSS for each benchmark/replay child
process. The Local Volatility replay peak is present exactly when
`local-volatility-replay.json` is retained. Allocation counters require an
instrumented allocator and remain recorded explicitly as unavailable rather
than silently omitted. Later instrumentation must add fields without silently
changing the v0.1 workload.
