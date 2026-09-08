# European Black–Scholes benchmark baseline v0.1

Status: baseline harness; measured artifacts are produced by CI

The benchmark suite is a regression and optimization baseline, not a latency
service-level agreement. Pull-request CI runs it on the two MVP targets:
Apple Silicon macOS and Windows x86-64. Each run retains `rust.json`,
`python.json`, and `metadata.json` as a short-lived workflow artifact.

## Workload

- European equity Call, Spot and Strike 100, one-year expiry;
- continuously compounded rate 5%, continuous dividend yield 2%, volatility 20%;
- counter-based Pseudo-MC with 16,384 independent units and antithetic pairing;
- 32,768 evaluated Paths, two Rayon workers, Reduction block 256;
- full risk requests use Delta, Gamma with a 1% relative Spot bump, and Vega.

Rust records compile and evaluate timings for Price-only and full-risk Plans.
Python records full-risk compile/evaluate timings through the installed wheel and
the `PricingResult.value` getter overhead. Every case is warmed up before the
recorded samples, and reports minimum, median, maximum, sample count, and Paths
per second where applicable.

## Current measurement boundary

The v0.1 full-risk executor computes AAD Delta/Vega, central-bumped AAD Delta
for Gamma, and common-random-number validation bumps in one kernel invocation.
The harness therefore labels this measurement
`evaluate_aad_with_crn_bump_validation` and explicitly records that AAD and
Bump timing are not separable. It does not infer one cost by subtracting noisy
wall times. Separate AAD-only and Bump-only instrumentation remains an explicit
G8 task before the final conformance report.

## Host and resource metadata

`metadata.json` records the commit, runner OS/architecture, platform and CPU
strings, Python version, and full `rustc -vV`/`cargo -V` output. Portable peak
memory and allocation counters are not currently available without changing
the measured process or allocator, so the corresponding fields are `null` and
the reason is recorded. Later instrumentation must add fields without silently
changing the v0.1 workload.
