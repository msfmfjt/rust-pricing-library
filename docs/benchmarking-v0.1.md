# European Black–Scholes benchmark baseline v0.1

Status: baseline harness; measured artifacts are produced by CI

The benchmark suite is a regression and optimization baseline, not a latency
service-level agreement. Pull-request CI runs it on the two MVP targets:
Apple Silicon macOS and Windows x86-64. Each run retains `rust.json`,
`local-volatility-rust.json`, `python.json`, and `metadata.json` as a
short-lived workflow artifact.

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

## Local Volatility/VegaKT candidate workload

The suite also emits `local-volatility-rust.json` as candidate L8 evidence for
the Local Volatility/VegaKT slice. It records Rust compile and evaluate timings
for Price-only, AAD Local Vega, VegaKT decomposition with full bucket
covariance, and selected common-random-number bump workloads. The standalone
bump case evaluates five Price-only Plans: base, Spot-down/up, and uniform
Local-variance-down/up. It uses the same two-worker, Reduction block 256
execution policy as the replay fixture and records grid sizes, checkpoint
policy, AAD tile capacity, and covariance layout.

## Host and resource metadata

`metadata.json` records the commit, runner OS/architecture, platform and CPU
strings, Python version, and full `rustc -vV`/`cargo -V` output. Portable peak
memory and allocation counters are not currently available without changing
the measured process or allocator, so the corresponding fields are `null` and
the reason is recorded. Later instrumentation must add fields without silently
changing the v0.1 workload.
