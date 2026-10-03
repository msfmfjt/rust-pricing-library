# Rough Heston diffusion-coefficient cache

## Scope

This performance-only change starts from the six-family extension and Heston
validation stack at PR #131 (`df08a1273e63111f983aeeb784560bd5771daf46`).
It changes only the repeated history work for Rough Heston and Quadratic rough
Heston. The other four families, Fourier prices/Greeks/calibration, RNG layout,
compiled kernel, path/pricing fingerprints, public Rust/Python APIs and wire
contracts are unchanged. The [model definition](../../docs/models/rough-volatility-families.md)
and full-truncation/hybrid schemes are retained.

The pre-cache history loop recomputes each left-node diffusion coefficient for
every later node. Cache exactly these expressions once per left node:

```
Rough Heston:           fractional_scale * nu * sqrt(positive_variance[j])
Quadratic rough Heston: lambda * eta * sqrt(variance[j]) * fractional_scale
```

Multiplication association, subsequent multiplication by the innovation,
Neumaier summation order, drift terms and all finite checks are retained. The
last node is not a left-node input, so its unused coefficient is not computed.
This is neither a new discretization nor a change to positivity/truncation.

For n steps, diffusion-coefficient square roots fall from n(n+1)/2 to n.
Asset-evolution square roots are unchanged. Additional live working storage is
one n-element f64 vector (8n payload bytes and one allocation per path); at the
existing 2048-step cap, this is 16 KiB. The overall convolution remains O(n^2).
The cache is local to a path; it is not shared mutable state or retained in a
compiled plan. No new approximation, kernel table or sampling parameter is added.

## Arithmetic regression

`cache_tests.rs` retains the pre-cache Heston and quadratic recurrences, sharing
the current kernel and asset map deliberately. It is an arithmetic regression,
not an independent continuous-time price oracle. Compare every forward,
variance and latent-state bit plus diagnostic counts across 2304 path attempts: two
models, H=.01/.1/.3/.5, zero/moderate/high volatility randomness, uniform and
nonuniform grids, seeds91/1973 and antithetic signs. Negative raw variance must
actually occur. Some stress paths fail in the original recurrence; those must
return the exact same typed error, not an arbitrary error or a successful path.
Separate tests exercise zero/boundary parameters and identical
success/error behavior on overflow inputs, including a one-step grid.

The focused three-OS rough workflow runs these tests in debug, minimal and
release builds. Existing two-step independent pricing and estimator tests remain
active, along with the prior release refinement panel. No numerical tolerance
is changed. Existing workspace, wheel and primal-path tests provide additional
regression coverage; each delivery records which commands actually ran.

## Matched executable benchmark

Build `crates/pricing/examples/benchmark_rough_diffusion_cache.rs` with the same
pinned release toolchain against the parent and candidate production source.
The example accepts bounded path and repetition counts (defaults128 and5).
The recorded experiment uses 128 paths and3 timing repetitions per case:

- Rough Heston and Quadratic rough Heston; H=.1/.3/.5.
- 64/256/1024 time steps; t=j/n and t=(j/n)^2, T=1.
- Seed91, forward100. Heston(v0=.04,kappa=.7,theta=.055,nu=.18,rho=-.65).
  Quadratic(z0=.15,lambda=1.1,eta=.5,a=.8,b=.25,c=.02).

There are36 cases per executable. Compilation, random-number generation and
full-path hashing occur outside the timed interval. Timings include path
allocations, variance history and asset evolution, but **not** payoff, discount,
thread scheduling or Monte Carlo reduction. Each timed result is black-boxed.
Every untimed warmup path contributes all three state arrays and diagnostic
counts to a BLAKE3 digest. Digest, scheme and plan fingerprint must be identical
between builds. Do not substitute a terminal-payoff-only check for this replay.

Run baseline/candidate/candidate/baseline as separate processes while no local
compiler or test suite is running; retain every case, including slowdowns.
On Linux the recorded experiment pins each process to one permitted CPU. This
reduces migration, not interference from a shared host or variation in frequency.
The comparator computes per-case ratios of median times and a geometric mean,
checks complete matched cases, and does not enforce a wall-clock CI threshold.
Its mutation tests reject changed outputs, seeds, counts, missing/duplicate
cases and invalid measurements. A slower candidate must remain visible.

```
cargo build --locked --release -p pricing --example benchmark_rough_diffusion_cache
# Retain each build's executable separately, then run serially:
./baseline-benchmark 128 3 > baseline.json
./candidate-benchmark 128 3 > candidate.json
python scripts/compare_rough_diffusion_benchmarks.py baseline.json candidate.json
```

Measured timings and runtime/compiler identity belong in retained delivery
logs, not a hardware-independent performance promise. Any fixed paths shared
with training/validation are replay inputs, not new market data.

## Limits

This change cannot reduce the existing MC time-grid or full-truncation bias:
it intentionally computes the same paths. In particular, earlier rough-Heston
H=.1 discrepancies versus Fourier remain. It does not make a 5-IV-bp admission
claim. Accuracy/weak-scheme changes must be benchmarked separately at fixed
sampling and numerical budgets; do not combine them silently with this cache.
A path microbenchmark is not end-to-end throughput, and extra per-path memory
must be considered at large worker counts. Finite bitwise replay on one platform
is not a universal cross-architecture identity theorem. The models remain
experimental and price-only in the MC adapter.
