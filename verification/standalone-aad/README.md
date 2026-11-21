# Standalone AAD verification and benchmarks

Run the numeric comparison and application cases locally with Rust 1.98.1
and Python 3.9+. Check out rust-aad at the revision in this package's
Cargo.toml. From this repository root, with its checkout at ../rust-aad:

```sh
python3 ../rust-aad/benchmarks/run.py --manifest verification/standalone-aad/Cargo.toml --binary native_matrix --backend pricing-native --profile full --output /tmp/pricing-native.jsonl
python3 verification/standalone-aad/model_matrix.py --aad-root ../rust-aad --output /tmp/pricing-models.jsonl
```

The public native payoff compiler receives the identical graph, paths and
seeds as the standalone backend. Forward uses one reusable workspace. Native
reverse takes one output at a time and returns owned observation adjoints.
The adapter sums these with the supplied seeds and includes that work in
the timed call. The production source under crates/ is identical to the
baseline at f82a8b2c3fb2231959ae4104ec30778cdba683a1.

The model executable measures the Pricing Library's native algorithms. All
results carry 11 warmed wall times, separate heap counters, whole-process RSS
and host/build metadata. The path workload uses 32 Philox paths with seed
971, 120 steps and a terminal-state seed of one. Normal generation is outside
the timer. A path record contains the means of terminal forward, initial
forward and factor adjoints, squared-leverage grid adjoints, and spot and
orthogonal shock adjoints, in that order. Every path checks a joint central
difference in the initial forward, leverage and all Gaussian draws. The 1F
and 2F runs use their respective factor-major shock layouts.

The calibration case measures 256 particles (seed 8401) and the separate
pullback on the retained calibration. It checks every target-grid adjoint by
recalibrating under symmetric bumps. The hybrid case times plan construction
with calibration, price, and price plus native risks including market-IV
mapping. It uses 64 independent antithetic units with Brownian bridge and
one pricing worker, and retains every labeled derivative and price standard
error. The benchmark compares its price to the price-only entry point.
All four cases check every reported field against the fixed local reference
in model_fixtures.json with absolute tolerance 2e-10 × (1 + |reference|).
The path means are diagnostic aggregates. The native paths and calibration
pullbacks are separate from the portable numeric AD graphs.
