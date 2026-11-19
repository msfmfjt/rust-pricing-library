# Lean verification of the pricing library

The audit covers main at `2fadb7352972ade46b42ed8cdc87a3a38bcf1a86` and PR #216 at `9966d8053aa2951ed86248722c1d78cb4a25884b`. Two additional defects in public numerical APIs reproduce on both revisions. All 117 named Lean theorems passed. The strict replay of the two new counterexamples returned exit code 1 in both debug and release builds.

## Reproduced numerical defects

| Public operation | Reproducer | Mathematical contract | Rust result |
| --- | --- | --- | --- |
| `evaluate_black_forward` | Unit forward, strike, discount factor, notional and maturity. Set volatility to `2^-60`. | For either option side the ATM price is the integral of the standard normal density from `-σ/2` to `σ/2`. Lean proves a lower bound of `2^-63`. The price is approximately `3.460272696860437e-19`. | Call returns positive zero and put returns negative zero. Both CDF values in the subtraction round to the same value. At volatility 0.2 both prices are approximately 0.079655674554058. |
| `Complex64::div` | Evaluate `(M + iM) / (1 + i)` with `M = f64::MAX`. | The exact quotient is `(M, 0)`. Lean proves representability of `M` and that doubling it exceeds the finite binary64 exponent range. | The real numerator overflows before division by 2. Rust returns `(+∞, 0)`. The control with `M = 1` returns `(1, 0)`. |

The Black price itself lies well within the normal floating-point range. These are three failing public-API observations across two defects. The numerical checks keep reporting a failure while these inputs violate the contracts.

The earlier two discrepancies remain reproducible on main. For a smoothed up barrier at `x = 1 − 2^-25`, the complementary weight is `−2^-51` and its logarithm is NaN. For a cash dividend of 64 and forward-state input `2^-47`, the original validator rejects a positive post-dividend spot of `2^-48`. The corresponding price/delta and funded-dividend regression tests pass on PR #216.

## Coverage on PR #216

| Verification layer | Corpus | Result |
| --- | --- | --- |
| Named Lean theorems | 117 in Lean 4.19.0 with mathlib v4.19.0, including the 30 earlier barrier/dividend results | Checked by the kernel. The root module audits transitive dependencies against `propext`, `Classical.choice` and `Quot.sound`. |
| Rust MIR, all three library targets with all features | 25,144 bodies, 252,854 basic blocks, 367,510 edges | Exact structural reachability for six event classes, from every block in each extracted graph. The compiler lists 380 Rust source files in these builds. |
| Rust MIR, all three unit-test targets with default features | 29,274 bodies, 305,197 blocks, 444,775 edges | The unit builds include all 412 Rust source files under the three `src/` directories. Bodies in the two builds overlap. |
| Backtesting package | All 41 shipped Python modules, 577 CPython 3.12 code objects, 63,303 instructions | Six structural reachability checks on the extracted bytecode graphs, including exceptional edges. |
| Scalar getters | 194 eligible Rust methods in 50 source files | Every emitted field read matches the separately parsed source declaration. Each generated theorem quantifies over arbitrary field contents. |
| Wire schemas | All 43 published JSON schemas, including 215 `oneOf` nodes and 1,638 references | All choices have a required discriminator with disjoint values. Reference targets, required keys and enum uniqueness checks yield 691 distinct kernel-checked statements. |
| Retained AAD/finite-difference observations | 1,484 sweeps across 11 case names, found in a scan of 246 JSON files | Exact rational re-evaluation of the recorded binary64 inputs agrees on all 1,476 accepted sweeps and reproduces the 8 failed initial probes. |
| Philox4x32-10 | Arbitrary 128-bit counters for each fixed key, plus 180 blocks from the public Rust API | Injectivity by round inversion. All 720 output words match the Lean model and all 720 native uniform conversions match the exact midpoint bits. |
| Calendar, typed numbers and IDs | 3,652,059 valid proleptic Gregorian dates, 20,480 floating-point patterns, all 65,536 currency IDs | Native checks of date order, weekdays, ACT/365F and ACT/360 to or from the first date, signed zeros, subnormals, infinities, NaNs and checked ID increments. All three number constructors were tested on each floating-point pattern. |

[Source coverage for every shipped Rust and Python file](results/source-coverage.tsv) · [Scalar getter proofs by method](results/scalar-getters.tsv) · [PR #216 results and native output](results/summary.json) · [Main results and native output](results/main-summary.json)

On main, the ordinary library builds produced 25,141 MIR bodies and 252,839 blocks. The unit builds produced 29,263 bodies and 305,076 blocks. The scalar getter analysis covered 193 methods on main and 194 on PR #216. The source catalog selects the dyadic witness corresponding to each revision of the barrier implementation.

The standard Rust regression suites passed 1,191 tests, with 91 opt-in tests ignored by the default harness. All 330 backtesting Python tests passed. The supplemental CPython binding run recorded 635 passing test methods across 114 completed modules. Of the 740 discovered binding test methods, 105 have no final verdict in this snapshot. The full per-module status for the remaining simulation panels is in the results JSON. Four negative controls changed a field index, a Philox bit, a risk verdict and a discriminator tag. The Lean kernel rejected all four corrupted statements.

## Mathematical contracts

| Area | What the theorems establish |
| --- | --- |
| Pricing and interpolation | Black price/delta parity under a symmetric CDF, the vega/gamma relation, positive Gaussian ATM time value, forward carry, PCHIP knot and affine reproduction, SSVI ATM variance, the flat-slice Durrleman value. |
| Product graph and risk | For any finite composition of literal, spot, addition, subtraction, multiplication, division, negation, exponential and logarithm, the modeled reverse pass returns the seed times the analytic derivative. Shared spot occurrences accumulate. Denominators must be nonzero and logarithm inputs positive. Centered gamma is exact on quadratics and has an explicit quartic bias. |
| Barrier and cash dividends | Bounds and parity for arbitrary finite barrier histories, monotonicity of bridge hit probability with variance, complementary quintic smoothing, and reserve settlement and positive post-event spots for funded cash/proportional schedules. |
| Process models | Hull–White bond-loading composition including zero reversion, reconstruction of Gaussian driver covariances, the LSV leverage identity, Bergomi two-factor normalization and its singular case, lifted-Heston nonexpansion for a common increment, quadratic rough-Heston variance and mixed rough-Bergomi positivity and flat-vol-of-vol limits. |
| Numerical and Monte Carlo methods | Welford folds, Chan and covariance merges, Simpson exactness on cubics, Richardson cancellation, a damped normal-equation coordinate and complex arithmetic. On finite scenarios, the mean squared error is variance divided by sample count under equal diagonal and zero cross moments. Other results cover a weighted tail bound and the first-exercise dual gap. |
| Discrete and representation properties | Gregorian leap periods, ACT additivity, bounded ID exhaustion, binary64 sign/exponent/fraction splitting and sign guards for every encoding, open-interval samples, and Brownian bridge transpose consistency. |

[Mapping from source operations to mathematical contracts](contracts.tsv) · [Lean theorem and axiom manifest](PricingAudit.lean)

## Interpreting the coverage

The inventory counts 6,256 explicitly declared Rust functions and methods, including inline tests, and 295 Python functions and methods. Compiler-generated functions, CTFE duplicates, promoted constants, static items, lambdas and comprehensions use different counting units. The coverage table records literal source hashes, compiler dependency maps and matched contracts. Of the 453 source files, 67 have a mapped mathematical operation or a directly checked scalar getter. A dash in the specific-contract column and a zero getter count identify files with structural coverage only in this inventory.

The Rust graph categories are normal return, index guards, integer zero divisors, integer overflow, pointer alignment and null guards. The Python categories are return, explicit raise, conditional jump, back edge, call and exception-handler entry. Each nonzero rank has a strictly descending path to a marked block. A zero rank excludes such a path. Lean checks the local obligations for each distinct input and proves their relationship to reachability by induction. Calls are intraprocedural boundaries. The runtime predicate and the feasibility of a branch require separate proofs. Extraction relies on the pinned compiler, the MIR/bytecode parsers and the compiler dependency maps.

The getter audit covers same-file, unambiguously declared structs with a single-receiver inherent method that returns a primitive scalar via a single `self.field` expression. Independent Rust AST and MIR readers supply the field indices. The remaining mathematical contracts formalize specified formulas and hypotheses. Whole-engine Rust-to-Lean refinement, joint floating-point error bounds, particle-calibration convergence, strong/weak error of rough-volatility discretizations, and optimality of the fitted LSM policy are open. The stored risk sweep check replays the acceptance math on retained observations. The native probes run the current implementation.

## Reproduce

Use the repository's Rust 1.98.1 toolchain, Lean 4.19.0 via elan, and CPython 3.12 with its embedding library available to the linker. Run from the repository root.

```sh
cd verification/lean
lake exe cache get
python3 tools/verify.py
```

The runner writes stage logs, generated Lean statements, per-record mappings, and a machine-readable summary under `.lake/audit/`. Exit 0 means every stage passed, exit 1 identifies a native contract violation, and exit 2 identifies an incomplete or invalid verification run. A narrower replay of the two public API defects is available through the probe crate.

```sh
cargo run --locked --release --manifest-path rust-probes/Cargo.toml -- --strict
```

The prior barrier and dividend contracts can be replayed on either revision with this separate command. The result is two discrepancies on main and zero on PR #216.

```sh
cargo run --locked --release --manifest-path rust-probes/Cargo.toml --bin repaired_tails
```

To audit the same main revision, copy the verifier from the audit commit into a detached worktree. Run these commands from the root of a checkout that contains this directory.

```sh
git worktree add --detach ../pricing-main-audit 2fadb7352972ade46b42ed8cdc87a3a38bcf1a86
git archive HEAD verification/lean | tar -x -C ../pricing-main-audit
cd ../pricing-main-audit/verification/lean
lake exe cache get
python3 tools/verify.py
```
