# Andersen–Broadie primal-dual pricing

`pricing::dual::AndersenBroadiePlan` adds an opt-in Rust/Python price adapter to the
existing [LSM policy](early-exercise-calculation-specifications.md).
It returns a policy lower bound, a dual upper bound, their gap, standard errors
and an asymptotic 95% price bracket. These are statistical estimates: the
expectations bound the **declared finite exercise grid**. They do not establish
an upper bound for continuously exercisable options.

## Usage and scope

```rust,ignore
use pricing::dual::{AndersenBroadieConfig, AndersenBroadiePlan};
use pricing::mc::ExecutionPolicy;

// request: existing price-only AmericanVanilla PricingRequest with LsmConfig.
let nested = AndersenBroadieConfig::new(256, 128, 0x1234)?;
let plan = AndersenBroadiePlan::compile(
    &request, ExecutionPolicy::new(4, Some(64))?, nested,
)?;
let result = plan.evaluate()?;
```

Run the complete [put example](../../crates/pricing/examples/american_dual.rs):

```shell
cargo run --locked --release -p pricing --example american_dual
```

The Python boundary uses the same immutable `PricingRequest` and releases the
GIL during compilation and evaluation:

```python
import rust_pricing as rp

# request: price-only AmericanVanilla request with independent LSM training.
config = rp.AndersenBroadieConfig(256, 128, inner_seed=0x1234)
plan = rp.AndersenBroadiePlan.compile(
    request, config, worker_threads=4, reduction_block_size=64,
)
result = plan.evaluate()
print(result.lower_bound.value, result.upper_bound.value)
print(result.duality_gap.value, result.duality_gap.standard_error)
print(result.price_confidence_interval_95)
```

Run the complete [Python example](../../examples/python/american_dual.py) with
`python examples/python/american_dual.py`. It has the same inputs as the Rust
example. Each bound and the gap is an existing `DiagnosticEstimate`, exposing
`value`, `standard_error`, `confidence_interval`, `estimator`, and
`effective_sampling_units`. Result metadata includes `config`,
`policy_fingerprint`, `plan_fingerprint`, `outer_trajectories` (including
antithetic mates), and `exercise_date_count`.

Python config and compile failures use `ValidationError.issues`; evaluation
failures use `PricingError`. Negative or out-of-range unsigned integer arguments
raise `OverflowError` at the binding boundary. Config, plan, result and estimate
properties are read-only.

Supported: single-asset Call/Put, Black–Scholes or the existing Black-76 adapter,
deterministic curves, escrowed affine dividends, arbitrary declared dates
including valuation date, Pseudo-MC training and evaluation, and optional outer
antithetic pairs. Exercise at a dividend collision uses the existing post-dividend
convention. Config arguments are continuation-region inner paths, exercise-region
inner paths, and the inner seed. Counts must be positive.

RQMC, Local Volatility, stochastic rates/volatility, multi-asset products and
Greeks return explicit unsupported errors at this boundary. A versioned JSON
dual-result adapter is not included. Existing Rust/Python/JSON pricing APIs
and their results are unchanged.

## Calculation contract

The algorithm follows [Andersen and Broadie (2004), §3, equations (8)–(15)](https://business.columbia.edu/sites/default/files-efs/pubfiles/1233/primal_dual_ms_2004.pdf).
Train and freeze the existing LSM exercise rule. Write `Z_k = D_k h_k` for
discounted exercise payoff and `C_k` for the conditional discounted cashflow
obtained by skipping exercise at k, then following this rule. Inner paths
estimate `C_k`; they stop at the first future exercise recommendation.

On each outer path use `L_k = Z_k` if the rule exercises, otherwise `L_k = C_k`.
Start `A = 0`. At every date, first update
`gap = max(gap, Z_k - L_k - A)`, then, on a nonterminal exercise recommendation,
update `A += Z_k - C_k`. At maturity set `L_k = Z_k` without inner simulation.
Continue the outer path through maturity even after its lower-bound cashflow
has been selected. Average that cashflow for the lower estimate, and
`cashflow + gap` for the upper estimate.

This is a policy-value construction. It does not substitute the LSM regression
prediction for the inner conditional expectation. Finite inner sampling adds
upward bias to the expected upper estimator through the maximum operation.

## Uncertainty and replay

The returned estimates are conditional on one fitted policy. Lower payoff and
gap share an outer path. Upper standard error is therefore measured directly
from their sum, including covariance; it is not the square root of the sum of
their separate variances. With antithetics, each pair is one independent
sampling unit. The reported bracket uses the lower estimate's lower 95%
endpoint and the upper estimate's upper 95% endpoint (normal asymptotics).
It excludes continuous-exercise discretization and model error.

Scheme ID: `andersen-broadie-policy-nested-v1`. Existing training and valuation
Philox namespaces are reused unchanged. Inner paths reserve counter namespace
`0x4455414c`, separate from all existing `RandomDomain` IDs, even for equal
master seeds. With `stride=max(N2,N3)`, the path coordinate is
`outer_trajectory*stride + inner_path`; the dimension is
`branch_date*observation_count + future_observation`. The trajectory includes
the antithetic lane. Checked products reject coordinate overflow before execution.
Inner paths use chronological conditional Brownian increments. No outer future
shocks or inner stopping outcomes influence the frozen exercise rule.

The dual fingerprint hashes the scheme ID, existing simulation-plan fingerprint,
both inner counts and seed in big-endian format. The result also retains the
trained policy fingerprint. Worker count does not change estimates with fixed
reduction blocks. Zero volatility uses a single exact inner rollout.

Worst-case rollout work is O(outer paths × inner paths × exercise dates²).
Inner simulation is streamed; it does not allocate a nested path cube.
LSM training retains the existing matrix limits and typed numerical errors.

## Validation

See the [decision](../../design/adr/0027-american-dual.md) and
[validation record](../../design/validation/american-dual.md). The focused commands are:

```shell
cargo test --locked -p pricing --lib engine::risk::dual::tests
cargo test --locked -p pricing --test american_dual
cargo test --locked --release -p pricing --test american_dual -- --ignored --nocapture
python -m unittest discover -s tests/python -p test_american_dual.py -v
```
