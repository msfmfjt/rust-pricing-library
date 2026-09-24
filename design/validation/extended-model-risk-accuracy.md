# Extended-model AAD and VegaKT acceptance

This work starts from PR #70 at
`e389960f99fd6c86b9e5b975c5bd2d4e15879f30`, as requested. It adds a sensitivity
acceptance gate and leaves the inherited price-acceptance gate unchanged.
The deferred 2F constituent timestep case (ensemble SE 4.0124 bp versus 4)
remains a separate issue. No model algorithm, numerical policy, API, random
layout, fingerprint or price tolerance changes in this extension.

## What is measured

The [public-plan tests](../../crates/pricing/tests/extended_risk_acceptance.rs)
compare AAD against central differences of fully recompiled and recalibrated
prices. AAD, plus-bump and minus-bump use the same calibration seed, pricing
seed, grid, kernel, model parameters, factor ordering and payoff policy.
Three independent seed pairs are checked at each of two resolutions.

| Cases | Derivative contract |
| --- | --- |
| Deterministic-rate 1F/2F/rough LSV | Effective target Local-variance VJP: global, signed and isolated-node directions |
| 1F/2F/rough LSV + HW | Every explicit market-IV quote, parallel IV, Spot and selected discount/dividend log-DF pillars |
| Mixed 2F/1F and rough/2F multi-asset + HW | Every quote and parallel IV for each asset, both Spots and selected curve pillars |
| Joint 2F local correlation | Basket and constituent target-variance directions, BS sigma and both Spots |
| Joint 2F/rough local correlation + HW | Basket and LSV-constituent market-IV quotes, parallel IV, BS sigma, both Spots and selected curve pillars |

The deterministic-rate public API reports effective Local-variance risk. It is
not relabelled as market-IV VegaKT. For HW quote targets, the difference prices
regenerate both Dupire variance and T-forward log density before recalibration.
A separate check requires the full quote adjoint to differ measurably from a
variance-only transpose, guarding against an omitted density chain.

The market-IV surface has maturities `[0.5, 1]` and log-strike nodes `[-1, 0, 1]`
with a nonflat smile/term structure. The target grid uses log moneyness
`[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]`. Fixed cash plus proportional dividends
at time 0.37 exercise the affine carry paths. That date is explicitly included
between the uniform target times, as required by the single-asset affine API. Multi-asset
baskets use explicit compact-C2 payoff smoothing of width 3; finite differences
price that same surrogate. The single-asset payoff is an ordinary European call.

| Resolution | Calibration particles | Requested steps | Points per scramble | Scrambles |
| --- | ---: | ---: | ---: | ---: |
| Base | 1,024 | 4 | 256 | 4 |
| Refined | 2,048 | 8 | 512 | 4 |

Calibration seeds are 1709, 2903 and 4001. The master pricing seed is the
calibration seed XOR `0xd1b54a32d192ed03`; the second independently calibrated
asset uses calibration seed +101. Inserted dates can increase the compiled
grid. Refinement changes several numerical controls together and checks
derivative correctness at both resolutions; it is not a one-axis economic-risk
convergence claim.

## Acceptance rule

For each labelled input or direction, evaluate central differences at absolute
bumps `1e-5`, `1e-6`, `1e-7` and `1e-8` in that input's native units. The two finest
differences must each satisfy

```text
|AAD - FD| <= 2e-5 + 1e-4 * max(|AAD|, |FD|).
```

Their mutual difference must also be within twice the corresponding tolerance.
The two coarser bumps are retained as truncation/active-branch diagnostics; they
cannot be selected to rescue a failed finer comparison. All values must be finite.
The fast gate test rejects zeroed feedback, nonfinite values and inconsistent
finite differences. Independent-scramble risk errors must be finite/nonnegative
and have the correct shape, but do not widen the AAD/FD budget.

The price returned by AAD must equal the ordinary public-plan price. Quote
bucket sums must equal parallel Vega, and market scaling must equal raw Vega
times 0.01. Existing worker-replay, missing-trace, unsupported-risk and
analytic-limit regressions continue to run in the ordinary suite.

## Reproduction and evidence

```sh
cargo test --locked --release -p pricing --test extended_risk_acceptance -- --ignored --nocapture --test-threads=1
```

CI runs this command on all three native Rust platforms before the inherited
price gate. Each platform retains `extended-model-risk.log`, including the
commit/toolchain, every AAD/FD pair, eight bumped prices, four bump sizes, budgets,
seed pairs and joint-calibration projection/fallback counts.

The local Linux release run with pinned Rust 1.98.1 passed all 11 named tests:
66 scenarios, 738 directional/bucket sweeps and 5,904 bumped prices. Execution
took 158.97 seconds after compilation. Both required bump sizes passed every
comparison. The largest error was 3.82% of its allowed budget; the limit is 100%.

| Model group | Scenarios | Sweeps | Largest error / budget |
| --- | ---: | ---: | ---: |
| Deterministic-rate 1F/2F/rough | 18 | 72 | 3.72% |
| Single-asset 1F/2F/rough + HW | 18 | 180 | 0.57% |
| Mixed multi-asset + HW | 12 | 216 | 0.48% |
| Joint 2F local correlation | 6 | 42 | 0.37% |
| Joint 2F/rough local correlation + HW | 12 | 228 | 3.82% |

The [retained observations](extended-model-risk-observations.json) contain the
exact source hashes, toolchain, source base, all measured prices/derivatives,
seed pairs, diagnostics and initial fixture-qualification failures. The local
build uses `CARGO_INCREMENTAL=0`, `CARGO_BUILD_JOBS=1` and
`RUSTFLAGS='-C lto=off -C codegen-units=1 -C llvm-args=-threads=1'`, consistent
with the documented workspace workaround. These flags are not added to CI.

The all-features workspace regression run passed 516 tests; the minimal-feature
run passed 502, and the five statistical/path-dependence/early-exercise
acceptance tests passed. Clippy with warnings denied, formatting, API docs,
dependency direction, reference fixtures (67/114/61 checks), JSON schemas,
Markdown links and the committed source-archive check passed.
The inherited heavy price panel was deliberately not
rerun for this extension; its deferred failure is neither cleared nor softened.
Native CI results for this extension remain separate from these local results.

### Fixture qualification

An initial wider target grid spanning `[-0.8, 0.8]` encountered the existing
`non_positive_rate_corrected_variance` rejection for two refined rough/HW
combinations. This panel therefore uses the central grid specified above; it
does not claim derivative acceptance at the rejected tail nodes. The rejection
guard and its numerical policy are unchanged. The first single-asset HW attempt
also correctly rejected a target grid missing the cash-dividend date; the final
fixture supplies that required date explicitly.

The initial `1e-6` comparison showed changes between finite-difference scales
that largely disappeared at `1e-7`. The final panel retains those coarser
measurements and adds `1e-8`, requiring agreement at both finest scales for
every case. No seeds are selected or removed, and the absolute/relative error
budgets are unchanged. These initial findings are retained separately in the
observations, rather than counted as passing checks.

## Scope limits

These are derivatives of the finite regularized calibration-and-pricing
algorithm with fixed model parameters and locally fixed discrete decisions.
They do not establish infinite-particle convergence, optimal bandwidth,
statistical coverage across calibration seeds, physical quote fitting risk,
SV/HW parameter Greeks, unsmoothed digital/barrier Greeks or Gamma acceptance.
Conditional pricing-risk SE excludes calibration uncertainty. Passing this
panel does not resolve the deferred price-noise failure in its parent PR.
