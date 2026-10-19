# VegaKT hedging backtest: independent numerical validation v0.1

Date: 2026-09-29  
Status: **FAILED — H2/H3 acceptance remains open. Native backtest use is diagnostic only.**

This record corrects the interpretation of the earlier executable backtest
slice. Successful API replay and component tests did not establish the native
implementation's conformance to the paper decomposition. Earlier frozen LV
validation records are retained as historical evidence; their existence does
not resolve the end-to-end failures below. Synthetic P&L from the earlier
backtest example must not be used to rank hedge strategies.

## Reproducible panel

The [frozen policy](../../fixtures/backtesting/numerical-policy-v0.1.json) and
[runner](../../backtesting/rust_pricing_backtest/numerics.py) exercise a
zero-dividend, zero-rate European call with spot/strike 100, volatility 20%, and
128/365 years to expiry. The closed-form Black–Scholes oracle is independent of
the native implementation. The panel uses seeds 7/41/97, independent antithetic
sampling units 1,024/4,096/16,384, time intervals 2/4/8/16, and strike-grid nodes
9/17/33 over log-moneyness [-0.8, 0.8]. These are one-factor refinements around
4,096 × 8 × 17, not a full Cartesian sweep. There are 24 cases and 154 checks.

Common-random-number bumps of 0.0001 and 0.0005 shift the entire flat IV surface
and its matching local variance together. They do not assume calibrated quote
nodes are independent. Marginal bucket standard errors are not summed into a
parallel-Vega standard error. Quantity checks use a short target and calls with
strikes 95/105, fixed ridge and position bounds.

```shell
python -m rust_pricing_backtest.numerics --policy fixtures/backtesting/numerical-policy-v0.1.json --output benchmark-results/vega-kt-numerics
```

The command writes JSON and Markdown then exits 1 for a failed numerical gate.
The manually dispatched backtesting workflow retains the artifact even when
the job fails. The complete local per-bucket artifacts are under
`benchmark-results/vega-kt-numerics/{before,after}/`. The committed
[observations](../../fixtures/backtesting/numerical-observations-v0.1.json)
retain case totals, all check decisions, effective policy and binary hashes.
These are observations of a failing implementation, not golden accepted values.

## Defect corrected: density was summed as integrated sensitivity

The path reverse sweep already supplies integrated local-vol node adjoints
`A_i`. The old public execution path divided them by mass-lumped hat areas
`m_i` and passed `A_i/m_i` to a projection routine whose contract sums integrated
adjoints directly. For interior nodes on a uniform grid this introduced a
spurious `1 / delta_x` multiplier. Refining the grid doubled the total.

The correction passes the integrated node adjoints to the existing projection.
It restores that function's integration contract. It does **not** implement
the missing paper gamma pipeline or make this allocation a market-IV gradient.
The new native regression compares equal-width domains with different node
spacing and common simulation paths.

At seed 7, 4,096 sampling units and 8 time intervals:

| Strike nodes | Previous bucket sum | Corrected bucket sum | Corrected residual |
| ---: | ---: | ---: | ---: |
| 9 | 103.735374 | 20.747075 | 2.831866 |
| 17 | 207.470747 | 20.747075 | 2.831866 |
| 33 | 414.941494 | 20.747075 | 2.831866 |

Analytic Vega is **23.583438**. Native local parallel Vega is **23.578941**;
the paired bump is **23.578941**. The corrected bucket sum is still short by
**12.0101%**, exceeding the unchanged 3% threshold. Price, Delta and local Vega
remain unchanged in this comparison. Before: 60/154 checks failed. After:
54/154 failed; the six grid-total failures are resolved. Price/Delta analytic,
local-Vega CRN, price sampling-error reduction, and the configured seed/grid
quantity checks pass. The total-Vega and residual failures remain.

Native binary SHA-256:

- Before: `5fc31a9769a942030d4da26672e459cfefa40d9e84f8546229e3c6965277a00e`
- After: `ebe8262e8d941ce67f8a67c2e93fbe55bc22932a5cef1eacbbe93d6766c0f094`

The corrected binary was built in release mode; the preexisting binary's build
profile was not recorded. Their timings are not evidence of a speedup from the
correction. The installed shared environment was left unchanged; the corrected
wheel was unpacked separately for verification.

## Remaining mathematical blockers

1. **Initial-time adjoints are skipped.** With 2/4/8/16 time intervals, the
   observed residual fractions are 49.3067% / 23.9494% / 12.0101% / 6.4019%.
   The last is still above 3%. Assigning this residual arbitrarily to a tradable
   bucket is not a valid completion of the paper algorithm.
2. **The gamma pipeline is not connected.** The public path projects local
   adjoints directly. The equation (11) recovery and equations (3)–(5)
   transition/cell-integration primitives exist and have tests, but are not used
   in this execution path before equation (9) IV projection. The current paper
   policy label is therefore insufficient evidence of conformance.
3. **Residual metadata does not certify the decomposition.** The aggregate
   builder reconstructs residual as local parallel Vega minus bucket sum. This
   accounting identity holds even for the original defective output. It is not
   an independent sensitivity check. Edge statistics and time-dependent active
   domains also need to be accumulated through the actual paper calculation.
4. **Scalar conventions need reconciliation.** The native parallel local Vega
   and sum of canonical IV buckets are distinct in the present code. Passing
   the flat-vol total test will not prove that a non-flat surface has the right
   IV-risk definition or maturity/strike localization.

The next native implementation must connect the paper operators, derive the
initial-time treatment, and propagate genuine projection residual/edge
diagnostics. Acceptance then requires the unchanged panel plus vanilla
maturity/strike localization, non-flat surfaces, time/space refinement, and
hedge-error convergence. No independent quote-gradient substitute is accepted
under the current requirements.

## Guard and verification

`RustPricingProvider()` now raises `unvalidated_vega_kt` before importing the
native engine. `RustPricingProvider(diagnostic_only=True)` permits investigation
and marks every native risk, the manifest and the summary. The CLI and example
require `--diagnostic-only`. Finishing a replay only means its ledger finished.
The guard applies to the companion; the native library's existing public API
and wire formats are not disabled or relabelled in this bounded correction.

Locally verified:

- 25 companion ledger/optimizer/oracle/guard tests passed.
- 4 native adapter and four-strategy replay tests passed with the corrected wheel.
- 44 native Local Vol tests and 37 VegaKT primitive tests passed.
- Native release wheel build passed; remote CI has not been run.
- Independent numerical panel: **54 failures; not accepted**.

Tests of software mechanics and numerical acceptance are reported separately.
Thresholds were not relaxed to pass the existing implementation.

## Diagnostic capacity measurement

The [workload policy](../../fixtures/backtesting/performance-policy-v0.1.json)
declares 252 weekday sessions, 10 target contracts, 20 hedge candidates and a
5 × 9 reporting basis. All calls share one expiry beyond the history, so target
positions remain open at the end. Native pricing uses five local time intervals,
256 independent antithetic sampling units, seed 7 and one worker. The runner
verifies 20 candidates on each of 251 decisions and exactly 7,540 provider calls.
This is one VegaKT strategy run starting with an empty provider cache.

```shell
python -m rust_pricing_backtest.benchmark --policy fixtures/backtesting/performance-policy-v0.1.json --output benchmark-results/vega-kt-performance-full --diagnostic-only
```

Observed on macOS arm64, Python 3.13.1, corrected native release build:

| Measurement | Result |
| --- | ---: |
| Replay, including native adapter and optimization | 42.6045 s |
| Input generation + replay + serialization | 45.1185 s |
| Process peak RSS | 387.2969 MiB |
| Provider calls | 7,540 |

The [committed measurement](../../fixtures/backtesting/performance-observations-v0.1.json)
contains environment, policy/binary/source hashes and timing breakdown. The
complete result is local at `benchmark-results/vega-kt-performance-full/`.
The first diagnostic run excluded one illiquid candidate under ordinary spread
limits. It was not accepted as the declared complete-universe workload; the
benchmark now sets explicit spread/minimum-risk thresholds and checks counts.
This relaxes only diagnostic workload eligibility, not numerical acceptance.

No performance budget is approved, and 256 sampling units do not satisfy the
numerical gate. This measurement is not evidence of production accuracy,
tail-latency bounds, historical hedge effectiveness, or full H3 acceptance.
