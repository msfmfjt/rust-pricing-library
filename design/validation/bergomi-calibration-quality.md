# Supplementary one-factor Bergomi calibration diagnostics

This test-only panel preserves useful verification from
`9e017f71c961905919aa5b606077b44bd48238b8` (the former
`perf/bergomi-quality-speed` branch). Production calibration and path evolution
are supplied by the current implementation. No historical results JSON is
carried forward as evidence for the current revision.

The [Rust test](../../crates/pricing/tests/bergomi_calibration_quality.rs)
reuses each calibrated surface across 15 quotes and compares LSV and local-vol
paths with common spot noise. Its interleaved two-factor Brownian-bridge layout
is specific to this low-level diagnostic. It does not certify the public
pricing plan's random-coordinate layout, deterministic carry, or dividend
handling. Those contracts remain covered by
[extended-model price acceptance](extended-model-accuracy.md).

## Fixed panel and budgets

Spot is 100 with zero discount and dividend rates. Evaluation maturities are
0.25, 0.5 and 1 year; log strikes are -0.2, -0.1, 0, 0.1 and 0.2. The flat
surface is 20% IV. The skew surface is
`sqrt((0.04 + 0.004*T) * (1 - 0.25*x + 0.1*x*x))`. Market quote strikes
span -1 to 1 and the evaluation includes off-quote interpolation. Both local
variance targets must have no repairs.

Each surface uses four independent complete calibration-and-pricing runs,
with calibration seeds 42, 137, 711 and 2027. The corresponding pricing seed is
the calibration seed XOR `0xd1b54a32d192ed03`. The one-factor model has
mean reversion 2, volatility multiplier 0.7 and correlation -0.5. Every run uses
262,144 calibration particles, bandwidth 0.01, 1,024 time steps, 160 spatial
intervals and eight pricing scrambles of 32,768 points each.

The inherited numerical budgets are unchanged. One IV basis point is `1e-4`
in absolute volatility. These are acceptance limits, not measured outcomes.

| Metric | Limit |
| --- | --- |
| LSV absolute IV error / IV RMSE | 20 bp / 10 bp |
| Local-vol absolute IV error / IV RMSE | 10 bp / 5 bp |
| LSV total SE / conditional pricing SE | 5 bp / 3 bp |
| Worst calibration seed absolute IV error | 35 bp |
| Paired LSV-minus-local-vol price, divided by target vega | 15 IV bp |
| Target interpolation absolute IV error | 0.5 bp |
| Minimum ESS at both neighbors of each quote | 200 |
| Quote-neighbor fallbacks | 0 |
| Martingale deviation from 100, at each maturity | `4 * SE + 0.02` |
| Martingale SE | 0.02 |

Between-run uncertainty treats a complete calibration-and-pricing run as the
sampling unit; its variance already contains pricing noise. A pricing-only
floor prevents an accidentally small four-run variance from understating SE.
Invalid/no-time-value prices fail IV inversion; the panel never clips prices,
drops quotes or substitutes finite metrics for nonfinite results. Fallbacks
elsewhere in a full spatial row are reported but are not the quote-neighbor
fallback gate.

## Execution and evidence

Five ordinary Rust tests check the oracle/inversion, surface repairs,
uncertainty accounting, rejection gates and analytic target interpolation.
Existing workspace CI discovers them automatically. The Python tests exercise
incomplete logs, mismatched panels, invalid metrics and provenance handling:

```sh
cargo test --locked -p pricing --test bergomi_calibration_quality
python3 -m unittest discover -s scripts -p 'test_summarize_bergomi_quality.py' -v
```

Two expensive numerical gates are ignored by default. Run them explicitly
with a release build, serial test selection and complete stdout/stderr capture.
For Bash, preserve the producer's exit status through `tee`:

```sh
set -o pipefail
cargo test --locked --release -p pricing --test bergomi_calibration_quality \
  iv_round_trip_ -- --ignored --nocapture --test-threads=1 \
  2>&1 | tee bergomi-calibration-quality.log
```

Four further ignored tests are manual studies of refinement, bandwidth,
high vol-of-vol and error decomposition. Select them by exact test name when
needed. An unfiltered `--include-ignored` also executes those studies and is
not the release-gate command. No new scheduled or automatic heavy CI job is
introduced by this restoration.

The [summarizer](../../scripts/summarize_bergomi_quality.py) accepts only a
complete successful release log containing the two named numerical gates,
both smiles, all four seeds per smile, all 15 quotes, 48 raw estimates per run
and all three martingale/support dates. It rejects mismatched model/settings,
duplicate or missing entries, nonfinite/null metrics and failed/truncated test
completion. The Rust assertions own the numerical budgets: reports are
printed before those assertions, so report lines without a successful test
completion cannot be used as passing evidence.

Use an explicit output path and the actual tested commit, compiler and target.
The following example requires a committed, clean source checkout and a log
from the commands above on the same machine:

```sh
python3 scripts/summarize_bergomi_quality.py \
  --log bergomi-calibration-quality.log \
  --out bergomi-calibration-quality-current.json \
  --source-commit "$(git rev-parse HEAD)" \
  --platform "$(rustc -vV | sed -n 's/^host: //p')" \
  --compiler "$(rustc --version)"
```

When summarizing a foreign log, supply that run's commit/compiler/platform and
its exact Rust source via `--test-file`; local values would misattribute the
run. The generated document records SHA-256 hashes of the supplied test
source and log, complete raw runs, aggregate reports and the limited evidence
scope. The parser checks structural completeness and successful producer
completion, not independent source/build provenance or a second pricing
implementation. `--check` compares an existing document without writing it;
an invalid log is rejected before any output file is changed. Retain the log
with any generated evidence. Synthetic Python fixtures are parser tests only
and provide no numerical accuracy evidence.
