# Escrowed simulation and dividend-IV conversion validation

Date: 2026-09-21. Base: PR #71, `6a9f78a098289f489c3fd47792d918a7aed4fe9b`.
Decision: [ADR 0009](../adr/0009-escrowed-simulation-and-iv-conversion.md).
Current API: [escrowed quote coordinates](../../docs/models/hull-white-cash-dividends.md).

This is a historical validation record for the implementation before
[ADR 0010](../adr/0010-remove-paid-cash.md). The paid-cash converter and its
dedicated tests described below have since been removed. The captured results
are retained for that earlier revision and do not validate the removal change.

## Change and independent checks

All equity simulations use the funded escrow convention. The paid-cash HW
simulation is removed. IV quotes can use either deterministic dividend
coordinate through a separate bidirectional conversion boundary.

The conversion tests cover both directions, zero/expiry-date and future cash,
mixed proportional payouts, nonzero r/q, exact no-cash/proportional-only
identity, round trips, changed strike axes, invalid domains and the IV Jacobian
against central differences. An independent Black implementation reprices the
returned physical strike. Tests require physical-price preservation within
`2e-12` currency units and round-trip IV within `3e-11` in the quoted test grid.
These are pointwise checks, not a claim about interpolation between quotes or
unidentifiable deep-tail/zero-vega options.

Simulation tests include deterministic funded Black and independent HW Gaussian
references, time-zero/expiry jumps, unfunded-schedule rejection, future cash
without horizon/driver extension, fixed-cash Spot bumps, bridge Delta/Vega,
LSV/rough quote-node reverses, and worker replay. Multi-asset HW and joint local
correlation reverse the reserve and calibration. The joint cash test includes
a payout beyond expiry, so positive-time stochastic reserve loadings remain
active; curve bumps preserve the entire schedule and cross-Gamma is compared
with independently recompiled bumped-Delta reports.

The independent synthetic cash-price panel's terminal affine coefficients are
updated algebraically: after all payouts, `A=0` and
`B=product(1-beta)-D*exp(-(r-q)*ex_time)/S0` for its one-event fixture. Historical
raw evidence under the prior convention is preserved. No acceptance tolerance,
pricing/calibration seed, particle count or required finite-difference scale is
relaxed for this change.

## Local execution

Local checks use Rust 1.98.1 on Linux x86-64 and CPython 3.12. As in the parent
accuracy work, the constrained build uses `CARGO_INCREMENTAL=0`,
`CARGO_BUILD_JOBS=1` and
`RUSTFLAGS=-C lto=off -C codegen-units=1 -C llvm-args=-threads=1`.
Those build settings are not committed as repository defaults.

| Check | Result |
| --- | --- |
| All-features Rust regressions | 516 passed, 0 failed; 44 heavy tests ignored in this command |
| Minimal-feature Rust regressions | 502 passed, 0 failed; 44 heavy tests ignored in this command |
| Recalibrated AAD/VegaKT acceptance | 11 tests, 66 scenarios, 738 sweeps and 5,904 bumped prices passed |
| Selected extended price acceptance | 2 tests covering 8 deterministic/HW smile cases passed |
| Statistical, path-dependence and early-exercise acceptance | 5 passed |
| Final-wheel Python tests | 93 passed, including independent paid-cash Black prices at 30 quote points |
| All-targets check, Clippy with warnings denied, Rust API docs | Passed |
| Wheel ABI, metadata, SBOM, RECORD and type-stub checks | Passed |
| JSON schemas, Markdown links and reference fixtures | Passed; reference fixtures have 67/114/61 checks |

Both mandatory finite-difference scales (`1e-7` and `1e-8`) pass for every
risk sweep. The maximum absolute error divided by its unchanged error budget
is **0.0371591** (3.716%). Raw risk/price observations, log hashes and source
hashes are retained in [the execution evidence](escrowed-dividends-observations.json).
The selected price tests are `single_asset::deterministic_lsv_smile_repricing`
and `single_asset::stochastic_rate_lsv_smile_repricing`; this is not the full
inherited stress panel.

The Rust checks and final wheel build completed before the work was resumed.
On continuation, their logs were inspected and the final wheel was reinstalled
and tested against the current Python suite. The continuation changes only
documentation, one Rust comment, the archive manifest and an independent Python
price test. The Rust toolchain and build cache were no longer available in that
session, so those Rust checks were not rerun. The final wheel also verifies the
last diagnostic-normalization fix through the existing multi-asset Python test.

## Regressions found and addressed

- Old coordinate fixtures assumed a negative paid-cash offset and unreserved
  Spot. Expected coefficients now follow the independent funded formula;
  future-payout tests now require a changed reserve and price.
- Funded schedules fail at market construction, so invalid-input tests inspect
  that boundary instead of waiting for an invalid simulated path.
- Pure proportional events initially inherited the stricter fixed-cash grid
  gate. They cancel from F, zeta and h, so their prior off-grid calibration
  behavior is preserved. Contractual observation-time checks still apply.
- The IV root check initially measured roundoff relative to the small option
  price. It now uses the actual CDF terms' floating-point error scale, retaining
  the price-domain checks and independent repricing assertions.

- The multi-asset diagnostic initially exposed internal initial-Spot units.
  It is normalized back to `E[Dbar*F/S0]` at the public boundary, preserving
  the existing value-one contract in Rust and Python.

## Limits

All quoted IVs fed into a pricing plan are escrow-coordinate IVs. Conversion
returns source-IV Jacobians; a market bump holding paid-cash source quotes
fixed must reconvert and recompile. Automatic Spot/curve conversion chains,
physical-index basket quote conversion, uncertain dividend amounts, dividend
payment dates distinct from ex-dates, and dividend-amount sensitivities are
outside this change.

Finite-calibration AAD differentiates the realized program with support donors,
active sets and digital tail membership fixed. Pricing standard errors exclude
calibration noise, kernel bias and time-discretization bias. Native CI and the
complete inherited stress panel remain separate from local fast regressions.
The parent PR's explicitly deferred one-year 2F constituent ensemble-SE gate
(4.012373 bp versus 4 bp) and its threshold are unchanged.

The broader model-boundary refactor and performance baseline are still pending.
This change keeps quote conversion outside path loops and caches the HW
market-risk transpose, but it does not establish a before/after timing or
peak-memory guarantee.
