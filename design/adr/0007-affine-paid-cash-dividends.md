# Use paid-cash affine dividends by default with Hull–White

Date: 2026-09-14. Status: implemented, experimental model contract.
Baseline: main `244678f80f1f0c344d2d86b55ec8ceb874ce59e6`.

## Context and decision

The user requested the deterministic engine's continuous-equity / affine event
convention for HW rather than a future-dividend bond reserve. HW does not require
escrowing dividends. Preserve explicit escrowed entry points as a different model,
not an alias for the new default. Supersede the *default fixed-cash rejection* and
*cash always means escrowed* statements in ADRs 0001–0005 and the v0.1 HW/rough
contracts. Their escrowed equations continue to govern explicit escrowed mode.

Use continuous normalized equity U with initial value S0. Physical Spot is
`S(t)=b(t)*G0(t)*U(t)+c(t)`, where `G0=Q0/P0` and b is the product of already-paid
proportional factors. The paid-cash offset c carries at realized r-q, and at each
ex-date updates to `(1-beta)*c-D`; b updates to `(1-beta)*b`. Do not reserve future
cash or reinterpret a residual-equity smile as a continuous-equity smile.

The deterministic transform must also carry its A coefficient at deterministic
r-q. B remains piecewise constant because the existing f state already carries.
Keeping the previous constant cash offset would reproduce an incorrect physical
forward, not the zero-rate-volatility limit of this model.

## Impact analysis

- API: existing default Rust BS/LSV/rough constructors accept in-horizon fixed
  cash. Python omission/None selects this default. The only named legacy option
  remains `cash_dividend_model="escrowed"`; no new keyword or schema variant.
- Wire: schema v3, JSON migrations, dividend event validation and fixed-cash quote
  semantics are unchanged. Experimental adapter methods remain outside the wire.
- Numerical: default dividends are paid-cash offsets, not bond reserves. Standard
  BS/LV/LSV physical observations and forward now include post-ex-date cash carry.
- Calibration: default HW LSV/rough-LSV fits the continuous U target via the
  no-reserve discounted-particle equation; no escrow quadratic/covariance terms.
- Risk: reverse the offset recurrence and curve interpolation before the existing
  path/calibration VJP. D and beta remain fixed under Spot/curve/IV bumps.
- Reproducibility: a cash-carry numerical fingerprint tag distinguishes changed
  deterministic plans; HW additionally records `affine-paid-cash-realized-carry-v1`.
  No-cash/proportional behavior is retained. Cash dates use the existing step RNG
  blocks, not a new jump driver; future-only cash must not change the active grid.
- Compatibility: old explicit escrowed prices, Greeks, metadata and future-reserve
  rules remain. Historical expectations that default cash compilation fails are
  intentionally replaced with successful affine compilation assertions.

## Verification and limits

See [the numerical/API contract](../../docs/models/hull-white-affine-dividends.md) and
`crates/pricing/tests/hull_white_affine.rs`. Test deterministic limits against an
independent carried-cash forward and shifted-Black reference; check discounted
gains pathwise, event collisions, missing cash dates, nonpositive paths, future
payout exclusion, all curve-pillar AADs, recalibrated VegaKT and worker/RQMC replay.

The model is still experimental. Physical-spot quote conversion, dividend-amount
risk, HW parameter/correlation risk, Gamma, American exercise and continuous
Barrier monitoring in the HW adapters are not added by this change. Reject
nonpositive observed Spot instead of clipping, resampling or switching models.
