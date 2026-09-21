# ADR 0009: Escrowed simulation with separate bidirectional IV conversion

Date: 2026-09-20. Status: simulation accepted; IV conversion superseded by
[ADR 0010](0010-remove-paid-cash.md) on 2026-09-21.
Supersedes the simulation default in [ADR 0007](0007-affine-paid-cash-dividends.md).

## Decision

Every equity simulation uses escrowed fixed cash dividends. Future known
payouts are reserved even beyond the pricing horizon; residual equity must be
positive. Keep one evolution/calibration path, including deterministic rates,
shared Hull–White, Bergomi/rough-LSV and local correlation.

Both escrowed and paid-cash IV quotes remain supported as explicit input
conventions through a separate price-preserving bidirectional conversion API.
Paid-cash quote coordinates use deterministic initial curves. Remove the old
realized-rate paid-cash simulation. This paragraph records the original
decision; ADR 0010 subsequently removes the quote conversion as well. The current
[coordinate specification](../../docs/models/hull-white-cash-dividends.md)
defines the retained escrowed model.

## Impact

- **API:** existing default compilers now use escrowed dividends. Rust
  `*_with_cash_dividends` methods and Python `cash_dividend_model="escrowed"`
  are compatibility aliases for the same implementation. `None` also selects
  escrowed. New Rust/Python conversion APIs require both IV coordinates.
- **Serialized data:** payout/request schemas and quote-array shapes are
  unchanged. Their numerical interpretation is now escrowed. Old paid-cash IV
  data must be explicitly converted; the engine cannot infer its convention.
- **Numerics:** fixed-cash prices and Greeks can change; no-cash and proportional
  limits are preserved to floating-point accuracy. Unfunded schedules now fail
  at market construction. HW LSV uses the reserve diffusion and quadratic
  calibration. Joint basket calibration conditions on escrow quote coordinates
  and includes their rate diffusion and reserve drift shift.
- **Risk:** cash amounts stay fixed under Spot bumps. Reverse reserve and
  calibration effects feed Spot and both initial curves. Multi-asset Gamma
  recalibrates when those effects depend on Spot. Quote conversion exposes its
  diagonal IV Jacobian; market-bump conversion chains remain explicit.
- **Reproducibility:** dividend model fingerprints change. Ex-dates do not add
  Gaussian factors, and future-only payouts do not extend the pricing horizon.
  Worker-count replay and MC/RQMC uncertainty contracts remain required.
- **Performance:** quote conversion is outside simulation. Coefficient offsets
  and the market-risk transpose are compiled/cached. Additional recalibration
  is restricted to requested Gamma with cash-dependent HW models.
- **Gates:** revalidate funding/jumps, analytic limits, converted price/strike
  equality, both-direction round trips/Jacobians, recalibrated AAD, worker replay,
  Rust/Python facades and extended price/risk panels. Retain historical evidence
  under its original model convention; do not overwrite prior raw artifacts.

This change implements the user's dividend decision before the broader static
model-boundary refactor and performance-tuning work. It does not merge the
pending accuracy PRs or change the previously deferred CI threshold.
