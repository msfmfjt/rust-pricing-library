# Explicit escrowed cash dividends under Hull–White

Date: 2026-09-13. Status: experimental implementation decision.

The follow-on [AAD decision](0003-hull-white-aad.md) adds first-order risk through
this coordinate construction at fixed payout amounts and proportions.

The user requested dividend support after the initial HW extension. Proportional
dividends already worked; this change addresses scheduled fixed cash and mixed
cash/proportional payouts. Simply removing the cash rejection and reusing the
deterministic affine map would miss stochastic dividend present values.

Add an explicit escrowed model: split physical Spot into positive residual
equity and a reserve of Hull–White bonds for all future fixed payouts. Continue
the residual equity with the existing exact rate/log-Euler kernel. Reconstruct
physical pre/post-dividend Spot for the contractual graph. For LSV, calibrate a
deterministic escrow-coordinate target including the bond variance, equity/bond
covariance and stochastic-rate drift correction.

| Impact | Decision |
| --- | --- |
| Requirements | Implements the initial H4 cash-dividend scope; broad calibration acceptance remains open |
| Rust API | Add explicit `compile_bs_with_cash_dividends` and `compile_lsv_with_cash_dividends`, bond-reserve coordinate plans and conditional quadratic diagnostics |
| Python API | Add optional `cash_dividend_model="escrowed"`, result/plan model metadata and plan `risky_spot` |
| Compatibility | Default compile behavior and stable JSON remain unchanged; the new model is an opt-in because residual-equity volatility differs from spot volatility |
| Calibration inputs | Existing paired variance/density objects refer to the deterministic escrow coordinate when this mode is selected; ordinary spot-IV quotes must first be converted |
| Numerics | Exact bond reserve, positive residual equity, upper positive quadratic leverage root; no discriminant/variance clipping |
| Reproducibility | Same four Gaussian blocks; all in-horizon dividend nodes are required, and effective reserve coefficients enter the hybrid fingerprint |
| Tests | Jump continuity, rate loading, BS analytical value, LSV repricing with a post-expiry payout, worker replay, deterministic limits, barrier order and explicit failures |
| Pending | Hybrid AAD/DV01/VegaKT, dividend amount risk/calibration, separate dividend payment dates, broad H3/H4 refinement and stable hybrid wire format |

See the [cash-dividend calculation specifications](../../docs/models/hull-white-cash-dividends.md).
