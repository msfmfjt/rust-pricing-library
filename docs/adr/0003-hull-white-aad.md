# First-order AAD for the equity/Hull–White hybrid

Date: 2026-09-13. Status: experimental implementation decision.
The market-IV boundary below is extended by [ADR 0004](0004-hull-white-vegakt.md).

The user requested AAD after the stochastic-rate and escrowed-dividend extensions.
The deterministic-rate LSV reverse omits discounted calibration, stochastic bond
reserves and the quadratic leverage solve, so H5 needs its own reverse.

Add an explicit `evaluate_aad` method to the hybrid plan. Reverse the compiled
payoff, physical dividend observations and equity steps, then transpose the finite
particle calibration once per pricing mean (per scramble for RQMC). Transpose
reserve/scale coefficients and the payment discount into the initial curves.

| Impact | Decision |
| --- | --- |
| Active inputs | Spot, BS volatility, discount/dividend log-DF pillars, effective local-variance and paired forward-density samples |
| Fixed inputs | HW/Bergomi parameters, correlations, payout amounts/proportions, dates, grid axes, RNG and kernel/support settings |
| Calibration | Retain row checkpoints only when requested; differentiate particle motion, kernel moments, density denominators, escrow coordinates and the quadratic solve |
| Branches | Retain the existing price algorithm; its digital indicators, sort/support/donor choices and payoff branches have almost-everywhere derivatives with those decisions fixed |
| API | Rust `HullWhiteAadRisk`, recorded paths and calibration VJPs; Python `evaluate_aad`, `retain_reverse_trace=False` and immutable result properties |
| Compatibility | Existing price defaults and stable JSON remain unchanged; risk flags in the stable request still do not select hybrid Greeks |
| Replay | The trace opt-in changes the plan fingerprint; RNG coordinates and default price fingerprints do not change; fixed-block vector reductions preserve worker replay |
| Integrity | A retained trace fingerprints the public calibration arrays and rejects mutation before reverse |
| Evidence | Recompiled CRN Spot/curve/volatility/target bumps, paired-smile chain rule, Gaussian BS+HW Greeks, cash barrier jumps, deterministic limits, trace failures, MC/RQMC worker replay and Python ownership |
| Remaining | Gamma, HW/Bergomi parameter risk, dividend-amount risk, market-IV VegaKT/fitting adjoints, broad H5 risk-refinement acceptance and stable hybrid serialization |

The returned LSV target adjoints must be contracted through **both** variance
and density changes for a smile shock. This does not establish a market-IV
VegaKT mapping or an unbiased continuum Greek. See the
[AAD numerical contracts](../hull-white-aad-v0.1.md).
