# ADR 0042 · Exercise rights and continuous monitoring in Contract

Accepted for implementation on 2026-11-08. Builds on [ADR 0041](0041-contract-lifecycle-mc.md).

## Contract and pricing

Contract v2 records the holder's right to receive vanilla intrinsic value, its fixing identity and its cash currency. A Bermudan right enumerates legal dates. An American right records the beginning of its exercise period and the terminal fixing defines expiry. The date-based lifecycle can record exercise throughout that period. A continuous barrier specifies its inclusive window, direction, level, knock style, fixed cash rebate and payment date.

The single-asset task adapter lowers an exercise option to the native `AmericanVanillaSpec` and delegates independent training, valuation and fixed-strategy risk to the existing LSM solver. An American valuation requires an explicit discrete grid ending at expiry. A Bermudan valuation uses its contractual dates. The caller selects the regression basis and training engine with `LsmConfig`. Contract replay preserves legal rights even when two numerical tasks share a fingerprint because their exercise dates and all other inputs coincide.

Continuous monitoring delegates to the existing conditional Brownian-bridge survival estimator. Callers opt in to the model's interval law. The constant-volatility engines use their log-coordinate bridge. Local volatility uses the existing endpoint-variance approximation, simulation grid and affine-dividend jump handling. Live native option tasks support a single underlying, deterministic rates and native constant or local volatility. A live continuous bridge starts at valuation. A prior monitoring window requires a status report through valuation. Pricing a forward-starting monitoring window or a portfolio with nested unexercised rights requires a further native adapter. Unsupported combinations return errors.

## Lifecycle

An exercise notice identifies the option. A separate published fixing supplies its dated spot. Both are replayed before same-day confirmation or settlement. The notice replaces the remaining right with one intrinsic cashflow. At expiry the option settles its nonnegative intrinsic value. Unused past Bermudan dates lapse. American exercise accepts a notice on an eligible calendar date independently of the valuation grid. The lifecycle date resolution is one day.

Historical continuous monitoring requires a cumulative hit or no-hit report. An observed inclusive terminal touch establishes a hit. A no-hit report covers the window through its effective date. An absorbing hit can determine a rebate before its payment date. Contradictory reports are rejected. Simulated bridge weights never become historical events.

## Path storage

The forward evaluator allocates linear-tape registers according to last use, retaining outputs until collection. It uses the original SSA tape for reverse-mode derivatives and diagnostics. The compiler enables a compact register program when it reduces the register count and fits the graph's byte budget. Lazy control flow keeps memo flags only for shared subroutines. AAD records its execution trace on request. Pseudo-MC and RQMC LSM release the training matrices before allocating the valuation matrices.

## Compatibility and verification

V1 contracts retain their canonical JSON and fingerprints. V2 option terms and their lifecycle checkpoints carry explicit versions. The native task retains the materialized exercise grid or the continuous-monitoring product and can be replayed with the existing pricing-task reader. The option source retains the legal exercise style. See [Rust regressions](../../crates/pricing/tests/contract_advanced.rs), [installed-wheel tests](../../tests/python/test_contract_advanced.py) and the [paired benchmark](../../crates/pricing/examples/benchmark_contract_advanced.rs).
