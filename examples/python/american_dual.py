"""Andersen–Broadie bounds for a four-date put, matching the Rust example.

The statistical bounds apply to these exercise dates and are conditional on
the fitted policy. They do not bound continuous-exercise discretization error.
"""

import json
import math

import rust_pricing as rp


def main() -> None:
    request = rp.PricingRequest(
        valuation_date="2026-09-04",
        product=rp.Product.american_vanilla(
            underlying_id=1,
            currency_id=1,
            expiry="2027-09-04",
            strike=100.0,
            notional=1.0,
            side="put",
            exercise_dates=["2026-12-04", "2027-03-04", "2027-06-04", "2027-09-04"],
        ),
        market=rp.Market.equity(
            currency_id=1,
            underlying_id=1,
            spot=100.0,
            discount_curve=rp.DiscountCurve(1, [0.0, 1.0], [1.0, math.exp(-0.05)]),
            dividend_curve=rp.DiscountCurve(2, [0.0, 1.0], [1.0, 1.0]),
        ),
        model=rp.Model.black_scholes(0.2),
        engine=rp.Engine.pseudo_monte_carlo(71, 4096, antithetic=True),
        risk=rp.RiskRequest(),
        lsm=rp.LsmConfig(
            rp.Engine.pseudo_monte_carlo(17, 8192, antithetic=True),
            max_degree=3,
            max_basis_columns=8,
            max_total_exponents=8,
            max_matrix_elements=4_000_000,
        ),
    )
    config = rp.AndersenBroadieConfig(
        continuation_inner_paths=256,
        exercise_inner_paths=128,
        inner_seed=0x1234,
    )
    plan = rp.AndersenBroadiePlan.compile(
        request, config, worker_threads=4, reduction_block_size=64,
    )
    result = plan.evaluate()
    # Presentation only: the dual result has no versioned JSON wire adapter.
    print(json.dumps({
        "method": "andersen-broadie-policy-nested-v1",
        "scope": "declared four-date Bermudan grid; conditional on trained policy",
        "lower_bound": result.lower_bound.value,
        "lower_standard_error": result.lower_bound.standard_error,
        "upper_bound": result.upper_bound.value,
        "upper_standard_error": result.upper_bound.standard_error,
        "duality_gap": result.duality_gap.value,
        "gap_standard_error": result.duality_gap.standard_error,
        "price_confidence_interval_95": result.price_confidence_interval_95,
        "outer_trajectories": result.outer_trajectories,
        "policy_fingerprint": result.policy_fingerprint,
        "plan_fingerprint": result.plan_fingerprint,
    }, indent=2))


if __name__ == "__main__":
    main()
