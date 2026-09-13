"""American vanilla valuation with independent LSM training and valuation paths."""

from datetime import date

import numpy as np
import rust_pricing as rp

VALUATION_DATE = date(2026, 9, 4)
EXPIRY = date(2027, 9, 4)
EXERCISE_DATES = [
    date(2026, 12, 4),
    date(2027, 3, 4),
    date(2027, 6, 4),
    EXPIRY,
]

discount = rp.DiscountCurve(
    10,
    np.array([0.0, 1.0], dtype=np.float64),
    np.array([1.0, np.exp(-0.05)], dtype=np.float64),
)
dividend = rp.DiscountCurve(
    11,
    np.array([0.0, 1.0], dtype=np.float64),
    np.array([1.0, 1.0], dtype=np.float64),
)
product = rp.Product.american_vanilla(
    underlying_id=1,
    currency_id=2,
    expiry=EXPIRY,
    strike=100.0,
    notional=1.0,
    side="put",
    exercise_dates=EXERCISE_DATES,
)
market = rp.Market.equity(
    currency_id=2,
    underlying_id=1,
    spot=100.0,
    discount_curve=discount,
    dividend_curve=dividend,
)
training_engine = rp.Engine.pseudo_monte_carlo(
    master_seed=0x1020304050607080,
    independent_sampling_units=1024,
    antithetic=True,
)
valuation_engine = rp.Engine.pseudo_monte_carlo(
    master_seed=0x0123456789ABCDEF,
    independent_sampling_units=2048,
    antithetic=True,
)
lsm = rp.LsmConfig(training_engine, max_degree=3)
risk = rp.RiskRequest(delta=True, gamma_relative_bump=0.01, vega=True)


def evaluate(model: rp.Model) -> rp.PricingResult:
    request = rp.PricingRequest(
        valuation_date=VALUATION_DATE,
        product=product,
        market=market,
        model=model,
        engine=valuation_engine,
        risk=risk,
        lsm=lsm,
    )
    normalized = rp.PricingRequest.from_json(request.to_json())
    assert normalized.fingerprint == request.fingerprint
    return rp.PricingPlan.compile(
        normalized, worker_threads=2, reduction_block_size=256
    ).evaluate()


black_scholes = evaluate(rp.Model.black_scholes(0.20))
local_volatility = evaluate(
    rp.Model.local_volatility_from_grid(
        time_nodes=np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        log_forward_moneyness_nodes=np.array([-0.25, 0.0, 0.25], dtype=np.float64),
        local_variances=np.full(15, 0.04, dtype=np.float64),
        floor=1.0e-8,
        cap=4.0,
    )
)

for name, result in [
    ("Black-Scholes", black_scholes),
    ("Local Volatility", local_volatility),
]:
    exercise = result.early_exercise_diagnostics
    assert exercise is not None
    assert result.diagnostics.exercise_strategy_risk == "fixed_exercise_strategy"
    assert result.diagnostics.stopping_index_risk == "frozen_stopping_indices"
    restored = rp.PricingResult.from_json(result.to_json())
    restored_exercise = restored.early_exercise_diagnostics
    assert restored.to_json() == result.to_json()
    assert restored_exercise is not None
    assert restored_exercise.policy_fingerprint == exercise.policy_fingerprint
    assert restored_exercise.exercise_dates == exercise.exercise_dates
    assert restored_exercise.exercise_counts == exercise.exercise_counts
    assert restored_exercise.stopping_indices == exercise.stopping_indices
    assert restored_exercise.decisions[0].coefficients == exercise.decisions[0].coefficients
    print(name, "price:", result.value, "policy:", exercise.policy_fingerprint)
