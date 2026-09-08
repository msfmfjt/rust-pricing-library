"""Notebook-style European Black--Scholes valuation through the Python facade."""

# %% Imports and market data
from datetime import date
import json

import numpy as np
import rust_pricing as rp

discount = rp.DiscountCurve(10, np.array([0.0, 1.0]), np.array([1.0, 0.95]))
dividend = rp.DiscountCurve(11, np.array([0.0, 1.0]), np.array([1.0, 0.98]))

# %% Contract, model, engine, and risks
product = rp.Product.european_vanilla(
    underlying_id=2,
    currency_id=1,
    expiry=date(2027, 9, 4),
    strike=100.0,
    notional=1.0,
    side="call",
)
market = rp.Market.equity(
    currency_id=1,
    underlying_id=2,
    spot=100.0,
    discount_curve=discount,
    dividend_curve=dividend,
)
model = rp.Model.black_scholes(volatility=0.20)
engine = rp.Engine.pseudo_monte_carlo(
    master_seed=7,
    independent_sampling_units=4096,
    antithetic=True,
)
risks = rp.RiskRequest(delta=True, gamma_relative_bump=0.01, vega=True)

# %% Compile once, then evaluate without holding the Python GIL
request = rp.PricingRequest(
    valuation_date="2026-09-04",
    product=product,
    market=market,
    model=model,
    engine=engine,
    risk=risks,
)
plan = rp.PricingPlan.compile(request, worker_threads=2, reduction_block_size=256)
result = plan.evaluate()

# %% Structured result and diagnostics
print(json.dumps(json.loads(result.to_json()), indent=2))
print("plan:", plan.plan_fingerprint)
print("estimator:", result.diagnostics.estimator)
print("warnings:", [(warning.code, warning.message) for warning in result.warnings])
