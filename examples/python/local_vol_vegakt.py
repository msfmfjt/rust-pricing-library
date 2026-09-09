"""Notebook-style Local Volatility/VegaKT request construction."""

# %% Imports and market data
from datetime import date
import json

import numpy as np
import rust_pricing as rp

discount = rp.DiscountCurve(
    10,
    np.array([0.0, 0.5, 1.0], dtype=np.float64),
    np.array([1.0, 0.975, 0.95], dtype=np.float64),
)
dividend = rp.DiscountCurve(
    11,
    np.array([0.0, 0.5, 1.0], dtype=np.float64),
    np.array([1.0, 0.99, 0.98], dtype=np.float64),
)

# %% Contract, market, calibrated surface helper, and VegaKT request
product = rp.Product.european_vanilla(
    underlying_id=1,
    currency_id=2,
    expiry=date(2027, 9, 4),
    strike=100.0,
    notional=1.0,
    side="call",
)
market = rp.Market.equity(
    currency_id=2,
    underlying_id=1,
    spot=100.0,
    discount_curve=discount,
    dividend_curve=dividend,
)
model = rp.Model.local_volatility_from_standard_ssvi_power_law(
    np.array([0.25, 0.5, 1.0], dtype=np.float64),
    np.array([0.02, 0.03, 0.04], dtype=np.float64),
    0.02,
    -0.3,
    0.5,
    0.4,
    np.array([0.25, 0.5, 1.0], dtype=np.float64),
    np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float64),
    1.0e-8,
    4.0,
)
engine = rp.Engine.pseudo_monte_carlo(
    master_seed=7,
    independent_sampling_units=4096,
    antithetic=True,
)
risks = rp.RiskRequest(
    delta=True,
    gamma_relative_bump=0.01,
    vega=True,
    vega_kt_maturity_nodes=[date(2027, 3, 4), date(2027, 9, 4)],
    vega_kt_log_forward_moneyness_nodes=np.array([-0.2, 0.0, 0.2], dtype=np.float64),
    vega_kt_relative_density_threshold=1.0e-8,
    vega_kt_full_bucket_covariance=True,
    checkpoint_interval=16,
    aad_tile_capacity=256,
)

# %% Build the canonical request payload
request = rp.PricingRequest(
    valuation_date="2026-09-04",
    product=product,
    market=market,
    model=model,
    engine=engine,
    risk=risks,
)
round_trip = rp.PricingRequest.from_json(request.to_json())
payload = json.loads(round_trip.to_json())

# %% Inspect the materialized Local Volatility and VegaKT inputs
print(json.dumps(payload, indent=2))
print("request:", round_trip.fingerprint)
print("local variance shape:", payload["model"]["local_variance_grid"]["shape"])
print(
    "vega kt buckets:",
    len(payload["risk"]["vega_kt"]["maturity_nodes"])
    * len(payload["risk"]["vega_kt"]["log_forward_moneyness_nodes"]),
)
