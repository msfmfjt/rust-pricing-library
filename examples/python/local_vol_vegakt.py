"""Notebook-style Local Volatility/VegaKT valuation."""

# %% Imports and market data
from datetime import date

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

VALUATION_DATE = date(2026, 9, 4)
FIRST_BUCKET_MATURITY = date(2027, 3, 5)
EXPIRY = date(2027, 9, 4)


def year_fraction(start: date, end: date) -> float:
    return (end - start).days / 365.0


# %% Contract, market, explicit Local Volatility grids, and VegaKT request
product = rp.Product.european_vanilla(
    underlying_id=1,
    currency_id=2,
    expiry=EXPIRY,
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
bucket_maturities = np.array(
    [
        year_fraction(VALUATION_DATE, FIRST_BUCKET_MATURITY),
        year_fraction(VALUATION_DATE, EXPIRY),
    ],
    dtype=np.float64,
)
local_vol_times = np.array(
    [0.0, bucket_maturities[0], bucket_maturities[1]], dtype=np.float64
)
local_vol_log_moneyness = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float64)
local_variances = np.array(
    [
        0.038,
        0.039,
        0.040,
        0.041,
        0.042,
        0.037,
        0.039,
        0.040,
        0.042,
        0.044,
        0.036,
        0.038,
        0.041,
        0.044,
        0.047,
    ],
    dtype=np.float64,
)
reporting_log_moneyness = np.array([-0.2, 0.0, 0.2], dtype=np.float64)
reporting_implied_volatilities = np.array(
    [
        0.195,
        0.200,
        0.207,
        0.190,
        0.202,
        0.215,
    ],
    dtype=np.float64,
)
vega_kt_maturity_dates = [FIRST_BUCKET_MATURITY, EXPIRY]
model = rp.Model.local_volatility_from_grid_with_reporting_basis(
    local_vol_times,
    local_vol_log_moneyness,
    local_variances,
    1.0e-8,
    4.0,
    bucket_maturities,
    reporting_log_moneyness,
    reporting_implied_volatilities,
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
    vega_kt_maturity_nodes=vega_kt_maturity_dates,
    vega_kt_log_forward_moneyness_nodes=reporting_log_moneyness,
    vega_kt_relative_density_threshold=1.0e-8,
    vega_kt_full_bucket_covariance=True,
    checkpoint_interval=16,
    aad_tile_capacity=256,
)

# %% Build the canonical request payload
request = rp.PricingRequest(
    valuation_date=VALUATION_DATE,
    product=product,
    market=market,
    model=model,
    engine=engine,
    risk=risks,
)
round_trip = rp.PricingRequest.from_json(request.to_json())

# %% Inspect the materialized Local Volatility and VegaKT inputs
print(round_trip.to_pretty_json())
print("request:", round_trip.fingerprint)
print("request schema bytes:", len(rp.request_json_schema()))
print(
    "vega kt buckets:",
    len(vega_kt_maturity_dates) * len(reporting_log_moneyness),
)

# %% Evaluate and inspect the VegaKT result report
plan = rp.PricingPlan.compile(round_trip, worker_threads=2, reduction_block_size=512)
result = plan.evaluate()
vega_kt = result.vega_kt
assert vega_kt is not None

print("price:", result.value)
print("standard error:", result.standard_error)
print("delta:", result.delta_raw)
print("gamma:", result.gamma_raw)
print("vega:", result.vega_raw)
print("vega kt scalar:", vega_kt.projection.scalar_vega)
print("vega kt units:", vega_kt.raw_unit, "=>", vega_kt.market_scaled_unit)
print("first bucket:", vega_kt.estimates[0].raw_mean)
print("covariance layout:", vega_kt.covariance_layout)
