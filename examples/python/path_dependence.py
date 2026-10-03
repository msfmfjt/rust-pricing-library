"""Digital smoothing and continuous Barrier pricing with declared hit history."""

# %% Market and contract
import numpy as np
import rust_pricing as rp

discount = rp.DiscountCurve(10, np.array([0.0, 1.0]), np.array([1.0, 0.95]))
dividend = rp.DiscountCurve(11, np.array([0.0, 1.0]), np.array([1.0, 0.98]))
market = rp.Market.equity(2, 1, 100.0, discount, dividend)
product = rp.Product.digital(
    1,
    2,
    "2027-09-04",
    100.0,
    10.0,
    "call",
    "cash",
)
model = rp.Model.black_scholes(0.20)
engine = rp.Engine.pseudo_monte_carlo(7, 4096, antithetic=True)

# %% Exact contractual Price
exact_request = rp.PricingRequest(
    "2026-09-04",
    product,
    market,
    model,
    engine,
    rp.RiskRequest(),
)
exact = rp.PricingPlan.compile(
    exact_request, worker_threads=2, reduction_block_size=256
).evaluate()
assert exact.diagnostics.valuation_kind == "exact_contractual"
assert exact.diagnostics.payoff_smoothing_half_width is None

# %% Smoothed Price/Greeks and an explicitly ordered, non-adaptive ladder
smoothed_request = rp.PricingRequest(
    "2026-09-04",
    product,
    market,
    model,
    engine,
    rp.RiskRequest(
        delta=True,
        gamma_relative_bump=0.01,
        vega=True,
        payoff_smoothing_half_width=3.0,
        payoff_smoothing_width_ladder=[4.0, 2.0, 1.0],
    ),
)
smoothed_plan = rp.PricingPlan.compile(
    rp.PricingRequest.from_json(smoothed_request.to_json()),
    worker_threads=2,
    reduction_block_size=256,
)
ladder = smoothed_plan.evaluate_width_ladder()
assert ladder.primary.diagnostics.valuation_kind == "smoothed_surrogate"
assert ladder.primary.diagnostics.payoff_smoothing_half_width == 3.0
assert [entry.half_width for entry in ladder.entries] == [4.0, 2.0, 1.0]

# %% Structured comparison
print("exact price:", exact.value)
print("primary smoothed price:", ladder.primary.value)
for entry in ladder.entries:
    difference = entry.adjacent_difference
    print(
        "half-width:",
        entry.half_width,
        "price:",
        entry.result.value,
        "delta:",
        entry.result.delta_raw,
        "adjacent price difference:",
        None if difference is None else difference.price,
    )


# %% Continuous Barrier history in the shared Black–Scholes plan
# This state summarizes the whole past monitoring interval, not just endpoints.
for historical_hit in (False, True):
    barrier = rp.Product.barrier(
        1, 2, "2027-09-04", 100.0, 120.0, 2.0, "call", "up", "knock_out",
        "continuous", ["2026-09-03", "2027-09-04"], "2027-09-04",
        rebate=7.0, historical_hit=historical_hit,
    )
    request = rp.PricingRequest(
        "2026-09-04", barrier, market, model, engine,
        rp.RiskRequest(delta=True, gamma_relative_bump=0.01, vega=True),
    )
    result = rp.PricingPlan.compile(request, worker_threads=2).evaluate()
    print("historical hit:", historical_hit, "price:", result.value, "delta:", result.delta_raw)
    if historical_hit:
        assert abs(result.value - 7.0 * 0.95) < 1e-12
        assert (result.delta_raw, result.gamma_raw, result.vega_raw) == (0.0, 0.0, 0.0)
