"""One-currency BS/LSV + Hull-White, with independent calibration and pricing.

Rate parameters below are illustrative inputs, not a calibration to swaptions.
Pricing standard errors exclude particle calibration noise and numerical bias.
"""
import numpy as np
import rust_pricing as rp

discount = rp.DiscountCurve(10, [0.0, 1.0, 5.0], [1.0, 0.95, 0.95**5])
dividend = rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
rates = rp.HullWhiteModel(0.2, [0.0, 0.45], [0.012, 0.018])
engine = rp.Engine.randomized_quasi_monte_carlo(
    1024, 612, scramble_count=8, antithetic=True, brownian_bridge=True
)


def request(model):
    return rp.PricingRequest(
        "2026-09-04",
        rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
        rp.Market.equity(2, 1, 100.0, discount, dividend),
        model, engine, rp.RiskRequest(),
    )


bs = rp.HullWhiteEquityPlan.compile_bs(
    request(rp.Model.black_scholes(0.2)), rates,
    equity_rate_correlation=0.25, maximum_step=1.0, worker_threads=2,
).evaluate()
# Independent IV buckets and a finer particle grid. The interpolation uses
# natural-cubic total variance in log moneyness and linear total variance in time.
quote_times = [0.25, 0.5, 1.0]
quote_log_nodes = [-0.75, -0.25, 0.0, 0.25, 0.75]
target = rp.HullWhiteLsvTarget.from_market_iv(
    quote_times, quote_log_nodes, [0.2] * 15,
    np.linspace(0.0, 1.0, 33), np.linspace(-0.75, 0.75, 31),
)
plan = rp.HullWhiteEquityPlan.compile_lsv(
    request(target.model), target, rates,
    vol_mean_reversion=2.0, vol_of_vol=0.4,
    equity_vol_correlation=-0.5, equity_rate_correlation=0.25,
    vol_rate_correlation=-0.1, particle_count=8192, calibration_seed=712,
    log_bandwidth=0.14, minimum_effective_samples=20.0, worker_threads=2,
)
lsv = plan.evaluate()
print("Initial 5Y bond:", rates.bond_price(discount, 0.0, 5.0, 0.0))
print("BS + HW price / SE:", bs.value, bs.standard_error)
print("Recalibrated LSV + HW price / SE:", lsv.value, lsv.standard_error)
print("Uncertainty scope:", lsv.uncertainty_scope)
print("Terminal mean D/P0 (target 1):", plan.calibration_discount_means[-1])
print("Terminal mean D/P0 * normalized equity (target 100):",
      plan.calibration_discounted_equity_means[-1])
print("Terminal fallback nodes:", plan.fallback_nodes[-1])
print("Scheme / calibration:", lsv.scheme, lsv.calibration_method)
print("Plan:", plan.plan_fingerprint)

# Explicit escrowed model: the flat target refers to the normalized deterministic
# escrow coordinate. It is not an unadjusted spot Black implied-volatility smile.
# A cash payment after option expiry is included in the stochastic bond reserve.
cash_market = rp.Market.equity(
    2, 1, 100.0, discount, dividend,
    discrete_dividends=[
        rp.DividendEvent.fixed_cash(1, 0.5, 6.0),
        rp.DividendEvent.fixed_cash_and_proportional(2, 1.5, 4.0, 0.1),
    ],
)
cash_request = rp.PricingRequest(
    "2026-09-04",
    rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
    cash_market, target.model, engine, rp.RiskRequest(),
)
cash_plan = rp.HullWhiteEquityPlan.compile_lsv(
    cash_request, target, rates,
    vol_mean_reversion=2.0, vol_of_vol=0.4,
    equity_vol_correlation=-0.5, equity_rate_correlation=0.25,
    vol_rate_correlation=-0.1, particle_count=8192, calibration_seed=712,
    log_bandwidth=0.14, minimum_effective_samples=20.0, worker_threads=2,
    cash_dividend_model="escrowed", retain_reverse_trace=True,
)
cash_price = cash_plan.evaluate()
print("Cash dividend model:", cash_price.cash_dividend_model)
print("Initial risky equity (after dividend reserve):", cash_plan.risky_spot)
print("Cash-dividend LSV + HW price / SE:", cash_price.value, cash_price.standard_error)

# AAD includes particle recalibration. Target variance and forward density are
# separate active inputs; contract both for a smile bump. HW/Bergomi parameters,
# correlations and payout amounts are held fixed. Curve risk refits the initial
# HW curve and includes the reserve and payment discount.
cash_risk = cash_plan.evaluate_aad()
assert cash_risk.price.value == cash_price.value
print("Cash AAD Delta:", cash_risk.delta)
print("Cash AAD signed +1 bp discount-curve DV01:", cash_risk.parallel_discount_dv01)
print("Discount log-DF node adjoints:", cash_risk.discount_log_df_adjoints)
print("Paired target risk shape:", (len(cash_risk.time_nodes), len(cash_risk.log_moneyness_nodes)))
print("AAD method / uncertainty:", cash_risk.method, cash_risk.uncertainty_scope)

# In cash mode these are VegaKT buckets for the converted escrow-coordinate IV
# quotes above. Raw physical-S IV conversion is not differentiated by this API.
assert target.supports_vega_kt
vega_kt = np.array(cash_risk.vega_kt_market_scaled).reshape(
    len(cash_risk.vega_kt_maturity_nodes), len(cash_risk.vega_kt_log_moneyness_nodes)
)
assert np.isclose(sum(cash_risk.vega_kt_raw), cash_risk.vega)
print("Cash VegaKT (currency per +1 vol point), rows T / columns log(K_F/S0):")
print(vega_kt)
print("Parallel IV Vega / conditional SE:", cash_risk.vega,
      cash_risk.parallel_vega_standard_error)
print("VegaKT interpolation:", cash_risk.vega_kt_method)
