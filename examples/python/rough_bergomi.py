"""Experimental rough Bergomi and rough-LSV with Hull-White, cash, AAD and VegaKT.

Illustrative parameters and escrow-coordinate IV quotes; no parameter fitting.
H and eta are fixed during AAD. Pure rough uses flat initial forward variance.
"""
import numpy as np
import rust_pricing as rp

rough = rp.RoughBergomiModel(0.1, 0.8, equity_vol_correlation=-0.5)
rates = rp.HullWhiteModel(0.2, [0.0, 0.45], [0.005, 0.008])
# Set all rate volatilities to zero for deterministic rates.
discount = rp.DiscountCurve(10, [0.0, 1.0, 5.0], [1.0, 0.95, 0.95**5])
dividend = rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
market = rp.Market.equity(
    2, 1, 100.0, discount, dividend,
    discrete_dividends=[
        rp.DividendEvent.fixed_cash(1, 0.5, 6.0),
        rp.DividendEvent.fixed_cash_and_proportional(2, 1.5, 4.0, 0.1),
    ],
)
engine = rp.Engine.randomized_quasi_monte_carlo(
    1024, 612, scramble_count=8, antithetic=True, brownian_bridge=True)


def request(model):
    return rp.PricingRequest(
        "2026-09-04",
        rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
        market, model, engine, rp.RiskRequest(),
    )


# sigma0=20%, xi0(t)=sigma0^2. Eta multiplies log variance; it is twice the
# log-volatility coefficient used by the older one-factor Bergomi API.
pure = rp.HullWhiteEquityPlan.compile_rough_bergomi(
    request(rp.Model.black_scholes(0.2)), rough, rates,
    equity_rate_correlation=0.25, vol_rate_correlation=-0.1,
    maximum_step=1/32, worker_threads=2, cash_dividend_model="escrowed",
)
pure_risk = pure.evaluate_aad()
print("Pure rough Bergomi + HW price / SE:", pure_risk.price.value, pure_risk.price.standard_error)
print("Spot Delta / initial-volatility Vega:", pure_risk.delta, pure_risk.vega)
print("Gaussian blocks:", pure.random_factor_count)

# A flexible smile comes through rough-LSV leverage calibration. These are
# already converted F-coordinate IV inputs, not physical-S Black IV quotes.
target = rp.HullWhiteLsvTarget.from_market_iv(
    [0.25, 0.5, 1.0], [-0.75, -0.25, 0.0, 0.25, 0.75], [0.2]*15,
    np.linspace(0.0, 1.0, 33), np.linspace(-0.75, 0.75, 31),
)
plan = rp.HullWhiteEquityPlan.compile_rough_lsv(
    request(target.model), target, rough, rates,
    equity_rate_correlation=0.25, vol_rate_correlation=-0.1,
    particle_count=8192, calibration_seed=712, log_bandwidth=0.14,
    minimum_effective_samples=20.0, worker_threads=2,
    cash_dividend_model="escrowed", retain_reverse_trace=True,
)
price = plan.evaluate()
risk = plan.evaluate_aad()
assert risk.price.value == price.value
assert np.isclose(sum(risk.vega_kt_raw), risk.vega)
print("Rough-LSV + HW price / conditional SE:", price.value, price.standard_error)
print("Spot Delta / signed +1 bp discount-curve DV01:", risk.delta, risk.parallel_discount_dv01)
print("VegaKT: currency per +1 vol point, rows T / columns log(K_F/S0):")
print(np.asarray(risk.vega_kt_market_scaled).reshape(3, 5))
print("Parallel IV Vega / conditional SE:", risk.vega, risk.parallel_vega_standard_error)
print("Uncertainty scope:", risk.uncertainty_scope)
print("Terminal fallback nodes:", plan.fallback_nodes[-1])
print("Scheme / calibration:", price.scheme, price.calibration_method)
print("Plan:", plan.plan_fingerprint)
