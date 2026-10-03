"""Bass through PricingRequest/PricingPlan, including carry, cash and VegaKT.

Input IVs are quoted on residual equity: k = log((K - A*S0)/(B*F_f)).
When cash dividends are present these are not raw physical-spot Black IVs.
"""
from datetime import date
import rust_pricing as rp


def build_request():
    valuation = date(2026, 9, 4)
    expiries = [date(2027, 3, 4), date(2027, 9, 4)]
    times = [(d-valuation).days/365 for d in expiries]
    strikes = [-2.0, 0.0, 2.0]
    model = rp.Model.bass_local_volatility(
        times, strikes, [0.20]*6,
        [-2.0+4.0*i/800 for i in range(801)],
        config=rp.BassLvConfig(grid_points=401, cdf_tolerance=1e-7),
        iv_bump=1e-4,
    )
    market = rp.Market.equity(
        2, 1, 100.0,
        rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
        rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.25, 6.0),
                            rp.DividendEvent.proportional(2, 1.0, 0.1)],
    )
    product = rp.Product.arithmetic_asian(
        1, 2, 100.0, 1.0, "call",
        [rp.AsianObservation.known(date(2026, 8, 4), 0.25, 98.0),
         rp.AsianObservation.unknown(expiries[0], 0.25),
         rp.AsianObservation.unknown(expiries[1], 0.50)],
        date(2027, 12, 4),
    )
    engine = rp.Engine.randomized_quasi_monte_carlo(
        points_per_scramble=2048, scramble_count=8, master_scramble_seed=42,
        antithetic=True, brownian_bridge=True,
    )
    risk = rp.RiskRequest(
        delta=True, gamma_relative_bump=0.002, vega=True,
        vega_kt_maturity_nodes=expiries,
        vega_kt_log_forward_moneyness_nodes=strikes,
        vega_kt_relative_density_threshold=1e-5,
        vega_kt_full_bucket_covariance=True,
    )
    return rp.PricingRequest(valuation, product, market, model, engine, risk)


if __name__ == "__main__":
    request = build_request()
    # Typed and serialized requests take the same compiled/evaluated path.
    restored = rp.PricingRequest.from_json(request.to_json())
    assert restored.fingerprint == request.fingerprint
    plan = rp.PricingPlan.compile(restored, worker_threads=2)
    result = plan.evaluate()
    print(f"Asian price: {result.value:.6f} +/- {result.standard_error:.6f}")
    print(f"Delta: {result.delta_raw:.6f}; Gamma: {result.gamma_raw:.6f}")
    print(f"Parallel Vega / vol point: {result.vega_market_scaled:.6f}")
    print("Quote Vegas / vol point:", [e.market_scaled_mean for e in result.vega_kt.estimates])
    print("Vega method:", result.vega_kt.policy_label)
    print("Independent scrambles:", result.independent_sampling_units)
    print("Evaluated paths:", result.evaluated_paths)
    print("Maximum marginal CDF error:", max(d.marginal_cdf_error for d in plan.bass_calibration_diagnostics))
