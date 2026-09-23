"""Pure 1F/2F/rough Bergomi at deterministic rates, plus 2F with Hull–White."""
import rust_pricing as rp


request = rp.PricingRequest(
    "2026-09-04",
    rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
    rp.Market.equity(
        2, 1, 100.0,
        rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
        rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.5, 3.0)],
    ),
    rp.Model.black_scholes(0.2),  # sigma0; flat initial forward variance = 0.04
    rp.Engine.randomized_quasi_monte_carlo(
        256, 612, scramble_count=8, antithetic=True, brownian_bridge=True),
    rp.RiskRequest(),
)
execution = dict(maximum_step=1/16, worker_threads=2)
two_factor = dict(mean_reversions=[0.7, 2.1], vol_of_vol=0.6,
                  mixing_weight=0.35, spot_correlations=[-0.5, -0.3],
                  factor_correlation=0.25)
plans = {
    "1F": rp.StochasticVolatilityPlan.compile_bergomi(
        request, mean_reversion=0.7, vol_of_vol=0.6, correlation=-0.5, **execution),
    "2F": rp.StochasticVolatilityPlan.compile_bergomi_two_factor(
        request, **two_factor, **execution),
    "rough": rp.StochasticVolatilityPlan.compile_rough_bergomi(
        request, rp.RoughBergomiModel(0.1, 1.2, equity_vol_correlation=-0.5), **execution),
    "2F + HW": rp.HullWhiteEquityPlan.compile_bergomi_two_factor(
        request, rp.HullWhiteModel(0.2, [0.0], [0.01]),
        equity_rate_correlation=0.2, vol_rate_correlations=[-0.1, 0.1],
        **two_factor, **execution),
}
for name, plan in plans.items():
    risk = plan.evaluate_aad()
    assert risk.price.calibration_method is None
    assert risk.vega_kt_raw is None
    print(f"{name}: price={risk.price.value:.6f}, SE={risk.price.standard_error:.6f}, "
          f"Delta={risk.delta:.6f}, dPrice/dsigma0={risk.vega:.6f}")
