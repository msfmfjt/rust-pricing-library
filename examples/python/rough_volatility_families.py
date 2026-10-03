"""Small six-model price-only example (not a calibration or accuracy benchmark).

Run after `python -m maturin develop --locked` from the repository root.
The BlackScholes request is only a payoff/market/engine carrier. Each explicit
rough model supplies its own volatility level. Sampling SE excludes model and
numerical-scheme biases.
"""
import math
import rust_pricing as rp


def main() -> None:
    curve = rp.ForwardVarianceCurve.constant(0.04)
    heston = rp.RoughVolatilityModel.rough_heston(
        hurst=0.1, initial_variance=0.04, mean_reversion=1.2,
        long_run_variance=0.04, vol_of_vol=0.3, correlation=-0.6)
    models = [
        heston,
        rp.RoughVolatilityModel.lifted_heston_from_rough(heston, factors=20, ratio=2.5),
        rp.RoughVolatilityModel.quadratic_rough_heston(
            hurst=0.1, initial_state=0.1, mean_reversion=1.2, vol_of_vol=0.3,
            quadratic=0.3, shift=0.1, variance_floor=0.04),
        rp.RoughVolatilityModel.mixed_rough_bergomi(
            hurst=0.1, correlation=-0.6, weights=[0.3, 0.7], vol_of_vols=[0.3, 0.8],
            forward_variance=curve),
        rp.RoughVolatilityModel.rough_sabr(
            hurst=0.1, vol_of_vol=0.6, correlation=-0.6, beta=1.0,
            forward_variance=curve),
        rp.RoughVolatilityModel.rfsv(
            hurst=0.1, mean_reversion=1.2, vol_of_log_vol=0.1,
            mean_log_vol=math.log(0.2), initial_log_vol=math.log(0.2)),
    ]
    request = rp.PricingRequest(
        "2026-09-04",
        rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
        rp.Market.equity(
            2, 1, 100.0,
            rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
            rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
            discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.5, 3.0)]),
        rp.Model.black_scholes(0.2),
        rp.Engine.randomized_quasi_monte_carlo(
            64, 612, scramble_count=4, antithetic=True, brownian_bridge=True),
        rp.RiskRequest(),
    )
    for model in models:
        plan = rp.RoughVolatilityPlan.compile(
            request, model, maximum_step=1/16, worker_threads=1,
            reduction_block_size=64)
        price = plan.evaluate()
        print(f"{model.name:25s} {price.value:10.6f}  SE={price.standard_error:.6f}"
              f"  paths={price.evaluated_paths}")
        # Diagnostics are explicit per-path outputs, not hidden variance floors.
        path_plan = plan.path_plan
        sample = path_plan.evolve_path(100.0, path_plan.pseudo_shocks(612, 0))
        print(f"  {price.scheme}; negative variance nodes={sample.negative_variance_nodes}")


if __name__ == "__main__":
    main()
