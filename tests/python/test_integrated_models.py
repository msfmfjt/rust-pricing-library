"""Contract boundaries when the accepted baseline and experimental models coexist."""
import unittest

import rust_pricing as rp


class IntegratedModelsTest(unittest.TestCase):
    def test_experimental_adapters_reject_unimplemented_payoff_features(self):
        expiry = "2027-09-04"
        target = rp.HullWhiteLsvTarget.flat(0.2, [0.0, 0.5, 1.0], [-0.5, 0.0, 0.5])
        market = rp.Market.equity(
            2, 1, 100.0,
            rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
            rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
        )
        engine = rp.Engine.pseudo_monte_carlo(master_seed=17, independent_sampling_units=32, antithetic=True)
        training = rp.Engine.pseudo_monte_carlo(master_seed=18, independent_sampling_units=32, antithetic=True)
        rates = rp.HullWhiteModel(0.2, [0.0], [0.005])
        rough = rp.RoughBergomiModel(0.1, 0.8, equity_vol_correlation=-0.5)
        particles = dict(particle_count=64, calibration_seed=42, log_bandwidth=0.5,
                         minimum_effective_samples=2.0, worker_threads=1)
        correlations = dict(equity_rate_correlation=0.2, vol_rate_correlation=-0.1)
        compilers = [
            (rp.Model.black_scholes(0.2), lambda r: rp.HullWhiteEquityPlan.compile_bs(
                r, rates, equity_rate_correlation=0.2, maximum_step=0.25, worker_threads=1)),
            (rp.Model.black_scholes(0.2), lambda r: rp.HullWhiteEquityPlan.compile_rough_bergomi(
                r, rough, rates, **correlations, maximum_step=0.25, worker_threads=1)),
            (target.model, lambda r: rp.BergomiLsvPlan.compile(
                r, mean_reversion=2.0, vol_of_vol=0.4, correlation=-0.5,
                retain_reverse_trace=False, **particles)),
            (target.model, lambda r: rp.HullWhiteEquityPlan.compile_lsv(
                r, target, rates, vol_mean_reversion=2.0, vol_of_vol=0.4,
                equity_vol_correlation=-0.5, **correlations, **particles)),
            (target.model, lambda r: rp.HullWhiteEquityPlan.compile_rough_lsv(
                r, target, rough, rates, **correlations, **particles)),
        ]
        cases = [
            ("early exercise", rp.Product.american_vanilla(
                1, 2, expiry, 100.0, 1.0, "put", ["2027-03-04", expiry]),
             rp.RiskRequest(), rp.LsmConfig(training, max_degree=1)),
            ("continuous Barrier", rp.Product.barrier(
                1, 2, expiry, 100.0, 120.0, 1.0, "call", "up", "knock_out",
                "continuous", [expiry], expiry), rp.RiskRequest(), None),
            ("smoothing width ladder", rp.Product.digital(
                1, 2, expiry, 100.0, 1.0, "call", "cash"),
             rp.RiskRequest(payoff_smoothing_half_width=1.0,
                            payoff_smoothing_width_ladder=[0.5, 1.0, 2.0]), None),
        ]
        for feature, product, risk, lsm in cases:
            for i, (model, compile_model) in enumerate(compilers):
                with self.subTest(feature=feature, adapter=i):
                    request = rp.PricingRequest(
                        "2026-09-04", product, market, model, engine, risk, lsm=lsm)
                    # The general facade supports this request. The experimental
                    # adapter must not silently use an expiry-only payoff graph.
                    rp.PricingPlan.compile(request, worker_threads=1)
                    with self.assertRaisesRegex(rp.PricingError, feature):
                        compile_model(request)


if __name__ == "__main__":
    unittest.main()
