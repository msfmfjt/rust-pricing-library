"""Public LSV configuration defaults, validation metadata and validation order."""
import unittest

import rust_pricing as rp


def configuration_cases():
    """Small deterministic inputs for every shared configuration entry point."""
    common = dict(particle_count=256, calibration_seed=42, log_bandwidth=0.35,
                  minimum_effective_samples=5.0, worker_threads=2)
    one_factor = dict(mean_reversion=2.0, vol_of_vol=0.6, correlation=-0.5)
    two_factor = dict(mean_reversions=[0.8, 2.1], vol_of_vol=0.3, mixing_weight=0.35,
                      spot_correlations=[-0.4, -0.2], factor_correlation=0.3)
    rough = dict(hurst=0.1, vol_of_vol=0.6, correlation=-0.4)
    dividends = dict(dividend_mean_reversion=0.7, equity_linkage=0.6,
                     dividend_volatility=0.35, equity_dividend_correlation=-0.25,
                     maximum_step=0.25)

    def request(stochastic=False, continuous=False):
        product = (rp.Product.barrier(1, 2, "2027-09-04", 100.0, 80.0, 1.0,
                       "call", "down", "knock_out", "continuous", ["2027-09-04"], "2027-09-04")
                   if continuous else
                   rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"))
        return rp.PricingRequest(
            "2026-09-04", product,
            rp.Market.equity(2, 1, 100.0,
                rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
                rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
                discrete_dividends=([rp.DividendEvent.fixed_cash(1, 0.5, 6.0),
                                     rp.DividendEvent.fixed_cash(2, 1.4, 3.0)]
                                    if stochastic else [])),
            rp.Model.local_volatility_from_grid(
                [0.0, 0.5, 1.0], [-0.5, 0.0, 0.5], [0.04] * 9, 1e-8, 4.0),
            rp.Engine.randomized_quasi_monte_carlo(64, 91, scramble_count=4,
                antithetic=True, brownian_bridge=True), rp.RiskRequest())

    target = request()
    for name, factory, model in [
        ("bergomi", rp.BergomiLsvPlan.compile, one_factor),
        ("two_factor", rp.Bergomi2FactorLsvPlan.compile, two_factor),
        ("rough", rp.RoughBergomiLsvPlan.compile, rough),
    ]:
        yield name, factory, (target,), common | model | dict(retain_reverse_trace=True), "/lsv"
    model = rp.RoughVolatilityModel.rough_sabr(
        hurst=0.1, vol_of_vol=0.3, correlation=-0.6, beta=1.0,
        forward_variance=rp.ForwardVarianceCurve.constant(0.04))
    yield ("rough_family", rp.RoughFamilyLsvPlan.compile, (target, model),
           common | dict(retain_reverse_trace=True), "/lsv")
    target = request(stochastic=True)
    for name, factory, model in [
        ("dividend_bergomi", rp.StochasticDividendPlan.compile_bergomi_lsv,
         one_factor | dict(dividend_volatility_correlation=0.15)),
        ("dividend_two_factor", rp.StochasticDividendPlan.compile_bergomi_two_factor_lsv,
         two_factor | dict(dividend_volatility_correlations=[0.15, -0.1])),
        ("dividend_rough", rp.StochasticDividendPlan.compile_rough_bergomi_lsv,
         rough | dict(dividend_volatility_correlation=0.15)),
    ]:
        yield name, factory, (target,), common | dividends | model, "/stochastic_dividends"
    yield ("continuous", rp.StochasticDividendContinuousBarrierPlan.compile_rough_bergomi_lsv,
           (request(stochastic=True, continuous=True),),
           common | dividends | rough | dict(dividend_volatility_correlation=0.15),
           "/stochastic_dividends")


class LsvConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = list(configuration_cases())

    def assert_validation(self, factory, args, params, domain, message):
        with self.assertRaises(rp.ValidationError) as caught:
            factory(*args, **params)
        error = caught.exception
        code = ("invalid_lsv_configuration" if domain == "/lsv" else
                "invalid_stochastic_dividend_configuration")
        self.assertIsInstance(error, ValueError)
        self.assertEqual((error.code, error.stage, error.field), (code, "validation", domain))
        self.assertEqual(len(error.issues), 1)
        issue = error.issues[0]
        self.assertEqual((issue.pointer, issue.code, issue.message, issue.phase),
                         (domain, code, message, "domain"))

    def test_particle_validation_precedes_execution_validation(self):
        for name, factory, args, params, domain in self.cases:
            for updates, message in [
                (dict(particle_count=1, log_bandwidth=0.0, worker_threads=0),
                 "invalid LSV particle_count at index 0"),
                (dict(log_bandwidth=0.0, minimum_effective_samples=0.0, worker_threads=0),
                 "invalid LSV log_bandwidth at index 0"),
                (dict(minimum_effective_samples=0.0, worker_threads=0),
                 "invalid LSV minimum_effective_samples at index 0"),
            ]:
                with self.subTest(plan=name, updates=updates):
                    self.assert_validation(factory, args, params | updates, domain, message)

    def test_nonfinite_and_out_of_range_particle_values_are_rejected(self):
        for name, factory, args, params, domain in self.cases:
            for field, value in [("log_bandwidth", float("nan")),
                                 ("log_bandwidth", float("inf")),
                                 ("minimum_effective_samples", float("nan")),
                                 ("minimum_effective_samples", 257.0)]:
                with self.subTest(plan=name, field=field, value=value):
                    self.assert_validation(factory, args, params | {field: value}, domain,
                                           f"invalid LSV {field} at index 0")

    def test_model_validation_precedes_particles_and_execution(self):
        for name, factory, args, params, domain in self.cases:
            invalid = dict(particle_count=1, worker_threads=0)
            if "equity_linkage" in params:
                invalid["equity_linkage"] = 1.1
                with self.subTest(plan=name, stage="dividend_model"):
                    self.assert_validation(factory, args, params | invalid, domain,
                                           "invalid stochastic-dividend equity_linkage")
            if "vol_of_vol" in params:
                message = ("invalid Hull–White rough_vol_of_vol at index 0"
                           if "hurst" in params else
                           "bergomi_vol_of_vol must be non-negative; received bits 0xbff0000000000000")
                with self.subTest(plan=name, stage="volatility_model"):
                    self.assert_validation(factory, args,
                                           params | invalid | dict(vol_of_vol=-1.0), domain, message)

    def test_execution_validation_keeps_domain_and_order(self):
        for name, factory, args, params, domain in self.cases:
            for updates, message in [
                (dict(worker_threads=0, reduction_block_size=0),
                 "worker-thread count must be positive"),
                (dict(reduction_block_size=0), "reduction-block size must be positive"),
            ]:
                with self.subTest(plan=name, updates=updates):
                    self.assert_validation(factory, args, params | updates, domain, message)

    def test_explicit_defaults_keep_price_surface_and_fingerprint(self):
        for name, factory, args, params, _ in self.cases:
            with self.subTest(plan=name):
                defaults = dict(reduction_block_size=None)
                if name.startswith("dividend_"):
                    defaults["retain_reverse_trace"] = False
                if name == "continuous":
                    defaults["market_iv_surface"] = None
                implicit = factory(*args, **params)
                explicit = factory(*args, **(params | defaults))
                self.assertEqual(implicit.plan_fingerprint, explicit.plan_fingerprint)
                left, right = implicit.evaluate(), explicit.evaluate()
                self.assertEqual((left.value, left.standard_error, left.evaluated_paths),
                                 (right.value, right.standard_error, right.evaluated_paths))
                surface = "squared_leverage" if hasattr(implicit, "squared_leverage") else "lsv_squared_leverage"
                self.assertEqual(getattr(implicit, surface), getattr(explicit, surface))

    def test_reverse_trace_flag_keeps_price_and_controls_risk(self):
        for name, factory, args, params, _ in self.cases:
            if name == "continuous":
                continue
            with self.subTest(plan=name):
                plain = factory(*args, **(params | dict(retain_reverse_trace=False)))
                traced = factory(*args, **(params | dict(retain_reverse_trace=True)))
                self.assertNotEqual(plain.plan_fingerprint, traced.plan_fingerprint)
                left, right = plain.evaluate(), traced.evaluate()
                self.assertEqual((left.value, left.standard_error), (right.value, right.standard_error))
                with self.assertRaises(rp.PricingError):
                    plain.evaluate_local_variance_risk()
                risk = traced.evaluate_local_variance_risk()
                self.assertEqual(risk.price.value, right.value)
                self.assertEqual(len(risk.node_adjoints), 9)


if __name__ == "__main__":
    unittest.main()
