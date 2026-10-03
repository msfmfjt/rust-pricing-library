"""Python boundary checks; numerical duality oracles live in the Rust suite."""

import math
import unittest

import rust_pricing as rp


class AndersenBroadieTest(unittest.TestCase):
    def request(self, *, model=None, risk=None, engine=None, training=None,
                dates=None, dividends=False, antithetic=True, european=False,
                spot=100.0):
        expiry = "2027-09-04"
        dates = dates or ["2026-12-04", "2027-03-04", "2027-06-04", expiry]
        product = (
            rp.Product.european_vanilla(1, 2, expiry, 100.0, 1.0, "put")
            if european else
            rp.Product.american_vanilla(1, 2, expiry, 100.0, 1.0, "put", dates)
        )
        return rp.PricingRequest(
            "2026-09-04", product,
            rp.Market.equity(
                2, 1, spot,
                rp.DiscountCurve(10, [0.0, 1.0], [1.0, math.exp(-0.05)]),
                rp.DiscountCurve(11, [0.0, 1.0], [1.0, 1.0]),
                discrete_dividends=[
                    rp.DividendEvent.fixed_cash_and_proportional(1, 181 / 365, 2.0, 0.03),
                ] if dividends else [],
            ),
            model or rp.Model.black_scholes(0.2),
            engine or rp.Engine.pseudo_monte_carlo(71, 64, antithetic=antithetic),
            risk or rp.RiskRequest(),
            lsm=None if european else rp.LsmConfig(
                training or rp.Engine.pseudo_monte_carlo(17, 512, antithetic=True),
            ),
        )

    def plan(self, request=None, *, config=None, workers=1):
        return rp.AndersenBroadiePlan.compile(
            request or self.request(), config or rp.AndersenBroadieConfig(16, 8, 1234),
            worker_threads=workers, reduction_block_size=16,
        )

    @staticmethod
    def estimates(result):
        return tuple(
            (estimate.value, estimate.standard_error, estimate.confidence_interval,
             estimate.effective_sampling_units, estimate.estimator)
            for estimate in (result.lower_bound, result.upper_bound, result.duality_gap)
        )

    def test_lower_matches_lsm_with_dividends_and_sampling_unit_metadata(self):
        for model in [rp.Model.black_scholes(0.2), rp.Model.black_76(0.2)]:
            for antithetic in [False, True]:
                with self.subTest(model=model, antithetic=antithetic):
                    request = self.request(model=model, dividends=True, antithetic=antithetic)
                    plan = self.plan(request)
                    result = plan.evaluate()
                    lsm = rp.PricingPlan.compile(
                        request, worker_threads=1, reduction_block_size=16,
                    ).evaluate()
                    # Joint dual statistics and the scalar LSM reduction may round differently.
                    self.assertAlmostEqual(result.lower_bound.value, lsm.value, places=13)
                    self.assertAlmostEqual(result.lower_bound.standard_error, lsm.standard_error,
                                           places=13)
                    self.assertEqual(result.policy_fingerprint,
                                     lsm.early_exercise_diagnostics.policy_fingerprint)
                    self.assertEqual(result.plan_fingerprint, plan.plan_fingerprint)
                    self.assertEqual(result.outer_trajectories, 128 if antithetic else 64)
                    self.assertEqual(result.exercise_date_count, 4)
                    self.assertGreaterEqual(result.duality_gap.value, 0.0)
                    self.assertAlmostEqual(result.upper_bound.value,
                                           result.lower_bound.value + result.duality_gap.value)
                    self.assertEqual(result.price_confidence_interval_95, (
                        result.lower_bound.confidence_interval[0],
                        result.upper_bound.confidence_interval[1],
                    ))
                    for estimate in [result.lower_bound, result.upper_bound, result.duality_gap]:
                        self.assertIsInstance(estimate, rp.DiagnosticEstimate)
                        self.assertEqual(estimate.effective_sampling_units, 64)
                        self.assertEqual(estimate.estimator, "pseudo_monte_carlo")
                        self.assertGreaterEqual(estimate.standard_error, 0.0)

    def test_repeat_worker_and_request_round_trip_replay(self):
        request = self.request()
        plan = self.plan(request)
        result = plan.evaluate()
        for replay in [plan.evaluate(), self.plan(request, workers=3).evaluate(),
                       self.plan(rp.PricingRequest.from_json(request.to_json())).evaluate()]:
            self.assertEqual(self.estimates(result), self.estimates(replay))
            self.assertEqual(result.policy_fingerprint, replay.policy_fingerprint)

    def test_expiry_only_collapses_to_existing_price(self):
        request = self.request(dates=["2027-09-04"])
        result = self.plan(request).evaluate()
        lsm = rp.PricingPlan.compile(request, worker_threads=1, reduction_block_size=16).evaluate()
        self.assertAlmostEqual(result.lower_bound.value, lsm.value, places=13)
        self.assertEqual(result.upper_bound.value, result.lower_bound.value)
        # Scalar and joint reductions can differ by one floating-point rounding.
        self.assertAlmostEqual(result.upper_bound.standard_error, lsm.standard_error, places=14)
        self.assertEqual(result.duality_gap.value, 0.0)
        self.assertEqual(result.duality_gap.standard_error, 0.0)
        self.assertEqual(result.exercise_date_count, 1)

    def test_zero_volatility_immediate_exercise_is_exact(self):
        request = self.request(model=rp.Model.black_scholes(0.0), spot=80.0,
                               dates=["2026-09-04", "2027-03-04", "2027-09-04"])
        result = self.plan(request).evaluate()
        self.assertEqual(result.lower_bound.value, 20.0)
        self.assertEqual(result.upper_bound.value, 20.0)
        self.assertEqual(result.price_confidence_interval_95, (20.0, 20.0))
        self.assertEqual(result.duality_gap.value, 0.0)

    def test_config_validation_and_integer_boundaries(self):
        for counts, pointer in [((0, 8), "/continuation_inner_paths"),
                                ((8, 0), "/exercise_inner_paths")]:
            with self.subTest(counts=counts), self.assertRaises(rp.ValidationError) as error:
                rp.AndersenBroadieConfig(*counts, 1234)
            self.assertEqual(error.exception.issues[0].pointer, pointer)
            self.assertEqual(error.exception.issues[0].code, "zero_inner_paths")
        for arguments in [(-1, 8, 0), (2**32, 8, 0), (8, -1, 0),
                          (8, 2**32, 0), (8, 8, -1), (8, 8, 2**64)]:
            with self.subTest(arguments=arguments), self.assertRaises(OverflowError):
                rp.AndersenBroadieConfig(*arguments)
        config = rp.AndersenBroadieConfig(2**32 - 1, 2**32 - 1, 2**64 - 1)
        self.assertEqual(config.inner_seed, 2**64 - 1)

    def test_unsupported_requests_fail_at_compile(self):
        rqmc = rp.Engine.randomized_quasi_monte_carlo(64, 101, scramble_count=4)
        local_vol = rp.Model.local_volatility_from_grid(
            [0.0, 1.0], [-0.5, 0.5], [0.04] * 4, 1e-8, 4.0,
        )
        for case in [dict(risk=rp.RiskRequest(delta=True)),
                     dict(risk=rp.RiskRequest(vega=True)),
                     dict(risk=rp.RiskRequest(gamma_relative_bump=0.01)),
                     dict(engine=rqmc), dict(training=rqmc),
                     dict(model=local_vol), dict(european=True)]:
            with self.subTest(case=case), self.assertRaises(rp.ValidationError) as error:
                self.plan(self.request(**case))
            self.assertEqual(error.exception.issues[0].code, "dual_unsupported")

    def test_invalid_execution_policy(self):
        request, config = self.request(), rp.AndersenBroadieConfig(16, 8, 1234)
        for args in [dict(worker_threads=0), dict(worker_threads=1, reduction_block_size=0)]:
            with self.subTest(args=args), self.assertRaises(rp.ValidationError) as error:
                rp.AndersenBroadiePlan.compile(request, config, **args)
            self.assertEqual(error.exception.issues[0].code, "plan_compile_error")

    def test_config_replay_metadata_and_immutable_objects(self):
        config = rp.AndersenBroadieConfig(16, 8, 1234)
        plan = self.plan(config=config)
        result = plan.evaluate()
        for name in ["continuation_inner_paths", "exercise_inner_paths", "inner_seed"]:
            self.assertEqual(getattr(config, name), getattr(result.config, name))
        for candidate in [(17, 8, 1234), (16, 9, 1234), (16, 8, 1235)]:
            self.assertNotEqual(plan.plan_fingerprint,
                                self.plan(config=rp.AndersenBroadieConfig(*candidate)).plan_fingerprint)
        for obj, field in [(config, "inner_seed"), (plan, "plan_fingerprint"),
                           (result, "outer_trajectories"), (result.lower_bound, "value")]:
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(obj, field, 0)
        self.assertIn("AndersenBroadieConfig", repr(config))
        self.assertIn(plan.plan_fingerprint, repr(plan))
        self.assertIn("lower_bound=", repr(result))


if __name__ == "__main__":
    unittest.main()
