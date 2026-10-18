import importlib.util
import json
from pathlib import Path
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location("bass_request_example", ROOT / "examples/python/bass_lv_request.py")
example = importlib.util.module_from_spec(loader)
loader.loader.exec_module(example)


class BassRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = example.build_request()
        cls.plan = rp.PricingPlan.compile(cls.request, worker_threads=2)
        cls.result = cls.plan.evaluate()

    def test_common_request_result_and_wire(self):
        request = rp.PricingRequest.from_json(self.request.to_json())
        self.assertEqual(request.fingerprint, self.request.fingerprint)
        restored = rp.PricingResult.from_json(self.result.to_json())
        self.assertEqual(restored.value, self.result.value)
        self.assertEqual(restored.vega_kt.raw_buckets, self.result.vega_kt.raw_buckets)
        self.assertEqual(len(restored.vega_kt.full_bucket_covariance), 36)
        self.assertEqual(restored.vega_kt.policy_label, "recalibrated-central-crn-v1")
        self.assertEqual(self.result.independent_sampling_units, 8)
        self.assertEqual(self.result.evaluated_paths, 32768)
        self.assertIsNotNone(self.result.diagnostics.direction_checksum)
        self.assertIsNotNone(self.result.delta)
        self.assertIsNotNone(self.result.gamma)
        self.assertAlmostEqual(self.result.vega_market_scaled, 0.01*self.result.vega_raw)
        self.assertEqual(len(self.plan.bass_calibration_diagnostics), 2)
        self.assertEqual(len(self.plan.bass_projection_diagnostics), 2)
        self.assertEqual(len(self.plan.bass_vega_scenario_diagnostics), 14)

    def test_worker_replay_and_price_only_parity(self):
        replay = rp.PricingPlan.compile(self.request, worker_threads=1).evaluate()
        self.assertEqual(replay.value, self.result.value)
        self.assertEqual(replay.delta_raw, self.result.delta_raw)
        self.assertEqual(replay.vega_kt.raw_buckets, self.result.vega_kt.raw_buckets)
        v = json.loads(self.request.to_json())
        v["risk"] = {"delta": False, "vega": False, "smile_dynamics": {"type": "sticky_log_moneyness"}}
        result = rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(v)), worker_threads=1).evaluate()
        self.assertEqual(result.value, self.result.value)
        self.assertEqual(result.standard_error, self.result.standard_error)
        self.assertIsNone(result.vega_kt)

    def test_schema_accepts_new_model_and_old_versions_reject(self):
        # Check the exported schema and native strict request reader together.
        loader = importlib.util.spec_from_file_location("schema_gate", ROOT / "scripts/check_schemas.py")
        schema_gate = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(schema_gate)
        value = json.loads(self.request.to_json())
        schema = json.loads(rp.request_json_schema())
        schema_gate.check_v3_artifacts()
        variant = next(v for v in schema["$defs"]["model"]["oneOf"]
                       if v["properties"]["type"]["const"] == "bass_local_volatility")
        parameters = variant["properties"]["parameters"]
        self.assertEqual(set(parameters["required"]), set(value["model"]["parameters"]))
        self.assertFalse(parameters["additionalProperties"])
        malformed = json.loads(json.dumps(value))
        malformed["model"]["parameters"]["unknown_parameter"] = 1
        with self.assertRaises(rp.ValidationError):
            rp.PricingRequest.from_json(json.dumps(malformed))
        for version in [1, 2]:
            old = dict(value, schema_version=version)
            with self.assertRaises(rp.ValidationError):
                rp.PricingRequest.from_json(json.dumps(old))

    def test_unsupported_features_are_explicit(self):
        base = json.loads(self.request.to_json())
        for change in ["horizon", "axes", "smile", "continuous", "american"]:
            v = json.loads(json.dumps(base))
            if change == "horizon":
                v["model"]["parameters"]["maturity_nodes"] = [0.2, 0.3]
            elif change == "axes":
                v["risk"]["vega_kt"]["log_forward_moneyness_nodes"] = [-1.0, 0.0, 1.0]
            elif change == "smile":
                v["risk"]["smile_dynamics"]["type"] = "sticky_strike"
            elif change == "continuous":
                v["risk"] = {"delta": False, "vega": False, "smile_dynamics": {"type": "sticky_log_moneyness"}}
                v["product"] = {"type": "barrier", "underlying_id": 1, "currency_id": 2,
                    "expiry": "2027-09-04", "strike": 100.0, "barrier": 120.0, "notional": 1.0,
                    "side": {"type": "call"}, "direction": {"type": "up"}, "style": {"type": "knock_out"},
                    "monitoring": {"type": "continuous"}, "monitoring_dates": ["2027-09-04"], "payment_date": "2027-09-04"}
            else:
                product = rp.Product.american_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "put", ["2027-09-04"])
                market = rp.Market.equity(2, 1, 100.0, rp.DiscountCurve(1, [0.0, 1.0], [1.0, 0.95]), rp.DiscountCurve(2, [0.0, 1.0], [1.0, 0.98]))
                model = rp.Model.bass_local_volatility([0.5, 1.0], [-2.0, 0.0, 2.0], [0.2]*6, [-2+4*i/800 for i in range(801)])
                engine = rp.Engine.pseudo_monte_carlo(7, 64)
                request = rp.PricingRequest("2026-09-04", product, market, model, engine, rp.RiskRequest(), lsm=rp.LsmConfig(rp.Engine.pseudo_monte_carlo(8, 64)))
                with self.assertRaises(rp.ValidationError):
                    rp.PricingPlan.compile(request, worker_threads=1)
                continue
            with self.subTest(change=change), self.assertRaises(rp.ValidationError):
                rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(v)), worker_threads=1)

    def test_lookback_and_smoothed_digital_width_ladder(self):
        market = rp.Market.equity(2, 1, 100.0,
            rp.DiscountCurve(1, [0.0, 1.0], [1.0, 0.95]),
            rp.DiscountCurve(2, [0.0, 1.0], [1.0, 0.98]))
        model = rp.Model.bass_local_volatility([0.5, 1.0], [-2.0, 0.0, 2.0],
            [0.2]*6, [-2+4*i/800 for i in range(801)],
            config=rp.BassLvConfig(grid_points=401, cdf_tolerance=1e-7))
        engine = rp.Engine.pseudo_monte_carlo(7, 1024, antithetic=True)

        def compile_product(product, risk):
            request = rp.PricingRequest("2026-09-04", product, market, model, engine, risk)
            return rp.PricingPlan.compile(request, worker_threads=2)

        # With one observation and the past maximum at strike, lookback equals a call.
        lookback = rp.Product.fixed_lookback(1, 2, 100.0, 1.0, "call",
            ["2026-08-04", "2027-09-04"], "2027-09-04", historical_extremum=100.0)
        vanilla = rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call")
        self.assertEqual(compile_product(lookback, rp.RiskRequest()).evaluate().value,
                         compile_product(vanilla, rp.RiskRequest()).evaluate().value)

        digital = rp.Product.digital(1, 2, "2027-09-04", 100.0, 10.0, "call", "cash")
        risk = rp.RiskRequest(delta=True, gamma_relative_bump=0.002,
            payoff_smoothing_half_width=3.0, payoff_smoothing_width_ladder=[4.0, 2.0, 1.0])
        ladder = compile_product(digital, risk).evaluate_width_ladder()
        self.assertEqual(ladder.primary.diagnostics.valuation_kind, "smoothed_surrogate")
        for entry in ladder.entries:
            independent = compile_product(digital, rp.RiskRequest(delta=True,
                gamma_relative_bump=0.002, payoff_smoothing_half_width=entry.half_width)).evaluate()
            self.assertEqual(entry.result.value, independent.value)
            self.assertEqual(entry.result.delta_raw, independent.delta_raw)
            self.assertEqual(entry.result.gamma_raw, independent.gamma_raw)
            self.assertGreaterEqual(entry.result.value, 0.0)
            self.assertLessEqual(entry.result.value, 9.5)


if __name__ == "__main__":
    unittest.main()
