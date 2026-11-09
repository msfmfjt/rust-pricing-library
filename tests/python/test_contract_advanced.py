"""Installed Python interface for dated holder rights and continuous monitoring."""
import json
from pathlib import Path
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
START = "2026-09-04"
MIDDLE = "2027-03-04"
EXPIRY = "2027-09-04"


def specification(fixture):
    request = rp.PricingRequest.from_json((ROOT / "fixtures/v3" / fixture).read_text())
    return rp.PricingSpecification.from_request(request, worker_threads=2, reduction_block_size=32)


class AdvancedContractTests(unittest.TestCase):
    def setUp(self):
        self.builder = rp.ContractBuilder("right", 2, 1, "official.close")

    def test_bermudan_and_american_grid_match_native_lsm(self):
        source = specification("pricing_request_american.golden.json")
        native = rp.PricingTask.from_specification(source)
        grid = [MIDDLE, EXPIRY]
        contracts = [
            (self.builder.bermudan("put", 100., 1., grid), {}),
            (self.builder.american("put", 100., 1., START, EXPIRY), {"american_grid": grid}),
        ]
        self.assertNotEqual(contracts[0][0].fingerprint, contracts[1][0].fingerprint)
        expected = native.compile().evaluate().report.to_json()
        for option, settings in contracts:
            task = option.pricing_task(source, **settings)
            self.assertEqual(task.fingerprint, native.fingerprint)
            self.assertEqual(task.compile().evaluate().report.to_json(), expected)
            self.assertEqual(rp.PricingTask.from_json(task.to_json()).compile().evaluate().report.to_json(), expected)
            self.assertEqual(option.fingerprint, rp.Contract.from_json(option.to_json()).fingerprint)
        with self.assertRaises(ValueError):
            contracts[1][0].pricing_task(source)
        with self.assertRaises(ValueError):
            contracts[0][0].pricing_task(source, american_grid=grid)
        schema = json.loads(rp.Contract.json_schema(2))
        self.assertEqual(schema["properties"]["schema_version"]["const"], 2)

    def test_fixed_strategy_risk_on_constant_and_local_volatility(self):
        raw = json.loads((ROOT / "fixtures/v3/pricing_request_american.golden.json").read_text())
        raw["risk"]["delta"] = True
        raw["risk"]["vega"] = True
        option = self.builder.bermudan("put", 100., 1., [MIDDLE, EXPIRY])
        for model in (
            {"type": "black_scholes", "volatility": .2},
            {"type": "black_76", "volatility": .2},
            {"type": "local_volatility", "local_variance_grid": {
                "time_nodes": [0., .5, 1.], "log_forward_moneyness_nodes": [-.25, 0., .25],
                "shape": [3, 3], "values": [.04] * 9, "floor": 1e-8, "cap": 4.}},
        ):
            with self.subTest(model=model["type"]):
                raw["model"] = model
                source = rp.PricingSpecification.from_request(rp.PricingRequest.from_json(json.dumps(raw)),
                                                               worker_threads=2, reduction_block_size=32)
                native = rp.PricingTask.from_specification(source).compile().evaluate().report
                report = option.pricing_task(source).compile().evaluate().report
                self.assertGreater(len(report.risks), 0)
                self.assertEqual(report.to_json(), native.to_json())

    def test_model_and_grid_capabilities_are_checked_before_simulation(self):
        hybrid_doc = json.loads((ROOT / "fixtures/specification-v1/bergomi_hull_white.golden.json").read_text())
        hybrid = rp.PricingSpecification.from_json(json.dumps(hybrid_doc))
        product = hybrid_doc["product"]
        builder = rp.ContractBuilder("hybrid", product["currency_id"], product["underlying_id"], "close")
        right = builder.american("put", 100., 1., hybrid_doc["valuation_date"], product["expiry"])
        with self.assertRaisesRegex(ValueError, "deterministic rates"):
            right.pricing_task(hybrid, american_grid=[product["expiry"]])
        source = specification("pricing_request_american.golden.json")
        fine = [MIDDLE, "2027-06-04", EXPIRY]
        right = self.builder.american("put", 100., 1., START, EXPIRY)
        coarse_task = right.pricing_task(source, american_grid=[EXPIRY])
        fine_task = right.pricing_task(source, american_grid=fine)
        self.assertNotEqual(coarse_task.fingerprint, fine_task.fingerprint)
        self.assertGreater(coarse_task.compile().evaluate().report.value, 0.)
        self.assertGreater(fine_task.compile().evaluate().report.value, 0.)

    def test_actual_american_notice_and_off_grid_settlement(self):
        option = self.builder.american("put", 100., 3., START, EXPIRY)
        day = "2027-01-13"
        option_id = "right/option"
        observation = json.loads(option.exercise_observation_json(option_id, day))
        fixing = {"id": "close", "effective_date": day,
                  "event": {"type": "fixing", "observation": observation, "value": 87.}}
        notice = {"id": "instruction", "effective_date": day,
                  "event": {"type": "exercise", "exercise_id": option_id}}
        earlier = option.advance(json.dumps([notice, fixing, notice]), START)
        self.assertEqual(json.loads(earlier.outstanding_payments_json()), [])
        decided = earlier.advance("[]", day)
        self.assertEqual(json.loads(decided.outstanding_payments_json())[0]["amount"], 39.)
        replayed = rp.ContractLifecycle.from_json(decided.to_json())
        self.assertEqual(replayed.outstanding_payments_json(), decided.outstanding_payments_json())
        paid = {"id": "paid", "effective_date": day, "event": {"type": "settlement", "cashflow_id": option_id}}
        self.assertEqual(json.loads(replayed.advance(json.dumps([paid]), EXPIRY).outstanding_payments_json()), [])

    def test_continuous_bridge_agrees_with_knock_parity_and_rejects_unconfirmed_history(self):
        raw = json.loads((ROOT / "fixtures/v3/pricing_request.golden.json").read_text())
        raw["risk"]["delta"] = raw["risk"]["vega"] = False
        source = rp.PricingSpecification.from_request(rp.PricingRequest.from_json(json.dumps(raw)), worker_threads=2, reduction_block_size=32)
        vanilla = self.builder.european("call", 100., 1., EXPIRY, EXPIRY)
        european = vanilla.pricing_task(source).compile().evaluate().report.value
        for direction, level in [("up", 118.), ("down", 82.)]:
            values = []
            for style in ["knock_in", "knock_out"]:
                option = self.builder.continuous_barrier("call", 100., 1., EXPIRY, EXPIRY, START, EXPIRY,
                                                        direction, style, level, 0.)
                with self.assertRaises(ValueError):
                    option.pricing_task(source)
                report = option.pricing_task(source, brownian_bridge=True).compile().evaluate().report
                self.assertGreater(report.value, 0.)
                values.append(report.value)
                with self.assertRaises(ValueError):
                    option.advance("[]", MIDDLE)
            self.assertAlmostEqual(sum(values), european, places=10)
