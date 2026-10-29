"""Python facade parity, replay, diagnostics, ownership and validation errors."""
import copy
import json
from pathlib import Path
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def fixture(name):
    return json.loads((ROOT / f"fixtures/specification-v1/{name}.golden.json").read_text())


class PricingSpecificationTests(unittest.TestCase):
    def test_legacy_wrapper_keeps_native_result_and_replay_metadata(self):
        for version in [1, 2, 3]:
            text = (ROOT / f"fixtures/v{version}/pricing_request.golden.json").read_text()
            request = rp.PricingRequest.from_json(text)
            legacy = rp.PricingPlan.compile(request, worker_threads=2, reduction_block_size=32)
            source = rp.PricingSpecification.from_request(request, worker_threads=2, reduction_block_size=32)
            compiled = source.compile()
            result = compiled.evaluate()
            self.assertEqual(compiled.plan_fingerprint, legacy.plan_fingerprint)
            self.assertIsInstance(result.native_result, rp.PricingResult)
            self.assertEqual(result.native_result.to_json(), legacy.evaluate().to_json())
            self.assertEqual(result.specification_fingerprint, source.fingerprint)
            self.assertEqual(compiled.specification_fingerprint, source.fingerprint)
            self.assertEqual(result.value, result.native_result.value)
            self.assertEqual(result.standard_error, result.native_result.standard_error)
        with self.assertRaises(TypeError):
            rp.PricingSpecification.from_request(request)
        with self.assertRaises(rp.ValidationError) as caught:
            rp.PricingSpecification.from_request(request, worker_threads=0)
        self.assertEqual(caught.exception.issues[0].document_kind, "pricing_specification")

    def test_source_documents_replay_all_result_types_and_are_immutable(self):
        for name, result_type in [("black_scholes", rp.PricingResult),
                                  ("bergomi_hull_white", rp.HullWhitePrice),
                                  ("bergomi_lsv", rp.LsvPrice), ("rough_heston", rp.HullWhitePrice)]:
            with self.subTest(name=name):
                source = rp.PricingSpecification.from_json(json.dumps(fixture(name)))
                restored = rp.PricingSpecification.from_json(source.to_pretty_json())
                self.assertEqual(source.to_json(), restored.to_json())
                self.assertTrue(source.to_json().endswith("\n"))
                self.assertEqual(source.fingerprint, restored.fingerprint)
                a, b = source.compile(), restored.compile()
                self.assertEqual(a.plan_fingerprint, b.plan_fingerprint)
                result, replay = a.evaluate(), b.evaluate()
                self.assertIsInstance(result.native_result, result_type)
                self.assertEqual(result.value, replay.value)
                self.assertEqual(result.standard_error, replay.standard_error)
                self.assertEqual(result.value, result.native_result.value)
                self.assertEqual(result.standard_error, result.native_result.standard_error)
                with self.assertRaises(AttributeError):
                    source.fingerprint = "changed"
                with self.assertRaises(AttributeError):
                    result.value = 0.0

    def test_pure_sv_and_lsv_match_dedicated_python_apis(self):
        value = fixture("bergomi_hull_white")
        value["model"]["rates"] = {"type": "deterministic"}
        original = json.loads((ROOT / "fixtures/v3/pricing_request.golden.json").read_text())
        original["engine"] = value["engine"]
        request = rp.PricingRequest.from_json(json.dumps(original))
        native = rp.StochasticVolatilityPlan.compile_bergomi(request, mean_reversion=1.2,
            vol_of_vol=0.3, correlation=-0.4, maximum_step=0.25, worker_threads=2, reduction_block_size=32)
        plan = rp.PricingSpecification.from_json(json.dumps(value)).compile()
        self.assertEqual(plan.plan_fingerprint, native.plan_fingerprint)
        self.assertEqual(plan.evaluate().value, native.evaluate().value)
        value = fixture("bergomi_lsv")
        original["model"] = {"type": "local_volatility", **value["model"]["equity"]["target"]}
        request = rp.PricingRequest.from_json(json.dumps(original))
        native = rp.BergomiLsvPlan.compile(request, mean_reversion=1.2, vol_of_vol=0.3,
            correlation=-0.4, particle_count=64, calibration_seed=429, log_bandwidth=0.4,
            minimum_effective_samples=2.0, retain_reverse_trace=True, worker_threads=2,
            reduction_block_size=32)
        plan = rp.PricingSpecification.from_json(json.dumps(value)).compile()
        result = plan.evaluate()
        self.assertEqual(plan.plan_fingerprint, native.plan_fingerprint)
        self.assertEqual(result.value, native.evaluate().value)
        self.assertEqual(result.native_result.calibration_seed, native.evaluate().calibration_seed)
        self.assertEqual(result.native_result.scheme, native.evaluate().scheme)

    def test_schema_is_self_contained_and_errors_identify_the_source_document(self):
        schema = json.loads(rp.specification_json_schema())
        self.assertEqual(schema, json.loads((ROOT / "schemas/specification-v1/pricing_specification.schema.json").read_text()))
        self.assertEqual(schema["properties"]["document_kind"]["const"], "pricing_specification")
        base = fixture("bergomi_hull_white")
        for name in ["version", "ignored", "step", "risk", "initial_volatility", "correlation"]:
            value = copy.deepcopy(base)
            if name == "version":
                value["schema_version"] = 99
            elif name == "ignored":
                value["model"]["rates"] = {"type": "deterministic", "ignored": 1}
            elif name == "step":
                del value["numerics"]["maximum_step"]
            elif name == "risk":
                value["risk"]["delta"] = True
            elif name == "initial_volatility":
                value["model"]["equity"]["initial_volatility"] = -1.0
            else:
                value["model"]["rates"]["equity_rate_correlation"] = 0.99
                value["model"]["rates"]["vol_rate_correlations"] = [0.99]
            with self.subTest(name=name), self.assertRaises(rp.ValidationError) as caught:
                rp.PricingSpecification.from_json(json.dumps(value)).compile()
            issue = caught.exception.issues[0]
            self.assertEqual(issue.document_kind, "pricing_specification")
            self.assertEqual(issue.schema_version, 99 if name == "version" else 1)

    def test_model_and_execution_changes_change_source_fingerprint(self):
        value = fixture("bergomi_hull_white")
        source = rp.PricingSpecification.from_json(json.dumps(value))
        value["model"]["rates"]["volatilities"][0] += 0.001
        changed = rp.PricingSpecification.from_json(json.dumps(value))
        self.assertNotEqual(source.fingerprint, changed.fingerprint)
        value = fixture("bergomi_hull_white")
        value["numerics"]["worker_threads"] = 3
        changed = rp.PricingSpecification.from_json(json.dumps(value))
        self.assertNotEqual(source.fingerprint, changed.fingerprint)
        self.assertEqual(source.compile().evaluate().value, changed.compile().evaluate().value)


if __name__ == "__main__":
    unittest.main()
