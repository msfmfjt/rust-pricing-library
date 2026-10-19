from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rust_pricing_backtest import (BacktestConfig, BacktestError, BacktestResult, Contract, ExternalCashflow,
                                  NumeraireObservation, PaymentDiscount, Position, SettlementFixing,
                                  compare_strategies, run_backtest)
from rust_pricing_backtest.domain import data, fingerprint
from rust_pricing_backtest.ensemble import EnsembleConfig, ScenarioResults, aggregate_scenarios
from rust_pricing_backtest.serialization import input_json, load_input
from rust_pricing_backtest.validation import loads_document, schema_documents, validate_document, _validators
from test_backtest import START, EXPIRY, snapshot
from test_payments import DatedProvider, PAYMENT


REPO = Path(__file__).resolve().parents[2]
HAS_VALIDATOR = importlib.util.find_spec("jsonschema") is not None


def schema_case(*, stop_after=None, fail_at=None, strategy="vega_kt"):
    contracts = [Contract("target", EXPIRY, 100., payment_date=PAYMENT, settlement_rule="explicit_fixing", settlement_fixing_id="close"),
                 Contract("hedge", EXPIRY, 105., payment_date=PAYMENT, settlement_rule="explicit_fixing", settlement_fixing_id="close")]
    history = []
    for i in (0, 1, 4, 6):
        s = snapshot(i, spot=110. if i >= 4 else 100., prices={"hedge": (3.99,4.01)})
        s = replace(s, lending_rate=.01, borrowing_rate=.03,
                    numeraire=NumeraireObservation("common", 2, s.at.date(), 2.*math.exp(.01*i/365),
                                                  s.at, s.at, "schema_fixture", "v1"),
                    payment_discounts={PAYMENT: PaymentDiscount(s.at.date(), .94+.01*i, s.at, s.at, "schema_fixture", "v1")})
        if i == 1:
            s = replace(s, external_cashflows=(ExternalCashflow("deposit", s.at, 100., 2, s.at, "schema_fixture", "v1"),))
        if i == 4:
            s = replace(s, settlement_fixings={"close": SettlementFixing(EXPIRY, 110., s.at, s.at, "schema_fixture", "v1")})
        if i == fail_at:
            s = replace(s, surface_available_at=s.at+timedelta(seconds=1))
        history.append(s)
    return run_backtest(BacktestConfig(strategy=strategy, ridge=0, initial_cash=50., numeraire_id="common"),
                        history, contracts, [Position("target", -1., 9.5, .2)], DatedProvider(),
                        stop_after=stop_after, on_error="return", code_revision="schema_fixture_v1")


class SchemaSourceTests(unittest.TestCase):
    def test_generated_schemas_match_specification(self):
        result = subprocess.run([sys.executable, str(REPO / "scripts/generate_backtest_schemas.py"), "--check"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(schema_documents()), 26)
        frozen = {name: document for name, document in schema_documents().items()
                  if name in ("definitions-v1.schema.json", "request-v5.schema.json", "result-v9.schema.json",
                              "checkpoint-v7.schema.json", "ensemble-request-v1.schema.json", "ensemble-v1.schema.json")}
        self.assertEqual(fingerprint(frozen), "5ad1c06bd94ba9403ea97142b6a50d098f68e09a2189738f8336a7a9ff9401d3")
        frozen_v2 = {name: document for name, document in schema_documents().items()
                     if name in ("definitions-v2.schema.json", "request-v6.schema.json", "result-v10.schema.json",
                                 "checkpoint-v8.schema.json", "ensemble-v2.schema.json")}
        self.assertEqual(fingerprint(frozen_v2), "5062dca3364dc1e082dd418e1e53e506d3e6d93d63ec9eb67305c8320964629a")
        frozen_v3 = {name: document for name, document in schema_documents().items()
                     if name in ("definitions-v3.schema.json", "request-v7.schema.json", "result-v11.schema.json",
                                 "checkpoint-v9.schema.json", "ensemble-v3.schema.json")}
        self.assertEqual(fingerprint(frozen_v3), "911ef0e79da5d5820ad832a567c53f69788ebdea287c84b3a80a513c6ac9b31e")

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for payload in ('{"x":1,"x":2}', '{"a":{"x":1,"x":2}}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e9999}'):
            with self.subTest(payload=payload), self.assertRaisesRegex(BacktestError, "invalid_json"):
                loads_document(payload)
        self.assertEqual(loads_document('{"x":null}'), {"x": None})

    def test_non_json_python_values_are_not_silently_coerced(self):
        for value in ({"x": (1,2)}, {2: "value"}, {"x": float("nan")}):
            with self.subTest(value=value), self.assertRaisesRegex(BacktestError, "invalid_json"):
                validate_document(value)


@unittest.skipUnless(HAS_VALIDATOR, "install the companion validation extra to run schema tests")
class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = schema_case()
        cls.request = json.loads(input_json(cls.result))
        cls.document = json.loads(cls.result.to_json())

    def assert_valid(self, document):
        report = validate_document(document)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["semantic_checks"], "not_run")
        self.assertEqual(report["financial_acceptance"], "not_evaluated")

    def test_current_request_result_and_checkpoint_with_real_lifecycle(self):
        self.assert_valid(self.request)
        self.assert_valid(self.document)
        self.assert_valid(self.document["checkpoint"])
        self.assertTrue(self.document["ledger"][2]["receivables"])
        self.assertTrue(self.document["fills"])
        self.assertTrue(any(f["kind"] == "external_capital" for f in self.document["cashflows"]))

    def test_checked_in_wire_goldens(self):
        goldens = sorted((REPO / "fixtures/backtesting/wire").glob("*.golden.json"))
        self.assertEqual(len(goldens), 5)
        for path in goldens:
            with self.subTest(path=path.name):
                self.assert_valid(loads_document(path.read_text(encoding="utf-8")))

    def test_missing_optional_dependency_is_actionable(self):
        _validators.cache_clear()
        try:
            with patch.dict(sys.modules, {"jsonschema": None}), self.assertRaisesRegex(BacktestError, "missing_validation_dependency"):
                validate_document(self.request)
        finally:
            _validators.cache_clear()

    def test_partial_and_failures_before_and_after_first_commit(self):
        for result in (schema_case(stop_after=1), schema_case(fail_at=0), schema_case(fail_at=1), schema_case(fail_at=4)):
            with self.subTest(status=result.status, observations=len(result.ledger)):
                self.assert_valid(json.loads(result.to_json()))
                self.assert_valid(data(result.checkpoint))
        self.assertFalse(schema_case(fail_at=0).ledger)

    def test_all_strategies_and_disabled_optional_reports(self):
        for strategy in ("unhedged", "delta", "scalar_vega", "vega_kt"):
            self.assert_valid(json.loads(schema_case(strategy=strategy).to_json()))
        inputs = load_input(input_json(self.result))
        inputs["history"] = [replace(s, numeraire=None) for s in inputs["history"]]
        inputs["config"] = replace(inputs["config"], numeraire_id=None)
        self.assert_valid(json.loads(run_backtest(**inputs, provider=DatedProvider()).to_json()))

    def test_request_defaults_still_load(self):
        request = deepcopy(self.request)
        request["config"] = {"numeraire_id": "common", "constraint_mode": "execution_cash", "gross_notional_limit": None}
        for c in request["contracts"]:
            for name in ("side", "multiplier", "lot", "underlying_id", "currency_id"):
                del c[name]
        del request["portfolio"][0]["entry_fee"]
        for s in request["market_history"]:
            for name in ("lending_rate", "borrowing_rate", "stock_borrow_rate", "dividend_per_share", "pricing_requests"):
                del s[name]
            for name in ("method", "coordinate", "delta_convention", "unit"):
                del s["basis"][name]
        self.assert_valid(request)
        self.assertEqual(load_input(json.dumps(request))["config"].strategy, "vega_kt")

    def test_unknown_missing_and_wrong_type_fields(self):
        mutations = [(lambda d: d.update(unknown=True)), (lambda d: d.pop("portfolio")),
                     (lambda d: d["config"].update(rebalance_every=True)),
                     (lambda d: d["market_history"][0]["spot"].update(bid="100")),
                     (lambda d: d["market_history"][0].pop("external_cashflows"))]
        for mutate in mutations:
            request = deepcopy(self.request)
            mutate(request)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(BacktestError, "schema_fields"):
                validate_document(request)

    def test_invalid_calendar_dates_and_naive_timestamps(self):
        for value in ("2026-02-30", "2026-13-01", "0000-01-01"):
            request = deepcopy(self.request)
            request["contracts"][0]["expiry"] = value
            with self.subTest(value=value), self.assertRaisesRegex(BacktestError, "schema_fields"):
                validate_document(request)
        for value in ("2026-09-07T16:00:00", "2026-02-30T16:00:00+00:00", "2026-09-07 16:00:00+00:00"):
            request = deepcopy(self.request)
            request["market_history"][0]["at"] = value
            with self.subTest(value=value), self.assertRaisesRegex(BacktestError, "schema_fields"):
                validate_document(request)

    def test_hash_shape_nonpositive_numeraire_and_empty_basis(self):
        for path, value in ((["manifest", "full_history_hash"], "bad"),
                            (["ledger", 0, "numeraire", "value"], 0),
                            (["manifest", "market_history", 0, "basis", "maturity_dates"], [])):
            document = deepcopy(self.document)
            target = document
            for key in path[:-1]: target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaisesRegex(BacktestError, "schema_fields"):
                validate_document(document)

    def test_settlement_rule_requires_corresponding_fixing(self):
        request = deepcopy(self.request)
        request["contracts"][0]["settlement_fixing_id"] = None
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(request)
        result = deepcopy(self.document)
        del result["settlements"][0]["settlement"]["fixing"]
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(result)

    def test_failed_status_requires_details_and_null_terminal_error(self):
        result = deepcopy(self.document)
        result["status"] = "failed"
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(result)
        result = json.loads(schema_case(fail_at=1).to_json())
        result["summary"]["terminal_hedging_error"] = 0.
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(result)

    def test_diagnostic_iv_pnl_cannot_be_claimed_as_adopted(self):
        for table, field, value in (("iv_alignment", "pnl_adopted", True),
                                    ("greek_pnl", "vega_status", "validated")):
            result = deepcopy(self.document)
            result[table][0][field] = value
            with self.subTest(table=table), self.assertRaisesRegex(BacktestError, "schema_fields"):
                validate_document(result)

    def test_optional_gamma_and_reporting_surface_pair_contract(self):
        result = deepcopy(self.document)
        risk = result["risks"][0]["instruments"]["target"]
        risk["gamma"] = None
        risk["gamma_spot_bump"] = .01
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(result)
        risk["gamma_spot_bump"] = None
        risk["reporting_iv"] = [.2]*4
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(result)

    def test_unmatched_iv_node_has_no_invented_zero_shock(self):
        result = deepcopy(self.document)
        node = result["iv_alignment"][1]["nodes"][0]
        self.assertNotEqual(node["status"], "matched")
        node["delta_iv"] = 0.
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(result)

    def test_embedded_request_syntax_checked_without_native_engine(self):
        for payload in ('{"x":NaN}', '{"x":1,"x":2}', 'not-json', '[]'):
            request = deepcopy(self.request)
            request["market_history"][0]["pricing_requests"]["target"] = payload
            with self.subTest(payload=payload), self.assertRaisesRegex(BacktestError, "invalid_embedded_json"):
                validate_document(request)
        request["market_history"][0]["pricing_requests"]["target"] = '{"custom":true}'
        self.assert_valid(request)  # Native product/model semantics are deliberately separate.

    def test_legacy_readers_remain_separate_from_current_schema_validation(self):
        result = deepcopy(self.document)
        result["manifest"]["schema_version"] = 8
        result.pop("iv_alignment")
        self.assertEqual(BacktestResult.from_json(json.dumps(result)).iv_alignment, ())
        with self.assertRaisesRegex(BacktestError, "schema_version"):
            validate_document(result)

    def test_semantic_mismatch_is_not_mislabeled_as_reconciliation(self):
        result = deepcopy(self.document)
        result["ledger"][-1]["equity"] += 100
        result["checkpoint"]["state_hash"] = "0"*64
        self.assert_valid(result)  # Explicitly structural: engine remains authoritative.

    def test_metadata_extensions_preserve_unvalidated_status(self):
        result = deepcopy(self.document)
        result["manifest"]["risk_validation"] = {"status": "unvalidated", "diagnostic_only": True,
                                                 "details": {"case": 123, "pricing_requests": 1}}
        self.assert_valid(result)
        self.assertEqual(result["manifest"]["risk_validation"]["status"], "unvalidated")
        self.assertEqual(validate_document(result)["risk_validation"]["status"], "unvalidated")

    def test_complete_incomplete_and_dependent_ensemble_reports(self):
        inputs = load_input(input_json(self.result))
        inputs.pop("config")
        results = compare_strategies(self.result_config(), **inputs, provider=DatedProvider())
        scenario = ScenarioResults("one", "one", "schema_fixture", results)
        for mapping in (results, {"delta": results["delta"]}):
            report = aggregate_scenarios([replace(scenario, results=mapping)])
            self.assert_valid(json.loads(report.to_json()))
        request = {"document_kind": "hedging_backtest_ensemble_request", "schema_version": 1,
                   "config": {}, "scenarios": [{"scenario_id": "one", "replication_id": "one", "source": "fixture", "results": {}}]}
        self.assert_valid(request)
        request["config"]["sampling_design"] = "independent_simulation"
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(request)
        request["scenarios"][0]["outer_randomness"] = {"algorithm": "PCG64", "domain": "outer", "seed": 1}
        self.assert_valid(request)

    def result_config(self):
        return load_input(input_json(self.result))["config"]

    def test_tail_reference_only_disallows_interval(self):
        report = aggregate_scenarios([ScenarioResults("one", "one", "fixture", {"delta": schema_case(strategy="delta")})])
        document = json.loads(report.to_json())
        tail = document["strategies"]["delta"]["loss_tail"][0]
        tail.update(status="reference_only", confidence_intervals={"quantile": [1,2], "expected_shortfall": [2,3]})
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(document)

    def test_cli_reports_each_input_and_exports_offline_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"good.json").write_text(json.dumps(self.request))
            (root/"bad.json").write_text('{"a":NaN}')
            result = subprocess.run([sys.executable, "-m", "rust_pricing_backtest.validation", str(root/"good.json"),
                                     str(root/"bad.json"), "--export-schemas", str(root/"schemas")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            reports = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual([r["status"] for r in reports], ["valid", "invalid"])
            self.assertEqual(len(list((root/"schemas").glob("*.json"))), 26)

    def test_validation_does_not_retrieve_network_schemas(self):
        _validators.cache_clear()
        with patch("urllib.request.urlopen", side_effect=AssertionError("network access forbidden")):
            self.assert_valid(self.request)


if __name__ == "__main__":
    unittest.main()
