"""Native-adapter integration, also discovered by the installed-wheel smoke suite."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtesting"))

from rust_pricing_backtest import (BacktestConfig, BacktestError, BacktestResult, Contract, ExternalCashflow, NumeraireObservation, PaymentDiscount, Position, Quote,
                                  RiskBasis, RustPricingProvider, SettlementFixing, Snapshot, compare_strategies, run_backtest)
from rust_pricing_backtest.serialization import input_json, load_input
from rust_pricing_backtest.domain import fingerprint
from rust_pricing_backtest.ensemble import ScenarioResults, aggregate_scenarios


class NativeBacktestTests(unittest.TestCase):
    def test_fixed_market_shocks_native_replay_audits_and_resume(self):
        cfg = BacktestConfig(ridge=1e-4, market_shock_log_moneyness=(-.1, .1))
        provider = RustPricingProvider(diagnostic_only=True)
        portfolio = [Position("target", -1., 1.)]
        results = compare_strategies(cfg, self.history, self.contracts, portfolio, provider)
        for strategy, result in results.items():
            partial = run_backtest(replace(cfg, strategy=strategy), self.history, self.contracts,
                                   portfolio, provider, stop_after=2)
            resumed = run_backtest(**load_input(input_json(result)), provider=provider, checkpoint=partial.checkpoint)
            self.assertEqual(resumed.to_json(), result.to_json())
            self.assertEqual(result.manifest["schema_version"], 11)
            self.assertEqual(result.summary["risk_validation_status"], "unvalidated")
            for row in result.risks:
                audit = row["market_shocks"]
                self.assertEqual(audit["reporting_basis_id"], row["basis_id"])
                self.assertEqual(audit["market_basis"]["maturity_dates"], row["basis"]["maturity_dates"])
                self.assertEqual(len(audit["held"]["market_buckets"]), 2*len(row["basis"]["maturity_dates"]))
                self.assertAlmostEqual(sum(audit["held"]["market_buckets"]), sum(row["held_buckets"]), places=11)
                self.assertEqual(audit["held"]["complement"]["market_hedge_effectiveness"], "not_established")
            if importlib.util.find_spec("jsonschema") is not None:
                from rust_pricing_backtest.validation import validate_document
                for doc in [json.loads(result.to_json()), json.loads(input_json(result)),
                            json.loads(json.dumps(result.checkpoint))]:
                    self.assertEqual(validate_document(doc)["status"], "valid")
        ensemble = aggregate_scenarios([ScenarioResults("market-shocks", "one", "synthetic_native", results)])
        self.assertEqual(ensemble.manifest["schema_version"], 3)
        if importlib.util.find_spec("jsonschema") is not None:
            self.assertEqual(validate_document(json.loads(ensemble.to_json()))["status"], "valid")

    @unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "install companion constraints extra")
    def test_decision_cash_gross_native_resume_and_budget_audit(self):
        config = BacktestConfig(ridge=1e-4, solver_tolerance=1e-8, constraint_mode="decision_cash_gross",
                                gross_notional_limit=500., cash_floor=-500., option_fee=.01, stock_fee=.001)
        provider = RustPricingProvider(diagnostic_only=True)
        portfolio = [Position("target", -1., 1.)]
        for strategy in ("unhedged", "delta", "scalar_vega", "vega_kt"):
            with self.subTest(strategy=strategy):
                cfg = replace(config, strategy=strategy)
                result = run_backtest(cfg, self.history, self.contracts, portfolio, provider)
                partial = run_backtest(cfg, self.history, self.contracts, portfolio, provider, stop_after=2)
                resumed = run_backtest(**load_input(input_json(result)), provider=provider, checkpoint=partial.checkpoint)
                self.assertEqual(resumed.to_json(), result.to_json())
                self.assertEqual(result.manifest["risk_validation"]["status"], "unvalidated")
                self.assertTrue(result.decisions)
                for decision in result.decisions:
                    self.assertGreaterEqual(decision["budget"]["rounded"]["cash_after"], cfg.cash_floor)
                    self.assertLessEqual(decision["budget"]["rounded"]["gross_notional"], cfg.gross_notional_limit)
                if importlib.util.find_spec("jsonschema") is not None:
                    from rust_pricing_backtest.validation import validate_document
                    self.assertEqual(validate_document(json.loads(result.to_json()))["status"], "valid")

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import rust_pricing as rp
        cls.rp = rp
        cls.start = date(2026, 9, 7)
        expiry = cls.start + timedelta(days=4)
        cls.contracts = [Contract("target", expiry, 100, lot=.1), Contract("hedge", expiry, 101, lot=.1)]
        cls.history = []
        for i in range(5):
            day = cls.start + timedelta(days=i)
            at = datetime.combine(day, datetime.min.time(), timezone.utc)
            dates = tuple(day+timedelta(days=j) for j in range(1, max(2, 4-i)+1))
            times = np.array([(d-day).days/365 for d in dates])
            k = np.array([-.1, 0., .1])
            basis = RiskBasis(day, dates, tuple(k))
            requests = {}
            if day < expiry:
                t = (expiry-day).days/365
                grid = np.array([0., *times[times <= t]])
                model = rp.Model.local_volatility_from_grid_with_reporting_basis(
                    grid, k, np.full(len(grid)*3, .04), 1e-8, 4., times, k, np.full(len(times)*3, .2))
                market = rp.Market.equity(2, 1, 100., rp.DiscountCurve(10, np.array([0., 1.]), np.array([1., 1.])),
                                         rp.DiscountCurve(11, np.array([0., 1.]), np.array([1., 1.])))
                for c in cls.contracts:
                    request = rp.PricingRequest(day, rp.Product.european_vanilla(1, 2, expiry, c.strike, 1., "call"),
                                                market, model, rp.Engine.pseudo_monte_carlo(7, 256, antithetic=True),
                                                rp.RiskRequest(delta=True, vega=True, vega_kt_maturity_nodes=dates,
                                                               vega_kt_log_forward_moneyness_nodes=k,
                                                               vega_kt_relative_density_threshold=1e-10))
                    requests[c.contract_id] = request.to_json()
            cls.history.append(Snapshot(f"native-{i}", at, Quote(100, 100, at, at),
                                         {"hedge": Quote(.4, .5, at, at)}, basis, at,
                                         "synthetic_native_integration", requests))

    def test_unvalidated_native_provider_requires_diagnostic_mode(self):
        with self.assertRaisesRegex(BacktestError, "unvalidated_vega_kt"):
            RustPricingProvider()

    def test_native_risk_matches_public_plan(self):
        provider = RustPricingProvider(diagnostic_only=True)
        s, c = self.history[0], self.contracts[0]
        actual = provider.evaluate(s, c)
        expected = self.rp.PricingPlan.compile(self.rp.PricingRequest.from_json(s.pricing_requests[c.contract_id]),
                                              worker_threads=1, reduction_block_size=256).evaluate()
        self.assertEqual(provider.validation["status"], "unvalidated")
        self.assertIn("unvalidated_vega_kt", actual.warning_codes)
        self.assertEqual(actual.price, expected.value)
        self.assertEqual(actual.delta, expected.delta_raw)
        self.assertEqual(actual.buckets, tuple(expected.vega_kt.raw_buckets))
        self.assertEqual(actual.native_policy, "flat_variance_terminal_measure_pathwise_v1")
        self.assertEqual(expected.vega_kt.truncation_order, "not_applicable")
        replayed_native = self.rp.PricingResult.from_json(expected.to_json())
        self.assertEqual(replayed_native.vega_kt.policy_label, actual.native_policy)
        self.assertEqual(len(actual.bucket_standard_errors), s.basis.size)
        self.assertGreater(max(actual.bucket_standard_errors), 0)
        self.assertEqual(actual, provider.evaluate(s, c))

    def test_native_four_strategies_and_json_replay(self):
        provider = RustPricingProvider(diagnostic_only=True)
        positions = [Position("target", -1, provider.evaluate(self.history[0], self.contracts[0]).price)]
        results = compare_strategies(BacktestConfig(ridge=1e-4), self.history, self.contracts, positions, provider)
        for result in results.values():
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.summary["risk_validation_status"], "unvalidated")
            self.assertTrue(result.manifest["risk_validation"]["diagnostic_only"])
            self.assertTrue(result.summary["fully_settled"])
            replay = run_backtest(**load_input(input_json(result)), provider=RustPricingProvider(diagnostic_only=True))
            self.assertEqual(result.to_json(), replay.to_json())

    @unittest.skipUnless(importlib.util.find_spec("jsonschema") is not None, "install the companion validation extra")
    def test_native_outputs_match_current_schemas_without_changing_validation_status(self):
        from rust_pricing_backtest.validation import loads_document, validate_document
        contracts, history = self.delayed_payment_case()
        results = compare_strategies(BacktestConfig(ridge=1e-4), history, contracts, [Position("target", -1., 1.)],
                                     RustPricingProvider(diagnostic_only=True))
        for result in results.values():
            self.assertEqual(validate_document(loads_document(input_json(result)))["status"], "valid")
            report = validate_document(loads_document(result.to_json()))
            self.assertEqual(report["risk_validation"]["status"], "unvalidated")
            self.assertEqual(report["financial_acceptance"], "not_evaluated")
            self.assertEqual(validate_document(loads_document(json.dumps(result.checkpoint, allow_nan=False)))["status"], "valid")

    def test_native_ensemble_freezes_pricing_randomness_and_risk_controls(self):
        from copy import deepcopy
        provider = RustPricingProvider(diagnostic_only=True)
        positions = [Position("target", -1, provider.evaluate(self.history[0], self.contracts[0]).price)]
        results = compare_strategies(BacktestConfig(ridge=1e-4), self.history, self.contracts, positions, provider)
        scenario = ScenarioResults("native", "native", "synthetic_fixture", results)
        report = aggregate_scenarios([scenario])
        self.assertEqual(report.manifest["paired_sample_count"], 1)
        self.assertEqual(report.strategies["vega_kt"]["risk_validation"]["status"], "unvalidated")
        for section, key, value in (("engine", "master_seed", 8), ("engine", "independent_sampling_units", 512),
                                    ("risk", "delta", False)):
            altered = deepcopy(results)
            manifest = altered["delta"].manifest
            s = manifest["market_history"][0]
            request = json.loads(s["pricing_requests"]["target"])
            request[section][key] = value
            s["pricing_requests"]["target"] = json.dumps(request)
            manifest["full_history_hash"] = fingerprint(manifest["market_history"])
            with self.subTest(key=key), self.assertRaisesRegex(BacktestError, "comparison_conditions_mismatch"):
                aggregate_scenarios([replace(scenario, results=altered)])

    def test_example_outer_randomness_is_reproducible_and_separate_from_pricing(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("scenario_example", ROOT / "examples/python/vega_kt_backtest.py")
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        a, contracts, positions, _ = example.build_case(32, diagnostic_only=True, market_seed=42)
        repeated, _, repeated_positions, _ = example.build_case(32, diagnostic_only=True, market_seed=42)
        b, _, other_positions, _ = example.build_case(32, diagnostic_only=True, market_seed=43)
        self.assertEqual(fingerprint(a), fingerprint(repeated))
        self.assertEqual(positions, repeated_positions)
        self.assertEqual(positions, other_positions)
        self.assertEqual(a[0].spot, b[0].spot)
        self.assertNotEqual(a[-1].spot, b[-1].spot)
        for history in (a, b):
            for s in history:
                for payload in s.pricing_requests.values():
                    self.assertEqual(json.loads(payload)["engine"]["master_seed"], 7)

    def test_inconsistent_contract_surface_and_grid_rejected(self):
        s, c = self.history[0], self.contracts[0]
        for mutation, code in (("notional", "request_mismatch"), ("grid", "unsupported_grid"), ("coordinate", "unsupported_risk_method")):
            with self.subTest(mutation=mutation):
                payload = json.loads(s.pricing_requests[c.contract_id])
                if mutation == "notional":
                    payload["product"]["notional"] = 100
                if mutation == "grid":
                    payload["model"]["local_variance_grid"]["time_nodes"][1] /= 2
                altered = replace(s, pricing_requests={**s.pricing_requests, c.contract_id: json.dumps(payload)})
                if mutation == "coordinate":
                    altered = replace(altered, basis=replace(altered.basis, coordinate="physical_spot_iv"))
                with self.assertRaisesRegex(BacktestError, code):
                    RustPricingProvider(diagnostic_only=True).evaluate(altered, c)
        payload = json.loads(s.pricing_requests["hedge"])
        payload["model"]["reporting_iv_basis"]["implied_volatilities"][0] = .21
        altered = replace(s, pricing_requests={**s.pricing_requests, "hedge": json.dumps(payload)})
        provider = RustPricingProvider(diagnostic_only=True)
        provider.evaluate(altered, c)
        with self.assertRaisesRegex(BacktestError, "surface_mismatch"):
            provider.evaluate(altered, self.contracts[1])

    def test_invalid_native_requests_produce_failed_report(self):
        s = self.history[1]
        payload = json.loads(s.pricing_requests["target"])
        payload["engine"]["type"] = "invalid_engine"
        for raw, code in (("{", "invalid_pricing_request"), ("[]", "invalid_pricing_request"),
                          (json.dumps(payload), "native_pricing_failure")):
            with self.subTest(code=code, raw=raw[:20]):
                bad = replace(s, pricing_requests={**s.pricing_requests, "target": raw})
                result = run_backtest(BacktestConfig(strategy="delta"), [self.history[0], bad, *self.history[2:]],
                                      self.contracts, [Position("target", -1, 1)],
                                      RustPricingProvider(diagnostic_only=True), on_error="return")
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.failure["code"], code)
                self.assertEqual(result.failure["snapshot_index"], 1)
                self.assertEqual(result.fills, ())
                self.assertEqual(len(result.ledger), 1)

    def test_native_pricing_error_is_distinct_from_programming_error(self):
        provider = RustPricingProvider(diagnostic_only=True)
        provider.rp = SimpleNamespace(PricingRequest=self.rp.PricingRequest, PricingError=self.rp.PricingError,
                                      DiscountCurve=self.rp.DiscountCurve,
                                      PricingPlan=SimpleNamespace(compile=Mock(side_effect=self.rp.PricingError("injected numerical failure"))))
        with self.assertRaisesRegex(BacktestError, "native_pricing_failure"):
            provider.evaluate(self.history[0], self.contracts[0])
        provider.rp.PricingPlan.compile.side_effect = RuntimeError("unexpected bug")
        with self.assertRaisesRegex(RuntimeError, "unexpected bug"):
            provider.evaluate(self.history[0], self.contracts[0])

    def test_native_cli_failure_export_and_repaired_resume(self):
        config = BacktestConfig(strategy="delta")
        positions = [Position("target", -1, 1)]
        whole = run_backtest(config, self.history, self.contracts, positions, RustPricingProvider(diagnostic_only=True))
        request = json.loads(input_json(whole))
        request["market_history"][1]["pricing_requests"].pop("target")
        with tempfile.TemporaryDirectory() as path:
            source, output = Path(path) / "input.json", Path(path) / "out"
            source.write_text(json.dumps(request), encoding="utf-8")
            command = [sys.executable, "-m", "rust_pricing_backtest", str(source), "--output", str(output), "--diagnostic-only"]
            # The companion is source-loaded above, not bundled in the native wheel.
            # Give its CLI the same explicit import path in a clean working directory.
            cli_env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [
                str(ROOT / "backtesting"), os.environ.get("PYTHONPATH"),
            ])))
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                       cwd=path, env=cli_env)
            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "failed")
            failed = BacktestResult.from_json((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(failed.failure["code"], "missing_pricing_request")
            source.write_text(input_json(whole), encoding="utf-8")
            completed = subprocess.run([*command, "--checkpoint", str(output / "checkpoint.json")],
                                       capture_output=True, text=True, timeout=30, cwd=path, env=cli_env)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((output / "result.json").read_text(encoding="utf-8"), whole.to_json())

    def test_native_four_strategies_with_explicit_expiry_fixing(self):
        contracts = [replace(c, settlement_rule="explicit_fixing", settlement_fixing_id="close") for c in self.contracts]
        last = self.history[-1]
        fix = SettlementFixing(last.at.date(), 110., last.at, last.at, "synthetic_native_fixing", "v1")
        history = [*self.history[:-1], replace(last, settlement_fixings={"close": fix})]
        results = compare_strategies(BacktestConfig(ridge=1e-4), history, contracts, [Position("target", -1, 1.)],
                                     RustPricingProvider(diagnostic_only=True))
        for result in results.values():
            self.assertTrue(result.summary["fully_settled"])
            self.assertEqual(result.summary["risk_validation_status"], "unvalidated")
            paid = next(f for f in result.cashflows if f["kind"] == "targets_settlement")
            self.assertEqual(paid["amount"], -10.)
            self.assertEqual(paid["settlement"]["fixing_id"], "close")
            replay = run_backtest(**load_input(input_json(result)), provider=RustPricingProvider(diagnostic_only=True))
            self.assertEqual(replay.to_json(), result.to_json())

    def delayed_payment_case(self):
        payment = self.contracts[0].expiry + timedelta(days=2)
        contracts = [replace(c, payment_date=payment, settlement_rule="explicit_fixing", settlement_fixing_id="close")
                     for c in self.contracts]
        history = []
        for i in range(7):
            if i < 5:
                s = self.history[i]
            else:
                day = self.start + timedelta(days=i)
                at = datetime.combine(day, datetime.min.time(), timezone.utc)
                basis = RiskBasis(day, (day + timedelta(days=1), day + timedelta(days=2)), (-.1, 0., .1))
                s = Snapshot(f"payment-{i}", at, Quote(100., 100., at, at), {}, basis, at, "synthetic_native_payment")
            requests = {}
            for cid, raw in s.pricing_requests.items():
                payload = json.loads(raw)
                payload["market"]["discount_curve"]["discount_factors"] = [1., math.exp(-.05)]
                requests[cid] = json.dumps(payload)
            discounts = {payment: PaymentDiscount(s.at.date(), math.exp(-.05*(payment-s.at.date()).days/365.),
                                                  s.at, s.at, "native_valuation_curve", "v1")} if i < 6 else {}
            fixings = {"close": SettlementFixing(s.at.date(), 110., s.at, s.at, "synthetic_native_fixing", "v1")} if i == 4 else {}
            history.append(replace(s, pricing_requests=requests, payment_discounts=discounts, settlement_fixings=fixings))
        return contracts, history

    def test_delayed_payment_scales_price_risks_and_uncertainty(self):
        contracts, history = self.delayed_payment_case()
        provider = RustPricingProvider(diagnostic_only=True)
        delayed = provider.evaluate(history[0], contracts[0])
        immediate = provider.evaluate(history[0], replace(contracts[0], payment_date=None))
        ratio = math.exp(-.05 * 2 / 365.)
        self.assertAlmostEqual(delayed.payment_scaling_factor, ratio, places=14)
        self.assertEqual(delayed.payment_date, contracts[0].payment_day)
        for field in ("price", "delta", "signed_residual", "pre_projection", "price_standard_error", "edge_sensitivity_abs"):
            self.assertAlmostEqual(getattr(delayed, field), getattr(immediate, field) * ratio, places=12)
        for actual, base in zip(delayed.buckets, immediate.buckets):
            self.assertAlmostEqual(actual, base * ratio, places=12)
        # Positive carry uses exact market-IV replication: sample uncertainty
        # is not applicable and must not be fabricated as zero variance.
        self.assertEqual(immediate.native_policy, "terminal_payoff_replication_iv_v1")
        self.assertIsNone(immediate.bucket_standard_errors)
        self.assertIsNone(delayed.bucket_standard_errors)
        self.assertNotEqual(delayed.provenance, immediate.provenance)

    def test_delayed_payment_curve_mismatch_and_missing_factor_rejected(self):
        contracts, history = self.delayed_payment_case()
        day = contracts[0].payment_day
        for discounts, code in (({}, "missing_payment_discount"),
                                ({day: replace(history[0].payment_discounts[day], factor=.9)}, "payment_discount_mismatch")):
            with self.subTest(code=code), self.assertRaisesRegex(BacktestError, code):
                RustPricingProvider(diagnostic_only=True).evaluate(replace(history[0], payment_discounts=discounts), contracts[0])

    def test_native_capital_flows_four_strategies_and_resume_with_unpaid_claims(self):
        contracts, history = self.delayed_payment_case()
        history = [replace(s, numeraire=NumeraireObservation(
            "common_account", 2, s.at.date(), 2.*math.exp(.02*(s.at.date()-self.start).days/365.),
            s.at, s.at, "synthetic_common_account", "v1")) for s in history]
        for i, amount in ((1, 10.), (6, -3.)):
            s = history[i]
            history[i] = replace(s, external_cashflows=[ExternalCashflow(
                f"capital-{i}", s.at, amount, 2, s.at, "synthetic_investor_capital", "v1")])
        positions = [Position("target", -1., 1.)]
        config = BacktestConfig(ridge=1e-4, initial_cash=50., numeraire_id="common_account")
        results = compare_strategies(config, history, contracts, positions, RustPricingProvider(diagnostic_only=True))
        self.assertEqual(len({r.manifest["external_cashflow_policy"]["input_hash"] for r in results.values()}), 1)
        for strategy, result in results.items():
            self.assertEqual(result.status, "completed")
            self.assertTrue(result.summary["fully_settled"])
            self.assertEqual(result.summary["external_capital"], {"count": 2, "net_cashflow": 7., "gross_inflows": 10., "gross_outflows": 3.})
            self.assertAlmostEqual(result.summary["terminal_hedging_error"], result.ledger[-1]["cash"]-57.)
            expected = result.ledger[-1]["cash"]/history[-1].numeraire.value - 25. - 10./history[1].numeraire.value + 3./history[6].numeraire.value
            self.assertAlmostEqual(result.summary["discounted_pnl"]["terminal_hedging_error"], expected)
            self.assertEqual(result.summary["risk_validation_status"], "unvalidated")
            partial = run_backtest(replace(config, strategy=strategy), history, contracts, positions,
                                    RustPricingProvider(diagnostic_only=True), stop_after=5)
            self.assertEqual(set(partial.checkpoint["state"]["external_cashflow_ids"]), {"capital-1"})
            self.assertTrue(partial.checkpoint["state"]["receivables"])
            replay = run_backtest(**load_input(input_json(result)), provider=RustPricingProvider(diagnostic_only=True),
                                 checkpoint=partial.checkpoint)
            self.assertEqual(replay.to_json(), result.to_json())

    def test_native_cost_basis_four_strategies_fixing_and_payment(self):
        contracts, history = self.delayed_payment_case()
        config = BacktestConfig(ridge=1e-4, option_fee=.01, stock_fee=.005, slippage_bps=2.)
        results = compare_strategies(config, history, contracts, [Position("target", -1., 1., entry_fee=.02)],
                                     RustPricingProvider(diagnostic_only=True))
        for result in results.values():
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.summary["risk_validation_status"], "unvalidated")
            fixed = result.cost_basis[4]["groups"]["target"]
            self.assertEqual(fixed["realized_pnl"], -9.)
            self.assertAlmostEqual(fixed["unrealized_pnl"], 10.*(1.-math.exp(-.05*2./365.)))
            self.assertEqual(result.cost_basis[4]["claims"]["targets:target"]["signed_cost"], -10.)
            for row, ledger in zip(result.cost_basis, result.ledger):
                self.assertLessEqual(abs(row["realized_pnl"] + row["unrealized_pnl"] - ledger["cumulative_pnl"]),
                                     row["cumulative_tolerance"])
                self.assertLessEqual(abs(row["daily_total_pnl"] - ledger["daily_pnl"]), row["daily_tolerance"])
            self.assertEqual(result.summary["cost_basis"]["unrealized_pnl"], 0.)
            self.assertAlmostEqual(result.summary["cost_basis"]["realized_pnl"], result.summary["terminal_hedging_error"])
            self.assertEqual(result.cost_basis[-1]["claims"], {})

    def with_gamma(self, history, *, move_spot=False, relative_bump=.01):
        output = []
        for i, s in enumerate(history):
            spot = 100.+i if move_spot else s.spot.mid
            requests = {}
            for cid, raw in s.pricing_requests.items():
                payload = json.loads(raw)
                payload["market"]["spot"] = spot
                payload["risk"]["gamma"] = {"bump": {"type": "relative", "value": relative_bump}}
                requests[cid] = json.dumps(payload)
            output.append(replace(s, spot=replace(s.spot, bid=spot, ask=spot), pricing_requests=requests))
        return output

    def test_native_optional_gamma_public_identity_and_payment_scale(self):
        contracts, history = self.delayed_payment_case()
        provider = RustPricingProvider(diagnostic_only=True)
        baseline = provider.evaluate(history[0], contracts[0])
        self.assertIsNone(baseline.gamma)
        matched = provider.evaluate(self.with_gamma(history, relative_bump=.0001)[0], contracts[0])
        self.assertEqual(matched.delta, baseline.delta)
        self.assertEqual(matched.price, baseline.price)
        history = self.with_gamma(history)
        s, c = history[0], contracts[0]
        actual = provider.evaluate(s, c)
        expected = self.rp.PricingPlan.compile(self.rp.PricingRequest.from_json(s.pricing_requests[c.contract_id]),
                                              worker_threads=1, reduction_block_size=256).evaluate()
        self.assertIsNotNone(actual.gamma)
        self.assertAlmostEqual(actual.gamma, expected.gamma_raw*math.exp(-.05*2./365.))
        self.assertAlmostEqual(actual.delta, expected.delta_raw*math.exp(-.05*2./365.))
        self.assertEqual(actual.gamma_method, expected.diagnostics.gamma_method)
        self.assertEqual(actual.gamma_method, "central_bump")
        self.assertEqual(actual.gamma_spot_bump, expected.diagnostics.gamma_spot_bump)
        self.assertEqual(actual.gamma_spot_bump, 1.)

    def test_native_greek_explanations_four_strategies_and_resume(self):
        contracts, history = self.delayed_payment_case()
        history = self.with_gamma(history, move_spot=True)
        config = BacktestConfig(ridge=1e-4, option_fee=.01)
        portfolio = [Position("target", -1., 1.)]
        results = compare_strategies(config, history, contracts, portfolio, RustPricingProvider(diagnostic_only=True))
        for strategy, result in results.items():
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.summary["risk_validation_status"], "unvalidated")
            prior = result.risks[0]["instruments"]["target"]
            row = result.greek_pnl[1]
            self.assertEqual(row["spot_change"], 1.)
            self.assertAlmostEqual(row["groups"]["target"]["delta_pnl"], -prior["delta"])
            self.assertAlmostEqual(row["groups"]["target"]["gamma_pnl"], -.5*prior["gamma"])
            self.assertIn("unvalidated_vega_kt", row["risk_warning_codes"])
            self.assertIsNone(row["groups"]["target"]["vega_pnl"])
            self.assertEqual(result.summary["greek_pnl"]["intervals_with_complete_gamma"], 6)
            for row, ledger in zip(result.greek_pnl, result.ledger):
                self.assertLessEqual(abs(row["total"]-ledger["daily_pnl"]), row["tolerance"])
                self.assertLessEqual(abs(row["cumulative_total"]-ledger["cumulative_pnl"]), row["cumulative_tolerance"])
            partial = run_backtest(replace(config, strategy=strategy), history, contracts, portfolio,
                                    RustPricingProvider(diagnostic_only=True), stop_after=3)
            replay = run_backtest(**load_input(input_json(result)), provider=RustPricingProvider(diagnostic_only=True),
                                 checkpoint=partial.checkpoint)
            self.assertEqual(replay.to_json(), result.to_json())

    def test_native_reporting_iv_and_forward_levels_are_not_payment_scaled(self):
        contracts, history = self.delayed_payment_case()
        provider = RustPricingProvider(diagnostic_only=True)
        s, c = history[0], contracts[0]
        risk = provider.evaluate(s,c)
        immediate = provider.evaluate(s,replace(c,payment_date=None))
        native = self.rp.PricingPlan.compile(self.rp.PricingRequest.from_json(s.pricing_requests[c.contract_id]),
                                            worker_threads=1,reduction_block_size=256).evaluate()
        self.assertEqual(risk.reporting_iv,tuple(x.implied_volatility for x in native.vega_kt.coordinates))
        self.assertEqual(risk.reporting_iv,immediate.reporting_iv)
        self.assertEqual(risk.reporting_forwards,immediate.reporting_forwards)
        for day,forward in zip(s.basis.maturity_dates,risk.reporting_forwards):
            self.assertAlmostEqual(forward,100.*math.exp(.05*(day-s.at.date()).days/365.),places=12)

    def test_native_iv_alignment_four_strategies_and_resume(self):
        contracts, history = self.delayed_payment_case()
        config = BacktestConfig(ridge=1e-4)
        portfolio = [Position("target",-1.,1.)]
        results = compare_strategies(config,history,contracts,portfolio,RustPricingProvider(diagnostic_only=True))
        for strategy,result in results.items():
            self.assertEqual(result.status,"completed")
            row = result.iv_alignment[1]
            self.assertEqual(row["total_nodes"],12)
            self.assertEqual(row["matched_nodes"],6)
            self.assertEqual(sum(n["status"] == "expired_maturity" for n in row["nodes"]),3)
            self.assertEqual(sum(n["status"] == "outside_strike_domain" for n in row["nodes"]),3)
            self.assertFalse(row["pnl_adopted"])
            self.assertIn("unvalidated_vega_kt",row["blockers"])
            self.assertIsNone(result.greek_pnl[1]["groups"]["target"]["vega_pnl"])
            self.assertEqual(result.summary["risk_validation_status"],"unvalidated")
            self.assertTrue(all(not r["pnl_adopted"] for r in result.iv_alignment))
            partial = run_backtest(replace(config,strategy=strategy),history,contracts,portfolio,
                                    RustPricingProvider(diagnostic_only=True),stop_after=3)
            replay = run_backtest(**load_input(input_json(result)),provider=RustPricingProvider(diagnostic_only=True),
                                 checkpoint=partial.checkpoint)
            self.assertEqual(replay.to_json(),result.to_json())

    def test_native_common_numeraire_and_delayed_payment_replay(self):
        contracts, history = self.delayed_payment_case()
        history = [replace(s, numeraire=NumeraireObservation(
            "common_account", 2, s.at.date(), 2.*math.exp(.02*(s.at.date()-self.start).days/365.),
            s.at, s.at, "synthetic_common_account", "v1")) for s in history]
        positions = [Position("target", -1., 1.)]
        config = BacktestConfig(ridge=1e-4, initial_cash=50., numeraire_id="common_account")
        results = compare_strategies(config, history, contracts, positions, RustPricingProvider(diagnostic_only=True))
        self.assertEqual(len({r.manifest["numeraire_policy"]["input_hash"] for r in results.values()}), 1)
        for strategy, result in results.items():
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.summary["risk_validation_status"], "unvalidated")
            self.assertTrue(result.summary["fully_settled"])
            for ledger, pnl in zip(result.ledger, result.pnl_components):
                expected = ledger["equity"]/ledger["numeraire"]["value"] - 25.
                self.assertAlmostEqual(ledger["discounted_cumulative_pnl"], expected)
                self.assertLessEqual(abs(pnl["discounted"]["cumulative_total"]-expected),
                                     pnl["discounted"]["cumulative_tolerance"])
            expected = result.ledger[-1]["cash"]/history[-1].numeraire.value - 25.
            self.assertAlmostEqual(result.summary["discounted_pnl"]["terminal_hedging_error"], expected)
            partial = run_backtest(replace(config, strategy=strategy), history, contracts, positions,
                                    RustPricingProvider(diagnostic_only=True), stop_after=5)
            replay = run_backtest(**load_input(input_json(result)), provider=RustPricingProvider(diagnostic_only=True),
                                 checkpoint=partial.checkpoint)
            self.assertEqual(replay.to_json(), result.to_json())

    def test_native_delayed_payments_four_strategies_and_resume(self):
        contracts, history = self.delayed_payment_case()
        positions = [Position("target", -1., 1.)]
        config = BacktestConfig(ridge=1e-4)
        results = compare_strategies(config, history, contracts, positions, RustPricingProvider(diagnostic_only=True))
        for strategy, result in results.items():
            self.assertEqual(result.status, "completed")
            self.assertFalse(result.ledger[4]["fully_settled"])
            self.assertTrue(result.summary["fully_settled"])
            target_claim = result.ledger[4]["receivables"]["targets:target"]
            self.assertEqual(target_claim["amount"], -10.)
            self.assertAlmostEqual(target_claim["present_value"], -10.*math.exp(-.05*2/365.), places=12)
            payments = [f for f in result.cashflows if f["kind"] == "targets_settlement"]
            self.assertEqual(len(payments), 1)
            self.assertEqual(payments[0]["at"], history[6].at.isoformat())
            self.assertEqual(payments[0]["amount"], -10.)
            self.assertEqual(len(result.pnl_components), len(history))
            for row, ledger in zip(result.pnl_components, result.ledger):
                self.assertLessEqual(abs(row["total"]-ledger["daily_pnl"]), row["tolerance"])
                self.assertLessEqual(abs(row["cumulative_total"]-ledger["cumulative_pnl"]), row["cumulative_tolerance"])
            self.assertAlmostEqual(result.summary["pnl_components"]["target"], -9.)
            self.assertTrue(result.summary["pnl_reconciliation"]["passed"])
            partial = run_backtest(replace(config, strategy=strategy), history, contracts, positions,
                                    RustPricingProvider(diagnostic_only=True), stop_after=5)
            replay = run_backtest(**load_input(input_json(result)), provider=RustPricingProvider(diagnostic_only=True),
                                 checkpoint=partial.checkpoint)
            self.assertEqual(replay.to_json(), result.to_json())


if __name__ == "__main__":
    unittest.main()
