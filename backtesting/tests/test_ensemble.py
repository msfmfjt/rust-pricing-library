from copy import deepcopy
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rust_pricing_backtest import (BacktestConfig, BacktestError, Contract, NumeraireObservation,
                                  Position, compare_strategies, run_backtest)
from rust_pricing_backtest.domain import canonical, data, fingerprint
from rust_pricing_backtest.ensemble import (EnsembleConfig, ScenarioResults, STRATEGIES,
                                          aggregate_scenarios, _improvement, _moments, _tail)
from test_backtest import START, EXPIRY, FixtureProvider, snapshot


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(ridge=0)
        self.contracts = [Contract("target", EXPIRY, 100), Contract("hedge", EXPIRY, 105)]
        self.portfolio = [Position("target", -1, 5)]
        self.provider = FixtureProvider({"target": (5., .5, (10., 20., 0., 0.)),
                                         "hedge": (3., .2, (5., 10., 0., 0.))})

    def scenario(self, index, *, config=None, numeraire=False, stop_after=None, history_length=5):
        history = [snapshot(i, spot=100. if i < 4 else 100.+index, prices={"hedge": 3.}) for i in range(history_length)]
        if numeraire:
            history = [replace(s, numeraire=NumeraireObservation("common", 2, s.at.date(), 2.+i/10,
                                                               s.at, s.at, "fixture", "v1"))
                       for i, s in enumerate(history)]
        results = compare_strategies(config or self.config, history, self.contracts, self.portfolio,
                                     self.provider, stop_after=stop_after)
        return ScenarioResults(f"s{index:03}", f"rep{index}", "hand_computed_fixture", results,
                               {"algorithm": "fixture_declared", "domain": "outer_market", "seed": index+100})

    def independent(self, **kwargs):
        return EnsembleConfig(sampling_design="independent_simulation", bootstrap_resamples=100,
                              minimum_ci_samples=3, **kwargs)

    def test_hand_computed_moments_and_tail_fractional_mass(self):
        moments = _moments([-10., -4., 2., 8.])
        self.assertAlmostEqual(moments["mean"], -1.)
        self.assertAlmostEqual(moments["standard_deviation"], math.sqrt(60.))
        self.assertAlmostEqual(moments["rmse"], math.sqrt(46.))
        # Sorted losses [-8, -2, 4, 10], upper 37.5% mass = 1.5 samples.
        tail = _tail([-10., -4., 2., 8.], .625)
        self.assertEqual(tail["quantile"], 4.)
        self.assertEqual(tail["effective_tail_observations"], 1.5)
        self.assertAlmostEqual(tail["expected_shortfall"], 8.)
        self.assertEqual(_tail([-10., -4., 2., 8.], .5)["quantile"], -2.)
        self.assertEqual(_tail([-10., -4., 2., 8.], .5)["expected_shortfall"], 7.)

    def test_loss_sign_ties_and_exact_tail_threshold(self):
        self.assertEqual(_tail([10., 20.], .95)["expected_shortfall"], -10.)
        self.assertEqual(_tail([-7.]*40, .95)["expected_shortfall"], 7.)
        self.assertEqual(_tail(range(400), .95)["effective_tail_observations"], 20.)
        self.assertEqual(_tail([1.], .99)["quantile"], -1.)

    def test_paired_actual_replays_and_no_daily_pseudoreplication(self):
        scenarios = [self.scenario(i) for i in (0, 2, 4)]
        report = aggregate_scenarios(scenarios)
        self.assertEqual(report.manifest["paired_sample_count"], 3)
        self.assertEqual(report.strategies["unhedged"]["mean"], 3.)
        self.assertAlmostEqual(report.strategies["delta"]["mean"], 4.)
        self.assertAlmostEqual(report.comparisons["unhedged"]["variance_improvement"], -3.)
        self.assertEqual(report.comparisons["unhedged"]["mean_error_difference"], -1.)
        self.assertIsNone(report.strategies["delta"]["confidence_intervals"])
        self.assertEqual(report.strategies["delta"]["confidence_interval_reason"], "dependence_not_modelled")
        self.assertEqual(report.manifest["overlapping_calendar_window_pairs"], 3)

    def test_partial_failed_and_missing_remove_whole_pair(self):
        good, partial, failed, missing = [self.scenario(i) for i in range(4)]
        partial.results["vega_kt"] = self.scenario(1, stop_after=2).results["vega_kt"]
        original = failed.results["scalar_vega"]
        bad = replace(original, status="failed", ledger=original.ledger[:2],
                      summary={**original.summary, "status": "failed", "observations": 2,
                               "fully_settled": False, "terminal_hedging_error": None},
                      failure={"code": "fixture_failure"})
        failed.results["scalar_vega"] = bad
        del missing.results["delta"]
        report = aggregate_scenarios([good, partial, failed, missing])
        self.assertEqual(report.manifest["paired_sample_count"], 1)
        self.assertEqual(report.manifest["status"], "incomplete_cohort")
        self.assertEqual(report.strategies["vega_kt"]["run_counts"]["partial"], 1)
        self.assertEqual(report.strategies["scalar_vega"]["run_counts"]["failed"], 1)
        self.assertEqual(report.strategies["delta"]["run_counts"]["missing"], 1)
        self.assertEqual(report.strategies["unhedged"]["terminal_eligible_count"], 4)
        self.assertEqual(report.scenarios[2]["runs"]["scalar_vega"]["failure_code"], "fixture_failure")
        for stats in report.strategies.values():
            self.assertEqual(stats["sample_count"], 1)
            self.assertIsNone(stats["standard_deviation"])

    def test_no_common_samples_and_unavailable_discounted_metric(self):
        report = aggregate_scenarios([self.scenario(0)], EnsembleConfig(metric="discounted"))
        self.assertEqual(report.manifest["paired_sample_count"], 0)
        self.assertIsNone(report.strategies["delta"]["mean"])
        self.assertIsNone(report.strategies["delta"]["rmse"])
        self.assertEqual(report.strategies["delta"]["loss_tail"][0]["status"], "unavailable")
        self.assertEqual(report.scenarios[0]["runs"]["delta"]["reason"], "metric_unavailable")

    def test_entire_missing_scenario_remains_in_planned_population(self):
        a, b = self.scenario(0), replace(self.scenario(1), results={})
        report = aggregate_scenarios([a, b])
        self.assertEqual(report.manifest["scenario_count"], 2)
        self.assertEqual(report.manifest["paired_sample_count"], 1)
        self.assertIsNone(report.scenarios[1]["full_history_hash"])
        for stats in report.strategies.values():
            self.assertEqual(stats["run_counts"]["missing"], 1)
        with self.assertRaisesRegex(BacktestError, "missing_comparison_identity"):
            aggregate_scenarios([b])

    def test_completed_with_open_holdings_is_not_terminal(self):
        config = replace(self.config, end_policy="mark_to_market")
        scenario = self.scenario(3, config=config)
        # Delta stock remains in mark-to-market mode even after target expiry.
        self.assertFalse(scenario.results["delta"].summary["fully_settled"])
        report = aggregate_scenarios([scenario])
        self.assertEqual(report.scenarios[0]["runs"]["delta"]["reason"], "unsettled")
        self.assertEqual(report.manifest["paired_sample_count"], 0)

    def test_discounted_samples_use_common_account_and_initial_normalization(self):
        config = replace(self.config, numeraire_id="common")
        scenarios = [self.scenario(i, config=config, numeraire=True) for i in (0, 2)]
        report = aggregate_scenarios(scenarios, EnsembleConfig(metric="discounted"))
        self.assertAlmostEqual(report.strategies["unhedged"]["mean"], 4./2.4)
        altered = deepcopy(scenarios[1])
        for result in altered.results.values():
            result.manifest["market_history"][0]["numeraire"]["value"] = 1.
            result.manifest["full_history_hash"] = fingerprint(result.manifest["market_history"])
        with self.assertRaisesRegex(BacktestError, "comparison_conditions_mismatch"):
            aggregate_scenarios([scenarios[0], altered], EnsembleConfig(metric="discounted"))

    def test_shared_bootstrap_preserves_pair_relationship_and_order(self):
        scenarios = [self.scenario(i) for i in range(1, 9)]
        policy = self.independent(loss_levels=(.5,), minimum_tail_observations=2)
        report = aggregate_scenarios(scenarios, policy)
        reverse = aggregate_scenarios(list(reversed(scenarios)), policy)
        self.assertEqual(report.to_json(), reverse.to_json())
        ci = report.strategies["unhedged"]["confidence_intervals"]
        self.assertIsNotNone(ci["mean"])
        # For positive terminal spot moves, error_unhedged = 2*error_delta - 5.
        interval = report.comparisons["unhedged"]["confidence_intervals"]["variance_improvement"]
        np.testing.assert_allclose(interval, [-3., -3.], atol=1e-12)
        self.assertIsNotNone(report.strategies["delta"]["loss_tail"][0]["confidence_intervals"])
        self.assertEqual(report.strategies["delta"]["loss_tail"][0]["status"], "estimate")

    def test_small_tail_sample_suppresses_tail_interval_not_point_value(self):
        report = aggregate_scenarios([self.scenario(i) for i in range(1, 9)], self.independent())
        tail = report.strategies["delta"]["loss_tail"][0]
        self.assertEqual(tail["status"], "reference_only")
        self.assertIsNotNone(tail["expected_shortfall"])
        self.assertIsNone(tail["confidence_intervals"])
        self.assertEqual(tail["confidence_interval_reason"], "insufficient_tail_samples")

    def test_undefined_bootstrap_denominators_are_not_silently_dropped(self):
        report = aggregate_scenarios([self.scenario(i) for i in (1, 2, 3)], self.independent())
        comparison = report.comparisons["unhedged"]
        self.assertGreater(comparison["undefined_variance_bootstrap_resamples"], 0)
        self.assertIsNone(comparison["confidence_intervals"]["variance_improvement"])
        self.assertIsNotNone(comparison["confidence_intervals"]["mean_error_difference"])
        self.assertEqual(_improvement([1, 2], [1, 1], 1e-12), None)
        self.assertEqual(_improvement([1, 2], [0, 1e-8], 1e-12), None)

    def test_independence_requires_provenance_and_unique_replications(self):
        a, b = self.scenario(0), self.scenario(1)
        for scenarios, error in (([replace(a, outer_randomness=None)], "missing_outer_randomness"),
                                 ([a, replace(b, replication_id=a.replication_id)], "dependent_replications"),
                                 ([a, replace(b, outer_randomness=a.outer_randomness)], "dependent_replications"),
                                 ([a, replace(b, source="another_generator")], "scenario_generator_mismatch")):
            with self.subTest(error=error), self.assertRaisesRegex(BacktestError, error):
                aggregate_scenarios(scenarios, self.independent())
        report = aggregate_scenarios([a, replace(b, replication_id=a.replication_id)])
        self.assertEqual(report.manifest["repeated_replication_count"], 1)
        self.assertIsNone(report.strategies["delta"]["confidence_intervals"])

    def test_duplicate_identity_or_history_rejected(self):
        a = self.scenario(0)
        with self.assertRaisesRegex(BacktestError, "duplicate_scenario"):
            aggregate_scenarios([a, a])
        with self.assertRaisesRegex(BacktestError, "duplicate_market_history"):
            aggregate_scenarios([a, replace(a, scenario_id="other")])

    def test_cross_strategy_market_change_rejected(self):
        a = self.scenario(0)
        a.results["delta"] = self.scenario(1).results["delta"]
        with self.assertRaisesRegex(BacktestError, "paired_history_mismatch"):
            aggregate_scenarios([a])

    def test_costs_book_source_and_horizon_must_match(self):
        a = self.scenario(0)
        variants = (("config", "option_fee", .02), ("config", "initial_cash", 100.),
                    (None, "implementation_hash", "changed"), (None, "provider", "changed"))
        for section, key, value in variants:
            b = deepcopy(self.scenario(1))
            manifest = b.results["delta"].manifest
            (manifest[section] if section else manifest)[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(BacktestError, "comparison_conditions_mismatch"):
                aggregate_scenarios([a, b])
        b = self.scenario(1, config=replace(self.config, option_fee=.02))
        with self.assertRaisesRegex(BacktestError, "comparison_conditions_mismatch"):
            aggregate_scenarios([a, b])
        with self.assertRaisesRegex(BacktestError, "comparison_conditions_mismatch"):
            aggregate_scenarios([a, self.scenario(1, history_length=4)])

    def test_corrupt_hash_status_and_nonfinite_terminal_rejected(self):
        for kind in ("hash", "completed", "terminal", "infinite"):
            a = deepcopy(self.scenario(0))
            result = a.results["delta"]
            if kind == "hash":
                result.manifest["full_history_hash"] = "bad"
            elif kind == "completed":
                a.results["delta"] = replace(result, ledger=result.ledger[:1])
            else:
                result.summary["terminal_hedging_error"] = float("inf") if kind == "infinite" else 123.
            with self.subTest(kind=kind), self.assertRaisesRegex(BacktestError, "invalid_ensemble_result"):
                aggregate_scenarios([a])

    def test_unvalidated_risk_status_never_upgraded(self):
        scenario = self.scenario(0)
        validation = {"status": "unvalidated", "diagnostic_only": True, "blocker": "paper_pipeline"}
        for result in scenario.results.values():
            result.manifest["risk_validation"] = validation
        report = aggregate_scenarios([scenario])
        self.assertEqual(report.manifest["risk_validation"], validation)
        self.assertEqual(report.strategies["vega_kt"]["risk_validation"], validation)

    def test_large_finite_moments_and_overflow(self):
        self.assertAlmostEqual(_moments([1e200, -1e200])["rmse"]/1e200, 1.)
        with self.assertRaisesRegex(BacktestError, "ensemble_nonfinite"):
            _moments([1.7e308, -1.7e308])

    def test_policy_validation(self):
        for args in ({"loss_levels": [.99, .95]}, {"loss_levels": [1.]}, {"loss_levels": [.95, .95]},
                     {"bootstrap_resamples": 1}, {"minimum_ci_samples": 1}, {"bootstrap_seed": True},
                     {"minimum_tail_observations": 0}, {"minimum_baseline_variance": -1.}):
            with self.subTest(args=args), self.assertRaisesRegex(BacktestError, "invalid_ensemble_config"):
                EnsembleConfig(**args)

    def test_cli_round_trip_csv_and_incomplete_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenario = self.scenario(2)
            for strategy, result in scenario.results.items():
                result.write(root / strategy)
            request = {"document_kind": "hedging_backtest_ensemble_request", "schema_version": 1,
                       "config": data(EnsembleConfig()), "scenarios": [{"scenario_id": scenario.scenario_id,
                       "replication_id": scenario.replication_id, "source": scenario.source,
                       "outer_randomness": scenario.outer_randomness,
                       "results": {k: f"{k}/result.json" for k in STRATEGIES}}]}
            request_path = root / "request.json"
            request_path.write_text(canonical(request), encoding="utf-8")
            command = [sys.executable, "-m", "rust_pricing_backtest.ensemble", str(request_path), "--output", str(root / "out")]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = aggregate_scenarios([scenario]).to_json()
            self.assertEqual((root / "out/ensemble.json").read_text(), expected)
            with (root / "out/scenario_runs.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            del request["scenarios"][0]["results"]["vega_kt"]
            request_path.write_text(canonical(request), encoding="utf-8")
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            with (root / "out/scenario_runs.csv").open() as stream:
                missing = next(r for r in csv.DictReader(stream) if r["strategy"] == "vega_kt")
            self.assertEqual(missing["terminal_error"], "")
            request["schema_version"] = 123
            request_path.write_text(canonical(request), encoding="utf-8")
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)


if __name__ == "__main__":
    unittest.main()
