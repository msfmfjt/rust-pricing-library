from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rust_pricing_backtest import (BacktestConfig, BacktestError, BacktestResult, BacktestRunError, Contract,
                                  Position, Quote, Risk, RiskBasis, Snapshot, run_backtest,
                                  compare_strategies)
from rust_pricing_backtest.optimizer import solve_hedge, round_lot
from rust_pricing_backtest.serialization import input_json, load_input


START = date(2026, 9, 7)
EXPIRY = START + timedelta(days=4)


def snapshot(i, spot=100.0, prices=None, **kwargs):
    at = datetime.combine(START + timedelta(days=i), datetime.min.time(), timezone.utc) + timedelta(hours=16)
    def q(price):
        if isinstance(price, tuple):
            return Quote(*price, at, at)
        return Quote(price, price, at, at)
    basis = RiskBasis(at.date(), (at.date()+timedelta(days=7), at.date()+timedelta(days=21)), (-.2, .2))
    return Snapshot(f"s{i}", at, q(spot), {k: q(v) for k, v in (prices or {}).items()}, basis,
                    at, "hand_computed_fixture", **kwargs)


class FixtureProvider:
    identity = "hand_computed_fixture_v1"

    def __init__(self, values):
        self.values = values

    def evaluate(self, s, c):
        price, delta, buckets = self.values[c.contract_id]
        if callable(price):
            price = price(s)
        return Risk(price, delta, tuple(buckets), s.basis.basis_id,
                    pre_projection=math.fsum(buckets), provenance=self.identity)


class SolverTests(unittest.TestCase):
    def test_exact_matrix_and_units(self):
        a = np.array([[2., 1.], [1., 3.]])
        expected = np.array([-4., 2.])
        v = -a @ expected
        result = solve_hedge(v, a, [0, 0], [-100, -100], [100, 100], [0, 0], ridge=0, tolerance=1e-12)
        np.testing.assert_allclose(result.quantities, expected, atol=1e-8)
        self.assertEqual(result.rank, 2)
        raw = solve_hedge([10], [[2]], [0], [-100], [100], [0], ridge=0)
        scaled = solve_hedge([.1], [[.02]], [0], [-100], [100], [0], ridge=0)
        self.assertEqual(raw.quantities, scaled.quantities)

    def test_box_l1_and_rank_deficiency(self):
        result = solve_hedge([10], [[1]], [0], [-3], [3], [1], ridge=0)
        self.assertEqual(result.quantities, (-3.,))
        result = solve_hedge([10], [[1]], [0], [-100], [100], [1], ridge=0, turnover_penalty=1)
        self.assertEqual(result.quantities, (0.,))
        result = solve_hedge([1, 2], [[1, 1], [2, 2]], [0, 0], [-10, -10], [10, 10], [0, 0], ridge=0)
        self.assertEqual(result.rank, 1)
        self.assertIsNone(result.condition_number)
        self.assertLess(result.objective_after, 1e-20)

    def test_known_active_constraint_optimum(self):
        # Independent dense grid oracle for a non-diagonal constrained case.
        v, a = np.array([2., -.5]), np.array([[1., .5], [.2, 1.]])
        result = solve_hedge(v, a, [0, 0], [-1, -1], [1, 1], [0, 0], ridge=.0001, tolerance=1e-12)
        best = min(np.linalg.norm(.01*(v+a @ np.array([x, y])))**2 + .0001*(x*x+y*y)
                   for x in np.linspace(-1, 1, 101) for y in np.linspace(-1, 1, 101))
        self.assertLessEqual(result.objective_after, best + 1e-12)

    def test_lot_bounds_and_invalid_inputs(self):
        self.assertEqual(round_lot(.95, 1, .9, 2), 1)
        self.assertEqual(round_lot(-1.9, 1), -1)
        with self.assertRaisesRegex(BacktestError, "lot_infeasible"):
            round_lot(.5, 1, .2, .9)
        with self.assertRaisesRegex(BacktestError, "solver_nonfinite"):
            solve_hedge([float("nan")], [[1]], [0], [-1], [1], [0])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.contracts = [Contract("target", EXPIRY, 100), Contract("hedge", EXPIRY, 105)]
        self.positions = [Position("target", -1, 5)]
        self.provider = FixtureProvider({"target": (5., .5, (10., 20., 0., 0.)),
                                         "hedge": (3., .2, (5., 10., 0., 0.))})
        self.history = [snapshot(i, prices={"target": 5., "hedge": 3.}) for i in range(5)]

    def run_case(self, config=None, history=None, **kwargs):
        return run_backtest(config or BacktestConfig(ridge=0), history or self.history,
                            self.contracts, self.positions, self.provider, **kwargs)

    def test_no_trades_no_changes(self):
        result = self.run_case(BacktestConfig(strategy="unhedged", end_policy="mark_to_market"), self.history[:3])
        self.assertEqual([r["daily_pnl"] for r in result.ledger], [0, 0, 0])
        self.assertEqual(result.fills, ())

    def test_vega_then_delta_and_settlement(self):
        result = self.run_case()
        self.assertAlmostEqual(result.orders[0]["desired"]["hedge"], 2.)
        self.assertAlmostEqual(result.orders[0]["stock"], .1)
        self.assertEqual(result.fills[0]["at"], self.history[1].at.isoformat())
        self.assertEqual(result.fills[0]["decision_at"], self.history[0].at.isoformat())
        self.assertEqual(result.ledger[-1]["hedges"], {})
        self.assertTrue(result.summary["fully_settled"])
        self.assertEqual(len([f for f in result.cashflows if f["kind"] == "targets_settlement"]), 1)
        self.assertEqual(len([f for f in result.cashflows if f["kind"] == "hedges_settlement"]), 1)
        self.assertAlmostEqual(result.summary["terminal_hedging_error"], -1.)

    def test_multiplier_and_identical_contract(self):
        contracts = [Contract("target", EXPIRY, 100, multiplier=10), Contract("copy", EXPIRY, 100, multiplier=2)]
        provider = FixtureProvider({"target": (5, .5, (10., 0., 0., 0.)), "copy": (5, .5, (10., 0., 0., 0.))})
        history = [snapshot(i, prices={"copy": 5.}) for i in range(5)]
        result = run_backtest(BacktestConfig(ridge=0, allow_identical_hedge=True), history, contracts,
                              [Position("target", -1, 5)], provider)
        self.assertEqual(result.orders[0]["desired"]["copy"], 5.)
        self.assertEqual(result.orders[0]["stock"], 0.)
        self.assertEqual(result.summary["terminal_hedging_error"], 0.)
        excluded = run_backtest(BacktestConfig(ridge=0), history, contracts, [Position("target", -1, 5)], provider)
        self.assertEqual(excluded.decisions[0]["eligible"], [])

    def test_spread_and_fees_counted_once(self):
        config = BacktestConfig(ridge=0, option_fee=.1, stock_fee=.01, slippage_bps=10, end_policy="mark_to_market", rebalance_every=10)
        history = [snapshot(0, prices={"hedge": (2.9, 3.1)}), snapshot(1, spot=(99.9, 100.1), prices={"hedge": (2.9, 3.1)})]
        result = self.run_case(config, history)
        paid = math.fsum(f["execution_cost"] for f in result.fills)
        self.assertAlmostEqual(result.summary["cumulative_pnl"], -paid)
        self.assertAlmostEqual(result.ledger[-1]["cash"], 5 + math.fsum(f["cash_change"] for f in result.fills))

    def test_hand_computed_funding_and_dividend(self):
        # Long target with Delta -1 implies +1 stock, costing 100 on day 1.
        provider = FixtureProvider({"target": (5, -1, (0., 0., 0., 0.))})
        history = [snapshot(0, borrowing_rate=.1), snapshot(1, borrowing_rate=.1), snapshot(2, dividend_per_share=2)]
        result = run_backtest(BacktestConfig(strategy="delta", end_policy="mark_to_market", rebalance_every=10), history,
                              [self.contracts[0]], [Position("target", 1, 5)], provider)
        c1 = -5 * math.exp(.1/365) - 100
        expected = c1 * math.exp(.1/365) + 2
        self.assertAlmostEqual(result.ledger[-1]["cash"], expected)
        self.assertEqual(result.ledger[-1]["stock"], 1.)

    def test_short_stock_borrow_and_dividend(self):
        provider = FixtureProvider({"target": (5, 1, (0., 0., 0., 0.))})
        history = [snapshot(0), snapshot(1, stock_borrow_rate=.1), snapshot(2, dividend_per_share=2)]
        result = run_backtest(BacktestConfig(strategy="delta", end_policy="mark_to_market", rebalance_every=10), history,
                              [self.contracts[0]], [Position("target", 1, 5)], provider)
        self.assertAlmostEqual(result.ledger[-1]["cash"], 95 - 100*.1/365 - 2)

    def test_future_market_changes_do_not_change_past_orders(self):
        first = self.run_case()
        changed = self.history[:2] + [snapshot(i, spot=120, prices={"hedge": 12., "target": 25.}) for i in range(2, 5)]
        second = self.run_case(history=changed)
        for x, y in zip(first.orders[:2], second.orders[:2]):
            self.assertEqual(x["desired"], y["desired"])
            self.assertEqual(x["stock"], y["stock"])
        self.assertNotEqual(first.fills[0]["price"], second.fills[-1]["price"])

    def test_reject_future_surface_and_stale_held_quote(self):
        bad = replace(self.history[0], surface_available_at=self.history[0].at + timedelta(seconds=1))
        with self.assertRaisesRegex(BacktestError, "future_surface"):
            self.run_case(history=[bad, *self.history[1:]])
        stale = Quote(3, 3, self.history[0].at, self.history[0].at)
        bad = replace(self.history[2], option_quotes={"hedge": stale})
        with self.assertRaisesRegex(BacktestError, "stale_quote"):
            self.run_case(history=[*self.history[:2], bad, *self.history[3:]])

    def test_expiry_snapshot_required(self):
        with self.assertRaisesRegex(BacktestError, "missing_settlement"):
            self.run_case(history=[*self.history[:4], snapshot(5, prices={"hedge": 3})])

    def test_checkpoint_roundtrip_and_corruption(self):
        whole = self.run_case()
        first = self.run_case(stop_after=2)
        resumed = self.run_case(checkpoint=json.loads(json.dumps(first.checkpoint)))
        self.assertEqual(first.status, "partial")
        self.assertEqual(whole.to_json(), resumed.to_json())
        from_partial_input = run_backtest(**load_input(input_json(first)), provider=self.provider, checkpoint=first.checkpoint)
        self.assertEqual(whole.to_json(), from_partial_input.to_json())
        self.assertEqual(BacktestResult.from_json(whole.to_json()).to_json(), whole.to_json())
        broken = json.loads(json.dumps(first.checkpoint))
        broken["state"]["cash"] += 1
        with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
            self.run_case(checkpoint=broken)
        changed = [replace(self.history[0], source="revised"), *self.history[1:]]
        with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
            self.run_case(history=changed, checkpoint=first.checkpoint)

    def test_held_contract_outside_candidates_remains_in_risk(self):
        # A widened spread excludes new trading, but must not erase the two held contracts.
        history = [*self.history[:2], snapshot(2, prices={"hedge": (0., 6.)}), *self.history[3:]]
        result = self.run_case(history=history)
        self.assertEqual(result.ledger[2]["hedges"], {"hedge": 2.})
        self.assertEqual(result.decisions[2]["eligible"], [])
        self.assertEqual(result.orders[2]["desired"]["hedge"], 2.)
        self.assertEqual(result.risks[2]["held_buckets"], [0., 0., 0., 0.])

    def test_atomic_cash_rejection(self):
        config = BacktestConfig(ridge=0, cash_floor=-1, end_policy="mark_to_market")
        result = self.run_case(config, self.history[:2])
        self.assertEqual(result.orders[0]["status"], "rejected_cash_limit")
        self.assertEqual(result.ledger[-1]["hedges"], {})
        self.assertEqual(result.ledger[-1]["stock"], 0)
        self.assertEqual(result.fills, ())

    def test_basis_mismatch_and_quality(self):
        class Wrong(FixtureProvider):
            def evaluate(self, s, c):
                return replace(super().evaluate(s, c), basis_id="wrong")
        with self.assertRaisesRegex(BacktestError, "basis_mismatch"):
            run_backtest(BacktestConfig(), self.history, self.contracts, self.positions, Wrong(self.provider.values))
        class Noisy(FixtureProvider):
            def evaluate(self, s, c):
                return replace(super().evaluate(s, c), price_standard_error=10)
        with self.assertRaisesRegex(BacktestError, "risk_quality"):
            run_backtest(BacktestConfig(max_price_standard_error=1), self.history, self.contracts, self.positions,
                         Noisy(self.provider.values))

    def test_four_strategies_and_files(self):
        results = compare_strategies(BacktestConfig(ridge=0), self.history, self.contracts, self.positions, self.provider)
        self.assertEqual(list(results), ["unhedged", "delta", "scalar_vega", "vega_kt"])
        with tempfile.TemporaryDirectory() as path:
            results["vega_kt"].write(path)
            self.assertEqual(len(list(Path(path).glob("*.csv"))), 11)
            self.assertEqual(BacktestResult.from_json(Path(path, "result.json").read_text()).summary,
                             results["vega_kt"].summary)

    def test_negative_integer_configuration_rejected(self):
        with self.assertRaises(BacktestError):
            BacktestConfig(option_fee=-1)

    def test_scalar_roll_closes_previous_contract(self):
        contracts = [self.contracts[0], Contract("near100", EXPIRY, 101, lot=.1), Contract("near110", EXPIRY, 111, lot=.1)]
        provider = FixtureProvider({"target": (5, .5, (10., 0., 0., 0.)),
                                    "near100": (3, .2, (5., 0., 0., 0.)),
                                    "near110": (3, .2, (5., 0., 0., 0.))})
        history = [snapshot(i, spot=100 if i == 0 else 110, prices={"near100": 3, "near110": 3}) for i in range(3)]
        result = run_backtest(BacktestConfig(strategy="scalar_vega", ridge=0, end_policy="mark_to_market"), history,
                              contracts, self.positions, provider)
        self.assertEqual(result.orders[0]["desired"], {"near100": 2.})
        self.assertEqual(result.orders[1]["desired"], {"near100": 0., "near110": 2.})
        self.assertEqual(result.ledger[-1]["hedges"], {"near110": 2.})

    def test_risk_price_does_not_read_future_quotes(self):
        future = replace(self.history[0].option_quotes["hedge"], available_at=self.history[1].at)
        altered = replace(self.history[0], option_quotes={"hedge": future})
        result = self.run_case(history=[altered, *self.history[1:]])
        self.assertEqual(result.decisions[0]["eligible"], [])
        self.assertEqual(result.orders[0]["desired"], {})

    def test_failure_after_fills_rolls_back_and_resumes_exactly(self):
        history = [replace(s, lending_rate=.02, borrowing_rate=.04) for s in self.history]
        partial = self.run_case(history=history, stop_after=1)
        whole = self.run_case(history=history)
        original = self.provider.evaluate

        def fail_second_mark(s, c):
            if s.snapshot_id == "s1":
                raise BacktestError("risk_quality", "injected after pending fills")
            return original(s, c)

        with patch.object(self.provider, "evaluate", side_effect=fail_second_mark):
            with self.assertRaises(BacktestRunError) as caught:
                self.run_case(history=history)
        failed = caught.exception.result
        self.assertIsInstance(caught.exception.__cause__, BacktestError)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.checkpoint, partial.checkpoint)
        self.assertEqual(failed.orders, partial.orders)
        self.assertEqual(failed.fills, ())
        self.assertEqual(failed.cashflows, partial.cashflows)
        self.assertEqual(failed.failure["code"], "risk_quality")
        self.assertEqual(failed.failure["snapshot_index"], 1)
        self.assertEqual(failed.failure["last_completed_at"], history[0].at.isoformat())
        self.assertEqual(failed.failure["pending_order"]["status"], "pending")
        self.assertEqual(failed.failure["remaining_state"], {"cash": 5., "stock": 0., "targets": {"target": -1.}, "hedges": {}, "receivables": {}})
        self.assertEqual(failed.summary["coverage_fraction"], .2)
        self.assertIsNone(failed.summary["terminal_hedging_error"])
        resumed = self.run_case(history=history, checkpoint=failed.checkpoint)
        self.assertEqual(resumed.to_json(), whole.to_json())

    def test_failure_after_mark_and_before_order_discards_day(self):
        from rust_pricing_backtest.optimizer import solve_hedge as original
        calls = 0

        def fail_second_solver(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise BacktestError("solver_nonconvergence", "injected after mark")
            return original(*args, **kwargs)

        partial = self.run_case(stop_after=1)
        with patch("rust_pricing_backtest.engine.solve_hedge", side_effect=fail_second_solver):
            failed = self.run_case(on_error="return")
        self.assertEqual(failed.checkpoint, partial.checkpoint)
        self.assertEqual(failed.risks, partial.risks)
        self.assertEqual(failed.ledger, partial.ledger)
        self.assertEqual(failed.decisions, partial.decisions)

    def test_initial_failure_has_null_pnl_and_resumable_entry(self):
        bad = replace(self.history[0], surface_available_at=self.history[0].at + timedelta(seconds=1))
        failed = self.run_case(history=[bad, *self.history[1:]], on_error="return")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.ledger, ())
        self.assertEqual(failed.checkpoint["state"]["next_index"], 0)
        self.assertEqual(failed.failure["remaining_state"]["cash"], 5.)
        self.assertIsNone(failed.failure["last_completed_at"])
        for key in ("cumulative_pnl", "daily_mean", "daily_standard_deviation", "daily_rmse", "worst_day",
                    "max_drawdown_currency", "terminal_hedging_error", "fully_settled"):
            self.assertIsNone(failed.summary[key], key)
        self.assertEqual(failed.summary["coverage_fraction"], 0.)
        replayed = BacktestResult.from_json(failed.to_json())
        resumed = self.run_case(checkpoint=replayed.checkpoint)
        self.assertEqual(resumed.to_json(), self.run_case().to_json())
        self.assertEqual(len([f for f in resumed.cashflows if f["kind"] == "target_entry"]), 1)

    def test_failure_during_settlement_keeps_original_positions(self):
        # The first target settles, but the next is missing its expiry snapshot.
        contracts = [Contract("first", START + timedelta(days=2), 90),
                     Contract("missed", START + timedelta(days=1), 90)]
        portfolio = [Position("first", 1, 10), Position("missed", -2, 10)]
        provider = FixtureProvider({cid: (10, 0, (0, 0, 0, 0)) for cid in ("first", "missed")})
        history = [snapshot(0), snapshot(2)]
        config = BacktestConfig(strategy="unhedged")
        partial = run_backtest(config, history, contracts, portfolio, provider, stop_after=1)
        failed = run_backtest(config, history, contracts, portfolio, provider, on_error="return")
        self.assertEqual(failed.failure["code"], "missing_settlement")
        self.assertEqual(failed.checkpoint, partial.checkpoint)
        self.assertEqual(failed.failure["remaining_state"]["targets"], {"first": 1, "missed": -2})
        self.assertFalse(any("settlement" in f["kind"] for f in failed.cashflows))

    def test_failed_terminal_liquidation_and_retry(self):
        config = BacktestConfig(strategy="delta", cash_floor=0)
        positions = [Position("target", 1, 5)]
        config = replace(config, initial_cash=5.)
        history = [snapshot(0), snapshot(1), snapshot(2, spot=200)]
        provider = FixtureProvider({"target": (5, 1, (0, 0, 0, 0))})
        failed = run_backtest(config, history, self.contracts, positions, provider, on_error="return")
        self.assertEqual(failed.failure["code"], "cash_limit")
        self.assertEqual(failed.failure["remaining_state"]["stock"], -1.)
        self.assertEqual(failed.failure["remaining_state"]["cash"], 100.)
        self.assertEqual(len(failed.fills), 1)
        self.assertEqual(len(failed.ledger), 2)
        self.assertFalse(any(o["kind"] == "precommitted_terminal_liquidation" for o in failed.orders))
        repaired = [*history[:2], snapshot(2, spot=100)]
        resumed = run_backtest(config, repaired, self.contracts, positions, provider, checkpoint=failed.checkpoint)
        whole = run_backtest(config, repaired, self.contracts, positions, provider)
        self.assertEqual(resumed.to_json(), whole.to_json())

    def test_comparison_retains_failures_without_fabricating_observations(self):
        original = self.provider.evaluate

        def unavailable_hedge(s, c):
            if c.contract_id == "hedge":
                raise BacktestError("missing_pricing_request", c.contract_id)
            return original(s, c)

        with patch.object(self.provider, "evaluate", side_effect=unavailable_hedge):
            results = compare_strategies(BacktestConfig(), self.history, self.contracts,
                                         self.positions, self.provider, on_error="return")
        self.assertEqual([r.status for r in results.values()], ["completed", "completed", "failed", "failed"])
        self.assertEqual(results["vega_kt"].ledger, ())
        self.assertIsNone(results["vega_kt"].summary["cumulative_pnl"])

    def test_cli_writes_failure_and_returns_nonzero(self):
        from rust_pricing_backtest.__main__ import main
        bad = replace(self.history[1], surface_available_at=self.history[1].at + timedelta(seconds=1))
        result = self.run_case(history=[self.history[0], bad, *self.history[2:]], on_error="return")
        with tempfile.TemporaryDirectory() as path:
            source = Path(path) / "request.json"
            source.write_text(input_json(result), encoding="utf-8")
            output = Path(path) / "output"
            with patch("sys.argv", ["backtest", str(source), "--output", str(output), "--diagnostic-only"]), \
                 patch("rust_pricing_backtest.__main__.RustPricingProvider", return_value=self.provider), \
                 patch("builtins.print"):
                self.assertEqual(main(), 1)
            loaded = BacktestResult.from_json((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded.to_json(), result.to_json())
            self.assertEqual(json.loads((output / "checkpoint.json").read_text())["state"]["next_index"], 1)
            self.assertEqual(len(list(output.glob("*.csv"))), 11)

    def test_failure_reporting_preserves_preflight_and_unexpected_errors(self):
        with self.assertRaises(BacktestError) as caught:
            self.run_case(history=[self.history[1], self.history[0]], on_error="return")
        self.assertNotIsInstance(caught.exception, BacktestRunError)
        with patch.object(self.provider, "evaluate", side_effect=RuntimeError("unexpected programming error")):
            with self.assertRaisesRegex(RuntimeError, "unexpected programming error"):
                self.run_case(on_error="return")
        old = json.loads(self.run_case().to_json())
        old["manifest"]["schema_version"] = 1
        del old["failure"]
        self.assertIsNone(BacktestResult.from_json(json.dumps(old)).failure)


if __name__ == "__main__":
    unittest.main()
