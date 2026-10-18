from copy import deepcopy
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rust_pricing_backtest import BacktestConfig, BacktestError, BacktestResult, Contract, Position, run_backtest
from rust_pricing_backtest.pnl import COMPONENTS, attribute_pnl
from test_backtest import EXPIRY, FixtureProvider, snapshot


class PnlTests(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(ridge=0., rebalance_every=10, initial_cash=1000.,
                                     option_fee=.25, stock_fee=.1, slippage_bps=10.)
        self.contracts = [Contract("target", EXPIRY, 100., multiplier=10.),
                          Contract("hedge", EXPIRY, 105., multiplier=5.)]
        self.positions = [Position("target", -2., 5., entry_fee=1.5)]
        self.provider = FixtureProvider({"target": (lambda s: {"s0": 5., "s1": 6., "s2": 8.}[s.snapshot_id],
                                                    .5, (10., 0., 0., 0.)),
                                         "hedge": (3., .2, (5., 0., 0., 0.))})
        self.history = [snapshot(0, prices={"hedge": (2.8, 3.2)}, lending_rate=.365),
                        snapshot(1, spot=(101.9, 102.1), prices={"hedge": (3.8, 4.2)}, lending_rate=.365),
                        snapshot(2, spot=(103.9, 104.1), prices={"hedge": (4.8, 5.2)}, dividend_per_share=1.2)]

    def run_case(self, **kwargs):
        args = dict(config=self.config, history=self.history, contracts=self.contracts,
                    portfolio=self.positions, provider=self.provider)
        return run_backtest(**{**args, **kwargs})

    def assert_reconciles(self, result):
        self.assertEqual(len(result.pnl_components), len(result.ledger))
        for row, ledger in zip(result.pnl_components, result.ledger):
            self.assertEqual(row["at"], ledger["at"])
            self.assertLessEqual(abs(math.fsum(row[k] for k in COMPONENTS) - ledger["daily_pnl"]), row["tolerance"])
            self.assertLessEqual(abs(math.fsum(row["cumulative"].values()) - ledger["cumulative_pnl"]),
                                 row["cumulative_tolerance"])
        for k in COMPONENTS:
            self.assertAlmostEqual(result.summary["pnl_components"][k], math.fsum(r[k] for r in result.pnl_components))

    def test_hand_computed_trades_multipliers_costs_carry_and_liquidation(self):
        result = self.run_case()
        day0, day1, day2 = result.pnl_components
        self.assertEqual(result.orders[0]["desired"]["hedge"], 8.)
        self.assertEqual(result.orders[0]["stock"], 2.)
        self.assertEqual(day0["target"], 0.)
        self.assertEqual(day0["entry_fees"], -1.5)
        self.assertEqual([r["target"] for r in result.pnl_components], [0., -20., -40.])
        self.assertEqual([r["option_hedge"] for r in result.pnl_components], [0., 0., 40.])
        self.assertEqual([r["stock"] for r in result.pnl_components], [0., 0., 4.])
        # Eight option contracts (multiplier five), two shares; commissions are per contract/share.
        self.assertAlmostEqual(day1["trading_fees"], -2.2)
        self.assertAlmostEqual(day2["trading_fees"], -2.2)
        opening_cost = 40*(4.2*1.001-4.) + 2*(102.1*1.001-102.)
        closing_cost = 40*(5.-4.8*.999) + 2*(104.-103.9*.999)
        self.assertAlmostEqual(day1["spread_slippage"], -opening_cost)
        self.assertAlmostEqual(day2["spread_slippage"], -closing_cost)
        first_interest = 1098.5 * math.expm1(.001)
        cash_after_purchase = 1098.5 + first_interest - 40*4.2*1.001 - 2*102.1*1.001 - 2.2
        self.assertAlmostEqual(day1["funding"], first_interest)
        self.assertAlmostEqual(day2["funding"], cash_after_purchase * math.expm1(.001))
        self.assertAlmostEqual(day2["dividend"], 2.4)
        self.assertAlmostEqual(result.summary["cumulative_pnl"],
                               -60.+40.+4.+2.4-1.5-4.4-opening_cost-closing_cost
                               +first_interest+cash_after_purchase*math.expm1(.001))
        self.assert_reconciles(result)

    def test_entry_lots_and_model_vs_market_marks(self):
        positions = [Position("target", 1., 4., entry_fee=.7), Position("target", 1., 6., entry_fee=.3)]
        history = [snapshot(0, prices={"target": 7.})]
        for mark, target in (("model", 0.), ("market", 40.)):
            with self.subTest(mark=mark):
                result = self.run_case(config=BacktestConfig(strategy="unhedged", target_mark=mark),
                                       portfolio=positions, history=history)
                row = result.pnl_components[0]
                self.assertEqual(row["target"], target)
                self.assertEqual(row["details"]["target_entry_premium"], -100.)
                self.assertEqual(row["entry_fees"], -1.)
                self.assertEqual(row["total"], target-1.)
                self.assert_reconciles(result)

    def test_short_stock_borrow_dividend_and_switch_from_borrowing_to_lending(self):
        history = [snapshot(0, borrowing_rate=.365),
                   snapshot(1, spot=101., lending_rate=.365, stock_borrow_rate=.365),
                   snapshot(2, spot=105., dividend_per_share=2.)]
        result = self.run_case(config=BacktestConfig(strategy="delta", rebalance_every=10), history=history,
                               contracts=[Contract("target", EXPIRY, 100.)], portfolio=[Position("target", 1., 5.)],
                               provider=FixtureProvider({"target": (5., 1., (0., 0., 0., 0.))}))
        self.assertAlmostEqual(result.pnl_components[1]["funding"], -5.*math.expm1(.001))
        row = result.pnl_components[2]
        self.assertAlmostEqual(row["funding"], (101.-5.*math.exp(.001))*math.expm1(.001))
        self.assertAlmostEqual(row["stock_borrow"], -.101)
        self.assertEqual(row["dividend"], -2.)
        self.assertEqual(row["stock"], -4.)
        self.assert_reconciles(result)

    def test_rejected_basket_has_no_execution_pnl(self):
        result = self.run_case(config=replace(self.config, initial_cash=0., cash_floor=0., end_policy="mark_to_market"))
        self.assertEqual(result.orders[0]["status"], "rejected_cash_limit")
        for row in result.pnl_components:
            for k in ("option_hedge", "stock", "trading_fees", "spread_slippage"):
                self.assertEqual(row[k], 0.)
        self.assert_reconciles(result)

    def test_large_cash_float_rounding_keeps_reported_difference(self):
        result = self.run_case(config=replace(self.config, initial_cash=1e12))
        self.assert_reconciles(result)
        self.assertTrue(any(r["reconciliation_error"] != 0. for r in result.pnl_components))

    def test_reconciliation_detects_corruption_and_rolls_back_before_resume(self):
        whole = self.run_case()

        def corrupted(current, previous, fills, cashflows, **kwargs):
            if current["at"] == self.history[1].at.isoformat():
                fills = deepcopy(fills)
                fills[0]["price"] += 1.  # Cash was booked correctly; damage only execution evidence.
            return attribute_pnl(current, previous, fills, cashflows, **kwargs)

        with patch("rust_pricing_backtest.engine.attribute_pnl", side_effect=corrupted):
            failed = self.run_case(on_error="return")
        self.assertEqual(failed.failure["code"], "pnl_reconciliation")
        self.assertEqual(failed.checkpoint, self.run_case(stop_after=1).checkpoint)
        self.assertEqual(len(failed.pnl_components), 1)
        self.assertEqual(failed.fills, ())
        self.assertEqual(failed.orders[0]["status"], "pending")
        self.assertEqual(self.run_case(checkpoint=failed.checkpoint).to_json(), whole.to_json())
        # Cumulative equality is checked independently of today's equality too.
        current = dict(whole.ledger[0], cumulative_pnl=123.)
        with self.assertRaisesRegex(BacktestError, "pnl_reconciliation"):
            attribute_pnl(current, None, [], whole.cashflows[:1], initial_cash=1000.,
                          entry_premium=100., entry_fees=1.5)

    def test_failed_first_day_has_no_attribution_and_entry_is_counted_once_on_retry(self):
        with patch.object(self.provider, "evaluate", side_effect=BacktestError("unavailable", "fixture")):
            failed = self.run_case(on_error="return")
        self.assertEqual(failed.pnl_components, ())
        self.assertIsNone(failed.summary["pnl_components"])
        self.assertIsNone(failed.summary["pnl_reconciliation"]["passed"])
        self.assertEqual(self.run_case(checkpoint=failed.checkpoint).to_json(), self.run_case().to_json())

    def test_export_resume_and_legacy_results(self):
        whole = self.run_case()
        partial = self.run_case(stop_after=2)
        self.assertEqual(self.run_case(checkpoint=json.loads(json.dumps(partial.checkpoint))).to_json(), whole.to_json())
        with tempfile.TemporaryDirectory() as directory:
            whole.write(directory)
            self.assertEqual(BacktestResult.from_json(Path(directory, "result.json").read_text()).to_json(), whole.to_json())
            with Path(directory, "pnl_components.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(float(rows[2]["option_hedge"]), 40.)
            self.assertEqual(json.loads(rows[2]["cumulative"]), whole.summary["pnl_components"])
        self.assertEqual(whole.manifest["schema_version"], 10)
        self.assertEqual(whole.manifest["input_schema_version"], 6)
        self.assertEqual(whole.checkpoint["schema_version"], 8)
        for version in (1, 2, 3):
            old = json.loads(whole.to_json())
            old["manifest"]["schema_version"] = version
            del old["pnl_components"]
            self.assertEqual(BacktestResult.from_json(json.dumps(old)).pnl_components, ())
        old_checkpoint = dict(partial.checkpoint, schema_version=2)
        with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
            self.run_case(checkpoint=old_checkpoint)


if __name__ == "__main__":
    unittest.main()
