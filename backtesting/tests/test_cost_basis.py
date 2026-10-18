from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rust_pricing_backtest import BacktestConfig, BacktestError, BacktestResult, Contract, Position, Risk, run_backtest
from rust_pricing_backtest import cost_basis
from test_backtest import START, EXPIRY, FixtureProvider, snapshot
from test_capital import movement
from test_payments import DatedProvider, PAYMENT, discount
import test_pnl


class CostBasisTests(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(strategy="unhedged")
        self.contract = Contract("target", EXPIRY, 100., multiplier=10.)
        self.positions = [Position("target", 1., 4.), Position("target", 1., 6.)]
        self.provider = FixtureProvider({"target": (7., 0., (0., 0., 0., 0.))})
        self.history = [snapshot(0), snapshot(1)]

    def run_case(self, **kwargs):
        args = dict(config=self.config, history=self.history, contracts=[self.contract],
                    portfolio=self.positions, provider=self.provider)
        return run_backtest(**{**args, **kwargs})

    def assert_reconciles(self, result):
        self.assertEqual(len(result.cost_basis), len(result.ledger))
        for row, ledger in zip(result.cost_basis, result.ledger):
            self.assertLessEqual(abs(row["realized_pnl"] + row["unrealized_pnl"] - ledger["cumulative_pnl"]),
                                 row["cumulative_tolerance"])
            self.assertLessEqual(abs(row["daily_realized_pnl"] + row["daily_unrealized_change"] - ledger["daily_pnl"]),
                                 row["daily_tolerance"])

    def test_entry_average_multiplier_fees_and_mark_policy(self):
        positions = [replace(self.positions[0], entry_fee=.7), replace(self.positions[1], entry_fee=.3)]
        for policy, mark, unrealized in (("model", 7., 40.), ("market", 9., 80.)):
            result = self.run_case(portfolio=positions, history=[snapshot(0, prices={"target": 9.})],
                                   config=replace(self.config, target_mark=policy))
            row = result.cost_basis[0]
            self.assertEqual(row["positions"]["targets"]["target"],
                             {"quantity": 2., "multiplier": 10., "average_price": 5., "signed_cost": 100.})
            self.assertEqual(row["groups"]["target"]["market_value"], 20.*mark)
            self.assertEqual(row["realized_pnl"], -1.)
            self.assertEqual(row["unrealized_pnl"], unrealized)
            self.assert_reconciles(result)

    def test_opposite_entry_lots_use_input_order_and_reopen_at_new_price(self):
        lots = [Position("target", 10., 100.), Position("target", 10., 120.), Position("target", -15., 130.)]
        provider = FixtureProvider({"target": (125., 0., (0., 0., 0., 0.))})
        a = self.run_case(portfolio=lots, provider=provider).cost_basis[-1]
        b = self.run_case(portfolio=[lots[0], lots[2], lots[1]], provider=provider).cost_basis[-1]
        self.assertEqual((a["realized_pnl"], a["unrealized_pnl"], a["total_pnl"]), (3000., 750., 3750.))
        self.assertEqual((b["realized_pnl"], b["unrealized_pnl"], b["total_pnl"]), (3500., 250., 3750.))
        self.assertEqual(b["positions"]["targets"]["target"]["average_price"], 120.)
        closed = self.run_case(portfolio=[Position("target", 1., 4.), Position("target", -1., 6.)])
        self.assertEqual(closed.summary["cost_basis"]["realized_pnl"], 20.)
        self.assertEqual(closed.cost_basis[-1]["positions"]["targets"], {})

    def test_stock_fills_partial_close_reversal_and_short_cover(self):
        class SequenceProvider:
            identity = "cost_basis_stock_sequence_v1"

            def evaluate(self, s, c):
                delta = (-10., -20., -15., 10., 6., 0.)[int(s.snapshot_id[1:])]
                return Risk(5., delta, (0.,)*4, s.basis.basis_id)

        history = [snapshot(i, spot=price) for i, price in enumerate((100., 100., 120., 130., 90., 80.))]
        result = self.run_case(config=BacktestConfig(strategy="delta", end_policy="mark_to_market"), history=history,
                               contracts=[replace(self.contract, expiry=START+timedelta(days=10), multiplier=1.)],
                               portfolio=[Position("target", 1., 5.)], provider=SequenceProvider())
        expected = ((0., 0., None), (0., 0., 100.), (0., 200., 110.),
                    (100., 300., 110.), (-200., 0., 90.), (-160., 60., 90.))
        self.assertEqual([f["quantity"] for f in result.fills], [10., 10., -5., -25., 4.])
        for row, (realized, unrealized, average) in zip(result.cost_basis, expected):
            group = row["groups"]["stock"]
            self.assertEqual((group["realized_pnl"], group["unrealized_pnl"]), (realized, unrealized))
            if average is not None:
                self.assertEqual(row["positions"]["stock"]["__spot__"]["average_price"], average)
        self.assertEqual(result.cost_basis[-1]["positions"]["stock"]["__spot__"]["signed_cost"], -540.)
        self.assert_reconciles(result)

    def test_actual_execution_basis_includes_spread_but_expenses_fees(self):
        fixture = test_pnl.PnlTests()
        fixture.setUp()
        result = fixture.run_case()
        day1, final = result.cost_basis[1:]
        self.assertAlmostEqual(day1["positions"]["hedges"]["hedge"]["average_price"], 4.2*1.001)
        self.assertAlmostEqual(day1["groups"]["option_hedge"]["unrealized_pnl"], 40.*(4.-4.2*1.001))
        self.assertAlmostEqual(final["groups"]["option_hedge"]["realized_pnl"], 40.*(4.8*.999-4.2*1.001))
        self.assertAlmostEqual(final["groups"]["stock"]["realized_pnl"], 2.*(103.9*.999-102.1*1.001))
        self.assertEqual(final["groups"]["option_hedge"]["unrealized_pnl"], 0.)
        self.assertEqual(final["cash_realized_components"]["trading_fees"], -4.4)
        self.assertEqual(final["unrealized_pnl"], -60.)
        self.assertNotIn("spread_slippage", final["cash_realized_components"])
        self.assertAlmostEqual(final["cash_realized_components"]["dividend"], 2.4)
        self.assertAlmostEqual(final["cash_realized_components"]["funding"], result.summary["funding_pnl"])
        self.assert_reconciles(result)

    def payment_case(self, sign=1., **kwargs):
        return self.run_case(**{**dict(history=self._payment_history(), contracts=[replace(self.contract, multiplier=1., payment_date=PAYMENT)],
                                     portfolio=[Position("target", sign, 9.5)], provider=DatedProvider()), **kwargs})

    def test_long_and_short_fix_face_cost_then_accrete_and_pay(self):
        for sign in (1., -1.):
            with self.subTest(sign=sign):
                result = self.payment_case(sign)
                for i, unrealized in ((4, -.2), (5, -.1), (6, 0.)):
                    row = result.cost_basis[i]
                    self.assertAlmostEqual(row["realized_pnl"], sign*.5)
                    self.assertAlmostEqual(row["unrealized_pnl"], sign*unrealized)
                    self.assertEqual(row["positions"]["targets"], {})
                self.assertEqual(result.cost_basis[4]["claims"]["targets:target"]["signed_cost"], sign*10.)
                self.assertEqual(result.cost_basis[6]["claims"], {})
                self.assertEqual(result.cost_basis[6]["daily_realized_pnl"], 0.)
                self.assertTrue(result.summary["fully_settled"])
                self.assert_reconciles(result)

    def test_same_contract_target_and_hedge_keep_separate_basis_and_claims(self):
        result = self.payment_case(-1., config=BacktestConfig(strategy="vega_kt", ridge=0., rebalance_every=10,
                                                            allow_identical_hedge=True),
                                   history=[replace(s, option_quotes={"target": replace(s.spot, bid=9., ask=9.)})
                                            for s in self._payment_history()])
        row = result.cost_basis[4]
        self.assertEqual(row["groups"]["target"]["realized_pnl"], -.5)
        self.assertEqual(row["groups"]["option_hedge"]["realized_pnl"], 1.)
        self.assertAlmostEqual(row["groups"]["target"]["unrealized_pnl"], .2)
        self.assertAlmostEqual(row["groups"]["option_hedge"]["unrealized_pnl"], -.2)
        self.assertEqual(set(row["claims"]), {"targets:target", "hedges:target"})
        self.assert_reconciles(result)

    def _payment_history(self):
        # This fixture has a delayed payoff but no native pricing dependency.
        history = [snapshot(i, spot=110. if i == 4 else 100.) for i in range(7)]
        return [replace(s, payment_discounts={PAYMENT: discount(s, .94+.01*i)}) if i < 6 else s
                for i, s in enumerate(history)]

    def test_zero_and_same_day_payoffs_close_basis_once(self):
        for payment_date, spot, expected in ((PAYMENT, 90., -9.5), (None, 90., -9.5), (None, 110., .5)):
            history = [snapshot(i, spot=spot) for i in range(5)]
            result = self.run_case(history=history, contracts=[replace(self.contract, multiplier=1., payment_date=payment_date)],
                                   portfolio=[Position("target", 1., 9.5)], provider=DatedProvider())
            row = result.cost_basis[-1]
            self.assertEqual((row["realized_pnl"], row["unrealized_pnl"]), (expected, 0.))
            self.assertEqual(row["claims"], {})
            self.assertEqual(row["positions"]["targets"], {})
            self.assert_reconciles(result)

    def test_capital_is_excluded_and_carry_is_realized(self):
        history = [snapshot(0, borrowing_rate=.365), snapshot(1, spot=101., lending_rate=.365, stock_borrow_rate=.365),
                   snapshot(2, spot=105., dividend_per_share=2.)]
        history[1] = replace(history[1], external_cashflows=[movement(history[1], 50.)])
        result = self.run_case(config=BacktestConfig(strategy="delta", rebalance_every=10), history=history,
                               contracts=[replace(self.contract, multiplier=1.)], portfolio=[Position("target", 1., 5.)],
                               provider=FixtureProvider({"target": (5., 1., (0.,)*4)}))
        row = result.cost_basis[-1]
        self.assertAlmostEqual(row["cash_realized_components"]["stock_borrow"], -.101)
        self.assertEqual(row["cash_realized_components"]["dividend"], -2.)
        self.assertAlmostEqual(row["realized_pnl"], -4.-.101-2.+result.summary["funding_pnl"])
        self.assertEqual(row["unrealized_pnl"], 0.)
        self.assert_reconciles(result)

    def test_rejected_basket_does_not_book_basis(self):
        fixture = test_pnl.PnlTests()
        fixture.setUp()
        result = fixture.run_case(config=replace(fixture.config, initial_cash=0., cash_floor=0., end_policy="mark_to_market"))
        self.assertEqual(result.orders[0]["status"], "rejected_cash_limit")
        for row in result.cost_basis:
            self.assertEqual(row["positions"]["hedges"], {})
            self.assertEqual(row["positions"]["stock"], {})
            self.assertEqual(row["groups"]["option_hedge"]["realized_pnl"], 0.)
        self.assert_reconciles(result)

    def test_corruption_after_fill_rolls_back_book_and_all_output(self):
        fixture = test_pnl.PnlTests()
        fixture.setUp()
        whole = fixture.run_case()
        for field in ("quantity", "signed_cost", "realized", "spread_slippage"):
            def corrupted(book, ledger, *args):
                if ledger["at"] == fixture.history[1].at.isoformat():
                    if field in ("quantity", "signed_cost"):
                        book["positions"]["hedges"]["hedge"][field] += 1.
                    else:
                        book[field]["hedges"] += 1.
                return cost_basis.report(book, ledger, *args)

            # Patch the engine binding only, retaining the real report as oracle.
            from types import SimpleNamespace
            module = SimpleNamespace(**vars(cost_basis))
            module.report = corrupted
            with self.subTest(field=field), patch("rust_pricing_backtest.engine.cost_basis", module):
                failed = fixture.run_case(on_error="return")
            self.assertIn(failed.failure["code"], ("cost_basis_position_mismatch", "cost_basis_reconciliation"))
            self.assertEqual(failed.checkpoint, fixture.run_case(stop_after=1).checkpoint)
            self.assertEqual(fixture.run_case(checkpoint=failed.checkpoint).to_json(), whole.to_json())

    def test_first_day_failure_keeps_entries_for_exact_retry(self):
        with patch.object(self.provider, "evaluate", side_effect=BacktestError("unavailable", "fixture")):
            failed = self.run_case(on_error="return")
        self.assertEqual(failed.cost_basis, ())
        self.assertIsNone(failed.summary["cost_basis"])
        self.assertEqual(self.run_case(checkpoint=failed.checkpoint).to_json(), self.run_case().to_json())

    def test_resume_claim_export_legacy_and_immutable_history(self):
        whole, partial = self.payment_case(), self.payment_case(stop_after=5)
        before = deepcopy(partial.cost_basis)
        resumed = self.payment_case(checkpoint=json.loads(json.dumps(partial.checkpoint)))
        self.assertEqual(resumed.to_json(), whole.to_json())
        self.assertEqual(partial.cost_basis, before)
        self.assertEqual(whole.cost_basis[0]["positions"]["targets"]["target"]["signed_cost"], 9.5)
        with tempfile.TemporaryDirectory() as directory:
            whole.write(directory)
            self.assertEqual(len(list(Path(directory).glob("*.csv"))), 11)
            with Path(directory, "cost_basis.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertAlmostEqual(float(rows[4]["unrealized_pnl"]), -.2)
            self.assertEqual(json.loads(rows[4]["claims"]), whole.cost_basis[4]["claims"])
            self.assertEqual(BacktestResult.from_json(Path(directory, "result.json").read_text()).to_json(), whole.to_json())
        for version in range(1, 7):
            old = json.loads(whole.to_json())
            old["manifest"]["schema_version"] = version
            del old["cost_basis"]
            self.assertEqual(BacktestResult.from_json(json.dumps(old)).cost_basis, ())
        with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
            self.payment_case(checkpoint={**partial.checkpoint, "schema_version": 4})


if __name__ == "__main__":
    unittest.main()
