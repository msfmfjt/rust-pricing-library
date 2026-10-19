from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rust_pricing_backtest import (BacktestConfig, BacktestError, BacktestResult, Contract, ExternalCashflow,
                                  Position, compare_strategies, run_backtest)
from rust_pricing_backtest.domain import fingerprint
from rust_pricing_backtest.pnl import attribute_pnl
from rust_pricing_backtest.serialization import input_json, load_input
from test_backtest import EXPIRY, FixtureProvider, snapshot
from test_numeraire import account
from test_payments import DatedProvider, PAYMENT, discount


def movement(s, amount, flow_id=None):
    return ExternalCashflow(flow_id or f"capital-{s.snapshot_id}", s.at, amount, 2, s.at,
                            "hand_computed_capital", "v1")


class CapitalTests(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(strategy="unhedged", initial_cash=100., numeraire_id="common_cash_account")
        self.contract = Contract("target", EXPIRY, 100.)
        self.positions = [Position("target", 0., 0.)]
        self.provider = FixtureProvider({"target": (5., -1., (0., 0., 0., 0.))})
        self.history = [snapshot(i) for i in range(3)]
        self.history = [replace(s, numeraire=account(s, 2.)) for s in self.history]

    def run_case(self, **kwargs):
        args = dict(config=self.config, history=self.history, contracts=[self.contract],
                    portfolio=self.positions, provider=self.provider)
        return run_backtest(**{**args, **kwargs})

    def test_initial_deposit_later_withdrawal_are_not_pnl(self):
        history = [replace(s, external_cashflows=[movement(s, amount)])
                   for s, amount in zip(self.history, (10., 50., -20.))]
        result = self.run_case(history=history)
        self.assertEqual([r["cash"] for r in result.ledger], [110., 160., 140.])
        self.assertEqual([r["cumulative_external_net_cashflow"] for r in result.ledger], [10., 60., 40.])
        self.assertEqual([r["daily_pnl"] for r in result.ledger], [0., 0., 0.])
        self.assertEqual([r["discounted_daily_pnl"] for r in result.ledger], [0., 0., 0.])
        for key in ("cumulative_pnl", "terminal_hedging_error", "worst_day", "max_drawdown_currency", "daily_rmse"):
            self.assertEqual(result.summary[key], 0.)
        self.assertEqual(result.summary["discounted_pnl"]["terminal_hedging_error"], 0.)
        self.assertEqual(result.summary["external_capital"], {"count": 3, "net_cashflow": 40., "gross_inflows": 60., "gross_outflows": 20.})
        nominal_only = self.run_case(history=history, config=replace(self.config, numeraire_id=None))
        self.assertEqual(nominal_only.summary["cumulative_pnl"], 0.)
        self.assertIsNone(nominal_only.summary["discounted_pnl"])
        self.assertTrue(all(f["discounted_amount"] is None for f in nominal_only.cashflows if f["kind"] == "external_capital"))

    def test_each_flow_uses_its_own_numeraire_not_final_value(self):
        history = [replace(s, numeraire=account(s, value)) for s, value in zip(self.history, (2., 4., 8.))]
        history[1] = replace(history[1], external_cashflows=[movement(history[1], 50.)])
        history[2] = replace(history[2], external_cashflows=[movement(history[2], -20.)])
        result = self.run_case(history=history)
        self.assertEqual([r["discounted_external_net_cashflow"] for r in result.ledger], [0., 12.5, -2.5])
        self.assertEqual([r["discounted_daily_pnl"] for r in result.ledger], [0., -25., -18.75])
        self.assertEqual(result.summary["discounted_pnl"]["external_net_cashflow"], 10.)
        self.assertEqual(result.summary["discounted_pnl"]["terminal_hedging_error"], -43.75)
        self.assertEqual(result.summary["terminal_hedging_error"], 0.)

    def test_deposit_earns_interest_only_after_booking(self):
        history = [replace(s, lending_rate=.365, numeraire=account(s, 2.*math.exp(.001*i))) for i, s in enumerate(self.history)]
        history[1] = replace(history[1], external_cashflows=[movement(history[1], 100.)])
        result = self.run_case(history=history)
        first = 100.*math.exp(.001) + 100.
        final = first*math.exp(.001)
        self.assertAlmostEqual(result.ledger[1]["cash"], first)
        self.assertAlmostEqual(result.ledger[2]["cash"], final)
        self.assertAlmostEqual(result.summary["cumulative_pnl"], final-200.)
        self.assertAlmostEqual(result.summary["funding_pnl"], final-200.)
        self.assertAlmostEqual(result.summary["discounted_pnl"]["terminal_hedging_error"], 0.)

    def test_capital_can_fund_pending_hedge_but_not_initial_entry(self):
        config = replace(self.config, strategy="delta", initial_cash=5., cash_floor=0., end_policy="mark_to_market")
        positions = [Position("target", 1., 5.)]
        history = [self.history[0], replace(self.history[1], external_cashflows=[movement(self.history[1], 100.)]), self.history[2]]
        result = self.run_case(config=config, portfolio=positions, history=history)
        rejected = self.run_case(config=config, portfolio=positions)
        self.assertEqual(result.orders[0]["status"], "filled")
        self.assertEqual(result.ledger[1]["stock"], 1.)
        self.assertEqual(rejected.orders[0]["status"], "rejected_cash_limit")
        self.assertEqual(result.summary["cumulative_pnl"], 0.)
        with self.assertRaisesRegex(BacktestError, "initial premium"):
            self.run_case(config=replace(config, initial_cash=0.), portfolio=positions,
                          history=[replace(self.history[0], external_cashflows=[movement(self.history[0], 100.)]), *self.history[1:]])

    def test_capital_can_fund_delayed_option_payment_without_becoming_profit(self):
        history = [snapshot(i, spot=110. if i == 4 else 100.) for i in range(7)]
        history = [replace(s, numeraire=account(s, 2.), payment_discounts={PAYMENT: discount(s, .94+.01*i)} if i < 6 else {})
                   for i, s in enumerate(history)]
        history[6] = replace(history[6], external_cashflows=[movement(history[6], .5)])
        args = dict(config=replace(self.config, initial_cash=0., cash_floor=0.), history=history,
                    contracts=[replace(self.contract, payment_date=PAYMENT)], portfolio=[Position("target", -1., 9.5)],
                    provider=DatedProvider())
        result = self.run_case(**args)
        self.assertEqual(result.ledger[-1]["cash"], 0.)
        self.assertTrue(result.summary["fully_settled"])
        self.assertAlmostEqual(result.summary["terminal_hedging_error"], -.5)
        self.assertAlmostEqual(result.summary["discounted_pnl"]["terminal_hedging_error"], -.25)
        partial = self.run_case(**args, stop_after=6)
        self.assertEqual(self.run_case(**args, checkpoint=partial.checkpoint).to_json(), result.to_json())

    def test_excess_withdrawal_rolls_back_and_repair_resumes_once(self):
        s = self.history[1]
        bad = [self.history[0], replace(s, external_cashflows=[movement(s, -101.)]), self.history[2]]
        config = replace(self.config, cash_floor=0.)
        failed = self.run_case(config=config, history=bad, on_error="return")
        self.assertEqual(failed.failure["code"], "cash_limit")
        self.assertEqual(failed.checkpoint, self.run_case(config=config, stop_after=1).checkpoint)
        self.assertEqual(failed.summary["external_capital"]["count"], 0)
        repaired = [self.history[0], replace(s, external_cashflows=[movement(s, -40.)]), self.history[2]]
        self.assertEqual(self.run_case(config=config, history=repaired, checkpoint=failed.checkpoint).to_json(),
                         self.run_case(config=config, history=repaired).to_json())

    def test_failure_after_capital_and_fills_restores_ids_cash_and_pending_order(self):
        history = [self.history[0], replace(self.history[1], external_cashflows=[movement(self.history[1], 100.)]), self.history[2]]
        args = dict(config=replace(self.config, strategy="delta"), history=history, portfolio=[Position("target", 1., 5.)])
        original = self.provider.evaluate

        def unavailable(s, c):
            if s.snapshot_id == "s1":
                raise BacktestError("unavailable", "after fills")
            return original(s, c)

        with patch.object(self.provider, "evaluate", side_effect=unavailable):
            failed = self.run_case(**args, on_error="return")
        self.assertEqual(failed.checkpoint, self.run_case(**args, stop_after=1).checkpoint)
        self.assertEqual(failed.checkpoint["state"]["external_cashflow_ids"], {})
        self.assertEqual(failed.orders[0]["status"], "pending")
        self.assertEqual(self.run_case(**args, checkpoint=failed.checkpoint).to_json(), self.run_case(**args).to_json())

    def test_first_day_failure_does_not_commit_capital(self):
        history = [replace(self.history[0], external_cashflows=[movement(self.history[0], 10.)]), *self.history[1:]]
        args = dict(history=history, portfolio=[Position("target", 1., 5.)])
        with patch.object(self.provider, "evaluate", side_effect=BacktestError("unavailable", "fixture")):
            failed = self.run_case(**args, on_error="return")
        self.assertEqual(failed.ledger, ())
        self.assertEqual(failed.summary["external_capital"]["count"], 0)
        self.assertIsNone(failed.summary["discounted_pnl"])
        self.assertEqual(self.run_case(**args, checkpoint=failed.checkpoint).to_json(), self.run_case(**args).to_json())

    def test_duplicate_ids_are_rejected_even_with_changed_revision(self):
        first = movement(self.history[1], 10., "same")
        same_day = [self.history[0], replace(self.history[1], external_cashflows=[first, first]), self.history[2]]
        self.assertEqual(self.run_case(history=same_day, on_error="return").failure["code"], "duplicate_external_cashflow")
        repeated = replace(first, effective_at=self.history[2].at, available_at=self.history[2].at, amount=20., revision="v2")
        history = [self.history[0], replace(self.history[1], external_cashflows=[first]),
                   replace(self.history[2], external_cashflows=[repeated])]
        partial = self.run_case(history=history, stop_after=2)
        failed = self.run_case(history=history, checkpoint=partial.checkpoint, on_error="return")
        self.assertEqual(failed.failure["code"], "duplicate_external_cashflow")
        self.assertEqual(failed.checkpoint, partial.checkpoint)
        self.assertEqual(failed.summary["external_capital"]["count"], 1)
        repaired = [*history[:2], replace(history[2], external_cashflows=[replace(repeated, flow_id="new")])]
        self.assertEqual(self.run_case(history=repaired, checkpoint=failed.checkpoint).to_json(), self.run_case(history=repaired).to_json())

    def test_time_currency_and_availability_checks_precede_provider(self):
        s = self.history[1]
        for changes, code in (({"effective_at": s.at-timedelta(seconds=1)}, "external_cashflow_time_mismatch"),
                              ({"effective_at": s.at+timedelta(days=1)}, "external_cashflow_time_mismatch"),
                              ({"available_at": s.at+timedelta(seconds=1)}, "future_external_cashflow"),
                              ({"currency_id": 99}, "external_cashflow_currency_mismatch")):
            with self.subTest(code=code), patch.object(self.provider, "evaluate", wraps=self.provider.evaluate) as evaluated:
                history = [self.history[0], replace(s, external_cashflows=[replace(movement(s, 10.), **changes)]), self.history[2]]
                failed = self.run_case(history=history, portfolio=[Position("target", 1., 5.)], on_error="return")
                self.assertEqual(failed.failure["code"], code)
                self.assertTrue(all(call.args[0].snapshot_id == "s0" for call in evaluated.call_args_list))

    def test_capital_record_corruption_is_not_hidden_by_adjusted_pnl(self):
        history = [self.history[0], replace(self.history[1], external_cashflows=[movement(self.history[1], 10.)]), self.history[2]]

        def corrupt(current, previous, fills, cashflows, **kwargs):
            cashflows = deepcopy(cashflows)
            for flow in cashflows:
                if flow["kind"] == "external_capital":
                    flow["amount"] += 1.
            return attribute_pnl(current, previous, fills, cashflows, **kwargs)

        with patch("rust_pricing_backtest.engine.attribute_pnl", side_effect=corrupt):
            failed = self.run_case(history=history, on_error="return")
        self.assertEqual(failed.failure["code"], "external_cashflow_reconciliation")
        self.assertEqual(failed.checkpoint, self.run_case(history=history, stop_after=1).checkpoint)

    def test_validation_netting_and_overflow(self):
        s = self.history[0]
        for changes in ({"amount": 0.}, {"amount": math.inf}, {"amount": math.nan}, {"flow_id": " "},
                        {"source": ""}, {"revision": ""}, {"currency_id": True}, {"effective_at": s.at.replace(tzinfo=None)}):
            with self.subTest(changes=changes), self.assertRaises(BacktestError):
                replace(movement(s, 1.), **changes)
        with self.assertRaisesRegex(BacktestError, "invalid_external_cashflow"):
            replace(s, external_cashflows=[1.])
        flows = [movement(s, -200., "a-withdraw"), movement(s, 200., "z-deposit")]
        result = self.run_case(config=replace(self.config, cash_floor=0.), history=[replace(s, external_cashflows=flows)])
        self.assertEqual(result.ledger[0]["cash"], 100.)
        self.assertEqual(result.summary["external_capital"]["count"], 2)
        self.assertEqual(result.summary["cumulative_pnl"], 0.)
        bad = self.run_case(history=[replace(s, external_cashflows=[movement(s, 1e308, "a"), movement(s, 1e308, "b")])], on_error="return")
        self.assertEqual(bad.failure["code"], "nonfinite_external_cashflow")
        self.assertEqual(bad.ledger, ())

    def test_export_replay_legacy_and_processed_flow_change(self):
        s = self.history[1]
        history = [self.history[0], replace(s, external_cashflows=[movement(s, 50.)]), self.history[2]]
        whole = self.run_case(history=history)
        partial = self.run_case(history=history, stop_after=2)
        self.assertEqual(run_backtest(**load_input(input_json(whole)), provider=self.provider, checkpoint=partial.checkpoint).to_json(), whole.to_json())
        changed = [history[0], replace(s, external_cashflows=[movement(s, 60.)]), history[2]]
        with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
            self.run_case(history=changed, checkpoint=partial.checkpoint)
        flow = next(f for f in whole.cashflows if f["kind"] == "external_capital")
        self.assertEqual(flow["external_cashflow_hash"], fingerprint(history[1].external_cashflows[0]))
        self.assertEqual(flow["discounted_amount"], 25.)
        with tempfile.TemporaryDirectory() as directory:
            whole.write(directory)
            self.assertEqual(BacktestResult.from_json(Path(directory, "result.json").read_text()).to_json(), whole.to_json())
            self.assertEqual(len(list(Path(directory).glob("*.csv"))), 11)
            with Path(directory, "cashflows.csv").open(newline="") as stream:
                exported = [r for r in csv.DictReader(stream) if r["kind"] == "external_capital"]
            self.assertEqual(len(exported), 1)
            self.assertEqual(json.loads(exported[0]["external_cashflow"])["flow_id"], flow["flow_id"])
        baseline = self.run_case()
        legacy = json.loads(input_json(baseline))
        legacy["schema_version"] = 4
        del legacy["config"]["constraint_mode"], legacy["config"]["gross_notional_limit"]
        for day in legacy["market_history"]:
            del day["external_cashflows"]
        self.assertEqual(run_backtest(**load_input(json.dumps(legacy)), provider=self.provider).to_json(), baseline.to_json())
        malformed = json.loads(input_json(whole))
        malformed["market_history"][0]["external_cashflows"] = None
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            load_input(json.dumps(malformed))
        with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
            self.run_case(history=history, checkpoint={**partial.checkpoint, "schema_version": 3})

    def test_four_strategies_share_capital_and_later_changes_do_not_rewrite_past(self):
        history = [self.history[0], replace(self.history[1], external_cashflows=[movement(self.history[1], 50.)]), self.history[2]]
        results = compare_strategies(self.config, history, [self.contract], self.positions, self.provider)
        self.assertEqual(len({r.manifest["external_cashflow_policy"]["input_hash"] for r in results.values()}), 1)
        for result in results.values():
            self.assertEqual(result.summary["external_capital"]["net_cashflow"], 50.)
        changed = [*history[:2], replace(history[2], external_cashflows=[movement(history[2], -20.)])]
        first, second = self.run_case(history=history), self.run_case(history=changed)
        self.assertEqual(first.ledger[:2], second.ledger[:2])
        self.assertEqual(first.pnl_components[:2], second.pnl_components[:2])
        self.assertEqual(self.run_case(history=changed, checkpoint=self.run_case(history=history, stop_after=2).checkpoint).to_json(), second.to_json())


if __name__ == "__main__":
    unittest.main()
