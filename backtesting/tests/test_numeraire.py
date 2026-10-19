from dataclasses import replace
from datetime import timedelta
import json
import math
import unittest
from unittest.mock import patch

from rust_pricing_backtest import (BacktestConfig, BacktestError, BacktestResult, Contract,
                                  NumeraireObservation, Position, compare_strategies, run_backtest)
from rust_pricing_backtest.numeraire import discounted_ledger
from rust_pricing_backtest.serialization import input_json, load_input
from test_backtest import START, EXPIRY, FixtureProvider, snapshot
from test_payments import DatedProvider, PAYMENT, discount


def account(s, value):
    return NumeraireObservation("common_cash_account", 2, s.at.date(), value, s.at, s.at, "hand_computed_account", "v1")


class NumeraireTests(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(strategy="delta", initial_cash=100., numeraire_id="common_cash_account")
        self.contract = Contract("target", EXPIRY, 100.)
        self.positions = [Position("target", 1., 5., entry_fee=.2)]
        self.provider = FixtureProvider({"target": (5., 1., (0., 0., 0., 0.))})
        self.history = [snapshot(i, spot=100.+i) for i in range(5)]
        self.history = [replace(s, numeraire=account(s, 2.*math.exp(.001*i))) for i, s in enumerate(self.history)]

    def run_case(self, **kwargs):
        args = dict(config=self.config, history=self.history, contracts=[self.contract],
                    portfolio=self.positions, provider=self.provider)
        return run_backtest(**{**args, **kwargs})

    def assert_reconciles(self, result):
        base = result.manifest["config"]["initial_cash"] / result.ledger[0]["numeraire"]["value"]
        previous = base
        for row, pnl in zip(result.ledger, result.pnl_components):
            equity = row["equity"] / row["numeraire"]["value"]
            self.assertAlmostEqual(row["discounted_daily_pnl"], equity-previous)
            self.assertAlmostEqual(row["discounted_cumulative_pnl"], equity-base)
            d = pnl["discounted"]
            self.assertAlmostEqual(d["total"], d["pnl_at_current_numeraire"] + d["numeraire_carry"])
            self.assertLessEqual(abs(d["total"]-row["discounted_daily_pnl"]), d["tolerance"])
            self.assertLessEqual(abs(d["cumulative_total"]-row["discounted_cumulative_pnl"]), d["cumulative_tolerance"])
            previous = equity

    def test_cash_in_same_account_has_zero_discounted_gain_with_nonunit_initial_value(self):
        for rate in (.365, -.365):
            for fee in (0., 2.):
                with self.subTest(rate=rate, entry_fee=fee):
                    history = [snapshot(i, lending_rate=rate) for i in (0, 1, 4)]
                    history = [replace(s, numeraire=account(s, 2.*math.exp(rate*(s.at.date()-START).days/365.))) for s in history]
                    result = self.run_case(config=replace(self.config, strategy="unhedged"), history=history,
                                           portfolio=[Position("target", 0., 0., entry_fee=fee)])
                    self.assertAlmostEqual(result.ledger[-1]["cash"], (100.-fee)*math.exp(rate*4/365.))
                    for row in result.ledger:
                        self.assertAlmostEqual(row["discounted_cumulative_pnl"], -fee/2.)
                    self.assertAlmostEqual(result.summary["discounted_pnl"]["terminal_hedging_error"], -fee/2.)
                    self.assert_reconciles(result)

    def test_unchanged_nominal_equity_still_has_numeraire_carry(self):
        history = [replace(s, numeraire=account(s, b)) for s, b in zip(self.history[:3], (2., 4., 8.))]
        result = self.run_case(config=replace(self.config, strategy="unhedged"), history=history,
                               portfolio=[Position("target", 1., 5.)])
        self.assertEqual([r["daily_pnl"] for r in result.ledger], [0., 0., 0.])
        self.assertEqual([r["discounted_daily_pnl"] for r in result.ledger], [0., -25., -12.5])
        self.assertEqual(result.summary["discounted_pnl"]["cumulative_pnl"], -37.5)
        self.assertIsNone(result.summary["discounted_pnl"]["terminal_hedging_error"])
        self.assertEqual([r["discounted"]["numeraire_carry"] for r in result.pnl_components], [0., -25., -12.5])
        self.assert_reconciles(result)

    def test_delayed_claim_and_final_cash_use_the_same_numeraire_once(self):
        history = [snapshot(i, spot=110. if i == 4 else 100.) for i in range(7)]
        history = [replace(s, numeraire=account(s, 2.*1.01**i),
                           payment_discounts={PAYMENT: discount(s, .94+.01*i)} if i < 6 else {}) for i, s in enumerate(history)]
        result = self.run_case(config=replace(self.config, strategy="unhedged", initial_cash=20.), history=history,
                               contracts=[replace(self.contract, payment_date=PAYMENT)],
                               portfolio=[Position("target", 1., 9.5)], provider=DatedProvider())
        for i, equity in ((4, 20.3), (5, 20.4), (6, 20.5)):
            self.assertAlmostEqual(result.ledger[i]["discounted_cumulative_pnl"], equity/(2.*1.01**i)-10.)
        self.assertAlmostEqual(result.summary["discounted_pnl"]["terminal_hedging_error"], 20.5/(2.*1.01**6)-10.)
        self.assertNotAlmostEqual(result.summary["discounted_pnl"]["terminal_hedging_error"], .5/(2.*1.01**6))
        self.assert_reconciles(result)

    def test_optional_mode_does_not_guess_funding_or_payment_discounts(self):
        result = self.run_case(config=replace(self.config, numeraire_id=None))
        self.assertIsNone(result.summary["discounted_pnl"])
        self.assertFalse(result.manifest["numeraire_policy"]["enabled"])
        self.assertTrue(all(r["discounted_daily_pnl"] is None for r in result.ledger))
        self.assertTrue(all(r["discounted"] is None for r in result.pnl_components))

    def test_all_strategies_share_account_and_nominal_execution_is_unchanged(self):
        results = compare_strategies(self.config, self.history, [self.contract], self.positions, self.provider)
        hashes = {r.manifest["numeraire_policy"]["input_hash"] for r in results.values()}
        self.assertEqual(len(hashes), 1)
        for strategy, result in results.items():
            nominal = self.run_case(config=replace(self.config, strategy=strategy, numeraire_id=None))
            self.assertEqual(result.orders, nominal.orders)
            self.assertEqual(result.fills, nominal.fills)
            self.assertEqual(result.cashflows, nominal.cashflows)
            self.assertEqual([r["daily_pnl"] for r in result.ledger], [r["daily_pnl"] for r in nominal.ledger])
            self.assert_reconciles(result)

    def test_account_checks_fail_before_provider_and_resume_exactly(self):
        s = self.history[1]
        cases = ((None, "missing_numeraire"),
                 (replace(s.numeraire, account_id="other"), "numeraire_account_mismatch"),
                 (replace(s.numeraire, currency_id=99), "numeraire_currency_mismatch"),
                 (replace(s.numeraire, valuation_date=START), "numeraire_date_mismatch"),
                 (replace(s.numeraire, available_at=s.at+timedelta(seconds=1)), "future_numeraire"),
                 (replace(s.numeraire, observed_at=s.at-timedelta(days=2)), "stale_numeraire"))
        partial = self.run_case(stop_after=1)
        whole = self.run_case()
        for point, code in cases:
            with self.subTest(code=code), patch.object(self.provider, "evaluate", wraps=self.provider.evaluate) as evaluated:
                failed = self.run_case(history=[self.history[0], replace(s, numeraire=point), *self.history[2:]], on_error="return")
                self.assertEqual(failed.failure["code"], code)
                self.assertEqual(failed.checkpoint, partial.checkpoint)
                self.assertTrue(all(call.args[0].snapshot_id == "s0" for call in evaluated.call_args_list))
                self.assertIsNone(failed.summary["discounted_pnl"]["terminal_hedging_error"])
                self.assertEqual(self.run_case(checkpoint=failed.checkpoint).to_json(), whole.to_json())

    def test_future_account_changes_cannot_rewrite_past_or_trades(self):
        whole = self.run_case()
        changed = [*self.history[:2], *(replace(s, numeraire=replace(s.numeraire, value=s.numeraire.value*2.)) for s in self.history[2:])]
        other = self.run_case(history=changed)
        self.assertEqual(whole.ledger[:2], other.ledger[:2])
        self.assertEqual(whole.pnl_components[:2], other.pnl_components[:2])
        self.assertEqual(whole.orders, other.orders)
        self.assertEqual(whole.fills, other.fills)
        self.assertNotEqual(whole.summary["discounted_pnl"]["cumulative_pnl"], other.summary["discounted_pnl"]["cumulative_pnl"])
        partial = self.run_case(stop_after=2)
        self.assertEqual(self.run_case(history=changed, checkpoint=partial.checkpoint).to_json(), other.to_json())
        with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
            self.run_case(history=[replace(self.history[0], numeraire=account(self.history[0], 3.)), *self.history[1:]],
                          checkpoint=partial.checkpoint)

    def test_first_day_failure_and_partial_run_do_not_claim_terminal_error(self):
        failed = self.run_case(history=[replace(self.history[0], numeraire=None), *self.history[1:]], on_error="return")
        self.assertEqual(failed.failure["code"], "missing_numeraire")
        self.assertIsNone(failed.summary["discounted_pnl"])
        self.assertEqual(self.run_case(checkpoint=failed.checkpoint).to_json(), self.run_case().to_json())
        partial = self.run_case(stop_after=2)
        self.assertIsNone(partial.summary["discounted_pnl"]["terminal_hedging_error"])

    def test_nonfinite_conversion_and_reconciliation_corruption_roll_back(self):
        failed = self.run_case(history=[replace(self.history[0], numeraire=account(self.history[0], 1e-320)), *self.history[1:]],
                               on_error="return")
        self.assertEqual(failed.failure["code"], "nonfinite_discounted_pnl")
        self.assertEqual(failed.ledger, ())

        def corrupted(row, *args, **kwargs):
            result = discounted_ledger(row, *args, **kwargs)
            if row["at"] == self.history[1].at.isoformat():
                result["discounted_cumulative_pnl"] += 1.
            return result

        with patch("rust_pricing_backtest.engine.discounted_ledger", side_effect=corrupted):
            failed = self.run_case(on_error="return")
        self.assertEqual(failed.failure["code"], "discounted_pnl_reconciliation")
        self.assertEqual(failed.checkpoint, self.run_case(stop_after=1).checkpoint)
        self.assertEqual(self.run_case(checkpoint=failed.checkpoint).to_json(), self.run_case().to_json())

    def test_input_validation(self):
        s = self.history[0]
        for changes in ({"value": 0.}, {"value": -1.}, {"value": math.nan}, {"value": math.inf},
                        {"account_id": " "}, {"currency_id": True}, {"source": ""}, {"revision": ""},
                        {"valuation_date": s.at}, {"available_at": s.at-timedelta(seconds=1)}):
            with self.subTest(changes=changes), self.assertRaises(BacktestError):
                replace(s.numeraire, **changes)
        with self.assertRaisesRegex(BacktestError, "invalid_numeraire"):
            replace(s, numeraire=1.)
        with self.assertRaisesRegex(BacktestError, "invalid_config"):
            replace(self.config, numeraire_id="")

    def test_json_replay_and_legacy_input_preserves_disabled_mode(self):
        whole = self.run_case()
        partial = self.run_case(stop_after=2)
        loaded = load_input(input_json(whole))
        self.assertEqual(run_backtest(**loaded, provider=self.provider, checkpoint=partial.checkpoint).to_json(), whole.to_json())
        self.assertEqual(BacktestResult.from_json(whole.to_json()).to_json(), whole.to_json())
        baseline = self.run_case(config=replace(self.config, numeraire_id=None),
                                 history=[replace(s, numeraire=None) for s in self.history])
        legacy = json.loads(input_json(baseline))
        legacy["schema_version"] = 3
        del legacy["config"]["constraint_mode"], legacy["config"]["gross_notional_limit"]
        del legacy["config"]["numeraire_id"]
        for s in legacy["market_history"]:
            del s["numeraire"], s["external_cashflows"]
        self.assertEqual(run_backtest(**load_input(json.dumps(legacy)), provider=self.provider).to_json(), baseline.to_json())
        malformed = json.loads(input_json(whole))
        del malformed["market_history"][0]["numeraire"]
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            load_input(json.dumps(malformed))
        old = json.loads(baseline.to_json())
        old["manifest"]["schema_version"] = 4
        del old["summary"]["discounted_pnl"]
        for row in old["ledger"]:
            for key in ("numeraire", "numeraire_hash", "discounted_equity", "discounted_daily_pnl", "discounted_cumulative_pnl"):
                del row[key]
        for row in old["pnl_components"]:
            del row["discounted"]
        restored = BacktestResult.from_json(json.dumps(old))
        self.assertNotIn("discounted_pnl", restored.summary)
        self.assertNotIn("discounted_equity", restored.ledger[0])


if __name__ == "__main__":
    unittest.main()
