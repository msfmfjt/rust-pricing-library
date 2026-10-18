from dataclasses import replace
from datetime import timedelta
import json
import math
from pathlib import Path
import tempfile
import unittest

from rust_pricing_backtest import (BacktestConfig, BacktestError, Contract, PaymentDiscount,
                                  Position, Risk, run_backtest)
from rust_pricing_backtest.serialization import input_json, load_input
from test_backtest import START, EXPIRY, snapshot


PAYMENT = START + timedelta(days=6)


def discount(s, factor):
    return PaymentDiscount(s.at.date(), factor, s.at, s.at, "hand_computed_curve", "v1")


class DatedProvider:
    identity = "payment_ledger_fixture_v1"

    def evaluate(self, s, c):
        if s.at.date() >= c.expiry:
            raise AssertionError("expired contract must not be repriced")
        return Risk(9.5, 0., (10., 0., 0., 0.), s.basis.basis_id, payment_date=c.payment_day)


class PaymentTests(unittest.TestCase):
    def setUp(self):
        self.contract = Contract("target", EXPIRY, 100., payment_date=PAYMENT)
        self.history = [snapshot(i, spot=110. if i == 4 else 100.) for i in range(7)]
        for i in range(6):
            self.history[i] = replace(self.history[i], payment_discounts={PAYMENT: discount(self.history[i], .94 + .01*i)})
        self.config = BacktestConfig(strategy="unhedged")

    def run_case(self, **kwargs):
        args = dict(config=self.config, history=self.history, contracts=[self.contract],
                    portfolio=[Position("target", 1., 9.5)], provider=DatedProvider())
        return run_backtest(**{**args, **kwargs})

    def test_fix_discount_and_pay_without_double_counting(self):
        result = self.run_case()
        for i, pv, cash, equity in ((4, 9.8, -9.5, .3), (5, 9.9, -9.5, .4), (6, 0., .5, .5)):
            row = result.ledger[i]
            self.assertAlmostEqual(row["receivable_value"], pv)
            self.assertAlmostEqual(row["cash"], cash)
            self.assertAlmostEqual(row["equity"], equity)
            self.assertEqual(row["target_value"], 0.)
            self.assertEqual(row["delta"], 0.)
            self.assertEqual(row["raw_vega_l2"], 0.)
        self.assertAlmostEqual(result.ledger[5]["daily_pnl"], .1)
        self.assertAlmostEqual(result.ledger[6]["daily_pnl"], .1)
        self.assertEqual([r["event"] for r in result.settlements], ["fixed", "paid"])
        self.assertEqual([r["at"] for r in result.settlements], [self.history[4].at.isoformat(), self.history[6].at.isoformat()])
        paid = [f for f in result.cashflows if f["kind"] == "targets_settlement"]
        self.assertEqual(len(paid), 1)
        self.assertEqual(paid[0]["amount"], 10.)
        self.assertEqual(paid[0]["at"], self.history[6].at.isoformat())
        self.assertAlmostEqual(result.summary["terminal_hedging_error"], .5)
        for i, pnl in ((4, .3), (5, .1), (6, .1)):
            self.assertAlmostEqual(result.pnl_components[i]["target"], pnl)
            self.assertAlmostEqual(result.pnl_components[i]["funding"], 0.)
        payment = result.pnl_components[6]["details"]
        self.assertAlmostEqual(payment["target_claim_change"], -9.9)
        self.assertEqual(payment["target_settlement_cash"], 10.)

    def test_unpaid_equal_receivable_and_payable_do_not_count_as_settled(self):
        contracts = [self.contract, replace(self.contract, contract_id="short")]
        positions = [Position("target", 1., 9.5), Position("short", -1., 9.5)]
        result = self.run_case(history=self.history[:6], contracts=contracts, portfolio=positions)
        self.assertEqual(result.summary["cumulative_pnl"], 0.)
        self.assertEqual(result.summary["receivable_value"], 0.)
        self.assertEqual(result.summary["outstanding_payments"], 2)
        self.assertFalse(result.summary["fully_settled"])
        self.assertIsNone(result.summary["terminal_hedging_error"])
        self.assertEqual(sorted(r["amount"] for r in result.ledger[-1]["receivables"].values()), [-10., 10.])

    def test_interest_accrues_on_cash_not_unpaid_receivable(self):
        history = [replace(s, borrowing_rate=.365, lending_rate=.01) for s in self.history]
        result = self.run_case(history=history)
        self.assertAlmostEqual(result.ledger[5]["cash"], -9.5 * math.exp(.005))
        self.assertAlmostEqual(result.ledger[6]["cash"], -9.5 * math.exp(.006) + 10.)
        self.assertAlmostEqual(result.ledger[5]["receivable_value"], 9.9)

    def test_resume_unpaid_claim_and_export(self):
        partial = self.run_case(stop_after=5)
        resumed = self.run_case(checkpoint=json.loads(json.dumps(partial.checkpoint)))
        whole = self.run_case()
        self.assertEqual(resumed.to_json(), whole.to_json())
        self.assertEqual(whole.to_json(), run_backtest(**load_input(input_json(whole)), provider=DatedProvider()).to_json())
        with tempfile.TemporaryDirectory() as directory:
            resumed.write(directory)
            self.assertTrue(Path(directory, "settlements.csv").is_file())
            self.assertEqual(len(list(Path(directory).glob("*.csv"))), 11)

    def test_missing_discount_rolls_back_fixing_or_mark(self):
        for i in (4, 5):
            with self.subTest(day=i):
                history = list(self.history)
                history[i] = replace(history[i], payment_discounts={})
                failed = self.run_case(history=history, on_error="return")
                self.assertEqual(failed.failure["code"], "missing_payment_discount")
                self.assertEqual(failed.checkpoint, self.run_case(stop_after=i).checkpoint)
                self.assertEqual(self.run_case(checkpoint=failed.checkpoint).to_json(), self.run_case().to_json())

    def test_missing_payment_day_does_not_pay_late(self):
        result = self.run_case(history=[*self.history[:6], snapshot(7)], on_error="return")
        self.assertEqual(result.failure["code"], "missing_payment_snapshot")
        self.assertEqual(result.failure["remaining_state"]["cash"], -9.5)
        self.assertEqual(len(result.failure["remaining_state"]["receivables"]), 1)
        self.assertFalse(any(f["kind"] == "targets_settlement" for f in result.cashflows))

    def test_payment_cash_floor_failure_restores_liability(self):
        result = self.run_case(config=replace(self.config, cash_floor=0.),
                               portfolio=[Position("target", -1., 9.5)], on_error="return")
        self.assertEqual(result.failure["code"], "cash_limit")
        self.assertEqual(result.failure["snapshot_index"], 6)
        self.assertEqual(result.failure["remaining_state"]["cash"], 9.5)
        self.assertEqual(result.failure["remaining_state"]["receivables"]["targets:target"]["amount"], -10.)
        self.assertEqual([r["event"] for r in result.settlements], ["fixed"])

    def test_zero_payoff_does_not_leave_a_claim(self):
        history = [replace(s, spot=replace(s.spot, bid=90., ask=90.), payment_discounts={}) for s in self.history[:5]]
        result = self.run_case(history=history)
        self.assertTrue(result.summary["fully_settled"])
        self.assertEqual(result.summary["outstanding_payments"], 0)
        self.assertEqual([r["event"] for r in result.settlements], ["fixed_zero"])

    def test_discount_validation_and_negative_rates(self):
        s = self.history[5]
        for changes, code in (({"available_at": s.at + timedelta(seconds=1)}, "future_payment_discount"),
                              ({"valuation_date": START}, "payment_discount_date_mismatch"),
                              ({"observed_at": s.at - timedelta(days=2)}, "stale_payment_discount")):
            with self.subTest(changes=changes):
                history = list(self.history)
                history[5] = replace(s, payment_discounts={PAYMENT: replace(discount(s, .99), **changes)})
                result = self.run_case(history=history, on_error="return")
                self.assertEqual(result.failure["code"], code)
        for factor in (0., -1., math.inf, math.nan):
            with self.subTest(factor=factor), self.assertRaises(BacktestError):
                discount(s, factor)
        history = [*self.history[:5], replace(s, payment_discounts={PAYMENT: discount(s, 1.01)})]
        self.assertAlmostEqual(self.run_case(history=history).summary["receivable_value"], 10.1)
        with self.assertRaisesRegex(BacktestError, "invalid_payment_date"):
            replace(self.contract, payment_date=EXPIRY - timedelta(days=1))

    def test_provider_must_declare_delayed_risk_payment_date(self):
        from test_backtest import FixtureProvider
        provider = FixtureProvider({"target": (9.5, 0., (10., 0., 0., 0.))})
        result = self.run_case(provider=provider, on_error="return")
        self.assertEqual(result.failure["code"], "payment_risk_mismatch")
        self.assertEqual(result.summary["observations"], 0)

    def test_version_two_inputs_keep_same_day_payment(self):
        contract = replace(self.contract, payment_date=None)
        history = [replace(s, payment_discounts={}) for s in self.history]
        expected = self.run_case(contracts=[contract], history=history)
        doc = json.loads(input_json(expected))
        doc["schema_version"] = 2
        del doc["config"]["constraint_mode"], doc["config"]["gross_notional_limit"]
        del doc["config"]["numeraire_id"]
        for c in doc["contracts"]:
            del c["payment_date"]
        for s in doc["market_history"]:
            del s["payment_discounts"], s["numeraire"], s["external_cashflows"]
        actual = run_backtest(**load_input(json.dumps(doc)), provider=DatedProvider())
        self.assertEqual(actual.to_json(), expected.to_json())

    def test_target_and_hedge_claims_stay_separate_until_payment(self):
        history = [replace(s, option_quotes={"target": replace(s.spot, bid=9.5, ask=9.5)}) for s in self.history]
        result = self.run_case(history=history, config=BacktestConfig(ridge=0., allow_identical_hedge=True),
                               portfolio=[Position("target", -1., 9.5)])
        claims = result.ledger[4]["receivables"]
        self.assertEqual(set(claims), {"targets:target", "hedges:target"})
        self.assertEqual(claims["targets:target"]["amount"], -10.)
        self.assertEqual(claims["hedges:target"]["amount"], 10.)
        self.assertEqual(result.ledger[4]["receivable_value"], 0.)
        self.assertFalse(result.ledger[4]["fully_settled"])
        self.assertEqual(result.summary["terminal_hedging_error"], 0.)
        for i, pnl in ((4, .3), (5, .1), (6, .1)):
            self.assertAlmostEqual(result.pnl_components[i]["target"], -pnl)
            self.assertAlmostEqual(result.pnl_components[i]["option_hedge"], pnl)
        self.assertAlmostEqual(result.summary["pnl_components"]["target"], -.5)
        self.assertAlmostEqual(result.summary["pnl_components"]["option_hedge"], .5)


if __name__ == "__main__":
    unittest.main()
