from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import math
import unittest
from unittest.mock import Mock

from rust_pricing_backtest import (BacktestConfig, BacktestError, Contract, Position, Quote,
                                  SettlementFixing, run_backtest)
from rust_pricing_backtest.serialization import input_json, load_input
from test_backtest import START, EXPIRY, FixtureProvider, snapshot


def fixing(value=110.):
    at = snapshot(4).at
    return SettlementFixing(EXPIRY, value, at - timedelta(hours=1), at,
                            "synthetic_official_fix_fixture", "original-v1")


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.contracts = [Contract("target", EXPIRY, 100, settlement_rule="explicit_fixing", settlement_fixing_id="close"),
                          Contract("hedge", EXPIRY, 105, settlement_rule="explicit_fixing", settlement_fixing_id="close")]
        self.positions = [Position("target", -1, 5)]
        self.provider = FixtureProvider({"target": (5., .5, (10., 20., 0., 0.)),
                                         "hedge": (3., .2, (5., 10., 0., 0.))})
        self.history = [snapshot(i, prices={"hedge": 3.}) for i in range(5)]
        self.history[-1] = replace(self.history[-1], settlement_fixings={"close": fixing()})
        self.config = BacktestConfig(ridge=0)

    def run_case(self, **kwargs):
        inputs = dict(config=self.config, history=self.history, contracts=self.contracts,
                      portfolio=self.positions, provider=self.provider)
        return run_backtest(**{**inputs, **kwargs})

    def test_hand_computed_call_put_sign_and_multiplier(self):
        for side, price, quantity, multiplier, entry, expected in (
                ("call", 112., 2., 10., 5., 140.), ("put", 88., -3., 2., 4., -48.),
                ("put", 0., 1., 1., 4., 96.)):
            with self.subTest(side=side, price=price):
                contract = replace(self.contracts[0], side=side, multiplier=multiplier)
                history = [*self.history[:-1], snapshot(4, spot=95., settlement_fixings={"close": fixing(price)})]
                result = self.run_case(config=BacktestConfig(strategy="unhedged"), contracts=[contract],
                                       portfolio=[Position("target", quantity, entry)], history=history)
                self.assertAlmostEqual(result.summary["terminal_hedging_error"], expected)
                row = next(f for f in result.cashflows if f["kind"] == "targets_settlement")
                self.assertEqual(row["amount"], quantity * multiplier * contract.payoff(price))
                self.assertEqual(row["settlement"]["underlying_price"], price)
                self.assertEqual(row["settlement"]["fixing"]["revision"], "original-v1")
                self.assertEqual(result.manifest["snapshot_spot_settlement_contracts"], [])

    def test_target_and_held_hedge_use_shared_fixing_not_spot(self):
        history = [*self.history[:-1], snapshot(4, spot=98., settlement_fixings={"close": fixing()})]
        result = self.run_case(history=history)
        settlements = [f for f in result.cashflows if f["kind"].endswith("_settlement")]
        self.assertEqual([f["amount"] for f in settlements], [-10., 10.])
        self.assertEqual(settlements[0]["settlement"]["fixing_hash"], settlements[1]["settlement"]["fixing_hash"])
        self.assertAlmostEqual(result.summary["terminal_hedging_error"], -1.2)
        self.assertEqual(result.ledger[-1]["target_value"], 0.)
        self.assertEqual(result.ledger[-1]["hedge_value"], 0.)

    def test_missing_fixing_stops_and_repaired_resume_is_exact(self):
        history = [*self.history[:-1], replace(self.history[-1], settlement_fixings={})]
        failed = self.run_case(history=history, on_error="return")
        self.assertEqual(failed.failure["code"], "missing_settlement_fixing")
        self.assertEqual(failed.checkpoint["state"]["next_index"], 4)
        self.assertEqual(failed.failure["remaining_state"]["targets"], {"target": -1.})
        self.assertEqual(failed.failure["remaining_state"]["hedges"], {"hedge": 2.})
        self.assertFalse(any(f["kind"].endswith("_settlement") for f in failed.cashflows))
        resumed = self.run_case(checkpoint=failed.checkpoint)
        self.assertEqual(resumed.to_json(), self.run_case().to_json())
        self.assertEqual(len([f for f in resumed.cashflows if f["kind"].endswith("_settlement")]), 2)

    def test_unavailable_fixing_cannot_affect_earlier_orders(self):
        unavailable = replace(fixing(999.), available_at=self.history[-1].at + timedelta(seconds=1))
        history = [*self.history[:-1], replace(self.history[-1], settlement_fixings={"close": unavailable})]
        failed = self.run_case(history=history, on_error="return")
        partial = self.run_case(stop_after=4)
        self.assertEqual(failed.failure["code"], "future_settlement_fixing")
        self.assertEqual(failed.orders, partial.orders)
        self.assertEqual(failed.checkpoint, partial.checkpoint)
        provider = FixtureProvider(self.provider.values)
        provider.evaluate = Mock(wraps=provider.evaluate)
        bad_first = replace(self.history[0], settlement_fixings={"unrelated": unavailable})
        first = self.run_case(history=[bad_first, *self.history[1:]], provider=provider, on_error="return")
        self.assertEqual(first.summary["observations"], 0)
        provider.evaluate.assert_not_called()

    def test_wrong_fixing_date_or_observation_date_rejected(self):
        for wrong in (replace(fixing(), fixing_date=EXPIRY - timedelta(days=1)),
                      replace(fixing(), observed_at=fixing().observed_at - timedelta(days=1))):
            with self.subTest(wrong=wrong):
                history = [*self.history[:-1], replace(self.history[-1], settlement_fixings={"close": wrong})]
                result = self.run_case(history=history, on_error="return")
                self.assertEqual(result.failure["code"], "settlement_date_mismatch")
                self.assertEqual(result.checkpoint["state"]["next_index"], 4)

    def test_observation_date_uses_session_timezone(self):
        jst = timezone(timedelta(hours=9))
        at = datetime.combine(EXPIRY, datetime.min.time(), jst) + timedelta(minutes=30)
        observed = (at - timedelta(minutes=15)).astimezone(timezone.utc)
        self.assertNotEqual(observed.date(), EXPIRY)
        f = SettlementFixing(EXPIRY, 110., observed, at, "timezone_fixture", "v1")
        final = replace(snapshot(4), at=at, spot=Quote(100., 100., at, at), surface_available_at=at,
                        settlement_fixings={"close": f})
        result = self.run_case(config=BacktestConfig(strategy="unhedged"), contracts=self.contracts[:1],
                               history=[self.history[0], final])
        self.assertEqual(result.summary["terminal_hedging_error"], -5.)

    def test_snapshot_spot_rule_is_explicit_and_not_overridden(self):
        contract = Contract("target", EXPIRY, 100)
        history = [*self.history[:-1], snapshot(4, spot=105., settlement_fixings={"close": fixing(999.)})]
        result = self.run_case(config=BacktestConfig(strategy="unhedged"), history=history, contracts=[contract])
        self.assertEqual(result.summary["terminal_hedging_error"], 0.)
        self.assertEqual(result.manifest["snapshot_spot_settlement_contracts"], ["target"])
        row = next(f for f in result.cashflows if f["kind"] == "targets_settlement")
        self.assertEqual(row["settlement"]["rule"], "snapshot_spot")
        self.assertEqual(row["settlement"]["underlying_price"], 105.)
        self.assertNotIn("fixing", row["settlement"])

    def test_expired_unfilled_order_needs_no_fixing(self):
        contracts = [Contract("target", EXPIRY, 100),
                     replace(self.contracts[1], expiry=START + timedelta(days=1))]
        result = self.run_case(contracts=contracts, history=[replace(s, settlement_fixings={}) for s in self.history])
        self.assertEqual(result.orders[0]["expired_contracts"], ["hedge"])
        self.assertFalse(any(f["contract_id"] == "hedge" for f in result.fills))
        self.assertFalse(any(f["kind"] == "hedges_settlement" for f in result.cashflows))

    def test_fixing_definition_validation(self):
        for fields in ({"value": -1.}, {"value": math.nan}, {"value": math.inf},
                       {"source": ""}, {"revision": " "}, {"observed_at": datetime(2026, 9, 11)},
                       {"available_at": fixing().observed_at - timedelta(seconds=1)}):
            with self.subTest(fields=fields), self.assertRaises(BacktestError):
                replace(fixing(), **fields)
        for fields in ({"settlement_rule": "unknown"}, {"settlement_fixing_id": None},
                       {"settlement_fixing_id": " "}, {"settlement_rule": "snapshot_spot"}):
            with self.subTest(fields=fields), self.assertRaises(BacktestError):
                replace(self.contracts[0], **fields)
        with self.assertRaisesRegex(BacktestError, "invalid_settlement_fixing"):
            replace(self.history[0], settlement_fixings={"close": 100.})

    def test_request_roundtrip_and_legacy_v1_migration(self):
        result = self.run_case()
        payload = input_json(result)
        self.assertEqual(json.loads(payload)["schema_version"], 6)
        replay = run_backtest(**load_input(payload), provider=self.provider)
        self.assertEqual(replay.to_json(), result.to_json())
        legacy_contracts = [replace(c, settlement_rule="snapshot_spot", settlement_fixing_id=None) for c in self.contracts]
        legacy = self.run_case(contracts=legacy_contracts, history=[replace(s, settlement_fixings={}) for s in self.history])
        doc = json.loads(input_json(legacy))
        doc["schema_version"] = 1
        del doc["config"]["constraint_mode"], doc["config"]["gross_notional_limit"]
        del doc["config"]["numeraire_id"]
        for c in doc["contracts"]:
            del c["settlement_rule"], c["settlement_fixing_id"], c["payment_date"]
        for s in doc["market_history"]:
            del s["settlement_fixings"], s["payment_discounts"], s["numeraire"], s["external_cashflows"]
        loaded = load_input(json.dumps(doc))
        self.assertTrue(all(c.settlement_rule == "snapshot_spot" for c in loaded["contracts"]))
        self.assertEqual(run_backtest(**loaded, provider=self.provider).to_json(), legacy.to_json())
        malformed = json.loads(payload)
        del malformed["contracts"][0]["settlement_rule"]
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            load_input(json.dumps(malformed))

    def test_identical_contract_exclusion_includes_fixing_rule(self):
        for key, expected in (("close", []), ("other_close", ["hedge"])):
            with self.subTest(key=key):
                hedge = replace(self.contracts[1], strike=100., settlement_fixing_id=key)
                result = self.run_case(contracts=[self.contracts[0], hedge], stop_after=1)
                self.assertEqual(result.decisions[0]["eligible"], expected)

    def test_later_revision_does_not_rewrite_paid_cashflows(self):
        corrected = replace(fixing(120.), available_at=snapshot(5).at, revision="revised-v2")
        history = [*self.history, snapshot(5, settlement_fixings={"close": corrected})]
        result = self.run_case(config=BacktestConfig(strategy="unhedged"), history=history)
        paid = [f for f in result.cashflows if f["kind"] == "targets_settlement"]
        self.assertEqual(len(paid), 1)
        self.assertEqual(paid[0]["amount"], -10.)
        self.assertEqual(paid[0]["settlement"]["fixing"]["revision"], "original-v1")
        self.assertEqual(result.ledger[-1]["cash"], result.ledger[-2]["cash"])
        partial = self.run_case(config=BacktestConfig(strategy="unhedged"), history=history, stop_after=5)
        changed = [*history[:4], replace(history[4], settlement_fixings={"close": replace(fixing(), revision="changed")}), history[5]]
        with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
            self.run_case(config=BacktestConfig(strategy="unhedged"), history=changed, checkpoint=partial.checkpoint)


if __name__ == "__main__":
    unittest.main()
