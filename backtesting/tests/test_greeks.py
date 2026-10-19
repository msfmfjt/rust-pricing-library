from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rust_pricing_backtest import BacktestConfig, BacktestError, BacktestResult, Contract, Position, Risk, run_backtest
from rust_pricing_backtest import greeks
from test_backtest import EXPIRY, snapshot
from test_capital import movement
import test_payments
import test_pnl


class PolynomialProvider:
    identity = "independent_quadratic_price_fixture_v1"

    def __init__(self, theta=0., gamma=.1):
        self.theta, self.gamma = theta, gamma

    def evaluate(self, s, c):
        x = s.spot.mid - 100.
        return Risk(10.+.4*x+.05*x*x+self.theta*int(s.snapshot_id[1:]), .4+.1*x,
                    (0.,)*s.basis.size, s.basis.basis_id, gamma=self.gamma,
                    gamma_method="analytic_polynomial" if self.gamma is not None else None)


class GreekPnlTests(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(strategy="unhedged")
        self.contracts = [Contract("target", EXPIRY, 100., multiplier=10.)]
        self.positions = [Position("target", -2., 10.)]
        self.history = [snapshot(0, spot=100.), snapshot(1, spot=102.), snapshot(2, spot=105.)]
        self.provider = PolynomialProvider()

    def run_case(self, **kwargs):
        args = dict(config=self.config, history=self.history, contracts=self.contracts,
                    portfolio=self.positions, provider=self.provider)
        return run_backtest(**{**args, **kwargs})

    def assert_reconciles(self, result):
        self.assertEqual(len(result.greek_pnl), len(result.ledger))
        for row, ledger in zip(result.greek_pnl, result.ledger):
            known = row["available_greek_pnl"] if row["available_greek_pnl"] is not None else 0.
            self.assertLessEqual(abs(known+row["unexplained_pnl"]+row["non_asset_pnl"]-ledger["daily_pnl"]), row["tolerance"])
            self.assertLessEqual(abs(sum(row["cumulative"].values())-ledger["cumulative_pnl"]), row["cumulative_tolerance"])

    def test_quadratic_price_long_short_and_multiplier_oracle(self):
        for sign in (1., -1.):
            result = self.run_case(portfolio=[Position("target", sign*2., 10.)])
            row = result.greek_pnl[1]["groups"]["target"]
            self.assertAlmostEqual(row["opening_delta"], sign*8.)
            self.assertAlmostEqual(row["opening_gamma"], sign*2.)
            self.assertAlmostEqual(row["delta_pnl"], sign*16.)
            self.assertAlmostEqual(row["gamma_pnl"], sign*4.)
            self.assertAlmostEqual(row["actual_pnl"], sign*20.)
            self.assertAlmostEqual(row["unexplained_pnl"], 0.)
            self.assertEqual(result.greek_pnl[1]["delta_convention"], "frozen_local_variance_grid")
            self.assert_reconciles(result)

    def test_time_and_market_mark_differences_remain_in_residual(self):
        result = self.run_case(provider=PolynomialProvider(theta=.3))
        self.assertAlmostEqual(result.greek_pnl[1]["groups"]["target"]["unexplained_pnl"], -6.)
        history = [snapshot(0, prices={"target": 9.}), snapshot(1, spot=102., prices={"target": 13.})]
        result = self.run_case(history=history, config=replace(self.config, target_mark="market"))
        row = result.greek_pnl[1]["groups"]["target"]
        self.assertAlmostEqual(row["actual_pnl"], -80.)
        self.assertAlmostEqual(row["available_greek_pnl"], -20.)
        self.assertAlmostEqual(row["unexplained_pnl"], -60.)
        self.assert_reconciles(result)

    def test_opening_stock_excludes_todays_fill_and_terminal_liquidation(self):
        result = self.run_case(config=BacktestConfig(strategy="delta"),
                               contracts=[replace(self.contracts[0], multiplier=1.)], portfolio=[Position("target", 1., 10.)])
        day1, day2 = result.greek_pnl[1:]
        self.assertEqual(day1["groups"]["stock"]["delta_pnl"], 0.)
        self.assertAlmostEqual(day2["groups"]["stock"]["opening_delta"], -.4)
        self.assertAlmostEqual(day2["groups"]["stock"]["delta_pnl"], -1.2)
        self.assertAlmostEqual(day2["groups"]["stock"]["unexplained_pnl"], 0.)
        self.assertEqual(result.ledger[-1]["stock"], 0.)
        self.assertAlmostEqual(day2["available_greek_pnl"], 1.05)
        self.assert_reconciles(result)

    def test_missing_gamma_is_null_and_vega_is_never_naively_dotted(self):
        class MixedProvider(PolynomialProvider):
            def evaluate(self, s, c):
                return replace(super().evaluate(s, c), gamma=None, gamma_method=None) if c.contract_id == "missing" else super().evaluate(s, c)

        result = self.run_case(contracts=[*self.contracts, replace(self.contracts[0], contract_id="missing")],
                               portfolio=[*self.positions, Position("missing", 1., 10.)], provider=MixedProvider())
        row = result.greek_pnl[1]["groups"]["target"]
        self.assertIsNone(row["gamma_pnl"])
        self.assertIsNone(row["opening_gamma"])
        self.assertEqual(row["gamma_missing_contracts"], ["missing"])
        self.assertAlmostEqual(row["delta_pnl"], -8.)
        self.assertAlmostEqual(row["unexplained_pnl"], -2.)
        self.assertIsNone(row["vega_pnl"])
        self.assertEqual(result.greek_pnl[1]["vega_status"], "not_computed_risk_and_delta_iv_convention_unvalidated")
        # Absence is still absence when the spot did not move.
        flat = self.run_case(history=[snapshot(0), snapshot(1)], provider=PolynomialProvider(gamma=None))
        self.assertIsNone(flat.greek_pnl[1]["groups"]["target"]["gamma_pnl"])
        self.assertEqual(flat.greek_pnl[1]["groups"]["stock"]["gamma_pnl"], 0.)
        self.assert_reconciles(result)

    def test_current_risks_do_not_explain_past_interval(self):
        class ChangedProvider(PolynomialProvider):
            def evaluate(self, s, c):
                r = super().evaluate(s, c)
                return replace(r, delta=1000., gamma=-999., warning_codes=("future_risk",)) if s.snapshot_id != "s0" else r

        ordinary, changed = self.run_case(), self.run_case(provider=ChangedProvider())
        self.assertEqual(ordinary.greek_pnl[:2], changed.greek_pnl[:2])
        self.assertNotEqual(ordinary.greek_pnl[2], changed.greek_pnl[2])
        self.assertNotIn("future_risk", changed.greek_pnl[1]["risk_warning_codes"])

    def test_initial_entry_mark_has_no_greek_interval(self):
        result = self.run_case(portfolio=[Position("target", -2., 9., entry_fee=.5)])
        row = result.greek_pnl[0]
        self.assertIsNone(row["from_at"])
        self.assertIsNone(row["available_greek_pnl"])
        self.assertIsNone(row["groups"]["target"]["delta_pnl"])
        self.assertEqual(row["unexplained_pnl"], -20.)
        self.assertEqual(row["non_asset_components"]["entry_fees"], -.5)
        self.assertEqual(row["total"], -20.5)

    def test_expiry_and_claim_discount_changes_are_explained_as_residual(self):
        fixture = test_payments.PaymentTests()
        fixture.setUp()
        result = fixture.run_case()
        self.assertAlmostEqual(result.greek_pnl[4]["unexplained_pnl"], .3)
        for i in (5, 6):
            row = result.greek_pnl[i]["groups"]["target"]
            self.assertEqual(row["opening_delta"], 0.)
            self.assertEqual(row["opening_gamma"], 0.)
            self.assertAlmostEqual(row["unexplained_pnl"], .1)
        partial = fixture.run_case(stop_after=5)
        self.assertEqual(fixture.run_case(checkpoint=partial.checkpoint).to_json(), result.to_json())
        self.assert_reconciles(result)

    def test_capital_and_costs_use_exact_cash_components(self):
        fixture = test_pnl.PnlTests()
        fixture.setUp()
        history = list(fixture.history)
        history[1] = replace(history[1], external_cashflows=[movement(history[1], 100.)])
        result = fixture.run_case(history=history)
        for row, pnl in zip(result.greek_pnl, result.pnl_components):
            for key, value in row["non_asset_components"].items():
                self.assertEqual(value, pnl[key])
        self.assertNotIn("external_capital", result.greek_pnl[1]["non_asset_components"])
        self.assert_reconciles(result)

    def test_nonfinite_approximation_fails_with_atomic_recovery_boundary(self):
        provider = PolynomialProvider(gamma=1e308)
        failed = self.run_case(provider=provider, on_error="return")
        self.assertEqual(failed.failure["code"], "nonfinite_greek_pnl")
        self.assertEqual(failed.failure["snapshot_index"], 1)
        self.assertEqual(failed.checkpoint, self.run_case(provider=provider, stop_after=1).checkpoint)

    def test_failure_after_fills_and_on_first_day_resumes_exactly(self):
        original = greeks.explain
        for index in (0, 1):
            def corrupted(*args):
                if args[0].snapshot_id == f"s{index}":
                    args = (*args[:4], {**args[4], "target": args[4]["target"]+1.}, *args[5:])
                return original(*args)

            config = BacktestConfig(strategy="delta")
            with patch("rust_pricing_backtest.greeks.explain", side_effect=corrupted):
                failed = self.run_case(config=config, on_error="return")
            self.assertEqual(failed.failure["code"], "greek_pnl_reconciliation")
            self.assertEqual(len(failed.greek_pnl), index)
            if index == 0:
                self.assertIsNone(failed.summary["greek_pnl"]["cumulative"])
            else:
                self.assertEqual(failed.checkpoint, self.run_case(config=config, stop_after=1).checkpoint)
                self.assertEqual(failed.fills, ())
            self.assertEqual(self.run_case(config=config, checkpoint=failed.checkpoint).to_json(), self.run_case(config=config).to_json())

    def test_export_legacy_and_serialized_checkpoint(self):
        whole, partial = self.run_case(), self.run_case(stop_after=2)
        self.assertEqual(self.run_case(checkpoint=json.loads(json.dumps(partial.checkpoint))).to_json(), whole.to_json())
        with tempfile.TemporaryDirectory() as directory:
            whole.write(directory)
            self.assertEqual(len(list(Path(directory).glob("*.csv"))), 11)
            with Path(directory, "greek_pnl.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["available_greek_pnl"], "")
            self.assertAlmostEqual(json.loads(rows[1]["groups"])["target"]["gamma_pnl"], -4.)
            self.assertEqual(BacktestResult.from_json(Path(directory, "result.json").read_text()).to_json(), whole.to_json())
        for version in range(1, 8):
            old = json.loads(whole.to_json())
            old["manifest"]["schema_version"] = version
            del old["greek_pnl"]
            self.assertEqual(BacktestResult.from_json(json.dumps(old)).greek_pnl, ())
        with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
            self.run_case(checkpoint={**partial.checkpoint, "schema_version": 5})

    def test_gamma_value_and_metadata_validation(self):
        r = self.provider.evaluate(self.history[0], self.contracts[0])
        for change in ({"gamma": float("nan")}, {"gamma": float("inf")}, {"gamma_spot_bump": 0.},
                       {"gamma_spot_bump": -1.}, {"gamma": None}, {"gamma_method": ""}):
            with self.subTest(change=change), self.assertRaises(BacktestError):
                replace(r, **change)


if __name__ == "__main__":
    unittest.main()
