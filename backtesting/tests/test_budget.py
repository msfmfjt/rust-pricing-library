from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rust_pricing_backtest import BacktestConfig, BacktestError, BacktestResult, Contract, Position, run_backtest
from rust_pricing_backtest import budget
from rust_pricing_backtest.domain import data
from rust_pricing_backtest.serialization import input_json, load_input
from rust_pricing_backtest.validation import validate_document
from rust_pricing_backtest.ensemble import ScenarioResults, aggregate_scenarios
from test_backtest import EXPIRY, FixtureProvider, snapshot


HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_VALIDATOR = importlib.util.find_spec("jsonschema") is not None


class BudgetConfigurationTests(unittest.TestCase):
    def test_optional_dependency_and_legacy_config(self):
        self.assertEqual(BacktestConfig().constraint_mode, "execution_cash")
        for kwargs in ({"constraint_mode": "unknown"}, {"gross_notional_limit": 100.},
                       {"constraint_mode": "decision_cash_gross"},
                       *({"constraint_mode": "decision_cash_gross", "gross_notional_limit": cap}
                         for cap in (-1, float("inf"), float("nan"), True, "100"))):
            with self.subTest(kwargs=kwargs), self.assertRaises(BacktestError):
                BacktestConfig(**kwargs)
        original = __import__
        def without_scipy(name, *args, **kwargs):
            if name == "scipy":
                raise ImportError("simulated missing optional dependency")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=without_scipy):
            with self.assertRaisesRegex(BacktestError, "missing_constraint_dependency"):
                budget.backend()
            run_backtest(BacktestConfig(strategy="unhedged"), [snapshot(0)],
                         [Contract("target", EXPIRY, 100)], [Position("target", -1., 5.)],
                         FixtureProvider({"target": (5., .5, (10., 20., 0., 0.))}))

    def test_cost_coefficients_match_direct_buy_and_sell_cash(self):
        q = snapshot(0, prices={"h": (2., 4.)}).option_quotes["h"]
        linear, spread = budget.execution_coefficients(q, 10., .3, 20.)
        for change in (-3., 0., 2.):
            price = (q.ask if change > 0 else q.bid)*(1+math.copysign(.002, change))
            expected = 50-change*10*price-abs(change)*.3
            metrics = budget.estimate([1+change], [1], [linear], [spread], [1000], cash=50, fixed_gross=2000)
            self.assertAlmostEqual(metrics["cash_after"], expected)
            self.assertEqual(metrics["gross_notional"], 2000+abs(1+change)*1000)
        with self.assertRaisesRegex(BacktestError, "solver_shape"):
            budget.estimate([1], [], [1], [0], [1], cash=0, fixed_gross=0)


@unittest.skipUnless(HAS_SCIPY, "install companion constraints extra")
class BudgetSolverTests(unittest.TestCase):
    def solve(self, **overrides):
        inputs = dict(v=[-10.], matrix=[[5.]], previous=[0.], lower=[-100.], upper=[100.], costs=[0.],
                      deltas=[.2], target_delta=-.5, stock_previous=0., stock_lower=-100., stock_upper=100.,
                      linear=[3., 100.], spreads=[0., 0.], gross_weights=[100., 100.],
                      cash=5., cash_floor=-1000., fixed_gross=100., gross_limit=1000., ridge=0.)
        return budget.solve_budget(**(inputs | overrides))

    def test_analytic_unconstrained_and_gross_active_solutions(self):
        loose, tight = self.solve(), self.solve(gross_limit=200.)
        self.assertAlmostEqual(loose.quantities[0], 2.)
        # 100 target + 100h + 100(.5-.2h) <= 200 implies h <= .625.
        self.assertAlmostEqual(tight.quantities[0], .625)
        self.assertAlmostEqual(tight.stock, .375)
        self.assertAlmostEqual(tight.gross_notional, 200.)
        self.assertIsNone(tight.projected_gradient)
        self.assertLessEqual(max(tight.primal_violation, tight.kkt_stationarity, tight.kkt_complementarity), 1e-9)

    def test_cash_constraint_changes_options_and_stock_jointly(self):
        # cash_after=5-3h-100(.5-.2h)=-45+17h.
        result = self.solve(cash_floor=0.)
        self.assertAlmostEqual(result.quantities[0], 45/17)
        self.assertAlmostEqual(result.cash_after, 0.)
        self.assertAlmostEqual(result.stock, .5-.2*45/17)
        with self.assertRaisesRegex(BacktestError, "budget_infeasible"):
            self.solve(cash_floor=0., gross_limit=310.)

    def test_stock_bounds_are_part_of_option_optimization(self):
        result = self.solve(v=[0.], stock_upper=.2)
        self.assertAlmostEqual(result.quantities[0], 1.5)
        self.assertAlmostEqual(result.stock, .2)

    def test_two_dimensional_dense_grid_oracle_with_costs(self):
        v, a = np.array([-2., .5]), np.array([[1., .5], [.2, 1.]])
        delta, linear, spreads = np.array([.2, .1]), np.array([3., 4., 100.]), np.array([.1, .2, .05])
        result = self.solve(v=v, matrix=a, previous=[0., 0.], lower=[-1., -1.], upper=[1., 1.],
            costs=[.1,.2], deltas=delta, linear=linear, spreads=spreads,
            gross_weights=[100., 100., 100.], cash_floor=-40., gross_limit=220., ridge=.0001, turnover_penalty=.001)
        best = math.inf
        for x in np.linspace(-1, 1, 151):
            for y in np.linspace(-1, 1, 151):
                h = np.array([x,y]); stock = .5-delta@h; held = np.array([x,y,stock])
                if 5-linear@held-spreads@np.abs(held) < -40 or 100+100*np.sum(np.abs(held)) > 220:
                    continue
                best = min(best, np.linalg.norm(.01*(v+a@h))**2+.0001*(h@h)+.001*np.array([.1,.2])@np.abs(h))
        self.assertTrue(math.isfinite(best))
        self.assertLessEqual(result.objective_after, best+1e-12)

    def test_empty_universe_requires_feasible_exact_delta_hedge(self):
        inputs = dict(v=[1.], matrix=[[]], previous=[], lower=[], upper=[], costs=[], deltas=[],
                      linear=[100.], spreads=[0.], gross_weights=[100.])
        self.assertAlmostEqual(self.solve(**inputs).stock, .5)
        for override in ({"stock_upper": .1}, {"cash_floor": 0.}, {"gross_limit": 149.}):
            with self.subTest(override=override), self.assertRaisesRegex(BacktestError, "budget_infeasible"):
                self.solve(**(inputs | override))

    def test_fixed_positions_and_rank_deficiency(self):
        fixed = self.solve(lower=[1.], upper=[1.])
        self.assertAlmostEqual(fixed.quantities[0], 1.)
        result = self.solve(matrix=[[5.,5.]], previous=[0.,0.], lower=[-10.,-10.], upper=[10.,10.],
            costs=[0.,0.], deltas=[.2,.2], linear=[3.,3.,100.], spreads=[0.,0.,0.], gross_weights=[100.]*3)
        self.assertEqual(result.rank, 1)
        self.assertIsNone(result.condition_number)
        self.assertAlmostEqual(sum(result.quantities), 2.)

    def test_invalid_inputs_and_false_optimizer_success(self):
        for kwargs, code in (({"linear": [1]}, "solver_shape"), ({"cash": float("nan")}, "solver_nonfinite"),
                             ({"lower": [2], "upper": [1]}, "solver_constraints")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(BacktestError, code):
                self.solve(**kwargs)
        scipy, linprog, _, nnls = budget.backend()
        fake = lambda *args, **kwargs: SimpleNamespace(x=np.zeros(5), nit=1, message="fake success", success=True)
        with patch.object(budget, "backend", return_value=(scipy, linprog, fake, nnls)):
            with self.assertRaisesRegex(BacktestError, "solver_not_converged"):
                self.solve()


@unittest.skipUnless(HAS_SCIPY, "install companion constraints extra")
class BudgetReplayTests(unittest.TestCase):
    def setUp(self):
        self.contracts = [Contract("target", EXPIRY, 100), Contract("hedge", EXPIRY, 105)]
        self.positions = [Position("target", -1, 5)]
        self.provider = FixtureProvider({"target": (5., .5, (10., 20., 0., 0.)), "hedge": (3., .2, (5., 10., 0., 0.))})
        self.history = [snapshot(i, prices={"target": 5., "hedge": 3.}) for i in range(5)]
        self.config = BacktestConfig(ridge=0., constraint_mode="decision_cash_gross", gross_notional_limit=500.)

    def run_case(self, config=None, history=None, contracts=None, **kwargs):
        return run_backtest(config or self.config, history or self.history, contracts or self.contracts,
                            self.positions, self.provider, **kwargs)

    def test_loose_limit_keeps_legacy_economics(self):
        old = self.run_case(config=BacktestConfig(ridge=0.))
        new = self.run_case()
        for table in ("ledger", "fills", "cashflows", "settlements", "pnl_components", "cost_basis"):
            self.assertEqual(getattr(new, table), getattr(old, table))
        self.assertEqual(new.summary, old.summary)

    def test_gross_cap_and_rounding_diagnostics(self):
        result = self.run_case(config=replace(self.config, gross_notional_limit=200.))
        decision = result.decisions[0]
        self.assertAlmostEqual(decision["solver"]["quantities"][0], .625)
        self.assertEqual(result.orders[0]["desired"]["hedge"], 0.)
        self.assertEqual(decision["budget"]["rounded"]["gross_notional"], 150.)
        self.assertEqual(decision["budget"]["rounded_delta"], 0.)

    def test_fees_multipliers_and_spread_estimates_match_unchanged_execution(self):
        contracts = [replace(c, multiplier=10., lot=.01) for c in self.contracts]
        history = [snapshot(i, spot=(99.9,100.1), prices={"target": 5., "hedge": (2.9,3.1)}) for i in range(5)]
        result = self.run_case(config=replace(self.config, gross_notional_limit=5000., option_fee=.1, stock_fee=.03,
                                              slippage_bps=10.), contracts=contracts, history=history)
        metrics = result.decisions[0]["budget"]["rounded"]
        self.assertAlmostEqual(metrics["cash_after"], result.ledger[1]["cash"])
        for key, value in metrics.items():
            self.assertAlmostEqual(value, result.orders[0]["execution_budget"][key])
        self.assertAlmostEqual(metrics["gross_notional"], 3100.)

    def test_rounding_can_make_cash_infeasible_without_silent_repair(self):
        result = self.run_case(config=replace(self.config, cash_floor=0.), on_error="return")
        self.assertEqual(result.failure["code"], "rounded_budget_infeasible")
        self.assertEqual(result.checkpoint["state"]["next_index"], 0)
        self.assertEqual(result.fills, ())
        self.assertEqual(result.orders, ())

    def test_delta_infeasibility_is_not_silently_clipped(self):
        result = self.run_case(config=replace(self.config, strategy="delta", stock_position_limit=.1), on_error="return")
        self.assertEqual(result.failure["code"], "budget_infeasible")

    def test_next_quote_cash_and_gross_rejections_are_atomic(self):
        for config, next_snapshot, status in (
            (replace(self.config, cash_floor=-20.), snapshot(1, prices={"hedge": 100.}), "rejected_cash_limit"),
            (replace(self.config, gross_notional_limit=320.), snapshot(1, spot=110., prices={"hedge": 3.}), "rejected_gross_limit")):
            with self.subTest(status=status):
                history = [self.history[0], next_snapshot, *self.history[2:]]
                result = self.run_case(config=replace(config, rebalance_every=3), history=history, stop_after=2)
                self.assertEqual(result.orders[0]["status"], status)
                self.assertEqual(result.fills, ())
                self.assertEqual(result.ledger[1]["cash"], 5.)
                self.assertEqual(result.ledger[1]["stock"], 0.)

    def test_market_move_breach_rolls_back_and_preserves_checkpoint(self):
        config = replace(self.config, gross_notional_limit=320., rebalance_every=3)
        partial = self.run_case(config=config, stop_after=2)
        history = [*self.history[:2], snapshot(2, spot=120., prices={"hedge": 3.}), *self.history[3:]]
        failed = self.run_case(config=config, history=history, on_error="return")
        self.assertEqual(failed.failure["code"], "gross_limit")
        self.assertEqual(failed.checkpoint, partial.checkpoint)
        whole = self.run_case(config=config)
        self.assertEqual(self.run_case(config=config, checkpoint=failed.checkpoint).to_json(), whole.to_json())

    def test_future_changes_do_not_change_prior_decisions(self):
        original = self.run_case(stop_after=2)
        changed = self.run_case(history=[*self.history[:2], snapshot(2, spot=120., prices={"hedge": 30.}), *self.history[3:]], stop_after=2)
        self.assertEqual(original.decisions, changed.decisions)
        self.assertEqual(original.orders, changed.orders)

    def test_all_strategies_resume_and_wire_roundtrip(self):
        for strategy in ("unhedged", "delta", "scalar_vega", "vega_kt"):
            with self.subTest(strategy=strategy):
                config = replace(self.config, strategy=strategy)
                whole, partial = self.run_case(config=config), self.run_case(config=config, stop_after=2)
                self.assertEqual(self.run_case(config=config, checkpoint=partial.checkpoint).to_json(), whole.to_json())
                self.assertEqual(run_backtest(**load_input(input_json(whole)), provider=self.provider).to_json(), whole.to_json())
                self.assertEqual(BacktestResult.from_json(whole.to_json()).to_json(), whole.to_json())

    def test_backend_version_is_part_of_checkpoint_identity(self):
        partial = self.run_case(stop_after=2)
        scipy, *functions = budget.backend()
        with patch.object(budget, "backend", return_value=(SimpleNamespace(__version__="other"), *functions)):
            with self.assertRaisesRegex(BacktestError, "checkpoint_mismatch"):
                self.run_case(checkpoint=partial.checkpoint)

    @unittest.skipUnless(HAS_VALIDATOR, "install companion validation extra")
    def test_current_schemas_accept_constrained_and_failed_runs(self):
        results = {}
        for strategy in ("unhedged", "delta", "scalar_vega", "vega_kt"):
            result = self.run_case(config=replace(self.config, strategy=strategy))
            results[strategy] = result
            for doc in (json.loads(result.to_json()), json.loads(input_json(result)), json.loads(json.dumps(data(result.checkpoint)))):
                self.assertEqual(validate_document(doc)["status"], "valid")
        report = aggregate_scenarios([ScenarioResults("budget", "budget", "fixture", results)])
        self.assertEqual(validate_document(json.loads(report.to_json()))["status"], "valid")
        failed = self.run_case(config=replace(self.config, cash_floor=0.), on_error="return")
        self.assertEqual(validate_document(json.loads(failed.to_json()))["status"], "valid")
        request = json.loads(input_json(result))
        request["config"]["gross_notional_limit"] = None
        with self.assertRaisesRegex(BacktestError, "schema_fields"):
            validate_document(request)

    @unittest.skipUnless(HAS_VALIDATOR, "install companion validation extra")
    def test_historical_v5_reader_and_new_ensemble_of_v9_results(self):
        root = Path(__file__).resolve().parents[2]/"fixtures/backtesting/wire"
        request = load_input((root/"request-v5.golden.json").read_text())
        self.assertEqual(request["config"].constraint_mode, "execution_cash")
        result = BacktestResult.from_json((root/"result-v9.golden.json").read_text())
        report = aggregate_scenarios([ScenarioResults("legacy", "legacy", "frozen_fixture", {"vega_kt": result})])
        self.assertEqual(report.manifest["schema_version"], 2)
        self.assertEqual(validate_document(json.loads(report.to_json()))["status"], "valid")


if __name__ == "__main__":
    unittest.main()
