"""Daily fixed-shock objectives, physical constraints, audits and safe replay."""
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import importlib.util
import json
import unittest

import numpy as np

from rust_pricing_backtest import (BacktestConfig, BacktestError, BacktestResult, Contract,
    Position, Quote, Risk, RiskBasis, Snapshot, compare_strategies, run_backtest)
from rust_pricing_backtest.ensemble import ScenarioResults, aggregate_scenarios
from rust_pricing_backtest.serialization import input_json, load_input
from rust_pricing_backtest.validation import validate_document


START = date(2026, 1, 5)
EXPIRY = START+timedelta(days=4)
HAS_SCIPY = importlib.util.find_spec('scipy') is not None
HAS_VALIDATOR = importlib.util.find_spec('jsonschema') is not None


class AtomicRiskProvider:
    identity = 'hand_computed_atomic_risk_v1'
    validation = {'status': 'synthetic_fixture'}

    def evaluate(self, s, c):
        target = c.contract_id == 'target'
        n = len(s.basis.log_moneyness)
        risk = np.zeros(s.basis.size)
        # Off-node market hat, exact reporting node. No native risk is reused
        # to construct this hand-computed optimization fixture.
        risk[-n:] = (10. if target else 5.)*np.array([
            np.interp(0. if target else .5, s.basis.log_moneyness, e) for e in np.eye(n)])
        return Risk(5. if target else 3., .5 if target else .2,
                    tuple(risk), s.basis.basis_id, provenance=self.identity)


def fixture(count=5, **config):
    contracts = [Contract('target', EXPIRY, 100., lot=.3), Contract('hedge', EXPIRY, 105., lot=.3)]
    history = []
    for i in range(5):
        day = START+timedelta(days=i)
        at = datetime.combine(day, datetime.min.time(), timezone.utc)
        basis = RiskBasis(day, (day+timedelta(days=7), day+timedelta(days=21)), tuple(np.linspace(-1., 1., count)))
        history.append(Snapshot(f'day-{i}', at, Quote(100., 100., at, at),
            {'target': Quote(5., 5., at, at), 'hedge': Quote(3., 3., at, at)}, basis, at, 'hand_computed_fixture'))
    return dict(config=BacktestConfig(**({'ridge': 0., 'market_shock_log_moneyness': (-1., 0., 1.)}|config)),
                history=history, contracts=contracts, portfolio=[Position('target', -1., 5.)], provider=AtomicRiskProvider())


class MarketShockReplayTests(unittest.TestCase):
    def test_market_objective_changes_only_vega_kt_and_is_grid_invariant(self):
        coarse, fine = [run_backtest(**fixture(n)) for n in (5, 9)]
        old = run_backtest(**fixture(market_shock_log_moneyness=None))
        self.assertAlmostEqual(coarse.decisions[0]['solver']['quantities'][0], 2.)
        self.assertEqual(coarse.orders[0]['desired'], fine.orders[0]['desired'])
        self.assertAlmostEqual(coarse.orders[0]['desired']['hedge'], 1.8)
        self.assertEqual(old.orders[0]['desired']['hedge'], 0.)
        for strategy in ('unhedged', 'delta', 'scalar_vega'):
            a = run_backtest(**fixture(strategy=strategy))
            b = run_backtest(**fixture(strategy=strategy, market_shock_log_moneyness=None))
            self.assertEqual(a.ledger, b.ledger)
            self.assertEqual(a.orders, b.orders)
        self.assertEqual(old.manifest['schema_version'], 10)
        self.assertNotIn('market_shock_log_moneyness', old.manifest['config'])
        self.assertNotIn('market_shocks', old.risks[0])

    def test_rounded_proposed_and_filled_holdings_keep_unmodeled_risk(self):
        r = run_backtest(**fixture())
        d = r.decisions[0]['market_shocks']
        self.assertEqual(d['objective'], 'fixed_market_iv')
        self.assertNotEqual(d['continuous'], d['proposed'])
        self.assertEqual(d['proposed_reporting_buckets'], [0.]*7+[-10., 9., 0.])
        np.testing.assert_allclose(d['proposed']['market_buckets'], [0., 0., 0., 0., -5.5, 4.5])
        self.assertEqual(d['proposed'], r.risks[1]['market_shocks']['held'])
        self.assertGreater(d['proposed']['complement']['complement_l2'], 0.)
        self.assertEqual(r.risks[-1]['market_shocks']['held']['complement']['reporting_l2'], 0.)
        self.assertEqual(r.manifest['market_shock_policy']['hedge_effectiveness'], 'not_established')

    def test_ineligible_held_contract_stays_in_audit_and_optimization(self):
        args = fixture()
        s = args['history'][1]
        args['history'][1] = replace(s, option_quotes={**s.option_quotes, 'hedge': Quote(0., 6., s.at, s.at)})
        r = run_backtest(**args)
        self.assertEqual(r.decisions[1]['eligible'], [])
        self.assertEqual(r.decisions[1]['candidates'], ['hedge'])
        self.assertAlmostEqual(r.orders[1]['desired']['hedge'], 1.8)
        self.assertEqual(r.decisions[1]['market_shocks']['proposed'], r.risks[1]['market_shocks']['held'])

    def test_candidate_eligibility_uses_configured_market_risk(self):
        class OrthogonalProvider(AtomicRiskProvider):
            def evaluate(self, s, c):
                r = super().evaluate(s, c)
                return replace(r, buckets=(0.,)*5+(-.5, 1., -.5, 0., 0.)) if c.contract_id == 'hedge' else r
        args = fixture()
        args['provider'] = OrthogonalProvider()
        r = run_backtest(**args)
        self.assertEqual(r.decisions[0]['eligible'], [])
        self.assertEqual(r.orders[0]['desired'], {})
        self.assertGreater(r.risks[0]['market_shocks']['held']['complement']['complement_l2'], 0.)

    def test_future_grid_failure_is_atomic_and_repair_resumes_exactly(self):
        args = fixture()
        whole = run_backtest(**args)
        broken = list(args['history'])
        s = broken[2]
        broken[2] = replace(s, basis=replace(s.basis, log_moneyness=(-1., -.5, .1, .5, 1.)))
        r = run_backtest(**(args|{'history': broken}), on_error='return')
        self.assertEqual(r.status, 'failed')
        self.assertEqual(r.failure['code'], 'market_shock_not_nested')
        self.assertEqual(r.checkpoint['state']['next_index'], 2)
        partial = run_backtest(**args, stop_after=2)
        self.assertEqual(r.checkpoint, partial.checkpoint)
        self.assertEqual(r.orders, partial.orders)
        self.assertEqual(run_backtest(**args, checkpoint=r.checkpoint).to_json(), whole.to_json())

    def test_wire_versions_roundtrip_and_changed_nodes_block_resume(self):
        args = fixture()
        r = run_backtest(**args)
        partial = run_backtest(**args, stop_after=2)
        self.assertEqual(r.manifest['schema_version'], 11)
        self.assertEqual(r.manifest['input_schema_version'], 7)
        self.assertEqual(partial.checkpoint['schema_version'], 9)
        self.assertEqual(BacktestResult.from_json(r.to_json()).to_json(), r.to_json())
        loaded = load_input(input_json(r))
        self.assertEqual(loaded['config'].market_shock_log_moneyness, (-1., 0., 1.))
        self.assertEqual(run_backtest(**loaded, provider=args['provider'], checkpoint=partial.checkpoint).to_json(), r.to_json())
        for cfg in [replace(args['config'], market_shock_log_moneyness=(-1., -.5, 0., .5, 1.)),
                    replace(args['config'], market_shock_log_moneyness=None)]:
            with self.assertRaisesRegex(BacktestError, 'checkpoint_mismatch'):
                run_backtest(**(args|{'config': cfg}), checkpoint=partial.checkpoint)
        corrupt = deepcopy(partial.checkpoint)
        corrupt['state']['risks'][0]['market_shocks']['held']['market_buckets'][0] += 1.
        with self.assertRaisesRegex(BacktestError, 'checkpoint_mismatch'):
            run_backtest(**args, checkpoint=corrupt)

    def test_invalid_configuration_and_versioned_input_rejected(self):
        for nodes in [[], [0.], [0., 0.], [1., 0.], [-1., float('inf')], [-1., True], '(-1,1)']:
            with self.subTest(nodes=nodes), self.assertRaises(BacktestError):
                BacktestConfig(market_shock_log_moneyness=nodes)
        doc = json.loads(input_json(run_backtest(**fixture())))
        for change in ('downgrade', 'missing', 'null'):
            altered = deepcopy(doc)
            if change == 'downgrade':
                altered['schema_version'] = 6
            elif change == 'missing':
                altered['config'].pop('market_shock_log_moneyness')
            else:
                altered['config']['market_shock_log_moneyness'] = None
            with self.subTest(change=change), self.assertRaisesRegex(BacktestError, 'schema_fields'):
                load_input(json.dumps(altered))

    @unittest.skipUnless(HAS_SCIPY, 'install constraints extra')
    def test_cash_gross_solver_and_rounding_use_same_market_objective(self):
        args = fixture(constraint_mode='decision_cash_gross', gross_notional_limit=250., solver_tolerance=1e-8)
        args['contracts'] = [replace(c, lot=.1) for c in args['contracts']]
        r = run_backtest(**args)
        d = r.decisions[0]
        # h*=1.25 from gross=150+80*h<=250; lot rounding leaves h=1.2.
        self.assertAlmostEqual(d['solver']['quantities'][0], 1.25, places=6)
        self.assertAlmostEqual(r.orders[0]['desired']['hedge'], 1.2)
        self.assertAlmostEqual(d['budget']['rounded']['gross_notional'], 246.)
        self.assertEqual(d['market_shocks']['proposed'], r.risks[1]['market_shocks']['held'])
        partial = run_backtest(**args, stop_after=2)
        self.assertEqual(run_backtest(**args, checkpoint=partial.checkpoint).to_json(), r.to_json())
        if HAS_VALIDATOR:
            validate_document(json.loads(r.to_json()))

    @unittest.skipUnless(HAS_VALIDATOR, 'install validation extra')
    def test_schemas_require_audit_and_policy_and_support_ensembles(self):
        args = fixture()
        results = compare_strategies(**args)
        for r in results.values():
            for document in [json.loads(r.to_json()), json.loads(input_json(r)), json.loads(json.dumps(r.checkpoint))]:
                self.assertEqual(validate_document(document)['status'], 'valid')
        partial = run_backtest(**args, stop_after=2)
        validate_document(json.loads(partial.to_json()))
        broken = fixture(market_shock_log_moneyness=(-1., .1, 1.))
        failed = run_backtest(**broken, on_error='return')
        validate_document(json.loads(failed.to_json()))
        r = results['vega_kt']
        for field in ('market_shocks',):
            altered = json.loads(r.to_json())
            altered['risks'][0].pop(field)
            with self.assertRaises(BacktestError):
                validate_document(altered)
        scenario = ScenarioResults('fixed-shocks', 'one', 'synthetic_fixture', results)
        ensemble = aggregate_scenarios([scenario])
        self.assertEqual(ensemble.manifest['schema_version'], 3)
        validate_document(json.loads(ensemble.to_json()))
        changed = deepcopy(results)
        changed['delta'].manifest['config']['market_shock_log_moneyness'] = [-1., 1.]
        with self.assertRaisesRegex(BacktestError, 'comparison_conditions_mismatch'):
            aggregate_scenarios([replace(scenario, results=changed)])


if __name__ == '__main__':
    unittest.main()
