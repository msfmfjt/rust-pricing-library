"""Native fixed-market-shock gate; preserve the independently failed v1 gate."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backtesting'))
from rust_pricing_backtest.domain import BacktestError
from rust_pricing_backtest.market_shock_numerics import run_panel


class NativeMarketShockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads((ROOT/'fixtures/backtesting/market-shock-policy-v0.2.json').read_text())
        cls.base = json.loads((ROOT/'fixtures/backtesting/numerical-policy-v0.1.json').read_text())
        cls.report = run_panel(cls.policy, cls.base)

    def test_fixed_shock_gate_and_joint_statistics_pass_but_full_acceptance_stays_open(self):
        self.assertEqual(self.report['status'], 'passed')
        self.assertEqual(len(self.report['observations']), 27)
        self.assertEqual(len(self.report['checks']), 307)
        self.assertTrue(all(c['passed'] for c in self.report['checks']))
        self.assertEqual(self.report['full_acceptance'], 'not_established')
        self.assertEqual(len(self.report['covariance_observations']), 3)

    def test_original_failure_records_are_not_rewritten_or_dropped(self):
        original = self.report['original_acceptance']
        self.assertEqual(original['status'], 'failed')
        self.assertEqual(len(original['checks']), 154)
        self.assertEqual(sum(not c['passed'] for c in original['checks']), 6)
        retained = [{k:v for k,v in c.items() if k != 'origin'} for c in self.report['checks']
                    if c['origin'] == 'unchanged_v1']
        self.assertEqual(retained, [c for c in original['checks'] if c['check'] != 'hedge_grid_stability'])
        self.assertEqual(self.report['superseded_original_checks'],
                         [c for c in original['checks'] if c['check'] == 'hedge_grid_stability'])

    def test_omitted_risk_is_returned_with_a_real_price_effect(self):
        audit = self.report['independent_shock_audit']
        self.assertGreater(audit['complement_l2'], 30.)
        self.assertAlmostEqual(audit['omitted_shock_price_derivative'], audit['complement_l2'], delta=2e-6)
        self.assertEqual(audit['market_hedge_effectiveness'], 'not_established')
        self.assertTrue(all('complement_audit' in r for r in self.report['observations']))

    def test_changed_original_policy_is_rejected(self):
        altered = deepcopy(self.base)
        altered['hedge_quantity_relative_tolerance'] = 1.
        with self.assertRaisesRegex(BacktestError, 'market_gate_policy'):
            run_panel(self.policy, altered)


if __name__ == '__main__':
    unittest.main()
