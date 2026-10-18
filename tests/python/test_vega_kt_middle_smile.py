from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
import rust_pricing as rp
from test_vega_kt_three_step import model as three_step_model,compile,directions,policy
from test_vega_kt_selection import remove_plan_fingerprints
from rust_pricing_backtest.middle_smile import evaluate_middle_terminal_market_gamma

ROOT=Path(__file__).resolve().parents[2]
BINS=[80.,90.,100.,110.,120.]
def model(kind="call",samples=65536,seed=7): return three_step_model(kind,samples=samples,seed=seed)
def middle(p,workers=1,**kwargs): return compile(p,workers).evaluate_vega_kt_middle_smile_gamma_audit(BINS,**kwargs)
def readout(p,D,R,settings=None):
    nodes=p['model']['local_variance_grid']['log_forward_moneyness_nodes']
    return compile(p).evaluate_vega_kt_three_step_terminal_readout_audit((100*np.exp(nodes)).tolist(),
        policy=rp.VegaKtGammaRecoveryPolicy(**(settings or policy())),variance_conservation_directions=D.tolist(),variance_risk_directions=R.tolist())

class MiddleGammaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((ROOT/'fixtures/local-vol/middle-smile-gamma-v0.1.json').read_text())
        cls.results={c['product']:middle(c['request']) for c in cls.fixture['cases']}

    def test_conditional_Gamma_matches_independent_integration_by_parts_for_all_payoffs(self):
        for case in self.fixture['cases']:
            with self.subTest(product=case['product']):
                d=self.results[case['product']];f=d['fine_history'];ref=case['reference']
                self.assertTrue(d['recovery_passed'])
                # Small-sample Asian tail cells may have no payoff hits even
                # after source stratification. Keep that explicit MC failure;
                # the full numerical panel requires 524288 units for all cases.
                if case['product']=='asian':
                    self.assertEqual(d['status'],'validation_failed');self.assertIsNone(d['estimate'])
                    self.assertEqual(d['native_middle_variance_sensitivities'][10]['mean'],0.)
                    self.assertGreater(f['variance_node_sensitivities'][10],d['functional_tolerance'])
                else:
                    self.assertEqual(d['status'],'middle_variance_functional_passed');self.assertIsNotNone(d['estimate'])
                np.testing.assert_allclose(f['variance_node_sensitivities'],ref['middle_variance_sensitivities'],atol=2e-6,rtol=0)
                self.assertAlmostEqual(f['recovered_price'],ref['price'],delta=d['functional_tolerance'])
                self.assertAlmostEqual(sum(f['gamma_bin_variance_sensitivities']),sum(f['variance_node_sensitivities']),delta=1e-10)
                self.assertLess(f['inner_bin_mass_refinement'],d['mass_tolerance'])
                self.assertLess(f['variance_tail_bound'],1e-20)
                self.assertEqual(d['sampling_unit'],'antithetic_stratified_sweep')
                self.assertGreater(d['validation_history_strata'],30)
                self.assertEqual(d['full_market_iv_bucket_acceptance'],'not_established')
                json.dumps(d,allow_nan=False)

    def test_worker_replay_and_notional_preserve_measure(self):
        p=model('fixed_asian');a=self.results['fixed_asian'];b=middle(p,4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b))
        p['product']['notional']=2.5;c=middle(p)
        np.testing.assert_allclose(c['estimate'],2.5*np.array(a['estimate']),atol=2e-8,rtol=0)
        np.testing.assert_allclose(c['fine_history']['history_averaged_gamma_bin_masses'],
            2.5*np.array(a['fine_history']['history_averaged_gamma_bin_masses']),atol=2e-7,rtol=0)
        for x,y in zip(a['native_middle_variance_sensitivities'],c['native_middle_variance_sensitivities']):
            self.assertAlmostEqual(y['sample_variance'],6.25*x['sample_variance'],delta=1e-7)

    def test_variance_slope_cusp_retains_negative_gamma_atom(self):
        p=model();g=p['model']['local_variance_grid'];nodes=np.array(g['log_forward_moneyness_nodes'])
        g['values'][66:]=(.06-.02*abs(nodes)).tolist()*2
        d=compile(p).evaluate_vega_kt_middle_smile_gamma_audit([99.999,100.,100.001])
        dt=g['time_nodes'][3]-g['time_nodes'][2];sd=np.sqrt(.06*dt)
        atom=-.04*dt*np.exp(-sd*sd/8)/(np.sqrt(2*np.pi)*2*sd)
        self.assertTrue(d['recovery_passed'])
        self.assertLess(d['fine_history']['history_averaged_gamma_bin_masses'][2],-.01)
        self.assertAlmostEqual(d['fine_history']['history_averaged_gamma_bin_masses'][2],atom,delta=d['mass_tolerance'])

    def test_numerical_and_statistical_failures_withhold_estimates(self):
        coarse=middle(model('asian'),subdivisions=64,relative_tolerance=1e-10)
        self.assertFalse(coarse['recovery_passed']);self.assertIsNone(coarse['estimate'])
        sparse=middle(model('lookback',samples=2,seed=1))
        self.assertTrue(sparse['recovery_passed']);self.assertFalse(sparse['validation_passed']);self.assertIsNone(sparse['estimate'])

    def test_scope_and_precision_fail_explicitly(self):
        plan=compile(model())
        for bounds,kwargs in [([],{}),([100.,100.],{}),([100.],dict(subdivisions=63)),([100.],dict(relative_tolerance=float('nan')))]:
            with self.assertRaises(rp.PricingError):plan.evaluate_vega_kt_middle_smile_gamma_audit(bounds,**kwargs)
        p=model();p['model']['local_variance_grid']['values'][0]+=.001
        with self.assertRaises(rp.PricingError):middle(p)

    def test_terminal_dependent_readouts_preserve_pathwise_variance_and_zero(self):
        p=model();D=directions(p);R=np.vstack([D[0]+D[1],-2*D[0],np.zeros(33)])
        d=readout(p,D,R);self.assertIsNotNone(d['estimate']);self.assertEqual(d['variance_risk_directions'],R.tolist())
        rows=d['validation'];self.assertEqual(d['estimate'][-1]['mean'],0.);self.assertEqual(d['estimate'][-1]['sample_variance'],0.)
        for actual,recovered in zip(rows['variance_direction_risks'],rows['recovered_direction_risks']):
            self.assertAlmostEqual(actual['mean'],recovered['mean'],delta=1e-8)
            self.assertAlmostEqual(actual['sample_variance'],recovered['sample_variance'],delta=1e-6)
        for invalid in [np.ones((1,32)),np.full((1,33),float('nan')),np.eye(33)[[16]],np.empty((0,33))]:
            with self.assertRaises(rp.PricingError):readout(p,D,invalid)

    def test_joint_market_wrapper_keeps_independent_failure_gates(self):
        p=model('fixed_asian');d=evaluate_middle_terminal_market_gamma(p,BINS,policy=policy())
        self.assertEqual(d['status'],'middle_terminal_market_gamma_passed');self.assertEqual(len(d['estimate']),10)
        T=np.array(d['calibration_tangent']);fixed=T[:33].T@d['middle_gamma_audit']['estimate']
        np.testing.assert_allclose([v['mean'] for v in d['estimate']],fixed+[v['mean'] for v in d['terminal_gamma_audit']['estimate']],atol=1e-12,rtol=0)
        np.testing.assert_array_equal([v['standard_error'] for v in d['estimate']],[v['standard_error'] for v in d['terminal_gamma_audit']['estimate']])
        d=evaluate_middle_terminal_market_gamma(p,BINS,policy=policy(reconstruction_relative_limit=1e-12),subdivisions=64,relative_tolerance=1e-10)
        self.assertIsNone(d['estimate']);self.assertIsNone(d['middle_gamma_audit']['estimate']);self.assertIsNone(d['terminal_gamma_audit']['estimate'])

if __name__=='__main__':unittest.main()
