import json
from pathlib import Path
import unittest
import numpy as np
import rust_pricing as rp
from test_vega_kt_three_step import model,compile,policy
from test_vega_kt_selection import remove_plan_fingerprints
from rust_pricing_backtest.three_step_full_smile import evaluate_three_step_market_gamma
ROOT=Path(__file__).resolve().parents[2]
BINS=[80.,90.,100.,110.,120.]
def audit(p,workers=1,**kwargs):return compile(p,workers).evaluate_vega_kt_three_step_initial_gamma_audit(BINS,**kwargs)

class InitialThreeGammaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((ROOT/'fixtures/local-vol/initial-three-gamma-v0.1.json').read_text())
        cls.results={c['product']:audit(c['request']) for c in cls.fixture['cases']}

    def test_all_histories_match_independent_continuation_Delta_integrals(self):
        for case in self.fixture['cases']:
            with self.subTest(product=case['product']):
                d=self.results[case['product']];r=case['reference']
                self.assertEqual(d['status'],'initial_variance_functional_passed')
                self.assertAlmostEqual(d['estimate'],r['initial_variance_sensitivity'],delta=2e-6)
                self.assertAlmostEqual(d['fine']['interpolated_price'],r['price'],delta=d['functional_tolerance'])
                self.assertAlmostEqual(sum(d['fine']['gamma_bin_variance_sensitivities']),d['estimate'],delta=1e-10)
                self.assertEqual(d['refinement_subdivisions'],[512,1024,2048,4096])
                self.assertLess(d['continuation_variance_refinement_allowance'],d['functional_tolerance'])
                self.assertLess(d['variance_tail_bound'],1e-20);self.assertFalse(d['continuation_source_renormalized'])
                self.assertEqual(d['full_market_iv_bucket_acceptance'],'not_established');json.dumps(d,allow_nan=False)

    def test_flat_limit_equals_three_transition_Black_variance_derivative(self):
        p=model();g=p['model']['local_variance_grid'];g['values']=[.04]*132;d=audit(p)
        v=.04*g['time_nodes'][-1];expected=100*np.exp(-v/8)/np.sqrt(2*np.pi*v)*g['time_nodes'][1]/2
        self.assertAlmostEqual(d['estimate'],expected,delta=d['functional_tolerance'])

    def test_workers_notional_and_normal_pricing_are_unchanged(self):
        p=model();plan=compile(p,4);before=plan.evaluate().to_json();a=self.results['call'];b=plan.evaluate_vega_kt_three_step_initial_gamma_audit(BINS)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b));self.assertEqual(before,plan.evaluate().to_json())
        p['product']['notional']=2.5;c=audit(p)
        self.assertAlmostEqual(c['estimate'],2.5*a['estimate'],delta=1e-7)
        np.testing.assert_allclose(c['fine']['gamma_bin_masses'],2.5*np.array(a['fine']['gamma_bin_masses']),atol=2e-7,rtol=0)

    def test_variance_slope_cusp_preserves_negative_initial_Gamma(self):
        p=model();g=p['model']['local_variance_grid'];x=np.array(g['log_forward_moneyness_nodes'])
        g['values'][33:66]=(.06-.02*abs(x)).tolist();g['values'][66:]=[.04]*66
        d=compile(p).evaluate_vega_kt_three_step_initial_gamma_audit([99.999,100.,100.001])
        dt2=g['time_nodes'][2]-g['time_nodes'][1];dt3=g['time_nodes'][3]-g['time_nodes'][2];sd=np.sqrt(.06*dt2+.04*dt3)
        atom=-.04*dt2*np.exp(-sd*sd/8)/(2*sd*np.sqrt(2*np.pi))
        self.assertTrue(d['recovery_passed']);self.assertLess(d['fine']['gamma_bin_masses'][2],-.008)
        self.assertAlmostEqual(d['fine']['gamma_bin_masses'][2],atom,delta=d['mass_tolerance'])

    def test_failed_refinement_or_MC_withholds_initial_estimate(self):
        coarse=audit(model('asian'),subdivisions=64,relative_tolerance=1e-10)
        self.assertFalse(coarse['recovery_passed']);self.assertIsNone(coarse['estimate'])
        sparse=audit(model('lookback',samples=2,seed=3));self.assertTrue(sparse['recovery_passed']);self.assertFalse(sparse['validation_passed']);self.assertIsNone(sparse['estimate'])

    def test_invalid_scope_and_unresolved_transition_fail_before_acceptance(self):
        plan=compile(model())
        for bounds,n,tol in [([],512,1e-8),([100.,100.],512,1e-8),([100.],63,1e-8),([100.],4097,1e-8),([100.],512,float('nan'))]:
            with self.assertRaises(rp.PricingError):plan.evaluate_vega_kt_three_step_initial_gamma_audit(bounds,subdivisions=n,relative_tolerance=tol)
        p=model();p['model']['local_variance_grid']['values'][0]+=.001
        with self.assertRaises(rp.PricingError):audit(p)
        p=model();p['model']['local_variance_grid']['floor']=1e-20;p['model']['local_variance_grid']['values'][33:66]=[1e-16]*33
        with self.assertRaises(rp.PricingError):audit(p)
        p=model();p['model']['local_variance_grid']['floor']=1e-40;p['model']['local_variance_grid']['values'][:33]=[1e-32]*33
        with self.assertRaisesRegex(rp.PricingError,'below floating-point resolution'):audit(p)

    def test_all_eleven_market_coordinates_and_component_gates(self):
        p=model('lookback',samples=65536);d=evaluate_three_step_market_gamma(p,BINS,policy=policy())
        self.assertEqual(d['status'],'all_reachable_market_gamma_passed');self.assertEqual(len(d['estimate']),11)
        T=np.array(d['calibration_tangent']);fixed=T[0]*d['initial_gamma_audit']['estimate']+T[1:34].T@d['middle_gamma_audit']['estimate']
        np.testing.assert_allclose([x['mean'] for x in d['estimate']],fixed+[x['mean'] for x in d['terminal_gamma_audit']['estimate']],atol=1e-12,rtol=0)
        np.testing.assert_array_equal([x['standard_error'] for x in d['estimate']],[x['standard_error'] for x in d['terminal_gamma_audit']['estimate']])
        d=evaluate_three_step_market_gamma(model('asian',samples=2,seed=1),BINS,policy=policy(reconstruction_relative_limit=1e-12),subdivisions=64,relative_tolerance=1e-10)
        self.assertIsNone(d['estimate']);self.assertIsNone(d['initial_gamma_audit']['estimate']);self.assertIsNone(d['terminal_gamma_audit']['estimate'])

if __name__=='__main__':unittest.main()
