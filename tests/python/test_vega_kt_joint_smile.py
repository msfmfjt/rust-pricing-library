"""Joint calibrated risk preserves cross-maturity covariance and failed gates."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backtesting'));sys.path.insert(0,str(ROOT/'scripts'))
from test_vega_kt_nonflat_gamma import model, compile, directions
from test_vega_kt_selection import policy, remove_plan_fingerprints
from check_discrete_smile_calibration import base_request
from check_path_gamma_recovery import request
from rust_pricing_backtest.joint_smile import evaluate_joint_market_gamma
from rust_pricing_backtest.domain import BacktestError

BOUNDS=[60.,80.,90.,100.,110.,120.,140.]

def readouts(p):
    b=directions(p);a=np.r_[.4,-.3*b[0]+.1*b[1]];c=np.r_[0.,b[0]]
    return np.array([a,c,a+c,-2*a])

def evaluate(p, *, workers=1, settings=None, basis=None, risk=None, **kwargs):
    x=p['model']['local_variance_grid']['log_forward_moneyness_nodes']
    return compile(p,workers).evaluate_vega_kt_joint_gamma_audit((100*np.exp(x)).tolist(),BOUNDS,
        policy=rp.VegaKtGammaRecoveryPolicy(**(settings or policy())),
        variance_conservation_directions=(directions(p) if basis is None else basis).tolist(),
        risk_directions=(readouts(p) if risk is None else risk).tolist(),**kwargs)


class JointGammaTests(unittest.TestCase):
    def test_dependent_readouts_full_covariance_and_cross_maturity_terms(self):
        d=evaluate(model());self.assertEqual(d['status'],'joint_direction_policy_passed');s=d['estimate']
        C=np.array(s['joint_sample_covariance']).reshape(12,12);m=s['native_directions']+s['recovered_directions']+s['differences']
        np.testing.assert_allclose(np.diag(C),[v['sample_variance'] for v in m],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(C,C.T);self.assertGreater(np.linalg.eigvalsh(C)[0],-1e-9)
        for offset in [0,4,8]:
            self.assertAlmostEqual(m[offset+2]['mean'],m[offset]['mean']+m[offset+1]['mean'],places=10)
            self.assertAlmostEqual(C[offset+2,offset+2],C[offset,offset]+C[offset+1,offset+1]+2*C[offset,offset+1],places=9)
            self.assertAlmostEqual(m[offset+3]['mean'],-2*m[offset]['mean'],places=10)
            self.assertAlmostEqual(C[offset+3,offset+3],4*C[offset,offset],places=9)
        self.assertGreater(abs(C[4,5]),1.)
        T=np.column_stack([-np.eye(4),np.eye(4)])
        np.testing.assert_allclose(T@C[:8,:8]@T.T,C[8:,8:],atol=1e-9)
        self.assertEqual(s['sampling_unit'],'antithetic_pair');self.assertEqual(s['independent_sampling_units'],4096)
        for stat in m:self.assertAlmostEqual(stat['standard_error']**2,stat['sample_variance']/4096,places=12)
        json.dumps(d,allow_nan=False)

    def test_joint_native_readout_matches_same_path_two_row_price_bumps(self):
        p=model('lookback');d=evaluate(p)
        for w,m in zip(readouts(p),d['estimate']['native_directions']):
            values=[];h=1e-8
            for sign in [-1.,1.]:
                q=deepcopy(p);v=np.array(q['model']['local_variance_grid']['values']);v[:9]+=sign*h*w[0];v[9:18]+=sign*h*w[1:]
                q['model']['local_variance_grid']['values']=v.tolist();values.append(compile(q).evaluate().value)
            self.assertAlmostEqual((values[1]-values[0])/(2*h),m['mean'],delta=2e-6)

    def test_component_readout_and_worker_price_replay(self):
        p=model('fixed_asian');plan=compile(p);before=plan.evaluate().to_json();a=evaluate(p);b=evaluate(p,workers=4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b));self.assertEqual(before,plan.evaluate().to_json())
        second=np.array([x['mean'] for x in a['second']['validation']['recovered_direction_risks']])
        expected=.4*a['initial']['estimate']-.3*second[0]+.1*second[1]
        self.assertAlmostEqual(a['estimate']['recovered_directions'][0]['mean'],expected,places=10)
        self.assertEqual(a['market_calibration'],'not_verified_by_native')

    def test_initial_or_second_failure_withholds_joint_estimate(self):
        for kwargs,status in [(dict(subdivisions=64),'initial_recovery_failed'),
            (dict(settings=policy(reconstruction_relative_limit=1e-12)),'second_recovery_failed')]:
            d=evaluate(model(),**kwargs);self.assertEqual(d['status'],status);self.assertIsNone(d['estimate']);self.assertIsNone(d['joint_validation'])
        d=evaluate(model('asian',samples=2,seed=4));self.assertIsNone(d['estimate']);self.assertEqual(d['status'],'initial_recovery_failed')

    def test_invalid_readouts_and_dependent_constraint_basis_rejected(self):
        p=model()
        for r in [np.ones((0,10)),np.ones((17,10)),np.ones((1,9)),np.zeros((1,10)),np.full((1,10),np.inf)]:
            with self.assertRaises(rp.PricingError):evaluate(p,risk=r)
        with self.assertRaises(rp.PricingError):evaluate(p,basis=np.ones((2,9)))
        # Reject covariance allocation before any large simulation is started.
        with self.assertRaisesRegex(rp.PricingError,'storage budget'):evaluate(model(samples=10000000),risk=np.ones((16,10)))

    def test_unconserved_readout_failure_retains_raw_joint_statistics(self):
        d=evaluate(model('asian',samples=32768),settings=policy(candidate_regularizations=[.01]),
            basis=np.zeros((0,9)),risk=np.r_[0.,np.eye(9)[4]][None,:])
        self.assertIsNotNone(d['initial']['estimate']);self.assertIsNotNone(d['second']['estimate'])
        self.assertEqual(d['status'],'joint_validation_failed');self.assertIsNone(d['estimate'])
        s=d['joint_validation'];self.assertFalse(s['passed']);self.assertGreater(abs(s['differences'][0]['mean']),6*s['differences'][0]['standard_error'])
        self.assertEqual(d['second']['selected_regularization'],.01)

    def test_unpaired_brownian_bridge_and_notional_scaling(self):
        p=model('call',samples=8192);p['engine']['variance_reduction'].update(antithetic=False,brownian_bridge=True)
        a=evaluate(p);self.assertEqual(a['estimate']['sampling_unit'],'path')
        p['product']['notional']=2.5;b=evaluate(p)
        self.assertAlmostEqual(b['estimate']['recovered_directions'][0]['mean'],2.5*a['estimate']['recovered_directions'][0]['mean'],delta=1e-7)
        np.testing.assert_allclose(b['estimate']['joint_sample_covariance'],6.25*np.array(a['estimate']['joint_sample_covariance']),rtol=1e-7,atol=1e-7)

    def test_companion_validates_quotes_compiles_same_payload_and_keeps_scope(self):
        config=json.loads((ROOT/'fixtures/backtesting/joint-smile-gamma-policy-v0.1.json').read_text());config['gamma_samples']=8192
        p=base_request(config,'nonflat');p['product']=request('call',samples=16)['product'];before=deepcopy(p)
        d=evaluate_joint_market_gamma(p,BOUNDS,policy=policy())
        self.assertEqual(p,before);self.assertEqual(len(d['coordinate_order']),6)
        self.assertEqual(d['coordinate_order'][0],'first_maturity_parallel_iv')
        self.assertEqual(d['status'],'joint_direction_policy_passed')
        self.assertEqual(len(d['audit']['variance_conservation_directions']),5)
        self.assertEqual(d['full_market_iv_bucket_acceptance'],'not_established')
        # The compensated vanilla first-IV risk vanishes in population, whereas
        # its frozen-second initial component is nonzero.
        self.assertGreater(d['audit']['initial']['estimate'],1.)
        stat=d['estimate']['recovered_directions'][0]
        self.assertLess(abs(stat['mean']),6*stat['standard_error']+.002)
        p['model']['reporting_iv_basis']['implied_volatilities'][-1]+=.001
        with self.assertRaisesRegex(BacktestError,'base terminal quotes'):evaluate_joint_market_gamma(p,BOUNDS,policy=policy())

if __name__=='__main__':unittest.main()
