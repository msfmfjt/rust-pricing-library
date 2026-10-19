import json,sys
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backtesting'))
from multi_step_cases import model
from rust_pricing_backtest.multi_step_full_smile import MultiStepSmileCalibrator,evaluate_multi_step_market_gamma
from test_vega_kt_selection import remove_plan_fingerprints

def audit(p,**kwargs):
    return evaluate_multi_step_market_gamma(p,[80.,90.,100.,110.,120.],subdivisions=512,gamma_relative_standard_error_limit=.5,**kwargs)
def plan(p,workers=1):return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)

class PathGammaChainTests(unittest.TestCase):
    def test_four_six_intervals_and_complete_market_coordinate_order(self):
        for n in [4,6]:
            a=audit(model(n,samples=1024));self.assertEqual(a['status'],'path_gamma_chain_passed')
            self.assertEqual(len(a['estimate']),5*n//2);d=a['gamma_audit']
            self.assertEqual([r['variance_row'] for r in d['rows']],list(range(1,n,2)))
            for row in d['rows']:
                self.assertAlmostEqual(sum(x['mean'] for x in row['gamma_bin_variance_sensitivities']),sum(x['mean'] for x in row['recovered_variance_nodes']),delta=1e-10)
            self.assertEqual(a['full_market_iv_bucket_acceptance'],'not_established');json.dumps(a,allow_nan=False)
    def test_history_payoffs_notional_workers_and_unchanged_native_price(self):
        for kind in ['asian','fixed_asian','lookback']:
            p=model(4,kind,samples=1024);p['risk']=dict(delta=False,vega=False,smile_dynamics=dict(type='sticky_log_moneyness'));before=plan(p).evaluate().to_json();a=audit(p)
            self.assertEqual(a['status'],'path_gamma_chain_passed');self.assertEqual(before,plan(p).evaluate().to_json())
            b=audit(p,worker_threads=4);self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b))
        p=model(4,'asian',samples=1024);a=audit(p);p['product']['notional']=2.5;b=audit(p)
        for x,y in zip(a['estimate'],b['estimate']):
            self.assertAlmostEqual(y['mean'],2.5*x['mean'],delta=1e-8)
            self.assertAlmostEqual(y['sample_variance'],6.25*x['sample_variance'],delta=1e-6)
    def test_dependent_readout_uses_joint_same_path_statistics(self):
        p=model(4,samples=1024);c=MultiStepSmileCalibrator(p);t,_=c.all_maturity_iv_tangent();d=t.T.reshape(10,2,33)
        directions=np.array([d[2],d[7],d[2]+d[7],np.zeros_like(d[2])])
        a=plan(p).evaluate_vega_kt_multi_step_gamma_chain_audit(c.calibrated_rows,[90.,100.,110.],variance_risk_directions=directions.tolist(),subdivisions=512,gamma_relative_standard_error_limit=.5)
        self.assertIsNotNone(a['estimate']);r=a['recovered_directions'];self.assertAlmostEqual(r[2]['mean'],r[0]['mean']+r[1]['mean'],delta=1e-10)
        self.assertEqual(r[3]['mean'],0.);self.assertEqual(r[3]['standard_error'],0.)
        self.assertNotAlmostEqual(r[2]['standard_error'],r[0]['standard_error']+r[1]['standard_error'],places=8)
    def test_unpaired_bridge_and_failure_gates(self):
        p=model(4,samples=1024);p['engine']['variance_reduction'].update(antithetic=False,brownian_bridge=True)
        a=audit(p);self.assertEqual(a['status'],'path_gamma_chain_passed');self.assertEqual(a['gamma_audit']['sampling_unit'],'path')
        a=evaluate_multi_step_market_gamma(model(4,'lookback',samples=1024),[80.,100.,120.],subdivisions=16,relative_tolerance=1e-10)
        self.assertEqual(a['status'],'recovery_failed');self.assertIsNone(a['estimate'])
        a=audit(model(4,'asian',samples=2,seed=1));self.assertIsNone(a['estimate'])
    def test_precision_shapes_and_work_budget_are_explicit(self):
        p=model(4,samples=1024);c=MultiStepSmileCalibrator(p);t,_=c.all_maturity_iv_tangent();d=t.T.reshape(10,2,33).tolist()
        for rows,directions in [([1,1],d),([4,5],d),([1,3],d[:1][0]),([1,3],np.full((1,2,33),np.nan).tolist())]:
            with self.assertRaises((rp.PricingError,TypeError)):plan(p).evaluate_vega_kt_multi_step_gamma_chain_audit(rows,[100.],variance_risk_directions=directions)
        q=deepcopy(p);q['engine']['independent_sampling_units']=262144
        with self.assertRaises(rp.PricingError):plan(q).evaluate_vega_kt_multi_step_gamma_chain_audit([1,3],[100.],variance_risk_directions=d,subdivisions=1024)
        q=deepcopy(p);q['model']['local_variance_grid']['floor']=1e-40;q['model']['local_variance_grid']['values'][33:66]=[1e-32]*33
        with self.assertRaises(rp.PricingError):plan(q).evaluate_vega_kt_multi_step_gamma_chain_audit([1,3],[100.],variance_risk_directions=d)

if __name__=='__main__':unittest.main()
