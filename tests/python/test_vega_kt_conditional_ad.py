import json
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backtesting'))
from multi_step_cases import model
from rust_pricing_backtest.multi_step_full_smile import MultiStepSmileCalibrator


class ConditionalAdTests(unittest.TestCase):
    def test_sparse_middle_node_uses_conditional_AD_without_changing_raw_observations(self):
        for kind in ['asian','fixed_asian']:
            p=model(6,kind,samples=8192,seed=7)
            c=MultiStepSmileCalibrator(p);t,_=c.all_maturity_iv_tangent()
            weights=np.array([np.ones(15)/np.sqrt(15),np.array([(-1.)**i for i in range(15)])/np.sqrt(15),
                np.eye(15)[0]-np.eye(15)[-1],np.zeros(15)])
            directions=(weights@t.T).reshape(4,3,33)
            a=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256
                ).evaluate_vega_kt_multi_step_gamma_chain_audit([1,3,5],[80.,90.,100.,110.,120.],
                    variance_risk_directions=directions.tolist(),subdivisions=256)
            self.assertEqual(a['status'],'path_gamma_chain_passed')
            row=a['rows'][1];node=11
            self.assertEqual(row['native_variance_nodes'][node]['mean'],0.)
            self.assertEqual(row['native_variance_nodes'][node]['sample_variance'],0.)
            self.assertFalse(row['raw_node_validation_passed'])
            self.assertGreater(row['conditional_native_variance_nodes'][node]['mean'],0.)
            self.assertGreater(row['raw_ad_conditional_variances'][node]['mean'],0.)
            self.assertTrue(row['conditional_gamma_validation_passed'])
            self.assertTrue(row['raw_conditional_validation_passed'])
            self.assertIsNotNone(a['estimate'])
            for last in a['rows'][-1:]:
                np.testing.assert_allclose([x['mean'] for x in last['conditional_native_variance_nodes']],
                    [x['mean'] for x in last['recovered_variance_nodes']],atol=1e-10,rtol=1e-11)
            json.dumps(a,allow_nan=False)


if __name__=='__main__':unittest.main()
