"""Readouts preserve native estimates and shared counter order across contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from multi_step_cases import model
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.portfolio_market_risk import evaluate_portfolio
from test_vega_kt_selection import remove_plan_fingerprints


class JointPortfolioTests(unittest.TestCase):
    def test_readouts_leave_existing_audit_unchanged_and_are_ordered(self):
        p=model(4,'asian',samples=257);t,_=MultiStepExtensionCalibrator(p).all_maturity_iv_tangent()
        def run(workers,retain):
            return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256).evaluate_vega_kt_multi_step_gamma_chain_audit(
                [0,1,2,3],[80.,100.,120.],variance_risk_directions=t.T.reshape(10,4,33).tolist(),subdivisions=512,retain_samples=retain)
        plain=run(1,False);retained=run(1,True);parallel=run(4,True)
        self.assertEqual(remove_plan_fingerprints(retained),remove_plan_fingerprints(parallel))
        values=np.array(retained.pop('sampling_unit_values'));retained.pop('sampling_unit_values_layout')
        self.assertEqual(plain,retained)
        self.assertEqual(values.shape,(257,11))
        np.testing.assert_allclose(values[:,0].mean(),plain['native_price']['mean'],atol=1e-12)
        cov=np.cov(values[:,1:],rowvar=False,ddof=1)
        native=np.array(plain['joint_sample_covariance']).reshape(30,30)[10:20,10:20]
        np.testing.assert_allclose(cov,native,rtol=1e-10,atol=1e-10)

    def test_rqmc_covariance_uses_scramble_means_not_points(self):
        p=model(4,'asian',samples=256)
        p['risk'].pop('vega_kt',None);p['risk']['vega']=False
        p['engine']=dict(type='randomized_quasi_monte_carlo',points_per_scramble=32,scramble_count=8,
                         master_scramble_seed=7,variance_reduction=dict(antithetic=True,brownian_bridge=True))
        t,_=MultiStepExtensionCalibrator(p).all_maturity_iv_tangent()
        def run(workers):
            plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)
            audit=plan.evaluate_vega_kt_multi_step_gamma_chain_audit([0,1,2,3],[80.,100.,120.],
                variance_risk_directions=t.T.reshape(10,4,33).tolist(),subdivisions=512,retain_samples=True)
            return audit,plan.evaluate()
        a,price=run(1);parallel,_=run(4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(parallel))
        self.assertEqual(a['sampling_unit'],'scramble_mean')
        self.assertEqual(a['independent_sampling_units'],8)
        self.assertEqual(a['points_per_scramble'],32)
        self.assertEqual(a['conditional_ad_points_per_scramble'],2048)
        self.assertEqual(a['conditional_AD_sampling'],'same_scramble_nested_prefix_reference_refinement')
        self.assertEqual([r['conditional_ad_points_per_scramble'] for r in a['rows']],[2048,2048,2048,32])
        for moment in a['rows'][-1]['conditional_node_differences']:
            self.assertLess(abs(moment['mean']),1e-10)
            self.assertLess(moment['standard_error'],1e-10)
        values=np.array(a['sampling_unit_values']);self.assertEqual(values.shape,(8,11))
        expected=np.cov(values,rowvar=False,ddof=1)/8
        np.testing.assert_allclose(np.diag(expected)[1:],[m['standard_error']**2 for m in a['recovered_directions']],rtol=1e-10,atol=1e-12)
        self.assertAlmostEqual(a['native_price']['mean'],price.value,places=11)
        self.assertAlmostEqual(np.sqrt(expected[0,0]),price.standard_error,places=11)
        self.assertGreater(abs(expected[0,0]-np.var(values[:,0],ddof=1)/256),1e-8)


    def test_identical_opposite_instruments_cancel_risk_and_error(self):
        p=model(4,'call',samples=4096)
        r=evaluate_portfolio({'long':p,'short':deepcopy(p)},{'long':2.5,'short':-2.5},worker_threads=4,retain_samples=True)
        cube=np.array(r['sampling_unit_values'])
        np.testing.assert_array_equal(cube[:,0,:],cube[:,1,:])
        np.testing.assert_array_equal(r['net_buckets'],0.)
        np.testing.assert_array_equal(r['net_bucket_standard_errors'],0.)
        self.assertEqual(r['net_price'],0.)
        c=np.array(r['market_sample_covariance'])
        q=cube.shape[2]-1
        # BLAS may reduce identical covariance blocks in different orders.
        # The underlying samples and the net cancellation remain bitwise exact.
        np.testing.assert_allclose(c[:q,:q],c[:q,q:],rtol=1e-12,atol=1e-12)
        for field in ('master_seed','independent_sampling_units'):
            changed=deepcopy(p);changed['engine'][field]+=1
            with self.assertRaisesRegex(ValueError,'portfolio_stream_mismatch'):
                evaluate_portfolio({'a':p,'b':changed},{'a':1,'b':-1})

    def test_mixed_rqmc_resolutions_pair_by_scramble(self):
        from market_acceptance_cases import cases, requests
        p=requests(next(c for c in cases() if c['id']=='short_3_left_skew'))['call']
        p['engine']=dict(type='randomized_quasi_monte_carlo',points_per_scramble=64,scramble_count=16,
                         master_scramble_seed=7,variance_reduction=dict(antithetic=True,brownian_bridge=False))
        other=deepcopy(p);other['engine']['points_per_scramble']=128
        other['engine']['variance_reduction']['brownian_bridge']=True
        r=evaluate_portfolio({'a':p,'b':other},{'a':1.,'b':-1.},worker_threads=4,retain_samples=True,subdivisions={'a':512,'b':1024})
        cube=np.array(r['sampling_unit_values'])
        self.assertEqual(cube.shape,(16,2,7))
        direct=np.cov(cube[:,0,:]-cube[:,1,:],rowvar=False)/16
        np.testing.assert_allclose(r['net_price_and_market_mean_covariance'],direct,atol=1e-12)
        self.assertEqual(r['audits']['a']['points_per_scramble'],64)
        self.assertEqual(r['audits']['b']['points_per_scramble'],128)
        self.assertEqual(r['audits']['a']['subdivisions'],512)
        self.assertEqual(r['audits']['b']['subdivisions'],1024)
        self.assertGreater(np.max(np.diag(direct)),0.)
        for key in ('master_scramble_seed','scramble_count'):
            changed=deepcopy(other);changed['engine'][key]+=1
            with self.assertRaisesRegex(ValueError,'portfolio_stream_mismatch'):
                evaluate_portfolio({'a':p,'b':changed},{'a':1,'b':-1})
        changed=deepcopy(other);changed['engine']['variance_reduction']['antithetic']=False
        with self.assertRaisesRegex(ValueError,'portfolio_stream_mismatch'):
            evaluate_portfolio({'a':p,'b':changed},{'a':1,'b':-1})

    def test_failure_does_not_publish_partial_portfolio(self):
        p=model(4,'asian',samples=2,seed=1)
        with self.assertRaisesRegex(ValueError,'portfolio_gamma_failure'):
            evaluate_portfolio({'a':p},{'a':1.},worker_threads=4)

if __name__=='__main__':unittest.main()
