"""Joint atom-chain contracts; full-market IV acceptance is deliberately separate."""
import copy
import json
import math
import sys
from pathlib import Path
import unittest
import numpy as np
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from check_path_gamma_recovery import request
from test_vega_kt_selection import policy, remove_plan_fingerprints


def compile_plan(p, workers=1):
    return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)


def evaluate(p, *, config=None, workers=1, strikes=None):
    return compile_plan(p,workers).evaluate_vega_kt_atom_chain_diagnostic(
        strikes or [70.,85.,100.,115.,135.],policy=rp.VegaKtGammaRecoveryPolicy(**(config or policy())))


class AtomChainTests(unittest.TestCase):
    def test_joint_covariance_total_price_and_hedge_sign(self):
        p=request('asian',samples=4096,nodes=9);p['risk']['vega_kt']['full_bucket_covariance']=True
        d=evaluate(p);e=d['estimate'];self.assertIsNotNone(e);b=len(e['raw_buckets'])
        self.assertEqual(d['iv_bucket_acceptance'],'not_established')
        self.assertTrue(d['diagnostic_only']);self.assertEqual(e['independent_sampling_units'],4096)
        cov=np.array(e['full_bucket_sample_covariance']).reshape(b,b)
        np.testing.assert_array_equal(cov,cov.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-10)
        np.testing.assert_allclose(cov.diagonal(),[m['sample_variance'] for m in e['raw_buckets']],rtol=1e-12,atol=1e-13)
        self.assertAlmostEqual(cov.sum(),e['total_iv_vega']['sample_variance'],places=9)
        self.assertGreater(abs(cov.sum()-np.trace(cov)),1.)
        self.assertAlmostEqual(e['total_iv_vega']['standard_error']**2,cov.sum()/4096,places=12)
        self.assertAlmostEqual(sum(m['mean'] for m in e['raw_buckets']),e['total_iv_vega']['mean'],places=12)
        price_variance=e['native_price']['sample_variance']
        for i,c in enumerate(e['bucket_price_sample_covariance']):
            self.assertLessEqual(c*c,cov[i,i]*price_variance+1e-8)
        for i,(v,h) in enumerate(zip(e['raw_buckets'],e['hedge_quantities'])):
            row,col=divmod(i,9);t=d['reporting_maturities'][row];x=d['reporting_log_nodes'][col]
            sd=.2*math.sqrt(t);black=100*math.exp(-.5*(-x/sd+sd/2)**2)/math.sqrt(2*math.pi)*math.sqrt(t)
            self.assertAlmostEqual(h['mean']*black,-v['mean'],places=10)
            self.assertAlmostEqual(h['sample_variance']*black**2,v['sample_variance'],places=9)
        json.dumps(d,allow_nan=False)

    def test_hedge_uses_actual_reporting_node_vega_within_calibration_tolerance(self):
        p=request('asian',samples=4096,nodes=9)
        p['model']['reporting_iv_basis']['implied_volatilities'][3]*=1+5e-11
        d=evaluate(p);e=d['estimate'];i=3;t=d['reporting_maturities'][0];x=d['reporting_log_nodes'][i]
        sd=d['reporting_implied_volatilities'][i]*math.sqrt(t)
        black=100*math.exp(-.5*(-x/sd+sd/2)**2)/math.sqrt(2*math.pi)*math.sqrt(t)
        self.assertTrue(math.isclose(e['hedge_quantities'][i]['mean']*black,-e['raw_buckets'][i]['mean'],rel_tol=1e-13))

    def test_heldout_identity_and_existing_initial_statistics_unchanged(self):
        p=request('lookback',samples=4096,nodes=9);d=evaluate(p);r=d['recovery'];e=d['estimate']
        plan=compile_plan(p)
        self.assertEqual(r['initial'],plan.evaluate_vega_kt_initial_gamma_diagnostic([70.,85.,100.,115.,135.]))
        self.assertEqual(e['native_price'],r['initial']['native_price'])
        local=r['initial']['native_initial_local_vega']['mean']+sum(row['local_vega_total']['mean'] for row in r['validation']['diagnostic']['rows'])
        residual=r['initial']['initial_local_vega_difference']['mean']+sum(row['total_difference']['mean'] for row in r['validation']['diagnostic']['rows'])
        self.assertAlmostEqual(e['native_parallel_local_vega']['mean'],local,places=12)
        self.assertAlmostEqual(e['iv_minus_local_parallel']['mean'],residual,places=11)
        self.assertEqual(d['sampling_index_end_exclusive'],4096)

    def test_workers_and_optional_covariance_do_not_change_moments(self):
        p=request('fixed_asian',samples=4096,nodes=9)
        a=evaluate(p);b=evaluate(p,workers=4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b))
        p['risk']['vega_kt']['full_bucket_covariance']=True;c=evaluate(p)
        self.assertIsNone(a['estimate']['full_bucket_sample_covariance'])
        for k in a['estimate']:
            if k!='full_bucket_sample_covariance':self.assertEqual(a['estimate'][k],c['estimate'][k])

    def test_notional_scaling_price_and_plan_are_preserved(self):
        p=request('asian',samples=4096,nodes=9);p['risk']['vega_kt']['full_bucket_covariance']=True
        plan=compile_plan(p);before=plan.evaluate().to_json();a=evaluate(p)['estimate']
        self.assertEqual(before,plan.evaluate().to_json())
        p['product']['notional']=2.5;b=evaluate(p)['estimate']
        for key in ['raw_buckets','hedge_quantities']:
            for x,y in zip(a[key],b[key]):
                self.assertAlmostEqual(y['mean'],x['mean']*2.5,delta=1e-10*max(1,abs(y['mean'])))
                self.assertAlmostEqual(y['sample_variance'],x['sample_variance']*6.25,delta=1e-10*max(1,abs(y['sample_variance'])))
        np.testing.assert_allclose(b['full_bucket_sample_covariance'],np.array(a['full_bucket_sample_covariance'])*6.25,rtol=1e-10,atol=1e-10)

    def test_failure_gate_does_not_produce_zero_iv_or_retry(self):
        p=request('asian',samples=2,seed=1,nodes=9);d=evaluate(p)
        self.assertEqual(d['recovery']['status'],'validation_failed');self.assertIsNone(d['estimate'])
        self.assertEqual(d['recovery']['selected_regularization'],.01)
        p['engine']['independent_sampling_units']=128
        d=evaluate(p,config=policy(reconstruction_relative_limit=1e-12))
        self.assertEqual(d['recovery']['status'],'no_feasible_pilot_candidate');self.assertIsNone(d['estimate'])

    def test_scope_and_covariance_budget_reject_explicitly(self):
        base=request('asian',samples=16,nodes=9)
        for mutate in [lambda p:p['model']['reporting_iv_basis']['implied_volatilities'].__setitem__(0,.201),
                       lambda p:p['model']['local_variance_grid']['values'].__setitem__(1,.05),
                       lambda p:p['risk'].pop('vega_kt')]:
            p=copy.deepcopy(base);mutate(p)
            with self.assertRaises(rp.PricingError):evaluate(p)
        p=request('asian',samples=16,nodes=33);p['risk']['vega_kt']['full_bucket_covariance']=True
        with self.assertRaises(rp.PricingError):evaluate(p)

    def test_storage_budget_rejected_before_sampling(self):
        p=request('asian',samples=1000000,nodes=9)
        p['risk']['vega_kt']['full_bucket_covariance']=True
        plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=1)
        with self.assertRaisesRegex(rp.PricingError,'storage budget'):
            plan.evaluate_vega_kt_atom_chain_diagnostic([70.,100.,135.],policy=rp.VegaKtGammaRecoveryPolicy(**policy()))
        p['engine']['independent_sampling_units']=4096
        plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=1)
        with self.assertRaisesRegex(rp.PricingError,'storage budget'):
            plan.evaluate_vega_kt_atom_chain_diagnostic([70.,100.,135.],policy=rp.VegaKtGammaRecoveryPolicy(**policy(pilot_sampling_units=1000000)))

    def test_term_structure_and_multiple_observations(self):
        p=request('asian',samples=4096,nodes=9);g=p['model']['local_variance_grid'];t=g['time_nodes']
        sigmas=[.17,.26];g['values']=[sigmas[min(i,1)]**2 for i in range(3) for _ in range(9)]
        vols=[.17,math.sqrt((.17**2*t[1]+.26**2*(t[2]-t[1]))/t[2])]
        p['model']['reporting_iv_basis']['implied_volatilities']=[v for v in vols for _ in range(9)]
        d=evaluate(p);self.assertIsNotNone(d['estimate']);self.assertFalse(d['estimate']['constant_volatility_term_structure'])
        # Four actual fixings, two antithetic modes and Brownian bridge.
        from rust_pricing_backtest.numerics import flat_request
        config=json.loads((ROOT/'fixtures/backtesting/numerical-policy-v0.1.json').read_text())
        p=json.loads(flat_request(config,seed=7,samples=4096,steps=4,nodes=9).to_json())
        dates=p['risk']['vega_kt']['maturity_nodes'];p['product']=request('asian',samples=16)['product']
        p['product']['observations']=[{'date':d,'weight':.25,'value':{'type':'unknown'}} for d in dates]
        for anti in [False,True]:
            p['engine']['variance_reduction'].update(antithetic=anti,brownian_bridge=True)
            d=evaluate(p,config=policy(reconstruction_relative_limit=.9,gamma_relative_standard_error_limit=.95))
            self.assertIsNotNone(d['estimate']);self.assertEqual(d['estimate']['antithetic'],anti)
            e=d['estimate'];r=d['recovery'];residual=r['initial']['initial_local_vega_difference']['mean']+sum(row['total_difference']['mean'] for row in r['validation']['diagnostic']['rows'])
            self.assertAlmostEqual(e['iv_minus_local_parallel']['mean'],residual,places=10)

if __name__=='__main__':unittest.main()
