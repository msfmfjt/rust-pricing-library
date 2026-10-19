"""Path-dependent positive-time Gamma recovery and its explicit scope boundary."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from check_path_gamma_recovery import request, evaluate, operator, inverse

class GammaRecoveryTests(unittest.TestCase):
    def test_asian_fixed_history_lookback_and_vanilla_use_actual_compiled_ad(self):
        for kind in ['call','asian','fixed_asian','lookback']:
            p=request(kind,samples=256,nodes=9);nodes=p['model']['local_variance_grid']['log_forward_moneyness_nodes'];strikes=100*np.exp(nodes)
            plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=256)
            before=plan.evaluate();d=evaluate(p,strikes,.01)
            self.assertEqual(d['document_kind'],'vega_kt_gamma_recovery_diagnostic')
            self.assertEqual(d['initial_gamma_status'],'not_identifiable_from_initial_local_vega')
            self.assertEqual(d['iv_bucket_acceptance'],'not_established');self.assertTrue(d['diagnostic_only'])
            self.assertNotIn('raw_buckets',d);self.assertNotIn('signed_residual',d)
            for row in d['rows']:
                counts=row['local_vega_nonzero_sampling_units']
                self.assertEqual(len(counts),len(nodes))
                self.assertTrue(all(0<=n<=d['independent_sampling_units'] for n in counts))
                self.assertIn(0,counts)
                for n,v in zip(counts,row['local_vega']):
                    if n==0:
                        self.assertEqual(v['mean'],0.)
                        self.assertEqual(v['standard_error'],0.)
            total=d['unrecovered_initial_local_vega']['mean']+sum(r['local_vega_total']['mean'] for r in d['rows'])
            self.assertAlmostEqual(total,before.vega_raw,places=11)
            self.assertEqual(before.to_json(),plan.evaluate().to_json())
            other=evaluate(p,strikes,.01,workers=4)
            d.pop('plan_fingerprint');other.pop('plan_fingerprint');self.assertEqual(d,other)
            json.dumps(d,allow_nan=False)

    def test_matrix_matches_independent_svd_and_residual_statistics(self):
        p=request('fixed_asian',samples=1024);nodes=p['model']['local_variance_grid']['log_forward_moneyness_nodes'];strikes=100*np.exp(nodes)
        d=evaluate(p,strikes,.001);row=d['rows'][0];dt=row['recovered_maturity']-row['source_maturity']
        b=operator(100.,row['source_total_variance'],row['step_variance'],np.sqrt(row['step_variance']/dt)*dt,nodes,strikes)
        r=inverse(b,.001);local=np.array([x['mean'] for x in row['local_vega']]);gamma=np.array([x['mean'] for x in row['gamma_masses']])
        np.testing.assert_allclose(gamma,r@local,atol=2e-7,rtol=2e-7)
        np.testing.assert_allclose(b@gamma-local,[v['mean'] for v in row['reconstruction_difference']],atol=2e-9)
        self.assertTrue(all(v['standard_error']>0 for v in row['gamma_masses']))
        self.assertAlmostEqual(row['reconstructed_total']['mean']-row['local_vega_total']['mean'],row['total_difference']['mean'],places=12)
        self.assertGreater(row['inverse_noise_gain_bound'],0.)

    def test_notional_scales_means_and_variances_without_fake_independence(self):
        p=request('asian',samples=256,nodes=9);strikes=100*np.exp(p['model']['local_variance_grid']['log_forward_moneyness_nodes'])
        a=evaluate(p,strikes,.01);p['product']['notional']=2.5;b=evaluate(p,strikes,.01)
        for field in ['gamma_masses','local_vega','reconstruction_difference']:
            for x,y in zip(a['rows'][0][field],b['rows'][0][field]):
                self.assertAlmostEqual(y['mean'],2.5*x['mean'],places=10)
                self.assertAlmostEqual(y['sample_variance'],6.25*x['sample_variance'],delta=1e-9*max(1.,y['sample_variance']))
                self.assertAlmostEqual(y['standard_error'],2.5*x['standard_error'],places=10)

    def test_rejects_nonflat_carry_indicators_rqmc_and_invalid_inverse(self):
        base=request('asian',samples=16,nodes=9);strikes=100*np.exp(base['model']['local_variance_grid']['log_forward_moneyness_nodes'])
        cases=[]
        p=json.loads(json.dumps(base));p['model']['local_variance_grid']['values'][1]=.05;cases.append(p)
        p=json.loads(json.dumps(base));p['market']['discount_curve']['discount_factors']=[1.,.98];cases.append(p)
        p=json.loads(json.dumps(base));p['engine']={'type':'randomized_quasi_monte_carlo','points_per_scramble':16,'scramble_count':2,'master_scramble_seed':7,'variance_reduction':{'antithetic':True,'brownian_bridge':False}};cases.append(p)
        p=request('call',samples=16,nodes=9);expiry=p['product']['expiry'];p['product']={'type':'digital','underlying_id':1,'currency_id':2,'expiry':expiry,'strike':100.,'payout':1.,'side':{'type':'call'},'payout_kind':{'type':'cash'},'payment_date':expiry};p['risk']={'delta':False,'vega':False,'smile_dynamics':{'type':'sticky_log_moneyness'}};cases.append(p)
        for p in cases:
            with self.assertRaises(rp.PricingError):evaluate(p,strikes,.01)
        for ridge in [0.,1e-8,float('nan'),float('inf'),2.]:
            with self.assertRaises(rp.PricingError):evaluate(base,strikes,ridge)
        for ks in [[100.],[100.,100.],[-1.,100.],[100.,float('nan')]]:
            with self.assertRaises(rp.PricingError):evaluate(base,ks,.01)
        p=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(base)),worker_threads=1)
        with self.assertRaises(TypeError):p.evaluate_vega_kt_gamma_recovery_diagnostic(strikes)

if __name__=='__main__':unittest.main()
