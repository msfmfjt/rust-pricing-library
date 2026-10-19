"""Fixed linear conservation constraints retain inverse bias/spatial diagnostics."""
import json
import math
import sys
from pathlib import Path
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from check_path_gamma_recovery import request, operator
from check_conservative_gamma_market import kkt_inverse
from test_vega_kt_selection import policy, remove_plan_fingerprints


def evaluate(p, config, workers=1):
    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)
    return plan.evaluate_vega_kt_atom_chain_diagnostic([70.,85.,100.,115.,135.],policy=rp.VegaKtGammaRecoveryPolicy(**config))


class ConservativeGammaTests(unittest.TestCase):
    def test_explicit_policy_and_legacy_default_are_preserved(self):
        p=policy(preserve_local_vega_total=True)
        self.assertEqual(rp.VegaKtGammaRecoveryPolicy(**p).to_dict(),p)
        self.assertEqual(rp.VegaKtGammaRecoveryPolicy(**policy(preserve_local_vega_total=False)).to_dict(),policy())
        payload=request('asian',samples=4096,nodes=9)
        self.assertEqual(evaluate(payload,policy()),evaluate(payload,policy(preserve_local_vega_total=False)))

    def test_every_candidate_matches_kkt_without_erasing_spatial_residuals(self):
        p=request('asian',samples=4096,nodes=9);d=evaluate(p,policy(preserve_local_vega_total=True))
        self.assertIsNotNone(d['estimate']);self.assertIn('conservative',d['policy_label'])
        r=d['recovery'];self.assertEqual(r['pilot_index_start'],4096);self.assertEqual(r['pilot_index_end_exclusive'],8192)
        for c in r['pilot_candidates']+[r['validation']]:
            raw=c['diagnostic'];self.assertIn('local_vega_total_constraint',raw)
            row=raw['rows'][0];t=row['source_maturity'];dt=row['recovered_maturity']-t
            b=operator(100.,.04*t,.04*dt,.2*dt,raw['source_log_nodes'],np.array(raw['gamma_strikes']))
            inv=kkt_inverse(b,raw['relative_regularization'])
            local=np.array([m['mean'] for m in row['local_vega']]);q=np.array([m['mean'] for m in row['gamma_masses']])
            np.testing.assert_allclose(q,inv@local,rtol=1e-8,atol=1e-8)
            self.assertLess(abs(row['total_difference']['mean']),1e-10)
            self.assertLess(row['total_difference']['sample_variance'],1e-18)
            np.testing.assert_allclose([m['mean'] for m in row['reconstruction_difference']],b@q-local,rtol=1e-7,atol=1e-9)
            if raw['relative_regularization']==.01:
                self.assertGreater(sum(m['mean']**2 for m in row['reconstruction_difference']),1e-6)
        e=d['estimate'];initial=r['initial']['initial_local_vega_difference']
        self.assertAlmostEqual(e['iv_minus_local_parallel']['mean'],initial['mean'],places=11)
        self.assertAlmostEqual(e['iv_minus_local_parallel']['sample_variance'],initial['sample_variance'],places=9)
        self.assertEqual(d['iv_bucket_acceptance'],'not_established')

    def test_conservation_worker_replay_and_actual_failed_quality_gate(self):
        p=request('fixed_asian',samples=4096,nodes=9);config=policy(preserve_local_vega_total=True)
        a=evaluate(p,config);b=evaluate(p,config,workers=4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b))
        failed=evaluate(p,policy(preserve_local_vega_total=True,reconstruction_relative_limit=1e-12))
        self.assertIsNone(failed['estimate']);self.assertEqual(failed['recovery']['status'],'no_feasible_pilot_candidate')
        self.assertTrue(all(not c['passed'] for c in failed['recovery']['pilot_candidates']))

    def test_notional_and_covariance_scale_without_data_dependent_mass_normalization(self):
        p=request('lookback',samples=4096,nodes=9);p['risk']['vega_kt']['full_bucket_covariance']=True
        config=policy(preserve_local_vega_total=True);a=evaluate(p,config)
        p['product']['notional']=2.5;b=evaluate(p,config)
        self.assertEqual(a['recovery']['selected_regularization'],b['recovery']['selected_regularization'])
        for x,y in zip(a['estimate']['raw_buckets'],b['estimate']['raw_buckets']):
            self.assertAlmostEqual(y['mean'],2.5*x['mean'],delta=1e-10*max(1,abs(y['mean'])))
        np.testing.assert_allclose(b['estimate']['full_bucket_sample_covariance'],np.array(a['estimate']['full_bucket_sample_covariance'])*6.25,rtol=1e-10,atol=1e-10)
        json.dumps(b,allow_nan=False)

if __name__=='__main__':unittest.main()
