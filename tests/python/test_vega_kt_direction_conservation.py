"""Same-path direction statistics and fixed constrained Gamma policy contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from check_path_gamma_recovery import request, operator
from check_gamma_quote_conservation import constrained_inverse
from test_vega_kt_selection import policy, remove_plan_fingerprints


def config(p, **kwargs):
    nodes=np.array(p['model']['local_variance_grid']['log_forward_moneyness_nodes'])
    rows=len(p['model']['local_variance_grid']['time_nodes'])-2
    return policy(preserve_local_vega_total=True,
                  local_vega_conservation_directions=[[nodes.tolist(),(nodes*nodes).tolist()] for _ in range(rows)], **kwargs)


def evaluate(p, settings, workers=1):
    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)
    return plan.evaluate_vega_kt_atom_chain_diagnostic(100*np.exp(p['model']['local_variance_grid']['log_forward_moneyness_nodes']), policy=rp.VegaKtGammaRecoveryPolicy(**settings))


class DirectionConservationTests(unittest.TestCase):
    def test_policy_roundtrip_default_and_tensor_validation(self):
        p=request('asian',samples=4096,nodes=9);settings=config(p)
        self.assertEqual(rp.VegaKtGammaRecoveryPolicy(**settings).to_dict(),settings)
        self.assertEqual(rp.VegaKtGammaRecoveryPolicy(**policy(),local_vega_conservation_directions=None).to_dict(),policy())
        for directions in [[],[[]],[[[0.]*9]],[[[float('inf')]*9]],[[[1.]*130]],[[[1.]*9]*9],
                           [[[1.]*9],[[1.]*8]],[[[1.]*129]*8]*64]:
            with self.assertRaises(rp.ValidationError):rp.VegaKtGammaRecoveryPolicy(**policy(local_vega_conservation_directions=directions))

    def test_all_candidates_preserve_signed_projections_and_real_sampling_variance(self):
        p=request('asian',samples=4096,nodes=9);settings=config(p);d=evaluate(p,settings);r=d['recovery']
        self.assertIn('direction_constrained',d['policy_label']);self.assertIsNotNone(d['estimate'])
        self.assertEqual((r['pilot_index_start'],r['pilot_index_end_exclusive']),(4096,8192))
        for c in r['pilot_candidates']+[r['validation']]:
            raw=c['diagnostic'];self.assertEqual(raw['local_vega_conservation_directions'],settings['local_vega_conservation_directions'])
            row=raw['rows'][0];t=row['source_maturity'];dt=row['recovered_maturity']-t
            B=operator(100.,.04*t,.04*dt,.2*dt,raw['source_log_nodes'],raw['gamma_strikes'])
            directions=np.array(settings['local_vega_conservation_directions'][0]);R=constrained_inverse(B,raw['relative_regularization'],directions)
            b=np.array([m['mean'] for m in row['local_vega']]);q=np.array([m['mean'] for m in row['gamma_masses']])
            np.testing.assert_allclose(q,R@b,rtol=1e-8,atol=1e-8)
            np.testing.assert_allclose([m['mean'] for m in row['local_vega_directions']],directions@b,atol=1e-12)
            for a,b,e in zip(row['local_vega_directions'],row['reconstructed_directions'],row['direction_differences']):
                self.assertAlmostEqual(a['mean'],b['mean'],places=11)
                self.assertAlmostEqual(a['sample_variance'],b['sample_variance'],places=10)
                self.assertLess(abs(e['mean'])+6*e['standard_error'],1e-10)
                self.assertGreater(a['sample_variance'],0.)
            np.testing.assert_allclose([m['mean'] for m in row['reconstruction_difference']],B@q-[m['mean'] for m in row['local_vega']],atol=1e-9)
        self.assertEqual(d['iv_bucket_acceptance'],'not_established');json.dumps(d,allow_nan=False)

    def test_worker_replay_notional_scaling_and_native_price_are_unchanged(self):
        p=request('lookback',samples=4096,nodes=9);settings=config(p)
        plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1)
        before=plan.evaluate().to_json();a=evaluate(p,settings);b=evaluate(p,settings,4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b));self.assertEqual(before,plan.evaluate().to_json())
        p['product']['notional']=2.5;c=evaluate(p,settings)
        self.assertEqual(a['recovery']['selected_regularization'],c['recovery']['selected_regularization'])
        for x,y in zip(a['recovery']['validation']['diagnostic']['rows'][0]['reconstructed_directions'],c['recovery']['validation']['diagnostic']['rows'][0]['reconstructed_directions']):
            self.assertAlmostEqual(y['mean'],2.5*x['mean'],places=10)
            self.assertAlmostEqual(y['sample_variance'],6.25*x['sample_variance'],places=9)

    def test_rank_and_actual_row_shape_fail_before_large_pilot(self):
        p=request('asian',samples=1000000,nodes=9)
        for directions in [[[[1.]*9]],[[[1.]*8]],[[list(range(9))]]*2,
                           [[list(range(9)),list(range(9))]]]:
            with self.assertRaises(rp.PricingError):evaluate(p,policy(preserve_local_vega_total=True,local_vega_conservation_directions=directions))

    def test_failed_quality_never_becomes_a_zero_or_reselected_estimate(self):
        p=request('asian',samples=4096,nodes=9)
        d=evaluate(p,config(p,reconstruction_relative_limit=1e-12))
        self.assertEqual(d['recovery']['status'],'no_feasible_pilot_candidate');self.assertIsNone(d['estimate'])
        p=request('asian',samples=2,seed=1,nodes=9)
        d=evaluate(p,config(p,candidate_regularizations=[.01],gamma_relative_standard_error_limit=.2))
        self.assertEqual(d['recovery']['status'],'validation_failed');self.assertIsNone(d['estimate'])
        self.assertEqual(d['recovery']['selected_regularization'],.01)

    def test_multiple_positive_times_keep_each_rows_own_directions(self):
        from rust_pricing_backtest.numerics import flat_request
        fixture=json.loads((ROOT/'fixtures/backtesting/numerical-policy-v0.1.json').read_text())
        p=json.loads(flat_request(fixture,seed=7,samples=32768,steps=4,nodes=9).to_json())
        p['product']=request('asian',samples=16)['product']
        p['product']['observations']=[dict(date=d,weight=.25,value=dict(type='unknown')) for d in p['risk']['vega_kt']['maturity_nodes']]
        settings=config(p,pilot_sampling_units=32768,reconstruction_relative_limit=.9,gamma_relative_standard_error_limit=.99)
        settings['local_vega_conservation_directions'][1][0]=[-2*x for x in settings['local_vega_conservation_directions'][1][0]]
        p['engine']['variance_reduction']['brownian_bridge']=True
        for anti in [True,False]:
            p['engine']['variance_reduction']['antithetic']=anti;d=evaluate(p,settings)
            self.assertIsNotNone(d['estimate'])
            for row in d['recovery']['validation']['diagnostic']['rows']:
                for e in row['direction_differences']:self.assertLess(abs(e['mean'])+6*e['standard_error'],1e-10)


if __name__=='__main__':unittest.main()
