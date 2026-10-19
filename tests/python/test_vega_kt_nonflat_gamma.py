"""Nonflat two-step variance coordinates, held-out gates, and scope boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from check_path_gamma_recovery import request
from check_nonflat_gamma_recovery import variance_operator, constrained_inverse, audit
from test_vega_kt_selection import policy, remove_plan_fingerprints


def model(kind='asian', samples=4096, seed=7):
    p=request(kind,samples=samples,seed=seed,nodes=9)
    grid=p['model']['local_variance_grid'];x=np.array(grid['log_forward_moneyness_nodes'])
    v=.04+.02*x+.01*x*x
    grid['values'][9:]=v.tolist()*2
    return p


def directions(p):
    x=np.array(p['model']['local_variance_grid']['log_forward_moneyness_nodes'])
    return np.array([np.ones(len(x)),x])


def compile(p,workers=1):
    return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)


class NonflatGammaTests(unittest.TestCase):
    def test_independent_operator_inverse_and_same_path_projection_variance(self):
        p=model();d=audit(p,100*np.exp(np.linspace(-.8,.8,9)),policy(),directions(p))
        self.assertEqual(d['coordinate'],'annual_local_variance_nodes')
        self.assertEqual(d['initial_gamma_recovery'],'not_implemented')
        self.assertEqual(d['full_market_iv_bucket_acceptance'],'not_established')
        self.assertEqual(d['sampling_unit'],'antithetic_pair')
        self.assertEqual((d['pilot_index_start'],d['pilot_index_end_exclusive']),(4096,8192))
        self.assertIsNotNone(d['estimate'])
        B=variance_operator(100.,d['source_variance'],d['terminal_maturity']-d['source_maturity'],d['source_log_nodes'],d['second_step_variances'],d['gamma_strikes'])
        np.testing.assert_allclose(B,d['forward_matrix'],rtol=1e-10,atol=1e-11)
        for row in d['pilot_candidates']+[d['validation']]:
            R=constrained_inverse(B,row['relative_regularization'],directions(p),False)
            b=np.array([m['mean'] for m in row['variance_sensitivities']]);q=np.array([m['mean'] for m in row['gamma_masses']])
            np.testing.assert_allclose(q,R@b,atol=2e-8,rtol=1e-8)
            np.testing.assert_allclose(B@q-b,[m['mean'] for m in row['reconstruction_differences']],atol=1e-10)
            for a,b,e in zip(row['variance_direction_risks'],row['recovered_direction_risks'],row['direction_differences']):
                self.assertAlmostEqual(a['mean'],b['mean'],places=10)
                self.assertAlmostEqual(a['sample_variance'],b['sample_variance'],places=8)
                self.assertLess(abs(e['mean'])+6*e['standard_error'],1e-9)
                self.assertGreater(a['sample_variance'],0.)
        json.dumps(d,allow_nan=False)

    def test_exact_node_variance_ad_agrees_with_same_path_price_bumps(self):
        p=model('lookback');x=np.linspace(-.8,.8,9)
        d=audit(p,100*np.exp(x),policy(),directions(p))
        for direction,stat in zip(directions(p),d['validation']['variance_direction_risks']):
            prices=[];h=1e-8
            for sign in [1,-1]:
                q=deepcopy(p);grid=q['model']['local_variance_grid'];grid['values'][9:18]=(np.array(grid['values'][9:18])+sign*h*direction).tolist()
                prices.append(compile(q).evaluate().value)
            self.assertAlmostEqual((prices[0]-prices[1])/(2*h),stat['mean'],delta=2e-6)

    def test_worker_replay_signed_scaling_and_price_preservation(self):
        p=model('lookback');ks=100*np.exp(np.linspace(-.8,.8,9));plan=compile(p)
        before=plan.evaluate().to_json();a=audit(p,ks,policy(),directions(p));b=audit(p,ks,policy(),directions(p),4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b));self.assertEqual(before,plan.evaluate().to_json())
        p['product']['notional']=2.5;c=audit(p,ks,policy(),directions(p))
        self.assertEqual(a['selected_regularization'],c['selected_regularization'])
        for x,y in zip(a['estimate'],c['estimate']):
            self.assertAlmostEqual(y['mean'],2.5*x['mean'],places=9)
            self.assertAlmostEqual(y['sample_variance'],6.25*x['sample_variance'],places=7)

    def test_no_flat_initial_or_legacy_nonflat_fallback(self):
        plan=compile(model());ks=[70.,85.,100.,115.,135.]
        with self.assertRaises(rp.PricingError):plan.evaluate_vega_kt_gamma_recovery_audit(ks,policy=rp.VegaKtGammaRecoveryPolicy(**policy()))
        with self.assertRaises(rp.PricingError):plan.evaluate_vega_kt_initial_gamma_diagnostic(ks)
        p=model();p['model']['local_variance_grid']['values'][0]+=.001
        with self.assertRaises(rp.PricingError):audit(p,ks,policy(),directions(p))
        p=model();p['model']['local_variance_grid']['time_nodes'][1]+=.001
        with self.assertRaises(rp.PricingError):audit(p,ks,policy(),directions(p))

    def test_invalid_directions_and_old_volatility_flags_rejected(self):
        p=model(samples=1000000);ks=100*np.exp(np.linspace(-.8,.8,9))
        for ds in [np.ones((1,8)),np.ones((2,9)),np.zeros((1,9)),np.full((1,9),float('nan'))]:
            with self.assertRaises((rp.PricingError,rp.ValidationError)):audit(p,ks,policy(),ds)
        for changes in [dict(preserve_local_vega_total=True),dict(local_vega_conservation_directions=[[[1.]*9]])]:
            with self.assertRaises(rp.PricingError):audit(p,ks,policy(**changes),directions(p))

    def test_no_feasible_pilot_withholds_estimate(self):
        p=model();d=audit(p,100*np.exp(np.linspace(-.8,.8,9)),policy(reconstruction_relative_limit=1e-12),directions(p))
        self.assertEqual(d['status'],'no_feasible_pilot_candidate');self.assertIsNone(d['validation']);self.assertIsNone(d['estimate'])
        self.assertTrue(all(not c['passed'] for c in d['pilot_candidates']))

    def test_heldout_failure_does_not_reselect(self):
        p=model(samples=2,seed=1);d=audit(p,100*np.exp(np.linspace(-.8,.8,9)),policy(candidate_regularizations=[.01],gamma_relative_standard_error_limit=.2),directions(p))
        self.assertEqual(d['status'],'validation_failed');self.assertEqual(d['selected_regularization'],.01)
        self.assertIsNone(d['estimate']);self.assertEqual(d['validation']['independent_sampling_units'],2)

    def test_unconstrained_call_and_non_antithetic_sampling_remain_explicit(self):
        p=model('call',samples=8192);p['engine']['variance_reduction']['antithetic']=False
        p['engine']['variance_reduction']['brownian_bridge']=True
        d=compile(p).evaluate_vega_kt_two_step_smile_gamma_audit([70.,85.,100.,115.,135.],policy=rp.VegaKtGammaRecoveryPolicy(**policy()))
        self.assertEqual(d['variance_conservation_directions'],[]);self.assertEqual(d['sampling_unit'],'path')
        for row in d['pilot_candidates']:self.assertEqual(row['recovered_direction_risks'],[])


if __name__=='__main__':unittest.main()
