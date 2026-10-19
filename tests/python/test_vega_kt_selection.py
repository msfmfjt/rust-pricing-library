"""Selection uses disjoint pilots, freezes lambda, and never promotes IV risk."""
import json
from pathlib import Path
import sys
import unittest
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from check_gamma_selection import request, evaluate, quality


def policy(**changes):
    p=dict(candidate_regularizations=[.01,.003,.001,.0003,.0001],pilot_sampling_units=4096,
           reconstruction_relative_limit=.5,gamma_relative_standard_error_limit=.7,mc_margin_multiplier=1.)
    p.update(changes);return p


def remove_plan_fingerprints(value):
    if isinstance(value,dict):
        return {k:remove_plan_fingerprints(v) for k,v in value.items() if k!='plan_fingerprint'}
    if isinstance(value,list):return [remove_plan_fingerprints(v) for v in value]
    return value


class GammaSelectionTests(unittest.TestCase):
    def test_public_policy_validation_and_roundtrip(self):
        p=policy();self.assertEqual(rp.VegaKtGammaRecoveryPolicy(**p).to_dict(),p)
        for change in [dict(candidate_regularizations=[]),dict(candidate_regularizations=[.001,.01]),
                       dict(candidate_regularizations=[.01,.01]),dict(candidate_regularizations=[float('nan')]),
                       dict(pilot_sampling_units=1),dict(reconstruction_relative_limit=0.),
                       dict(gamma_relative_standard_error_limit=1.),dict(mc_margin_multiplier=float('inf'))]:
            with self.assertRaises(rp.ValidationError):rp.VegaKtGammaRecoveryPolicy(**policy(**change))

    def test_validation_failure_does_not_reselect_or_hide_samples(self):
        p=request('asian',samples=2,seed=1,nodes=9);ks=[70.,85.,100.,115.,135.];d=evaluate(p,ks,policy())
        self.assertEqual(d['status'],'validation_failed');self.assertEqual(d['selected_regularization'],.01)
        self.assertTrue(d['pilot_candidates'][0]['passed']);self.assertFalse(d['validation']['passed'])
        self.assertGreater(d['validation']['quality'][0]['gamma_relative_standard_error'],.7)
        plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=256)
        expected=plan.evaluate_vega_kt_gamma_recovery_diagnostic(ks,relative_regularization=.01)
        actual=dict(d['validation']['diagnostic']);self.assertEqual(actual.pop('sampling_index_start'),0)
        self.assertEqual(actual.pop('sampling_index_end_exclusive'),2);self.assertEqual(actual,expected)
        self.assertEqual(d['initial'],plan.evaluate_vega_kt_initial_gamma_diagnostic(ks))
        self.assertEqual(d['iv_bucket_acceptance'],'not_established');self.assertNotIn('raw_buckets',d)
        json.dumps(d,allow_nan=False)

    def test_no_pilot_candidate_is_not_reported_as_zero_risk(self):
        p=request('fixed_asian',samples=128,nodes=9);d=evaluate(p,[70.,85.,100.,115.,135.],policy(reconstruction_relative_limit=1e-12))
        self.assertEqual(d['status'],'no_feasible_pilot_candidate');self.assertIsNone(d['validation'])
        self.assertIsNone(d['selected_regularization']);self.assertTrue(all(not c['passed'] for c in d['pilot_candidates']))
        self.assertIn('recovered_initial_local_vega',d['initial'])

    def test_moments_counter_ranges_quality_and_worker_replay(self):
        p=request('asian',samples=4096,seed=7,nodes=9);ks=[70.,85.,100.,115.,135.];config=policy()
        a=evaluate(p,ks,config);b=evaluate(p,ks,config,workers=4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b))
        self.assertEqual(a['status'],'positive_time_policy_passed');self.assertEqual(a['pilot_index_start'],4096)
        self.assertEqual(a['pilot_index_end_exclusive'],8192)
        expected=None
        for candidate in a['pilot_candidates']:
            raw=candidate['diagnostic'];self.assertEqual(raw['sampling_index_start'],4096)
            self.assertEqual(raw['sampling_index_end_exclusive'],8192)
            for row,q in zip(raw['rows'],candidate['quality']):
                ref=quality(row,config);self.assertEqual(ref['passed'],q['passed'])
                self.assertAlmostEqual(ref['reconstruction_margin_ratio'],q['reconstruction_margin_ratio'],places=12)
                self.assertAlmostEqual(ref['gamma_relative_standard_error'],q['gamma_relative_standard_error'],places=12)
            if candidate['passed'] and expected is None:expected=raw['relative_regularization']
        self.assertEqual(a['selected_regularization'],expected)
        self.assertEqual(a['initial']['native_initial_local_vega'],a['validation']['diagnostic']['unrecovered_initial_local_vega'])

    def test_policy_is_notional_invariant_and_pricing_remains_unchanged(self):
        p=request('fixed_asian',samples=4096,seed=7,nodes=9);ks=[70.,85.,100.,115.,135.]
        plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=256)
        before=plan.evaluate().to_json();a=plan.evaluate_vega_kt_gamma_recovery_audit(ks,policy=rp.VegaKtGammaRecoveryPolicy(**policy()))
        self.assertEqual(before,plan.evaluate().to_json())
        p['product']['notional']=2.5;b=evaluate(p,ks,policy())
        self.assertEqual(a['selected_regularization'],b['selected_regularization']);self.assertEqual(a['status'],b['status'])
        for x,y in zip(a['pilot_candidates'],b['pilot_candidates']):
            self.assertEqual(x['passed'],y['passed'])
            for q,r in zip(x['quality'],y['quality']):
                for key in ['reconstruction_margin_ratio','gamma_relative_standard_error']:
                    self.assertAlmostEqual(q[key],r[key],places=11)

    def test_nonflat_scope_and_missing_policy_fail_explicitly(self):
        p=request('asian',samples=16,nodes=9);p['model']['local_variance_grid']['values'][1]=.05
        with self.assertRaises(rp.PricingError):evaluate(p,[80.,100.,120.],policy())
        p=request('asian',samples=16,nodes=9)
        plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1)
        with self.assertRaises(TypeError):plan.evaluate_vega_kt_gamma_recovery_audit([80.,100.,120.])

if __name__=='__main__':unittest.main()
