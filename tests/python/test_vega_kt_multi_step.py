import json
import sys
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backtesting'))
from multi_step_cases import model,models
from rust_pricing_backtest.multi_step_smile import MultiStepTerminalSmileCalibrator
from test_vega_kt_selection import policy,remove_plan_fingerprints

def compile(p,workers=1):return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)
def audit(p,workers=1,settings=None):
    c=MultiStepTerminalSmileCalibrator(p);D=c.terminal_iv_tangent()[0].T
    return compile(p,workers).evaluate_vega_kt_multi_step_terminal_gamma_audit((100*np.exp(c.nodes)).tolist(),policy=rp.VegaKtGammaRecoveryPolicy(**(settings or policy())),variance_conservation_directions=D.tolist())

class MultiStepGammaTests(unittest.TestCase):
    def test_actual_recursive_source_and_nonflat_first_reporting_maturity(self):
        for case in models():
            d=audit(case['request']);count=case['intervals']
            self.assertEqual(d['status'],'positive_time_policy_passed');self.assertEqual(d['interval_count'],count)
            self.assertEqual(d['unrecovered_variance_rows'],list(range(count-1)));self.assertEqual(d['recovered_variance_row'],count-1)
            np.testing.assert_allclose(d['forward_matrix'],case['forward_matrix'],atol=1e-10,rtol=0)
            self.assertEqual(len(d['operator_accuracy']['steps']),count-1);self.assertFalse(d['operator_accuracy']['renormalized'])
            for s in d['operator_accuracy']['steps']:
                self.assertLess(abs(s['probability']-1),1e-10);self.assertLess(abs(s['first_spot_moment']-100),1e-8)
            self.assertEqual(d['full_market_iv_bucket_acceptance'],'not_established');json.dumps(d,allow_nan=False)
    def test_conditioning_preserves_native_prices_and_reduces_cancellation_noise(self):
        from test_vega_kt_three_step import model as old_model,directions,audit as old_audit
        p=old_model('call');old=old_audit(p,policy(),directions(p));new=audit(p)
        self.assertEqual(old['validation']['native_price'],new['validation']['native_price'])
        self.assertEqual(old['validation']['unrecovered_initial_variance_sensitivity'],new['validation']['unrecovered_initial_variance_sensitivity'])
        np.testing.assert_allclose(old['forward_matrix'],new['forward_matrix'],atol=1e-10,rtol=0)
        self.assertLess(new['validation']['variance_sensitivity_total']['standard_error'],.5*old['validation']['variance_sensitivity_total']['standard_error'])
    def test_same_path_direction_conservation_for_all_histories(self):
        for kind in ['asian','fixed_asian','lookback']:
            d=audit(model(4,kind,samples=16384))
            self.assertIsNotNone(d['estimate'])
            for c in d['pilot_candidates']+[d['validation']]:
                for v in c['direction_differences']:self.assertLess(abs(v['mean'])+6*v['standard_error'],1e-8)
    def test_workers_notional_and_normal_pricing(self):
        p=model();plan=compile(p);before=plan.evaluate().to_json();a=audit(p);b=audit(p,4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b));self.assertEqual(before,plan.evaluate().to_json())
        p['product']['notional']=2.5;c=audit(p);self.assertEqual(a['selected_regularization'],c['selected_regularization'])
        for x,y in zip(a['estimate'],c['estimate']):self.assertAlmostEqual(y['mean'],2.5*x['mean'],delta=1e-8);self.assertAlmostEqual(y['sample_variance'],6.25*x['sample_variance'],delta=1e-6)
    def test_unpaired_bridge_and_failure_withholding(self):
        p=model();p['engine']['variance_reduction'].update(antithetic=False,brownian_bridge=True);d=audit(p)
        self.assertEqual(d['sampling_unit'],'path');self.assertIsNotNone(d['estimate'])
        d=audit(model(),settings=policy(reconstruction_relative_limit=1e-12));self.assertEqual(d['status'],'no_feasible_pilot_candidate');self.assertIsNone(d['estimate'])
        d=audit(model(samples=2,seed=1));self.assertEqual(d['status'],'validation_failed');self.assertIsNone(d['estimate'])
    def test_invalid_scope_precision_and_conservation_rejected(self):
        p=model();c=MultiStepTerminalSmileCalibrator(p);D=c.terminal_iv_tangent()[0].T
        for row,value in [(0,1e-32),(1,1e-16)]:
            q=deepcopy(p);q['model']['local_variance_grid']['floor']=1e-40;q['model']['local_variance_grid']['values'][row*33:(row+1)*33]=[value]*33
            with self.assertRaises(rp.PricingError):compile(q).evaluate_vega_kt_multi_step_terminal_gamma_audit([80.,100.,120.],policy=rp.VegaKtGammaRecoveryPolicy(**policy()),variance_conservation_directions=D.tolist())
        with self.assertRaises(rp.PricingError):audit(p,settings=policy(preserve_local_vega_total=True))
        q=model(kind='asian');q['model']['local_variance_grid']['time_nodes'][1]+=.001
        with self.assertRaises(rp.PricingError):compile(q).evaluate_vega_kt_multi_step_terminal_gamma_audit((100*np.exp(c.nodes)).tolist(),policy=rp.VegaKtGammaRecoveryPolicy(**policy()),variance_conservation_directions=D.tolist())
if __name__=='__main__':unittest.main()
