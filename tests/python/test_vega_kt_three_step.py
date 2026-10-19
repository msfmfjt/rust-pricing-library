from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backtesting'))
from rust_pricing_backtest.three_step_smile import ThreeStepTerminalSmileCalibrator
from check_nonflat_gamma_recovery import variance_operator
from test_vega_kt_selection import policy,remove_plan_fingerprints

def with_product(p,kind):
    from datetime import date,timedelta
    p=deepcopy(p);dates=p['risk']['vega_kt']['maturity_nodes'];expiry=dates[-1]
    common=dict(underlying_id=1,currency_id=2,strike=100.,notional=1.,side=dict(type='call'),payment_date=expiry)
    if kind in ['asian','fixed_asian']:
        obs=[dict(date=d,weight=1/3 if kind=='asian' else .25,value=dict(type='unknown')) for d in dates]
        if kind=='fixed_asian':obs.insert(0,dict(date=str(date.fromisoformat(p['valuation_date'])-timedelta(days=1)),weight=.25,value=dict(type='known',fixing=98.)))
        p['product']=dict(type='arithmetic_asian',observations=obs,**common)
    elif kind=='lookback':p['product']=dict(type='fixed_lookback',monitoring_dates=dates,**common)
    return p

def model(kind='call',samples=8192,seed=7):
    p=json.loads((ROOT/'fixtures/local-vol/three-step-gamma-v0.1.json').read_text())['request'];p=with_product(p,kind)
    p['engine'].update(independent_sampling_units=samples,master_seed=seed);return p
def compile(p,workers=1):return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)
def directions(p):return ThreeStepTerminalSmileCalibrator(p).terminal_iv_tangent()[0].T
def audit(p,settings,D,workers=1):
    nodes=p['model']['local_variance_grid']['log_forward_moneyness_nodes']
    return compile(p,workers).evaluate_vega_kt_three_step_terminal_gamma_audit((100*np.exp(nodes)).tolist(),
        policy=rp.VegaKtGammaRecoveryPolicy(**settings),variance_conservation_directions=D.tolist())

class ThreeStepGammaTests(unittest.TestCase):
    def test_operator_uses_non_gaussian_source_and_retains_scope(self):
        p=model();d=audit(p,policy(),directions(p));f=json.loads((ROOT/'fixtures/local-vol/three-step-gamma-v0.1.json').read_text())
        self.assertEqual(d['status'],'positive_time_policy_passed');self.assertIsNotNone(d['estimate'])
        np.testing.assert_allclose(d['forward_matrix'],f['forward_matrix'],atol=1e-10,rtol=0)
        self.assertEqual(d['unrecovered_variance_rows'],[0,1]);self.assertEqual(d['full_market_iv_bucket_acceptance'],'not_established')
        t=p['model']['local_variance_grid']['time_nodes'];flat=variance_operator(100.,.04*t[2],t[3]-t[2],d['source_log_nodes'],d['third_step_variances'],d['gamma_strikes'])
        self.assertGreater(np.max(abs(flat-np.array(d['forward_matrix']))),1e-3)
        a=d['operator_accuracy'];self.assertFalse(a['renormalized']);self.assertLess(abs(a['source_probability']-1),1e-10);self.assertLess(abs(a['source_first_spot_moment']-100),1e-8)
        json.dumps(d,allow_nan=False)

    def test_actual_third_row_AD_matches_same_path_price_differences(self):
        p=model('lookback');D=directions(p);d=audit(p,policy(),D)
        for direction,m in zip(D,d['validation']['variance_direction_risks']):
            h=1e-8;prices=[]
            for sign in [-1,1]:
                q=deepcopy(p);values=np.array(q['model']['local_variance_grid']['values']);values[66:99]+=sign*h*direction;q['model']['local_variance_grid']['values']=values.tolist()
                prices.append(compile(q).evaluate().value)
            self.assertAlmostEqual((prices[1]-prices[0])/(2*h),m['mean'],delta=2e-6)

    def test_worker_notional_and_price_replay(self):
        p=model('asian');D=directions(p);plan=compile(p);before=plan.evaluate().to_json();a=audit(p,policy(),D);b=audit(p,policy(),D,4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b));self.assertEqual(before,plan.evaluate().to_json())
        p['product']['notional']=2.5;c=audit(p,policy(),D)
        self.assertEqual(a['selected_regularization'],c['selected_regularization'])
        for x,y in zip(a['estimate'],c['estimate']):
            self.assertAlmostEqual(y['mean'],2.5*x['mean'],places=9);self.assertAlmostEqual(y['sample_variance'],6.25*x['sample_variance'],places=7)

    def test_failure_withholds_estimate_without_reselection(self):
        p=model();D=directions(p);d=audit(p,policy(reconstruction_relative_limit=1e-12),D)
        self.assertEqual(d['status'],'no_feasible_pilot_candidate');self.assertIsNone(d['estimate']);self.assertIsNone(d['validation'])
        p=model(samples=2,seed=1);d=audit(p,policy(candidate_regularizations=[.01]),D)
        self.assertEqual(d['status'],'validation_failed');self.assertEqual(d['selected_regularization'],.01);self.assertIsNone(d['estimate'])

    def test_scope_rank_and_legacy_volatility_flags_rejected(self):
        p=model();D=directions(p)
        for bad in [np.ones((2,33)),np.zeros((1,33)),np.ones((1,32)),np.full((1,33),float('inf'))]:
            with self.assertRaises((rp.PricingError,rp.ValidationError)):audit(p,policy(),bad)
        with self.assertRaises(rp.PricingError):audit(p,policy(preserve_local_vega_total=True),D)
        q=deepcopy(p);q['model']['local_variance_grid']['values'][0]+=.001
        with self.assertRaises(rp.PricingError):audit(q,policy(),D)
        # Asian observations retain the original dates, so shifting a model
        # knot now inserts another actual simulation interval.
        q=with_product(p,'asian');q['model']['local_variance_grid']['time_nodes'][2]+=.001
        with self.assertRaises(rp.PricingError):audit(q,policy(),D)
        with self.assertRaises(rp.PricingError):compile(p).evaluate_vega_kt_two_step_smile_gamma_audit([80.,100.,120.],policy=rp.VegaKtGammaRecoveryPolicy(**policy()))

    def test_unpaired_sampling_and_direction_sign(self):
        p=model('fixed_asian');p['engine']['variance_reduction'].update(antithetic=False,brownian_bridge=True);D=directions(p);a=audit(p,policy(),D);b=audit(p,policy(),-D)
        self.assertEqual(a['sampling_unit'],'path');self.assertEqual(a['selected_regularization'],b['selected_regularization'])
        for x,y in zip(a['estimate'],b['estimate']):
            self.assertAlmostEqual(x['mean'],-y['mean'],places=10);self.assertAlmostEqual(x['sample_variance'],y['sample_variance'],places=8)

if __name__=='__main__':unittest.main()
