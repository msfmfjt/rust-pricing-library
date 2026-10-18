"""Public compiled-payoff paper-chain adapter; diagnostic-only, no acceptance claim."""
from datetime import date, timedelta
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backtesting'))
from rust_pricing_backtest.numerics import flat_request

POLICY=json.loads((ROOT/'fixtures/backtesting/numerical-policy-v0.1.json').read_text())


def request(samples=64):
    return flat_request(POLICY,seed=7,samples=samples,steps=4,nodes=9)


def plan(r=None,workers=1):
    return rp.PricingPlan.compile(r or request(),worker_threads=workers,reduction_block_size=16)


def strikes():
    return (POLICY['spot']*np.exp(np.linspace(-.8,.8,9))).tolist()


class ChainPricingTest(unittest.TestCase):
    def test_diagnostic_payload_units_replay_and_ordinary_result_unchanged(self):
        r=request();p=plan(r);before=p.evaluate().to_json();d=p.evaluate_vega_kt_chain_diagnostic(strikes());obj=d.to_dict()
        self.assertEqual(d.policy_label,'flat_variance_weak_gamma_chain_diagnostic_v1')
        self.assertTrue(obj['diagnostic_only']);self.assertEqual(obj['document_kind'],'vega_kt_chain_diagnostic')
        self.assertNotIn('signed_residual',obj);self.assertNotIn('accepted',obj)
        self.assertEqual(obj['requested_initial_strikes'],strikes())
        self.assertGreater(len(obj['initial_strikes']),len(strikes()))
        self.assertGreater(len(obj['recovery_log_moneyness_nodes']),9)
        self.assertEqual(obj['rows'][1]['recovery_density_maturity'],obj['maturity_nodes'][1])
        self.assertEqual(obj['rows'][1]['recovery_source_maturity'],obj['maturity_nodes'][0])
        self.assertEqual(obj['request_fingerprint'],p.request_fingerprint);self.assertEqual(obj['plan_fingerprint'],p.plan_fingerprint)
        self.assertEqual(len(d.raw_buckets),36);self.assertEqual(len(d.standard_errors),36)
        self.assertEqual(d.raw_buckets,[m['mean'] for m in obj['position_iv_buckets']])
        self.assertAlmostEqual(math.fsum(d.raw_buckets),d.scalar_vega,places=10)
        self.assertEqual(d.scalar_standard_error,obj['total_position_iv_vega']['standard_error'])
        self.assertEqual(obj['sampling_unit'],'antithetic_pair');self.assertEqual(obj['evaluated_paths'],128)
        json.dumps(obj,allow_nan=False);self.assertEqual(before,p.evaluate().to_json())
        parallel=plan(r,4).evaluate_vega_kt_chain_diagnostic(strikes()).to_dict()
        self.assertNotEqual(obj['plan_fingerprint'],parallel['plan_fingerprint'])
        obj.pop('plan_fingerprint');parallel.pop('plan_fingerprint');self.assertEqual(obj,parallel)
        with self.assertRaises(AttributeError):d.scalar_vega=0.

    def test_unsupported_graphs_engines_carry_and_covariance_fail_explicitly(self):
        base=json.loads(request().to_json());cases=[]
        q=json.loads(json.dumps(base));q['risk']['vega_kt']['full_bucket_covariance']=True;cases.append(q)
        q=json.loads(json.dumps(base));q['engine']={'type':'randomized_quasi_monte_carlo','points_per_scramble':64,'scramble_count':2,'master_scramble_seed':7,'variance_reduction':{'antithetic':True,'brownian_bridge':False}};cases.append(q)
        q=json.loads(json.dumps(base));q['market']['discount_curve']['discount_factors']=[1.,.98];q['market']['dividend_curve']['discount_factors']=[1.,.98];cases.append(q)
        expiry=base['product']['expiry'];start=date.fromisoformat(base['valuation_date'])
        q=json.loads(json.dumps(base));q['product']={'type':'digital','underlying_id':1,'currency_id':2,'expiry':expiry,'strike':100.,'payout':1.,'side':{'type':'call'},'payout_kind':{'type':'cash'},'payment_date':expiry};q['risk']={'delta':False,'vega':False,'smile_dynamics':{'type':'sticky_log_moneyness'}};cases.append(q)
        for monitoring in ['discrete','continuous']:
            q=json.loads(json.dumps(base));q['product']={'type':'barrier','underlying_id':1,'currency_id':2,'expiry':expiry,'strike':100.,'barrier':130.,'notional':1.,'side':{'type':'call'},'direction':{'type':'up'},'style':{'type':'knock_out'},'monitoring':{'type':monitoring},'monitoring_dates':[(start+timedelta(days=32)).isoformat(),expiry],'payment_date':expiry}
            if monitoring=='discrete':q['risk']={'delta':False,'vega':False,'smile_dynamics':{'type':'sticky_log_moneyness'}}
            cases.append(q)
        for q in cases:
            with self.subTest(product=q['product']['type'],engine=q['engine']['type']):
                p=plan(rp.PricingRequest.from_json(json.dumps(q)))
                with self.assertRaises(rp.PricingError):p.evaluate_vega_kt_chain_diagnostic(strikes())
                self.assertTrue(math.isfinite(p.evaluate().value))

    def test_parameters_pricing_sample_minimum_and_invalid_input(self):
        with self.assertRaises(rp.ValidationError):plan(request(1))
        p=plan(request(2));d=p.evaluate_vega_kt_chain_diagnostic(strikes(),relative_bump=.02,subdivisions=4)
        self.assertIsNotNone(d.scalar_standard_error);self.assertTrue(all(x is not None for x in d.standard_errors))
        obj=d.to_dict();self.assertEqual(obj['relative_bump'],.02);self.assertEqual(obj['coarse_subdivisions'],4)
        for ks,h,q in [([100.],.04,2),(strikes(),float('nan'),2),(strikes(),.04,0)]:
            with self.assertRaises(rp.PricingError):p.evaluate_vega_kt_chain_diagnostic(ks,relative_bump=h,subdivisions=q)


if __name__=='__main__':unittest.main()
