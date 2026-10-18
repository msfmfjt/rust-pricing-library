"""General-smile terminal replication, acceptance boundaries and legacy closure."""
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backtesting'))
from rust_pricing_backtest import (BacktestConfig, BacktestError, Contract, Position, Quote, RiskBasis,
                                  RustPricingProvider, Snapshot, compare_strategies, run_backtest)
from rust_pricing_backtest.serialization import input_json, load_input
from rust_pricing_backtest.smile_numerics import METHOD, dispose_legacy, evaluate, quote_price, smile_request

POLICY = json.loads((ROOT/'fixtures/backtesting/smile-policy-v0.3.json').read_text())


def snapshot_for(p):
    day = date.fromisoformat(p['valuation_date']); at = datetime.combine(day, datetime.min.time(), timezone.utc)
    r = p['risk']['vega_kt']
    basis = RiskBasis(day, tuple(date.fromisoformat(d) for d in r['maturity_nodes']),
                      tuple(r['log_forward_moneyness_nodes']))
    contract = Contract('unit', date.fromisoformat(p['product']['expiry']), p['product']['strike'],
                        p['product']['side']['type'])
    return Snapshot('smile', at, Quote(100.,100.,at,at), {}, basis, at, 'synthetic_smile',
                    {'unit':json.dumps(p)}), contract


class SmileReplicationTests(unittest.TestCase):
    def test_price_and_local_greeks_unchanged_exact_risk_has_no_sample_moments(self):
        for smile in POLICY['smiles']:
            p = smile_request(POLICY, smile, strike=105., steps=8, reporting_nodes=17, samples=128)
            # Two reporting dates, including expiry, independent of local grid.
            b = p['model']['reporting_iv_basis']; r = p['risk']['vega_kt']
            b['maturity_nodes'] = [b['maturity_nodes'][3], b['maturity_nodes'][7]]
            b['implied_volatilities'] = b['implied_volatilities'][3*17:4*17]+b['implied_volatilities'][7*17:8*17]
            b['shape'] = [2,17]
            r['maturity_nodes'] = [r['maturity_nodes'][3],r['maturity_nodes'][7]]
            r['full_bucket_covariance'] = True
            out = evaluate(p); kt = out.vega_kt
            self.assertEqual(kt.policy_label, METHOD)
            self.assertTrue(all(e.sample_variance is None and e.price_covariance is None for e in kt.estimates))
            self.assertEqual(list(kt.full_bucket_covariance), [None]*34**2)
            self.assertEqual(list(kt.raw_buckets[:17]), [0.]*17)
            without = deepcopy(p); without['risk'].pop('vega_kt')
            original = evaluate(without)
            for field in ['value','standard_error','delta_raw','vega_raw']:
                self.assertEqual(getattr(out,field),getattr(original,field))
            other = deepcopy(p); other['engine']['master_seed'] = 97
            self.assertEqual(list(kt.raw_buckets), list(evaluate(other,workers=4).vega_kt.raw_buckets))
            self.assertEqual(rp.PricingResult.from_json(out.to_json()).to_json(), out.to_json())
            if importlib.util.find_spec('jsonschema') is not None:
                import jsonschema
                jsonschema.validate(json.loads(out.to_json()),json.loads(rp.result_json_schema()))

    def test_carry_put_and_signed_position_scaling_match_market_price_differences(self):
        for side in ['call','put']:
            p = smile_request(POLICY, POLICY['smiles'][0], side=side, strike=110., rate=.03, steps=8, samples=64)
            out = evaluate(p); b = p['model']['reporting_iv_basis']; h=1e-5
            n = len(b['implied_volatilities']); width=b['shape'][1]
            for j in range(n-width,n):
                shift = [0.]*n;shift[j]=h
                ref=(quote_price(p,shift=shift)-quote_price(p,shift=[-v for v in shift]))/(2*h)
                self.assertAlmostEqual(out.vega_kt.raw_buckets[j],ref,delta=2e-6)
            p['product']['notional']=2.5
            scaled=evaluate(p)
            for a,b in zip(out.vega_kt.raw_buckets,scaled.vega_kt.raw_buckets):
                self.assertAlmostEqual(2.5*a,b,places=11)

    def test_path_dependent_payoff_is_not_promoted(self):
        p = smile_request(POLICY,POLICY['smiles'][0],steps=8,samples=64,reporting_nodes=65)
        dates=p['risk']['vega_kt']['maturity_nodes']
        p['product']={'type':'arithmetic_asian','underlying_id':1,'currency_id':2,'strike':100.,'notional':1.,
            'side':{'type':'call'},'observations':[{'date':d,'weight':.5,'value':{'type':'unknown'}}
                for d in [dates[3],dates[-1]]],'payment_date':dates[-1]}
        self.assertEqual(evaluate(p).vega_kt.policy_label,'local_vol_node_allocation_diagnostic_v1')

    def test_scoped_provider_accepts_nested_grids_and_rejects_uncalibrated_model(self):
        p=smile_request(POLICY,POLICY['smiles'][0])
        s,c=snapshot_for(p)
        provider=RustPricingProvider(numerical_scope='p0_terminal_replication')
        r=provider.evaluate(s,c)
        self.assertEqual(r.native_policy,METHOD)
        self.assertIsNone(r.bucket_standard_errors)
        self.assertNotIn('unvalidated_vega_kt',r.warning_codes)
        self.assertEqual(provider.validation['status'],'numerically_validated_p0')
        bad=deepcopy(p)
        bad['model']['reporting_iv_basis']['implied_volatilities']=[1.]*len(bad['model']['reporting_iv_basis']['implied_volatilities'])
        s,c=snapshot_for(bad)
        with self.assertRaisesRegex(BacktestError,'native_market_calibration_failure'):
            RustPricingProvider(numerical_scope='p0_terminal_replication').evaluate(s,c)
        with self.assertRaisesRegex(BacktestError,'unsupported_numerical_scope'):
            RustPricingProvider(numerical_scope='all_products')
        with self.assertRaisesRegex(BacktestError,'conflicting_numerical_modes'):
            RustPricingProvider(diagnostic_only=True,numerical_scope='p0_terminal_replication')

    def test_scoped_four_strategy_resume_and_mode_identity(self):
        import test_vega_kt_backtest as fixture
        fixture.NativeBacktestTests.setUpClass()
        f=fixture.NativeBacktestTests
        provider=RustPricingProvider(numerical_scope='p0_terminal_replication')
        cfg=BacktestConfig(ridge=1e-4,market_shock_log_moneyness=(-.1,.1))
        portfolio=[Position('target',-1.,1.)]
        results=compare_strategies(cfg,f.history,f.contracts,portfolio,provider)
        for name,full in results.items():
            self.assertEqual(full.status,'completed')
            self.assertEqual(full.summary['risk_validation_status'],'numerically_validated_p0')
            partial=run_backtest(replace(cfg,strategy=name),f.history,f.contracts,portfolio,provider,stop_after=2)
            resumed=run_backtest(**load_input(input_json(full)),provider=provider,checkpoint=partial.checkpoint)
            self.assertEqual(resumed.to_json(),full.to_json())
            if importlib.util.find_spec('jsonschema') is not None:
                from rust_pricing_backtest.validation import validate_document
                self.assertEqual(validate_document(json.loads(full.to_json()))['status'],'valid')
            with self.assertRaises(BacktestError):
                run_backtest(**load_input(input_json(full)),provider=RustPricingProvider(diagnostic_only=True),checkpoint=partial.checkpoint)

    def test_legacy_disposition_preserves_failure_and_requires_passing_replacement(self):
        failure={'check':'hedge_grid_stability','passed':False,'seed':7,'nodes':9,'observed':.054,'limit':.05}
        replacement={'check':'fixed_market_hedge_grid_stability','passed':True,'seed':7,'reporting_nodes':9}
        report={'original_acceptance':{'checks':[failure]},'checks':[replacement]}
        result=dispose_legacy(report)
        self.assertEqual(result[0]['status'],'superseded_comparison')
        self.assertEqual(result[0]['original'],failure)
        self.assertFalse(failure['passed'])
        report['checks'][0]['passed']=False
        self.assertEqual(dispose_legacy(report)[0]['status'],'unresolved')
        report['checks']=[]
        self.assertEqual(dispose_legacy(report)[0]['status'],'unresolved')
        failure['check']='unexplained_risk_failure'
        report['checks']=[replacement | {'passed':True}]
        self.assertEqual(dispose_legacy(report)[0]['status'],'unresolved')


if __name__ == '__main__':
    unittest.main()
