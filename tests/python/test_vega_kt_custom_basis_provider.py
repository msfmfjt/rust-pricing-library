"""Named tensor bases: native risk, dated Call/Put ledgers and atomic failures."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backtesting'));sys.path.insert(0,str(ROOT/'scripts'))
from test_vega_kt_extension_provider import panel as call_panel, six_interval_history
from test_vega_kt_extension_put_provider import panel as put_panel, six_put_history
from test_vega_kt_path_ledger import fixture
from rust_pricing_backtest import BacktestConfig,BacktestError,Position,RustPricingProvider,run_backtest,compare_strategies
from rust_pricing_backtest.domain import data
from rust_pricing_backtest.named_variance_basis import POLICIES,named_variance_basis
from rust_pricing_backtest.path_history import resolve
from rust_pricing_backtest.serialization import input_json,load_input
from rust_pricing_backtest.validation import validate_document

PRODUCTS=('call','asian_call','fixed_asian_call','lookback_call','put','asian_put','fixed_asian_put',
          'lookback_put','lookback_past_put','lookback_today_put')
SCOPE='multi_step_custom_basis_gamma'
STATUS='numerically_validated_multi_step_named_basis_call_put'


def provider(policy):
    return RustPricingProvider(numerical_scope=SCOPE,variance_basis_policy=policy,worker_threads=4)


def panel(n,kind):
    return put_panel(n,kind.removesuffix('_put')) if kind.endswith('put') else call_panel(n,kind.removesuffix('_call'))


class CustomBasisProviderTests(unittest.TestCase):
    policies=POLICIES
    intervals=(4,6)
    evidence={}

    @classmethod
    def setUpClass(cls):
        cls.providers={p:provider(p) for p in cls.policies}
        cls.evidence=dict(panels=[],replays=[],snapshots=[],failures=[],samples={})

    def test_a_native_panels(self):
        for policy,pr in self.providers.items():
            for n in self.intervals:
                for kind in PRODUCTS:
                    with self.subTest(policy=policy,n=n,kind=kind):
                        c,s=panel(n,kind);risk=pr.evaluate(s,c);audit=pr.gamma_audit(s,c);market=audit['market_gamma']
                        self.assertEqual(audit['validation']['status'],STATUS)
                        self.assertEqual(audit['validation']['variance_basis_policy'],policy)
                        self.assertFalse(audit['validation']['diagnostic_only'])
                        self.assertEqual(tuple(x['mean'] for x in market['estimate']),risk.buckets)
                        np.testing.assert_allclose(np.diag(market['market_mean_covariance']),np.square(risk.bucket_standard_errors),rtol=1e-12)
                        p=json.loads(s.pricing_requests['target']);_,construction=named_variance_basis(p,policy)
                        self.assertEqual(audit['basis_construction'],construction)
                        p['risk'].pop('vega_kt');p['risk']['vega']=False
                        raw=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate()
                        self.assertEqual((risk.price,risk.delta),(raw.value,raw.delta_raw))
                        self.evidence['panels'].append(dict(intervals=n,basis=policy,product=kind,seed=7,risk=data(risk),audit=audit))
                        print(f'Accepted named basis: {policy} {n} {kind}',flush=True)

    def test_b_dated_ledgers(self):
        for policy,pr in self.providers.items():
            for n in self.intervals:
                for side in ('call','put'):
                    for kind in ('arithmetic_asian','fixed_lookback'):
                        with self.subTest(policy=policy,n=n,side=side,kind=kind):
                            if n==4: c,h,history=fixture(kind,side,on_fixing_days=True)
                            else: c,h,history=(six_put_history if side=='put' else six_interval_history)(kind)
                            self.check_replay(policy,pr,n,c,h,history)

    def check_replay(self,policy,pr,n,c,h,history):
        cfg=BacktestConfig(ridge=1e-4,market_shock_log_moneyness=(-.2,0.,.2));positions=[Position('target',-1.,5.)]
        results=compare_strategies(cfg,history,[c,h],positions,pr);book={}
        for s in history[:-1]:
            resolved=resolve(s,[c,h],book)
            for instrument in (c,h):
                risk=pr.evaluate(resolved,instrument);audit=pr.gamma_audit(resolved,instrument)
                self.assertEqual(audit['status'],'passed')
                self.evidence['snapshots'].append(dict(basis=policy,intervals=n,kind=c.product_type,target_side=c.side,
                    at=s.at.isoformat(),contract=instrument.contract_id,risk=data(risk),audit=audit))
        for strategy,whole in results.items():
            self.assertEqual(whole.status,'completed');self.assertTrue(whole.summary['fully_settled'])
            self.assertEqual(whole.summary['risk_validation_status'],STATUS)
            if strategy!='unhedged': self.assertTrue(whole.fills)
            target=next(x for x in whole.settlements if x['contract_id']=='target' and x['event'] in ('fixed','fixed_zero'))
            values=[x['fixing']['value'] for x in target['settlement']['observations']]
            underlying=sum(o.weight*x for o,x in zip(c.observations,values)) if c.product_type=='arithmetic_asian' else (min(values) if c.side=='put' else max(values))
            payoff=max((1 if c.side=='call' else -1)*(underlying-c.strike),0.)
            self.assertAlmostEqual(target['settlement']['payoff_per_unit'],payoff)
            inp=load_input(input_json(whole));partial=run_backtest(**inp,provider=pr,stop_after=2)
            resumed=run_backtest(**inp,provider=pr,checkpoint=partial.checkpoint)
            self.assertEqual(resumed.to_json(),whole.to_json())
            for doc in (json.loads(whole.to_json()),json.loads(input_json(whole)),partial.checkpoint):
                self.assertEqual(validate_document(doc)['status'],'valid')
            for other in (provider(next(x for x in POLICIES if x!=policy)),
                          RustPricingProvider(numerical_scope='multi_step_extension_put_gamma',variance_extension='all_rows',worker_threads=4)):
                with self.assertRaisesRegex(BacktestError,'checkpoint_mismatch'):
                    run_backtest(**inp,provider=other,checkpoint=partial.checkpoint)
            self.evidence['replays'].append(dict(basis=policy,intervals=n,kind=c.product_type,side=c.side,strategy=strategy,
                summary=whole.summary,fills=len(whole.fills),payoff=payoff,resume_exact=True))
            if c.product_type=='fixed_lookback' and c.side=='put' and strategy=='vega_kt':
                self.evidence['samples'][f'{n}-{policy}']=dict(request=json.loads(input_json(whole)),result=json.loads(whole.to_json()),checkpoint=partial.checkpoint)
        if n==4:
            first=run_backtest(cfg,history,[c,h],positions,pr,stop_after=1);fixing=history[1].settlement_fixings['f1']
            for code,fixes in [('missing_path_fixing',{}),
                    ('future_settlement_fixing',{'f1':replace(fixing,available_at=history[1].at+timedelta(hours=1))}),
                    ('path_fixing_revision',{**history[1].settlement_fixings,'f0':replace(history[0].settlement_fixings['f0'],value=99.)})]:
                bad=replace(history[1],settlement_fixings=fixes)
                failed=run_backtest(cfg,[history[0],bad,*history[2:]],[c,h],positions,pr,on_error='return')
                self.assertEqual(failed.failure['code'],code);self.assertEqual(failed.checkpoint,first.checkpoint)
                repaired=run_backtest(cfg,history,[c,h],positions,pr,checkpoint=failed.checkpoint)
                self.assertEqual(repaired.to_json(),results['vega_kt'].to_json())
                self.evidence['failures'].append(dict(basis=policy,side=c.side,kind=c.product_type,code=code,checkpoint_atomic=True,repaired_exact=True))
        print(f'Settled named basis {policy} {n} {c.side} {c.product_type}: four strategies',flush=True)

    def test_c_failed_risk_and_wrong_basis_never_release_orders(self):
        for policy,pr in self.providers.items():
            c,s=panel(self.intervals[0],'put');market=pr.gamma_audit(s,c)['market_gamma']
            for code in ('multi_step_gamma_acceptance_failure','custom_basis_mismatch'):
                failed=deepcopy(market)
                if code=='custom_basis_mismatch': failed['tangent_audit']['basis_hash']='wrong_basis'
                else: failed.update(status='validation_failed',estimate=None,market_mean_covariance=None)
                fresh=provider(policy)
                with patch('rust_pricing_backtest.multi_step_extension.evaluate_extended_market_gamma',return_value=failed):
                    out=run_backtest(BacktestConfig(end_policy='mark_to_market'),[s],[c],[Position('target',-1.,5.)],fresh,on_error='return')
                self.assertEqual(out.failure['code'],code);self.assertFalse(out.orders);self.assertFalse(out.fills)
                self.assertFalse(fresh._cache);self.assertFalse(fresh._shared);self.assertFalse(fresh._gamma_audits)
                with patch('rust_pricing_backtest.multi_step_extension.evaluate_extended_market_gamma',return_value=market):
                    self.assertEqual(fresh.evaluate(s,c),pr.evaluate(s,c))
                detached=fresh.gamma_audit(s,c);detached['basis_construction']['parameter_scales'][0][0]=999.
                self.assertNotEqual(fresh.gamma_audit(s,c)['basis_construction']['parameter_scales'][0][0],999.)
                self.evidence['failures'].append(dict(basis=policy,code=code,orders=len(out.orders),retry_exact=True))

    def test_d_scope_and_sampling_gates(self):
        for policy in (None,'custom','all_rows','linear_time'):
            with self.assertRaisesRegex(BacktestError,'custom_basis_provider_policy'): provider(policy)
        for policy in POLICIES:
            for scope in (None,'multi_step_path_gamma','multi_step_extension_put_gamma'):
                kwargs={'variance_extension':'all_rows'} if scope=='multi_step_extension_put_gamma' else {}
                with self.assertRaisesRegex(BacktestError,'custom_basis_provider_policy'):
                    RustPricingProvider(numerical_scope=scope,variance_basis_policy=policy,**kwargs)
            with self.assertRaisesRegex(BacktestError,'extension_provider_policy'):
                RustPricingProvider(numerical_scope=SCOPE,variance_basis_policy=policy,variance_extension='all_rows')
            pr=provider(policy)
            with self.assertRaises(AttributeError): pr.variance_basis_policy='row_hats'
            c,s=panel(self.intervals[0],'call')
            for changes,code in ((dict(independent_sampling_units=16),'multi_step_provider_scope'),
                                 (dict(variance_reduction={'antithetic':False}),'extension_provider_scope')):
                p=json.loads(s.pricing_requests['target']);p['engine'].update(changes)
                with self.assertRaisesRegex(BacktestError,code): pr.evaluate(replace(s,pricing_requests={'target':json.dumps(p)}),c)
            self.assertFalse(pr._cache);self.assertFalse(pr._shared)

    def test_e_cli_selects_named_policy(self):
        from rust_pricing_backtest.__main__ import main
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.json';path.write_text('{}')
            for policy in self.policies:
                result=SimpleNamespace(summary={},failure=None,status='completed',write=lambda _:None)
                with patch.object(sys,'argv',['backtest',str(path),'--output',tmp,'--validated-custom-basis',policy]), \
                        patch('rust_pricing_backtest.__main__.load_input',return_value={}), \
                        patch('rust_pricing_backtest.__main__.run_backtest',return_value=result) as run:
                    self.assertEqual(main(),0)
                self.assertEqual(run.call_args.kwargs['provider'].numerical_scope,SCOPE)
                self.assertEqual(run.call_args.kwargs['provider'].variance_basis_policy,policy)


if __name__=='__main__': unittest.main()
