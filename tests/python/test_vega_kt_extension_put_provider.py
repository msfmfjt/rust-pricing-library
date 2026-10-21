"""Put and mixed Call/Put ledgers under explicit full-row calibration rules."""
from copy import deepcopy
from dataclasses import replace
from datetime import date,datetime,timedelta,timezone
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backtesting'));sys.path.insert(0,str(ROOT/'scripts'))
from multi_step_cases import model
from extension_put_reference import request,PRODUCTS
from test_vega_kt_path_ledger import fixture
from test_vega_kt_extension_provider import six_interval_history
from rust_pricing_backtest import (BacktestConfig,BacktestError,Contract,PathContract,PathObservation,
    Position,Quote,RiskBasis,RustPricingProvider,SettlementFixing,Snapshot,run_backtest,compare_strategies)
from rust_pricing_backtest.domain import data
from rust_pricing_backtest.path_history import native_product,resolve
from rust_pricing_backtest.serialization import input_json,load_input
from rust_pricing_backtest.validation import validate_document


def provider(extension='all_rows'):
    return RustPricingProvider(numerical_scope='multi_step_extension_put_gamma',variance_extension=extension,worker_threads=4)


def panel(n,kind):
    p=request(model(n,samples=4096 if n==4 else 2048,seed=7),kind)
    day=date.fromisoformat(p['valuation_date']);at=datetime.combine(day,datetime.min.time(),timezone.utc)+timedelta(hours=16)
    expiry=day+timedelta(days=round(p['model']['local_variance_grid']['time_nodes'][-1]*365));fixes={}
    if kind=='put':c=Contract('target',expiry,100.,side='put',lot=.1)
    else:
        product=p['product'];rows=product.get('observations',[dict(date=d,weight=1.) for d in product.get('monitoring_dates',[])])
        obs=tuple(PathObservation(date.fromisoformat(o['date']),f'f{i}',o['weight']) for i,o in enumerate(rows))
        c=PathContract('target',expiry,100.,side='put',lot=.1,product_type=product['type'],observations=obs)
        for i,o in enumerate(rows):
            d=date.fromisoformat(o['date'])
            if d<=day:
                value=o['value']['fixing'] if 'value' in o else 90. if d==day else 98.
                fixed_at=datetime.combine(d,datetime.min.time(),timezone.utc)+timedelta(hours=12)
                fixes[f'f{i}']=SettlementFixing(d,value,fixed_at,fixed_at,'extension_put_fixture','r1')
    b=p['risk']['vega_kt'];basis=RiskBasis(day,tuple(date.fromisoformat(d) for d in b['maturity_nodes']),tuple(b['log_forward_moneyness_nodes']))
    s=Snapshot(f'{n}-{kind}',at,Quote(100.,100.,at,at),{},basis,at,'extension_put_fixture',{'target':json.dumps(p)},settlement_fixings=fixes)
    if kind!='put':assert native_product(c,s)==p['product']
    return c,s


def six_put_history(kind):
    c,h,history=six_interval_history(kind);c=replace(c,side='put');book={};out=[]
    for s in history:
        resolved=resolve(s,[c,h],book)
        if s.at.date()<c.expiry:
            p=json.loads(s.pricing_requests['target']);p['product']=native_product(c,resolved)
            s=replace(s,pricing_requests={**s.pricing_requests,'target':json.dumps(p)})
        out.append(s)
    return c,h,out


class PutExtensionProviderTests(unittest.TestCase):
    evidence={}

    @classmethod
    def setUpClass(cls):
        cls.providers={e:provider(e) for e in ['all_rows','linear_time']}
        cls.evidence=dict(panels=[],replays=[],snapshots=[],failures=[],samples={})

    def test_a_native_put_panels(self):
        for extension,pr in self.providers.items():
            for n in [4,6]:
                for kind in PRODUCTS:
                    with self.subTest(extension=extension,n=n,kind=kind):
                        c,s=panel(n,kind);risk=pr.evaluate(s,c);audit=pr.gamma_audit(s,c)
                        self.assertEqual(audit['status'],'passed');market=audit['market_gamma']
                        self.assertEqual(tuple(x['mean'] for x in market['estimate']),risk.buckets)
                        np.testing.assert_allclose(np.diag(market['market_mean_covariance']),np.square(risk.bucket_standard_errors),rtol=1e-12)
                        p=json.loads(s.pricing_requests['target']);p['risk'].pop('vega_kt');p['risk']['vega']=False
                        raw=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate()
                        self.assertEqual((risk.price,risk.delta),(raw.value,raw.delta_raw))
                        self.assertEqual(audit['validation']['status'],'numerically_validated_multi_step_extension_call_put')
                        self.evidence['panels'].append(dict(intervals=n,extension=extension,product=kind,seed=7,risk=data(risk),audit=audit))
                        print(f'Accepted Put provider: {extension} {n} {kind}',flush=True)

    def check_replay(self,extension,c,h,history,n):
        pr=self.providers[extension];cfg=BacktestConfig(ridge=1e-4,market_shock_log_moneyness=(-.2,0.,.2))
        positions=[Position('target',-1.,5.)];results=compare_strategies(cfg,history,[c,h],positions,pr)
        book={}
        for s in history[:-1]:
            resolved=resolve(s,[c,h],book)
            for instrument in [c,h]:
                risk=pr.evaluate(resolved,instrument);audit=pr.gamma_audit(resolved,instrument)
                self.assertEqual(audit['status'],'passed')
                self.evidence['snapshots'].append(dict(extension=extension,intervals=n,kind=c.product_type,side=instrument.side,
                    at=s.at.isoformat(),contract=instrument.contract_id,risk=data(risk),audit=audit))
        for strategy,whole in results.items():
            self.assertEqual(whole.status,'completed');self.assertTrue(whole.summary['fully_settled'])
            self.assertEqual(whole.summary['risk_validation_status'],'numerically_validated_multi_step_extension_call_put')
            if strategy!='unhedged':self.assertTrue(whole.fills)
            target=next(x for x in whole.settlements if x['contract_id']=='target' and x['event'] in ['fixed','fixed_zero'])
            values=[x['fixing']['value'] for x in target['settlement']['observations']]
            underlying=sum(o.weight*x for o,x in zip(c.observations,values)) if c.product_type=='arithmetic_asian' else min(values)
            self.assertAlmostEqual(target['settlement']['payoff_per_unit'],max(c.strike-underlying,0.))
            partial=run_backtest(**load_input(input_json(whole)),provider=pr,stop_after=2)
            resumed=run_backtest(**load_input(input_json(whole)),provider=pr,checkpoint=partial.checkpoint)
            self.assertEqual(resumed.to_json(),whole.to_json())
            for document in [json.loads(whole.to_json()),json.loads(input_json(whole)),partial.checkpoint]:
                self.assertEqual(validate_document(document)['status'],'valid')
            for other in [self.providers['linear_time' if extension=='all_rows' else 'all_rows'],
                          RustPricingProvider(numerical_scope='multi_step_extension_gamma',variance_extension=extension,worker_threads=4)]:
                with self.assertRaisesRegex(BacktestError,'checkpoint_mismatch'):
                    run_backtest(**load_input(input_json(whole)),provider=other,checkpoint=partial.checkpoint)
            self.evidence['replays'].append(dict(extension=extension,intervals=n,kind=c.product_type,strategy=strategy,
                summary=whole.summary,fills=len(whole.fills),payoff=max(c.strike-underlying,0.),resume_exact=True))
            if n==4 and c.product_type=='fixed_lookback' and strategy=='vega_kt':
                self.evidence['samples'][extension]=dict(request=json.loads(input_json(whole)),result=json.loads(whole.to_json()),checkpoint=partial.checkpoint)
        if n==4:
            first=run_backtest(cfg,history,[c,h],positions,pr,stop_after=1)
            fixing=history[1].settlement_fixings['f1']
            for code,fixes in [('missing_path_fixing',{}),
                    ('future_settlement_fixing',{'f1':replace(fixing,available_at=history[1].at+timedelta(hours=1))}),
                    ('path_fixing_revision',{**history[1].settlement_fixings,'f0':replace(history[0].settlement_fixings['f0'],value=99.)})]:
                bad=replace(history[1],settlement_fixings=fixes)
                failed=run_backtest(cfg,[history[0],bad,*history[2:]],[c,h],positions,pr,on_error='return')
                self.assertEqual(failed.failure['code'],code);self.assertEqual(failed.checkpoint,first.checkpoint)
                repaired=run_backtest(cfg,history,[c,h],positions,pr,checkpoint=failed.checkpoint)
                self.assertEqual(repaired.to_json(),results['vega_kt'].to_json())
                self.evidence['failures'].append(dict(extension=extension,kind=c.product_type,code=code,checkpoint_atomic=True,repaired_exact=True))
        print(f'Settled Put {extension} {n} {c.product_type}: four mixed-side strategies',flush=True)

    def test_b_fixing_day_mixed_ledgers(self):
        for extension in self.providers:
            for kind in ['arithmetic_asian','fixed_lookback']:
                with self.subTest(extension=extension,kind=kind):
                    self.check_replay(extension,*fixture(kind,'put',on_fixing_days=True),4)

    def test_c_six_interval_mixed_ledgers(self):
        for extension in self.providers:
            for kind in ['arithmetic_asian','fixed_lookback']:
                with self.subTest(extension=extension,kind=kind):
                    self.check_replay(extension,*six_put_history(kind),6)

    def test_d_scope_and_failed_native_risk_do_not_release_orders(self):
        for extension in [None,'custom']:
            with self.assertRaisesRegex(BacktestError,'extension_provider_policy'):
                RustPricingProvider(numerical_scope='multi_step_extension_put_gamma',variance_extension=extension)
        c,s=panel(4,'put')
        with self.assertRaisesRegex(BacktestError,'extension_provider_scope'):
            RustPricingProvider(numerical_scope='multi_step_extension_gamma',variance_extension='all_rows').evaluate(s,c)
        market=self.providers['all_rows'].gamma_audit(s,c)['market_gamma'];failed=deepcopy(market)
        failed.update(status='validation_failed',estimate=None,market_mean_covariance=None)
        fresh=provider()
        with patch('rust_pricing_backtest.multi_step_extension.evaluate_extended_market_gamma',return_value=failed):
            out=run_backtest(BacktestConfig(end_policy='mark_to_market'),[s],[c],[Position('target',-1.,5.)],fresh,on_error='return')
        self.assertEqual(out.failure['code'],'multi_step_gamma_acceptance_failure');self.assertFalse(out.orders);self.assertFalse(out.fills)
        self.assertFalse(fresh._cache.results);self.assertIsNone(fresh._cache.shared)
        with patch('rust_pricing_backtest.multi_step_extension.evaluate_extended_market_gamma',return_value=market):
            self.assertEqual(fresh.evaluate(s,c),self.providers['all_rows'].evaluate(s,c))
        self.evidence['failures'].append(dict(code=out.failure['code'],orders=len(out.orders),retry_exact=True))

    def test_e_cli_put_scope_is_explicit(self):
        from rust_pricing_backtest.__main__ import main
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.json';path.write_text('{}')
            for extension in self.providers:
                result=SimpleNamespace(summary={},failure=None,status='completed',write=lambda _:None)
                with patch.object(sys,'argv',['backtest',str(path),'--output',tmp,'--validated-put-extension',extension]), \
                        patch('rust_pricing_backtest.__main__.load_input',return_value={}), \
                        patch('rust_pricing_backtest.__main__.run_backtest',return_value=result) as run:
                    self.assertEqual(main(),0)
                self.assertEqual(run.call_args.kwargs['provider'].numerical_scope,'multi_step_extension_put_gamma')
                self.assertEqual(run.call_args.kwargs['provider'].variance_extension,extension)


if __name__=='__main__':unittest.main()
