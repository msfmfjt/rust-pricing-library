"""Explicit variance extensions through real conditional Gamma and the ledger."""
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
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
from test_vega_kt_path_ledger import fixture
from test_vega_kt_multi_step_provider import advance
from rust_pricing_backtest import (BacktestConfig,BacktestError,Contract,PathContract,PathObservation,
    Position,Quote,RiskBasis,RustPricingProvider,SettlementFixing,Snapshot,run_backtest,compare_strategies)
from rust_pricing_backtest.domain import data
from rust_pricing_backtest.path_history import native_product,resolve
from rust_pricing_backtest.serialization import input_json,load_input
from rust_pricing_backtest.validation import validate_document


def provider(extension='all_rows',**kwargs):
    return RustPricingProvider(numerical_scope='multi_step_extension_gamma',variance_extension=extension,
        worker_threads=4,**kwargs)


def panel(intervals,kind,seed=7):
    p=model(intervals,kind,samples=4096 if intervals==4 else 2048,seed=seed)
    day=date.fromisoformat(p['valuation_date']);at=datetime.combine(day,datetime.min.time(),timezone.utc)+timedelta(hours=16)
    expiry=day+timedelta(days=round(p['model']['local_variance_grid']['time_nodes'][-1]*365))
    fixings={}
    if kind=='call':c=Contract('target',expiry,100.,lot=.1)
    else:
        product=p['product'];asian=product['type']=='arithmetic_asian'
        rows=product['observations'] if asian else [dict(date=d,weight=1.) for d in product['monitoring_dates']]
        observations=tuple(PathObservation(date.fromisoformat(o['date']),f'f{i}',o['weight']) for i,o in enumerate(rows))
        c=PathContract('target',expiry,100.,lot=.1,product_type=product['type'],observations=observations)
        for i,o in enumerate(rows):
            if date.fromisoformat(o['date'])<=day:
                fixed_at=datetime.combine(date.fromisoformat(o['date']),datetime.min.time(),timezone.utc)+timedelta(hours=12)
                fixings[f'f{i}']=SettlementFixing(date.fromisoformat(o['date']),o['value']['fixing'],fixed_at,fixed_at,'extension_fixture','r1')
    b=p['risk']['vega_kt'];basis=RiskBasis(day,tuple(date.fromisoformat(d) for d in b['maturity_nodes']),tuple(b['log_forward_moneyness_nodes']))
    s=Snapshot(f'{intervals}-{kind}-{seed}',at,Quote(100.,100.,at,at),{},basis,at,'extension_fixture',
        {'target':json.dumps(p)},settlement_fixings=fixings)
    if kind!='call':assert native_product(c,s)==p['product']
    return c,s


def six_interval_history(kind):
    c,s=panel(6,'fixed_asian' if kind=='arithmetic_asian' else 'lookback')
    h=Contract('hedge',c.expiry,100.,lot=.1)
    base=json.loads(s.pricing_requests['target']);history=[]
    for i,spot in [(0,100.),(1,100.1)]:
        p=base if i==0 else advance(base,i,spot);day=date.fromisoformat(p['valuation_date'])
        at=s.at+timedelta(days=i)
        b=replace(s.basis,valuation_date=day)
        current=replace(s,snapshot_id=f's{i}',at=at,spot=Quote(spot,spot,at,at),basis=b,surface_available_at=at,
            option_quotes={'hedge':Quote(6.,6.,at,at)})
        p=deepcopy(p);p['product']=native_product(c,current)
        hedge=deepcopy(p);hedge['product']=dict(type='european_vanilla',underlying_id=1,currency_id=2,
            expiry=str(c.expiry),strike=100.,notional=1.,side=dict(type='call'))
        current=replace(current,pricing_requests={'target':json.dumps(p),'hedge':json.dumps(hedge)},
                        settlement_fixings=s.settlement_fixings if i==0 else {})
        history.append(current)
    at=datetime.combine(c.expiry,datetime.min.time(),timezone.utc)+timedelta(hours=16)
    fixes={}
    for i,o in enumerate(c.observations):
        if o.observation_date>s.at.date():
            fixed_at=datetime.combine(o.observation_date,datetime.min.time(),timezone.utc)+timedelta(hours=12)
            fixes[o.fixing_id]=SettlementFixing(o.observation_date,101.+i,fixed_at,fixed_at,'extension_fixture','r1')
    history.append(Snapshot('expiry',at,Quote(103.,103.,at,at),{},
        RiskBasis(c.expiry,(c.expiry+timedelta(days=32),c.expiry+timedelta(days=64)),s.basis.log_moneyness),
        at,'extension_fixture',settlement_fixings=fixes))
    return c,h,history


class ExtensionProviderTests(unittest.TestCase):
    evidence={}

    @classmethod
    def setUpClass(cls):
        cls.providers={e:provider(e) for e in ['all_rows','linear_time']}
        cls.evidence=dict(panels=[],replays=[],snapshots=[],failures=[])

    def test_a_native_panels_and_covariance(self):
        for extension,pr in self.providers.items():
            for n in [4,6]:
                for kind in ['call','asian','fixed_asian','lookback']:
                    c,s=panel(n,kind)
                    risk=pr.evaluate(s,c);audit=pr.gamma_audit(s,c);market=audit['market_gamma'];a=market['gamma_audit']
                    self.assertEqual(a['status'],'path_gamma_chain_passed')
                    self.assertEqual(audit['validation']['calibration_extension'],extension)
                    self.assertFalse(audit['validation']['diagnostic_only'])
                    self.assertEqual(market['extension_audit']['extension'],extension)
                    self.assertEqual([r['variance_row'] for r in a['rows']],list(range(n)))
                    self.assertEqual(tuple(x['mean'] for x in market['estimate']),risk.buckets)
                    np.testing.assert_allclose(np.diag(market['market_mean_covariance']),np.square(risk.bucket_standard_errors),rtol=1e-12)
                    p=json.loads(s.pricing_requests['target']);p['risk'].pop('vega_kt');p['risk']['vega']=False
                    ordinary=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate()
                    self.assertEqual((risk.price,risk.delta),(ordinary.value,ordinary.delta_raw))
                    self.assertIn('variance_extension_'+extension,risk.warning_codes)
                    self.assertNotIn('fixed_unreported_variance_rows',risk.warning_codes)
                    self.evidence['panels'].append(dict(extension=extension,intervals=n,kind=kind,seed=7,risk=data(risk),audit=audit))
                    print(f'Accepted {extension} {n} intervals {kind}',flush=True)

    def check_replay(self,extension,c,h,history,intervals):
        pr=self.providers[extension];cfg=BacktestConfig(ridge=1e-4,market_shock_log_moneyness=(-.2,0.,.2))
        results=compare_strategies(cfg,history,[c,h],[Position('target',-1.,5.)],pr)
        book={}
        for s in history[:-1]:
            resolved=resolve(s,[c,h],book)
            for instrument in [c,h]:
                r=pr.evaluate(resolved,instrument);a=pr.gamma_audit(resolved,instrument)
                self.assertEqual(a['status'],'passed')
                self.evidence['snapshots'].append(dict(extension=extension,intervals=intervals,
                    kind=c.product_type,at=s.at.isoformat(),contract=instrument.contract_id,risk=data(r),audit=a))
        for strategy,full in results.items():
            self.assertEqual(full.status,'completed');self.assertTrue(full.summary['fully_settled'])
            self.assertEqual(full.summary['risk_validation_status'],'numerically_validated_multi_step_extension_call')
            if strategy!='unhedged':self.assertTrue(full.fills)
            self.assertTrue(full.settlements)
            target=next(x for x in full.settlements if x['contract_id']=='target' and x['event'] in ['fixed','fixed_zero'])
            observed=[v['fixing']['value'] for v in target['settlement']['observations']]
            expected=max(sum(o.weight*x for o,x in zip(c.observations,observed))-c.strike,0.) if c.product_type=='arithmetic_asian' else max(max(observed)-c.strike,0.)
            self.assertAlmostEqual(target['settlement']['payoff_per_unit'],expected)
            partial=run_backtest(**load_input(input_json(full)),provider=pr,stop_after=2)
            resumed=run_backtest(**load_input(input_json(full)),provider=pr,checkpoint=partial.checkpoint)
            self.assertEqual(resumed.to_json(),full.to_json())
            for document in [json.loads(full.to_json()),json.loads(input_json(full)),partial.checkpoint]:
                self.assertEqual(validate_document(document)['status'],'valid')
            other=self.providers['linear_time' if extension=='all_rows' else 'all_rows']
            with self.assertRaisesRegex(BacktestError,'checkpoint_mismatch'):
                run_backtest(**load_input(input_json(full)),provider=other,checkpoint=partial.checkpoint)
            self.evidence['replays'].append(dict(extension=extension,intervals=intervals,kind=c.product_type,strategy=strategy,
                summary=full.summary,fills=len(full.fills),settlements=len(full.settlements),payoff=expected,resume_exact=True))
            if intervals==4 and c.product_type=='fixed_lookback' and strategy=='vega_kt':
                self.evidence.setdefault('samples',{})[extension]=dict(request=json.loads(input_json(full)),result=json.loads(full.to_json()),checkpoint=partial.checkpoint)
        print(f'Settled and resumed {extension} {intervals} intervals {c.product_type}: four strategies',flush=True)

    def test_b_fixing_day_ledgers(self):
        for extension in self.providers:
            for kind in ['arithmetic_asian','fixed_lookback']:
                with self.subTest(extension=extension,kind=kind):
                    self.check_replay(extension,*fixture(kind,on_fixing_days=True),4)

    def test_c_six_interval_ledgers(self):
        for extension in self.providers:
            for kind in ['arithmetic_asian','fixed_lookback']:
                with self.subTest(extension=extension,kind=kind):
                    self.check_replay(extension,*six_interval_history(kind),6)

    def test_d_policy_scope_and_failed_risks_are_not_cached(self):
        for scope,extension in [('multi_step_extension_gamma',None),('multi_step_extension_gamma','custom'),
                                ('multi_step_path_gamma','all_rows')]:
            with self.assertRaisesRegex(BacktestError,'extension_provider_policy'):
                RustPricingProvider(numerical_scope=scope,variance_extension=extension)
        self.assertNotEqual(self.providers['all_rows'].identity,self.providers['linear_time'].identity)
        with self.assertRaises(AttributeError):self.providers['all_rows'].variance_extension='linear_time'
        c,s=panel(4,'call');payload=json.loads(s.pricing_requests['target'])
        cases=[]
        def changed(code,mutate,contract=c):
            p=deepcopy(payload);mutate(p);cases.append((code,replace(s,pricing_requests={'target':json.dumps(p)}),contract))
        changed('extension_provider_scope',lambda p:p['product']['side'].update(type='put'),replace(c,side='put'))
        changed('multi_step_provider_scope',lambda p:p['engine'].update(independent_sampling_units=2048))
        changed('extension_provider_scope',lambda p:p['engine']['variance_reduction'].update(antithetic=False))
        changed('multi_step_base_calibration',lambda p:p['model']['reporting_iv_basis']['implied_volatilities'].__setitem__(0,.3))
        for code,snapshot,contract in cases:
            fresh=provider()
            with self.subTest(code=code),self.assertRaisesRegex(BacktestError,code):fresh.evaluate(snapshot,contract)
            self.assertFalse(fresh._cache);self.assertFalse(fresh._shared);self.assertFalse(fresh._gamma_audits)
        market=deepcopy(self.providers['all_rows'].gamma_audit(s,c)['market_gamma'])
        failed=deepcopy(market);failed.update(status='validation_failed',estimate=None,market_mean_covariance=None)
        fresh=provider()
        with patch('rust_pricing_backtest.multi_step_extension.evaluate_extended_market_gamma',return_value=failed):
            out=run_backtest(BacktestConfig(end_policy='mark_to_market'),[s],[c],[Position('target',-1.,5.)],fresh,on_error='return')
        self.assertEqual(out.failure['code'],'multi_step_gamma_acceptance_failure')
        self.assertFalse(out.orders);self.assertFalse(out.fills);self.assertFalse(out.ledger);self.assertFalse(fresh._cache)
        self.assertFalse(fresh._shared);self.assertFalse(fresh._gamma_audits)
        with patch('rust_pricing_backtest.multi_step_extension.evaluate_extended_market_gamma',return_value=market):
            self.assertEqual(fresh.evaluate(s,c),self.providers['all_rows'].evaluate(s,c))
        detached=fresh.gamma_audit(s,c);detached['market_gamma']['estimate'][0]['mean']=1e30
        self.assertNotEqual(fresh.gamma_audit(s,c)['market_gamma']['estimate'][0]['mean'],1e30)
        self.evidence['failures'].append(dict(kind='injected_native_validation',failure=out.failure,orders=len(out.orders),retry_exact=True))

    def test_e_native_resource_limit_stops_before_trading(self):
        c,s=panel(6,'asian');p=json.loads(s.pricing_requests['target']);p['engine']['independent_sampling_units']=4096
        s=replace(s,pricing_requests={'target':json.dumps(p)})
        fresh=provider();out=run_backtest(BacktestConfig(end_policy='mark_to_market'),[s],[c],[Position('target',-1.,5.)],fresh,on_error='return')
        self.assertEqual(out.status,'failed');self.assertEqual(out.failure['code'],'native_pricing_failure')
        self.assertIn('budget',out.failure['message'])
        self.assertFalse(out.orders);self.assertFalse(out.fills);self.assertFalse(fresh._cache);self.assertFalse(fresh._shared)
        self.evidence['failures'].append(dict(kind='real_native_resource_limit',failure=out.failure,orders=len(out.orders)))

    def test_f_cli_selects_explicit_extension(self):
        from rust_pricing_backtest.__main__ import main
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.json';path.write_text('{}')
            for extension in self.providers:
                result=SimpleNamespace(summary={},failure=None,status='completed',write=lambda _:None)
                with patch.object(sys,'argv',['backtest',str(path),'--output',tmp,'--validated-extension',extension]), \
                        patch('rust_pricing_backtest.__main__.load_input',return_value={}), \
                        patch('rust_pricing_backtest.__main__.run_backtest',return_value=result) as run:
                    self.assertEqual(main(),0)
                self.assertEqual(run.call_args.kwargs['provider'].variance_extension,extension)
                self.assertEqual(run.call_args.kwargs['provider'].numerical_scope,'multi_step_extension_gamma')


if __name__=='__main__':unittest.main()
