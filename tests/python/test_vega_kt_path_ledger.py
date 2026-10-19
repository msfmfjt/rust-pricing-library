"""Native Asian/Lookback valuation, immutable histories and cash settlement."""
from copy import deepcopy
from dataclasses import replace
from datetime import date,datetime,timedelta,timezone
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backtesting'));sys.path.insert(0,str(ROOT/'scripts'))
from multi_step_cases import model
from rust_pricing_backtest import (PathContract,PathObservation,Contract,SettlementFixing,Snapshot,Quote,
    RiskBasis,RustPricingProvider,BacktestConfig,Position,BacktestError,run_backtest,compare_strategies)
from rust_pricing_backtest.path_history import native_product
from rust_pricing_backtest.domain import data
from rust_pricing_backtest.serialization import input_json,load_input
from rust_pricing_backtest.validation import validate_document
from rust_pricing_backtest.multi_step_smile import source_marginals
from rust_pricing_backtest.discrete_smile import _black


def provider():return RustPricingProvider(numerical_scope='multi_step_path_gamma',worker_threads=4)


def fixture(kind='arithmetic_asian',side='call',*,on_fixing_days=False):
    p=model(4,samples=4096);start=date.fromisoformat(p['valuation_date'])
    days=[-1,64,128,192,256];values=[98.,110.,94.,106.,102.]
    c=PathContract('target',start+timedelta(days=256),100.,side=side,lot=.1,
        product_type=kind,observations=tuple(PathObservation(start+timedelta(days=d),f'f{i}',.2 if kind=='arithmetic_asian' else 1.)
            for i,d in enumerate(days)))
    h=Contract('hedge',c.expiry,100.,lot=.1)
    def fix(i):
        day=start+timedelta(days=days[i]);at=datetime.combine(day,datetime.min.time(),timezone.utc)+timedelta(hours=12)
        return SettlementFixing(day,values[i],at,at,'synthetic_path','r1')
    history=[];all_known={}
    valuation_days=[0,64,128,192,256] if on_fixing_days else [0,65,129,193,256]
    for day,spot in zip(valuation_days,[100.,108.,96.,104.,103.]):
        current=start+timedelta(days=day);at=datetime.combine(current,datetime.min.time(),timezone.utc)+timedelta(hours=16)
        newly={f'f{i}':fix(i) for i,d in enumerate(days) if d<=day and f'f{i}' not in all_known}
        all_known.update(newly)
        q=Quote(spot,spot,at,at)
        if day==256:
            s=Snapshot(f's{day}',at,q,{},RiskBasis(current,(current+timedelta(days=32),current+timedelta(days=64)),(-.2,-.1,0.,.1,.2)),at,'synthetic_path',settlement_fixings=newly)
        else:
            distances=sorted(d-day for d in days if d>day)
            while len(distances)<4:
                cuts=[0,*distances];left,right=max(zip(cuts,cuts[1:]),key=lambda ab:ab[1]-ab[0])
                distances.append((left+right)//2);distances.sort()
            times=np.array([0.,*[d/365 for d in distances]])
            payload=deepcopy(p);payload['valuation_date']=current.isoformat();payload['market']['spot']=spot
            g=payload['model']['local_variance_grid'];b=payload['model']['reporting_iv_basis'];g['time_nodes']=times.tolist()
            b['maturity_nodes']=times[2::2].tolist()
            maturities=tuple(current+timedelta(days=d) for d in distances[1::2])
            payload['risk']['vega_kt']['maturity_nodes']=[d.isoformat() for d in maturities]
            rows=np.array(g['values']).reshape(g['shape']);nodes=np.array(g['log_forward_moneyness_nodes'])
            strikes=spot*np.exp(b['log_forward_moneyness_nodes']);states=source_marginals(spot,times,nodes,rows,64);quotes=[]
            for index in [2,4]:
                src,w,hats,_=states[index-2]
                prices=w@_black(src[:,None],strikes[None,:],(hats@rows[index-1])[:,None]*(times[index]-times[index-1]))[0]
                lo,hi=np.full(5,1e-8),np.full(5,4.)
                for _ in range(90):
                    mid=(lo+hi)/2;below=_black(spot,strikes,times[index]*mid*mid)[0]<prices
                    lo,hi=np.where(below,mid,lo),np.where(below,hi,mid)
                quotes.extend(((lo+hi)/2).tolist())
            b['implied_volatilities']=quotes
            s=Snapshot(f's{day}',at,q,{'hedge':Quote(6.,6.,at,at)},RiskBasis(current,maturities,tuple(b['log_forward_moneyness_nodes'])),at,'synthetic_path',settlement_fixings=all_known)
            target=deepcopy(payload);target['product']=native_product(c,s)
            hedge=deepcopy(payload);hedge['product']=dict(type='european_vanilla',underlying_id=1,currency_id=2,strike=100.,notional=1.,side=dict(type='call'),expiry=c.expiry.isoformat())
            s=replace(s,pricing_requests={'target':json.dumps(target),'hedge':json.dumps(hedge)},settlement_fixings=newly)
        history.append(s)
    return c,h,history


class PathLedgerTests(unittest.TestCase):
    evidence={}

    @classmethod
    def setUpClass(cls):
        cls.provider=provider();cls.cases=[fixture(kind,side) for kind in ['arithmetic_asian','fixed_lookback'] for side in ['call','put']]
        cls.evidence=dict(panels=[],replays=[])

    def test_native_rolls_histories_and_all_strategies_settle_resume(self):
        from rust_pricing_backtest.path_history import resolve
        from rust_pricing_backtest.ensemble import ScenarioResults,aggregate_scenarios
        for c,h,history in self.cases:
            cfg=BacktestConfig(ridge=1e-4,market_shock_log_moneyness=(-.2,0.,.2))
            portfolio=[Position('target',-1.,5.)]
            results=compare_strategies(cfg,history,[c,h],portfolio,self.provider)
            book={}
            for s in history[:-1]:
                s=resolve(s,[c,h],book)
                risk=self.provider.evaluate(s,c);audit=self.provider.gamma_audit(s,c)
                self.assertEqual(audit['status'],'passed')
                p=json.loads(s.pricing_requests['target']);p['risk'].pop('vega_kt');p['risk']['vega']=False
                raw=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate()
                self.assertEqual((risk.price,risk.delta),(raw.value,raw.delta_raw))
                self.evidence['panels'].append(dict(kind=c.product_type,side=c.side,at=s.at.isoformat(),risk=data(risk),audit=audit))
            expected=2. if c.product_type=='arithmetic_asian' and c.side=='call' else 0. if c.product_type=='arithmetic_asian' else 10. if c.side=='call' else 6.
            for strategy,whole in results.items():
                self.assertEqual(whole.status,'completed');self.assertTrue(whole.summary['fully_settled'])
                self.assertEqual(whole.summary['risk_validation_status'],'numerically_validated_multi_step_path')
                target=next(x for x in whole.settlements if x['contract_id']=='target' and x['event'] in ['fixed','fixed_zero'])
                self.assertAlmostEqual(target['settlement']['payoff_per_unit'],expected)
                self.assertEqual(len(target['settlement']['observations']),5)
                partial=run_backtest(replace(cfg,strategy=strategy),history,[c,h],portfolio,self.provider,stop_after=2)
                resumed=run_backtest(**load_input(input_json(whole)),provider=self.provider,checkpoint=partial.checkpoint)
                self.assertEqual(resumed.to_json(),whole.to_json())
                for doc in [json.loads(whole.to_json()),json.loads(input_json(whole)),whole.checkpoint]:
                    self.assertEqual(validate_document(doc)['status'],'valid')
                self.evidence['replays'].append(dict(kind=c.product_type,side=c.side,strategy=strategy,summary=whole.summary,
                    fills=len(whole.fills),payoff=expected,resume_exact=True))
            ensemble=aggregate_scenarios([ScenarioResults('path','one','synthetic',results)])
            self.assertEqual(ensemble.manifest['schema_version'],4)
            self.assertEqual(validate_document(json.loads(ensemble.to_json()))['status'],'valid')
            if c.product_type=='fixed_lookback' and c.side=='call':
                self.evidence['sample_result']=json.loads(results['vega_kt'].to_json())
                self.evidence['sample_request']=json.loads(input_json(results['vega_kt']))
            print(f"Accepted {c.product_type} {c.side}: four dated valuations and four settled strategies",flush=True)

    def test_six_interval_known_histories_use_the_same_native_price_and_delta(self):
        for kind in ['arithmetic_asian','fixed_lookback']:
            for side in ['call','put']:
                p=model(6,samples=4096);day=date.fromisoformat(p['valuation_date'])
                dates=[day+timedelta(days=round(t*365)) for t in p['model']['local_variance_grid']['time_nodes'][1:]]
                dates.insert(0,day-timedelta(days=1))
                c=PathContract('six',dates[-1],100.,side=side,product_type=kind,
                    observations=tuple(PathObservation(d,f'q{i}',1/7 if kind=='arithmetic_asian' else 1.) for i,d in enumerate(dates)))
                at=datetime.combine(day,datetime.min.time(),timezone.utc)+timedelta(hours=16)
                fixed_at=at-timedelta(days=1)
                f=SettlementFixing(dates[0],98.,fixed_at,fixed_at,'synthetic_path','r1')
                b=p['risk']['vega_kt'];s=Snapshot('six-'+kind+side,at,Quote(100.,100.,at,at),{},
                    RiskBasis(day,tuple(date.fromisoformat(d) for d in b['maturity_nodes']),tuple(b['log_forward_moneyness_nodes'])),
                    at,'synthetic_path',settlement_fixings={'q0':f})
                p['product']=native_product(c,s);s=replace(s,pricing_requests={'six':json.dumps(p)})
                risk=self.provider.evaluate(s,c);audit=self.provider.gamma_audit(s,c)
                self.assertEqual(audit['status'],'passed')
                p['risk'].pop('vega_kt');p['risk']['vega']=False
                raw=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate()
                self.assertEqual((risk.price,risk.delta),(raw.value,raw.delta_raw))
                self.evidence['panels'].append(dict(kind=kind,side=side,intervals=6,at=at.isoformat(),risk=data(risk),audit=audit))

    def test_native_contract_mismatch_and_old_scope_are_rejected(self):
        c,h,history=self.cases[0];s=history[0]
        with self.assertRaisesRegex(BacktestError,'unsupported_path_scope'):
            RustPricingProvider(numerical_scope='multi_step_gamma').evaluate(s,c)
        for field in ['weight','fixing']:
            p=json.loads(s.pricing_requests['target'])
            if field=='weight':p['product']['observations'][0]['weight']+=.01
            else:p['product']['observations'][0]['value']['fixing']+=1
            with self.assertRaisesRegex(BacktestError,'path_request_mismatch'):
                provider().evaluate(replace(s,pricing_requests={'target':json.dumps(p)}),c)
        p=json.loads(s.pricing_requests['target']);p['product']['observations'][1]['value']=dict(type='known',fixing=110.)
        with self.assertRaisesRegex(BacktestError,'path_request_mismatch'):
            provider().evaluate(replace(s,pricing_requests={'target':json.dumps(p)}),c)


if __name__=='__main__':unittest.main()
