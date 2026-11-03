"""Numerical release gates and real ledger replay for calibrated Gamma risks."""
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backtesting'))
sys.path.insert(0, str(ROOT/'scripts'))
from multi_step_cases import model
from rust_pricing_backtest import (BacktestConfig, BacktestError, Contract, Position, Quote,
    RiskBasis, RustPricingProvider, Snapshot, compare_strategies, run_backtest)
from rust_pricing_backtest.domain import data
from rust_pricing_backtest.discrete_smile import _black
from rust_pricing_backtest.multi_step_smile import source_marginals
from rust_pricing_backtest.serialization import input_json, load_input


def provider():
    return RustPricingProvider(numerical_scope='multi_step_gamma', worker_threads=4)


def snapshot(payload, sid='s0', contracts=None):
    day = date.fromisoformat(payload['valuation_date'])
    at = datetime.combine(day, datetime.min.time(), timezone.utc)
    b = payload['risk']['vega_kt']
    basis = RiskBasis(day, tuple(date.fromisoformat(d) for d in b['maturity_nodes']), tuple(b['log_forward_moneyness_nodes']))
    contracts = contracts or [Contract('target', basis.maturity_dates[-1], 100., lot=.01),
                              Contract('hedge', basis.maturity_dates[-1], 100., side='put', lot=.01)]
    requests = {}
    for c in contracts:
        p = deepcopy(payload)
        p['product'].update(strike=c.strike, side=dict(type=c.side))
        requests[c.contract_id] = json.dumps(p)
    spot = payload['market']['spot']
    q = Quote(spot, spot, at, at)
    return Snapshot(sid, at, q, {c.contract_id: Quote(6., 6., at, at) for c in contracts}, basis,
                    at, 'synthetic_checked_multi_step', requests), contracts


def advance(payload, days, spot):
    """Synthetic calibrated surface: preserve absolute knots and recompute quotes."""
    p = deepcopy(payload)
    start = date.fromisoformat(p['valuation_date'])
    p['valuation_date'] = (start+timedelta(days=days)).isoformat()
    p['market']['spot'] = spot
    g = p['model']['local_variance_grid']; b = p['model']['reporting_iv_basis']
    # Integer-day conversion uses exactly the native ACT/365 date convention.
    times = np.array([0., *[(round(t*365)-days)/365 for t in g['time_nodes'][1:]]])
    g['time_nodes'] = times.tolist(); b['maturity_nodes'] = times[2::2].tolist()
    nodes = np.asarray(g['log_forward_moneyness_nodes']); rows = np.asarray(g['values']).reshape(g['shape'])
    strikes = spot*np.exp(b['log_forward_moneyness_nodes'])
    states = source_marginals(spot, times, nodes, rows, 64)
    quotes = []
    for index in range(2, len(times), 2):
        source, weights, hats, _ = states[index-2]
        prices = weights@_black(source[:,None], strikes[None,:],
            (hats@rows[index-1])[:,None]*(times[index]-times[index-1]))[0]
        lo, hi = np.full(5, 1e-8), np.full(5, 4.)
        for _ in range(90):
            mid = (lo+hi)/2
            below = _black(spot, strikes, times[index]*mid*mid)[0] < prices
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        quotes.extend(((lo+hi)/2).tolist())
    b['implied_volatilities'] = quotes
    return p


class MultiStepProviderTests(unittest.TestCase):
    evidence = {}

    @classmethod
    def setUpClass(cls):
        cls.provider = provider()
        cls.panels = []
        for intervals in (4, 6):
            for seed in (7, 41):
                p = model(intervals, samples=4096, seed=seed)
                s, cs = snapshot(p, sid=f'{intervals}-step-{seed}')
                cls.panels.append((p, s, cs))
                for c in cs:
                    cls.provider.evaluate(s, c)
        cls.evidence = dict(panels=[], replays=[])

    def test_calibrated_call_put_risks_match_exact_market_vega_oracle(self):
        for p, s, contracts in self.panels:
            for c in contracts:
                r = self.provider.evaluate(s, c); a = self.provider.gamma_audit(s, c)
                market = a['market_gamma']; native = market['gamma_audit']
                self.assertEqual(a['status'], 'passed')
                self.assertEqual(native['status'], 'path_gamma_chain_passed')
                self.assertEqual(tuple(x['mean'] for x in market['estimate']), r.buckets)
                self.assertEqual(tuple(x['standard_error'] for x in market['estimate']), r.bucket_standard_errors)
                covariance = np.array(market['market_mean_covariance'])
                np.testing.assert_allclose(np.diag(covariance), np.square(r.bucket_standard_errors), rtol=1e-12)
                self.assertEqual(r.payment_date, c.expiry)
                # At a directly calibrated terminal quote, all other quote
                # derivatives vanish. Put-call parity gives the same put Vega.
                t = (c.expiry-s.basis.valuation_date).days/365
                iv = r.reporting_iv[-3]
                oracle = np.zeros(s.basis.size)
                oracle[-3] = 2*t*iv*_black(s.spot.mid, c.strike, t*iv*iv)[1]
                limits = 6*np.array(r.bucket_standard_errors)+.002
                ratios = abs(np.array(r.buckets)-oracle)/limits
                self.assertTrue(np.all(ratios <= 1), ratios)
                ordinary = json.loads(s.pricing_requests[c.contract_id])
                ordinary['risk'].pop('vega_kt'); ordinary['risk']['vega'] = False
                price = rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(ordinary)),
                    worker_threads=4, reduction_block_size=256).evaluate()
                self.assertEqual((r.price, r.delta), (price.value, price.delta_raw))
                self.evidence['panels'].append(dict(intervals=len(p['model']['local_variance_grid']['time_nodes'])-1,
                    seed=p['engine']['master_seed'], side=c.side, risk=data(r), maximum_market_vega_acceptance_ratio=float(max(ratios)),
                    audit=a))

    def test_all_four_strategies_fill_settle_serialize_and_resume_exactly(self):
        p, first, cs = self.panels[0]
        history = [first]
        for i, spot in [(1,100.1),(2,99.9)]:
            s, _ = snapshot(advance(p,i,spot), f's{i}', cs)
            history.append(s)
        expiry = cs[0].expiry
        at = datetime.combine(expiry, datetime.min.time(), timezone.utc)
        history.append(Snapshot('expiry', at, Quote(101.,101.,at,at), {},
            RiskBasis(expiry,(expiry+timedelta(days=128),expiry+timedelta(days=256)),first.basis.log_moneyness),
            at,'synthetic_settlement'))
        cfg = BacktestConfig(ridge=1e-6, market_shock_log_moneyness=(-.2,0.,.2))
        portfolio = [Position('target',-1.,self.provider.evaluate(first,cs[0]).price)]
        results = compare_strategies(cfg,history,cs,portfolio,self.provider)
        from rust_pricing_backtest.validation import validate_document
        for strategy, full in results.items():
            self.assertEqual(full.status,'completed')
            self.assertTrue(full.summary['fully_settled'])
            self.assertEqual(full.summary['risk_validation_status'],'numerically_validated_multi_step_european')
            self.assertTrue(full.settlements)
            if strategy != 'unhedged': self.assertTrue(full.fills)
            partial = run_backtest(replace(cfg,strategy=strategy),history,cs,portfolio,self.provider,stop_after=2)
            resumed = run_backtest(**load_input(input_json(full)),provider=self.provider,checkpoint=partial.checkpoint)
            self.assertEqual(resumed.to_json(),full.to_json())
            for document in [json.loads(full.to_json()),json.loads(input_json(full)),full.checkpoint]:
                self.assertEqual(validate_document(document)['status'],'valid')
            with self.assertRaisesRegex(BacktestError,'checkpoint_mismatch'):
                run_backtest(**load_input(input_json(full)),provider=RustPricingProvider(numerical_scope='p0_terminal_replication'),checkpoint=partial.checkpoint)
            self.evidence['replays'].append(dict(strategy=strategy,summary=full.summary,
                fills=len(full.fills),settlements=len(full.settlements),resume_exact=True,schemas='valid'))
        self.evidence['sample_result'] = json.loads(results['vega_kt'].to_json())
        self.evidence['sample_request'] = json.loads(input_json(results['vega_kt']))

    def test_failed_native_gate_releases_no_risk_or_orders_and_retries_cleanly(self):
        _,s,cs = self.panels[0]
        failed = deepcopy(self.provider.gamma_audit(s,cs[0])['market_gamma'])
        failed['status']='native_validation_failed';failed['estimate']=None
        fresh = provider()
        with patch('rust_pricing_backtest.multi_step_provider.evaluate_multi_step_market_gamma',return_value=failed):
            out = run_backtest(BacktestConfig(end_policy='mark_to_market'),[s],cs,
                [Position('target',-1.,6.)],fresh,on_error='return')
        self.assertEqual(out.status,'failed')
        self.assertEqual(out.failure['code'],'multi_step_gamma_acceptance_failure')
        self.assertFalse(out.orders);self.assertFalse(out.fills);self.assertFalse(out.ledger)
        self.assertFalse(fresh._cache.results);self.assertIsNone(fresh._cache.shared)
        accepted = self.provider.gamma_audit(s,cs[0])['market_gamma']
        with patch('rust_pricing_backtest.multi_step_provider.evaluate_multi_step_market_gamma',return_value=accepted):
            self.assertEqual(fresh.evaluate(s,cs[0]),self.provider.evaluate(s,cs[0]))

    def test_scope_coordinates_and_calibration_fail_closed(self):
        p,s,cs = self.panels[0]
        cases = []
        def changed(code, mutate):
            q=deepcopy(p);mutate(q);cases.append((code,q))
        changed('basis_mismatch',lambda q:q['model']['reporting_iv_basis']['maturity_nodes'].__setitem__(0,.1))
        changed('multi_step_base_calibration',lambda q:q['model']['reporting_iv_basis']['implied_volatilities'].__setitem__(0,.3))
        changed('multi_step_scope',lambda q:q['market']['discount_curve']['discount_factors'].__setitem__(1,.99))
        changed('multi_step_provider_scope',lambda q:q['engine'].__setitem__('independent_sampling_units',1024))
        changed('request_mismatch',lambda q:q['product'].__setitem__('type','arithmetic_asian'))
        for code,q in cases:
            bad=replace(s,pricing_requests={cs[0].contract_id:json.dumps(q)})
            with self.subTest(code=code),self.assertRaisesRegex(BacktestError,code):
                provider().evaluate(bad,cs[0])
        with self.assertRaisesRegex(BacktestError,'multi_step_provider_scope'):
            provider().evaluate(s,replace(cs[0],payment_date=cs[0].expiry+timedelta(days=1)))
        with self.assertRaisesRegex(BacktestError,'multi_step_provider_scope'):
            RustPricingProvider(numerical_scope='multi_step_gamma',reduction_block_size=128).evaluate(s,cs[0])

    def test_sparse_tail_precision_failure_stops_real_native_replay(self):
        from rust_pricing_backtest.multi_step_full_smile import evaluate_multi_step_market_gamma
        p=model(6,samples=4096)
        c=Contract('target',date.fromisoformat(p['product']['expiry']),200.)
        s,cs=snapshot(p,contracts=[c]);captured=[]
        def observed(*args,**kwargs):
            actual=evaluate_multi_step_market_gamma(*args,**kwargs)
            captured.append(actual)
            return actual
        fresh=provider()
        with patch('rust_pricing_backtest.multi_step_provider.evaluate_multi_step_market_gamma',side_effect=observed):
            out=run_backtest(BacktestConfig(end_policy='mark_to_market'),[s],cs,
                [Position('target',-1.,0.)],fresh,on_error='return')
        self.assertEqual(out.failure['code'],'multi_step_gamma_acceptance_failure')
        self.assertFalse(out.orders);self.assertFalse(out.fills);self.assertFalse(fresh._cache.results)
        self.assertEqual(captured[0]['status'],'recovery_failed')
        self.assertIsNone(captured[0]['estimate'])
        self.assertIsNone(captured[0]['market_mean_covariance'])
        self.evidence['real_precision_failure']=dict(failure=out.failure,orders=len(out.orders),
            fills=len(out.fills),audit=captured[0])

    def test_shared_local_grid_and_detached_audit(self):
        _,s,cs = self.panels[0]
        q=json.loads(s.pricing_requests['hedge']);q['model']['local_variance_grid']['values'][35]*=1.00001
        bad=replace(s,pricing_requests={**s.pricing_requests,'hedge':json.dumps(q)})
        fresh=provider()
        # Seed only a successful evaluation on the exact same snapshot.
        accepted=self.provider.gamma_audit(s,cs[0])['market_gamma']
        with patch('rust_pricing_backtest.multi_step_provider.evaluate_multi_step_market_gamma',return_value=accepted):
            fresh.evaluate(bad,cs[0])
        with self.assertRaisesRegex(BacktestError,'surface_mismatch'):fresh.evaluate(bad,cs[1])
        a=self.provider.gamma_audit(s,cs[0]);a['market_gamma']['estimate'][0]['mean']=1e30
        self.assertNotEqual(self.provider.gamma_audit(s,cs[0])['market_gamma']['estimate'][0]['mean'],1e30)
        with self.assertRaisesRegex(BacktestError,'unsupported_numerical_scope'):
            RustPricingProvider(diagnostic_only=True).gamma_audit(s,cs[0])

    def test_cache_eviction_preserves_native_gamma_audit_and_covariance(self):
        _, s, contracts = self.panels[0]
        fresh = RustPricingProvider(numerical_scope='multi_step_gamma', worker_threads=4, cache_max_entries=1)
        risk = fresh.evaluate(s, contracts[0])
        audit = fresh.gamma_audit(s, contracts[0])
        evidence = fresh.pricing_evidence(s, contracts[0])
        self.assertEqual(evidence.scope, 'ordinary_price_and_greeks')
        self.assertIsNone(evidence.report.vega_kt)
        self.assertEqual(evidence.report.value, risk.price)
        self.assertEqual(next(r for r in evidence.report.risks if r.name == 'delta.raw').values, [risk.delta])
        self.assertEqual(evidence.task.compile().evaluate().report.to_json(), evidence.report.to_json())
        fresh.evaluate(s, contracts[1])
        self.assertEqual(fresh.gamma_audit(s, contracts[0]), audit)
        fresh.evaluate(replace(s, snapshot_id='another-snapshot'), contracts[1])
        self.assertEqual(fresh.evaluate(s, contracts[0]), risk)
        self.assertEqual(fresh.gamma_audit(s, contracts[0]), audit)
        fresh.clear_cache()
        self.assertEqual(fresh.gamma_audit(s, contracts[0]), audit)
        self.assertEqual(len(fresh._cache.results), 1)
        self.assertEqual(fresh.pricing_evidence(s, contracts[0]).report.to_json(), evidence.report.to_json())

    def test_cli_selects_the_new_explicit_scope(self):
        from rust_pricing_backtest.__main__ import main
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'input.json';source.write_text('{}')
            result=SimpleNamespace(summary={},failure=None,status='completed',write=lambda _:None)
            with patch.object(sys,'argv',['backtest',str(source),'--output',tmp,'--validated-multi-step']), \
                    patch('rust_pricing_backtest.__main__.load_input',return_value={}), \
                    patch('rust_pricing_backtest.__main__.run_backtest',return_value=result) as run:
                self.assertEqual(main(),0)
            self.assertEqual(run.call_args.kwargs['provider'].numerical_scope,'multi_step_gamma')


if __name__=='__main__':unittest.main()
