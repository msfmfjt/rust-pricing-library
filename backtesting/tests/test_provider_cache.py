"""Cache lifecycle contracts with synthetic pricing; native conformance is separate."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
from threading import Barrier, Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from rust_pricing_backtest import (BacktestError, Contract, Quote, Risk, RiskBasis,
                                  RustPricingProvider, Snapshot, PricingEvidence)
from rust_pricing_backtest.domain import fingerprint


class ProviderCacheTests(unittest.TestCase):
    def setUp(self):
        start = date(2026, 9, 7)
        at = datetime.combine(start, datetime.min.time(), timezone.utc)
        dates = (start + timedelta(days=30), start + timedelta(days=60))
        basis = RiskBasis(start, dates, (-.1, .1))
        self.contracts = [Contract(f'c{i}', dates[-1], strike) for i, strike in enumerate((95., 100., 105.))]
        requests = {}
        for c in self.contracts:
            requests[c.contract_id] = json.dumps(dict(
                valuation_date=start.isoformat(),
                product=dict(type='european_vanilla', expiry=c.expiry.isoformat(), strike=c.strike,
                             notional=1., side=dict(type=c.side), underlying_id=1, currency_id=2),
                market=dict(spot=100., underlying_id=1, currency_id=2),
                model=dict(type='local_volatility', reporting_iv_basis=dict(implied_volatilities=[.2]*4),
                           local_variance_grid=dict(values=[.04]*6)),
                risk=dict(delta=True, smile_dynamics=dict(type='sticky_log_moneyness'),
                          vega_kt=dict(maturity_nodes=[d.isoformat() for d in dates],
                                       log_forward_moneyness_nodes=[-.1, .1]))))
        self.snapshot = Snapshot('s0', at, Quote(100., 100., at, at), {}, basis, at,
                                 'cache_lifecycle_fixture', requests)
        native = SimpleNamespace(__file__=__file__, PricingError=RuntimeError)
        module = patch.dict('sys.modules', {'rust_pricing': native})
        module.start(); self.addCleanup(module.stop)
        self.calls = []
        evaluation = patch('rust_pricing_backtest.multi_step_provider.evaluate', new=self.price)
        evaluation.start(); self.addCleanup(evaluation.stop)

    def price(self, provider, payload, snapshot, contract, *, _cache=None):
        key = fingerprint(dict(snapshot=snapshot, contract=contract))
        self.calls.append(key)
        _cache.native[key] = dict(audit=dict(marker=key))
        return (Risk(5., .5, (1., 2., 3., 4.), snapshot.basis.basis_id, provenance=key),
                dict(marker=key, nested=dict(values=[1., 2.])),
                PricingEvidence(key, key, contract.payment_day, 1., 'ordinary_price_and_greeks'))

    def test_evidence_shares_cache_lifetime_and_acceptance(self):
        p = self.provider(1); s = self.snapshot; a, b, _ = self.contracts
        evidence = p.pricing_evidence(s, a)
        expected = evidence
        self.assertIs(p.pricing_evidence(s, a), evidence)
        p.evaluate(s, a)
        self.assertEqual(len(self.calls), 1)
        old = weakref.ref(evidence)
        p.evaluate(s, b)
        self.assertEqual(len(p._cache.results), 1)
        # Caller-held evidence remains usable, independently of provider state.
        self.assertEqual(evidence.task, fingerprint(dict(snapshot=s, contract=a)))
        self.assertEqual(p.pricing_evidence(s, a), expected)
        self.assertEqual(len(self.calls), 3)
        del evidence, expected
        self.assertIsNone(old())
        old = weakref.ref(p.pricing_evidence(s, a))
        p.clear_cache()
        self.assertIsNone(old())
        with patch('rust_pricing_backtest.multi_step_provider.evaluate', side_effect=BacktestError('rejected', 'gate')):
            with self.assertRaisesRegex(BacktestError, 'rejected'):
                p.pricing_evidence(s, a)
        self.assertFalse(p._cache.results)

    def provider(self, capacity=2):
        return RustPricingProvider(numerical_scope='multi_step_gamma', cache_max_entries=capacity)

    def test_same_snapshot_hits_and_detached_audits(self):
        p = self.provider(); s = self.snapshot; c = self.contracts[0]
        risk = p.evaluate(s, c)
        audit = p.gamma_audit(s, c)
        audit['nested']['values'][0] = 1e20
        self.assertEqual(p.evaluate(s, c), risk)
        self.assertEqual(p.gamma_audit(s, c)['nested']['values'], [1., 2.])
        self.assertEqual(len(self.calls), 1)

    def test_entry_eviction_recomputes_risk_and_audit_together(self):
        p = self.provider(); s = self.snapshot; a, b, c = self.contracts
        p.evaluate(s, a)
        expected = p.gamma_audit(s, b)
        p.evaluate(s, a)  # a is most recently used; b will be evicted.
        p.evaluate(s, c)
        self.assertEqual(len(p._cache.results), 2)
        self.assertEqual(len(p._cache.native), 2)
        self.assertEqual(p.gamma_audit(s, b), expected)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len(p._cache.results), 2)

    def test_long_history_retains_only_one_snapshot_and_revisits_recompute(self):
        p = self.provider(); s = self.snapshot; c = self.contracts[0]
        expected = p.gamma_audit(s, c)
        old = weakref.ref(p._cache)
        for i in range(1, 101):
            current = replace(s, snapshot_id=f's{i}')
            p.evaluate(current, c)
            self.assertEqual(p._cache.snapshot_hash, fingerprint(current))
            self.assertEqual(len(p._cache.results), 1)
            self.assertEqual(len(p._cache.native), 1)
        self.assertIsNone(old())
        self.assertEqual(p.gamma_audit(s, c), expected)
        self.assertEqual(len(self.calls), 102)

    def test_clear_releases_every_cache_and_preserves_identity(self):
        p = self.provider(); s = self.snapshot; c = self.contracts[0]
        expected = p.gamma_audit(s, c); identity = p.identity
        old = weakref.ref(p._cache)
        p.clear_cache()
        self.assertIsNone(old())
        self.assertFalse(p._cache.results); self.assertFalse(p._quote_control_cache)
        self.assertIsNone(p._cache.shared)
        self.assertEqual(p.identity, identity)
        self.assertEqual(p.gamma_audit(s, c), expected)
        self.assertEqual(len(self.calls), 2)

    def test_failed_calculation_publishes_neither_risk_nor_surface(self):
        p = self.provider(); s = self.snapshot; c = self.contracts[0]
        with patch('rust_pricing_backtest.multi_step_provider.evaluate', side_effect=BacktestError('rejected', 'gate')):
            with self.assertRaisesRegex(BacktestError, 'rejected'):
                p.evaluate(s, c)
        self.assertFalse(p._cache.results)
        self.assertIsNone(p._cache.shared)
        self.assertEqual(p.evaluate(s, c), self.provider().evaluate(s, c))

    def mismatched_snapshot(self):
        requests = dict(self.snapshot.pricing_requests)
        bad = json.loads(requests['c1'])
        bad['model']['local_variance_grid']['values'][0] = .05
        requests['c1'] = json.dumps(bad)
        return replace(self.snapshot, pricing_requests=requests)

    def test_surface_guard_survives_entry_eviction(self):
        p = self.provider(1); s = self.mismatched_snapshot()
        p.evaluate(s, self.contracts[0]); p.evaluate(s, self.contracts[2])
        with self.assertRaisesRegex(BacktestError, 'surface_mismatch'):
            p.evaluate(s, self.contracts[1])
        self.assertEqual(len(self.calls), 2)

    def test_parallel_first_results_recheck_the_same_surface_before_publication(self):
        p = self.provider(); s = self.mismatched_snapshot(); barrier = Barrier(2)
        def calculate(*args, **kwargs):
            barrier.wait(timeout=10)
            return self.price(*args, **kwargs)
        with patch('rust_pricing_backtest.multi_step_provider.evaluate', new=calculate), ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(p.evaluate, s, c) for c in self.contracts[:2]]
            errors = [f.exception(timeout=10) for f in futures]
        self.assertEqual(sum(e is None for e in errors), 1)
        self.assertEqual([e.code for e in errors if e is not None], ['surface_mismatch'])
        self.assertEqual(len(p._cache.results), 1)

    def test_inflight_audit_cannot_repopulate_a_cleared_or_different_snapshot(self):
        for clear in (True, False):
            with self.subTest(clear=clear):
                p = self.provider(); s = self.snapshot; c = self.contracts[0]
                entered, release = Event(), Event()
                def calculate(*args, **kwargs):
                    if args[2].snapshot_id == s.snapshot_id:
                        entered.set()
                        if not release.wait(timeout=10):
                            raise RuntimeError('test did not release pricing')
                    return self.price(*args, **kwargs)
                with patch('rust_pricing_backtest.multi_step_provider.evaluate', new=calculate), ThreadPoolExecutor(1) as pool:
                    future = pool.submit(p.gamma_audit, s, c)
                    try:
                        self.assertTrue(entered.wait(timeout=10))
                        if clear:
                            p.clear_cache()
                        else:
                            p.evaluate(replace(s, snapshot_id='new'), c)
                        current = p._cache
                    finally:
                        release.set()
                    audit = future.result(timeout=10)
                self.assertEqual(audit['marker'], fingerprint(dict(snapshot=s, contract=c)))
                self.assertIs(p._cache, current)
                self.assertEqual(len(current.results), 0 if clear else 1)
                self.assertEqual(len(current.native), 0 if clear else 1)

    def test_invalid_capacity_fails_before_loading_native_library(self):
        for capacity in (0, -1, True, 1.5, None, '2'):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(BacktestError, 'invalid_provider_cache'):
                self.provider(capacity)
        self.assertEqual(self.provider(1).identity, self.provider(128).identity)


if __name__ == '__main__':
    unittest.main()
