"""Evidence integrity and annual workload boundaries, without expensive sampling."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
import check_portfolio_holdout as holdout
import benchmark_accepted_replay as benchmark
import probe_portfolio_holdout_failures as probes
from rust_pricing_backtest.domain import BacktestError,fingerprint


class HoldoutEvidenceTests(unittest.TestCase):
    def test_probe_budget_failure_is_saved_and_later_settings_continue(self):
        import rust_pricing as rp
        policy,precision,requests,previous,provenance=holdout.inputs('regression_4')
        class FakePlan:
            calls=0
            @classmethod
            def compile(cls,*args,**kwargs):return cls()
            def evaluate_vega_kt_multi_step_gamma_chain_audit(self,*args,**kwargs):
                FakePlan.calls+=1
                if FakePlan.calls==1:raise rp.PricingError('test continuation budget exceeded')
                return dict(status='path_gamma_chain_passed',recovered_directions=[dict(mean=0.,standard_error=1e8) for _ in range(10)])
        body=dict(status='failed',case='regression_4',provenance=provenance,
            panels=[dict(instrument='asian_call',seed=policy['seeds'][0],accepted=False)])
        body['report_hash']=fingerprint(body)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'failed.json';output=Path(directory)/'probe.json'
            source.write_text(json.dumps(body))
            with patch.object(rp,'PricingPlan',FakePlan):probes.run(source,output)
            result=holdout.sealed(output)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['probes']),3)
            self.assertEqual(result['probes'][0]['native_error'],'test continuation budget exceeded')
            self.assertFalse(result['probes'][0]['probe_passed'])
            self.assertTrue(all(p['probe_passed'] for p in result['probes'][1:]))
            self.assertEqual(json.loads(source.read_text()),body)

    def test_changed_sealed_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'evidence.json'
            body={'status':'passed','values':[1.,2.]};body['report_hash']=fingerprint(body)
            path.write_text(json.dumps(body));self.assertEqual(holdout.sealed(path),body)
            body['values'][0]=3.;path.write_text(json.dumps(body))
            with self.assertRaisesRegex(BacktestError,'corrupted report'):holdout.sealed(path)

    def test_original_seeds_are_rejected(self):
        policy=json.loads(holdout.POLICY.read_text());policy['seeds']=[7,104729]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.json';path.write_text(json.dumps(policy))
            with patch.object(holdout,'POLICY',path):
                with self.assertRaisesRegex(BacktestError,'distinct unused seeds'):holdout.inputs('regression_4')

    def test_all_market_requests_match_sealed_reference_before_seed_change(self):
        for case in json.loads((ROOT/'fixtures/backtesting/portfolio-market-policy-v0.1.json').read_text())['cases']:
            policy,precision,requests,previous,provenance=holdout.inputs(case)
            self.assertEqual(len(requests),10)
            self.assertFalse(set(policy['seeds'])&set(precision['seeds']))
            self.assertEqual({cid:fingerprint(p) for cid,p in requests.items()},previous['request_hashes'])


class AnnualWorkloadTests(unittest.TestCase):
    def test_declared_universe_matches_native_inputs_and_no_expiry_inside_year(self):
        import rust_pricing as rp
        policy=json.loads(benchmark.POLICY.read_text())
        history,contracts,positions=benchmark.build_workload(policy)
        self.assertEqual((len(history),len(contracts),len(positions)),(252,30,10))
        self.assertEqual(len({s.at for s in history}),252)
        self.assertTrue(all(s.at.weekday()<5 for s in history))
        self.assertGreater(min(c.expiry for c in contracts),history[-1].at.date())
        self.assertEqual(len({(c.expiry,c.strike,c.side) for c in contracts}),30)
        for s in (history[0],history[125],history[-1]):
            self.assertEqual(s.basis.size,15)
            self.assertEqual(set(s.pricing_requests),{c.contract_id for c in contracts})
            for c in contracts:
                p=json.loads(s.pricing_requests[c.contract_id]);rp.PricingRequest.from_json(json.dumps(p))
                self.assertEqual((p['product']['underlying_id'],p['product']['currency_id']),(c.underlying_id,c.currency_id))
                self.assertEqual((p['market']['underlying_id'],p['market']['currency_id']),(c.underlying_id,c.currency_id))
                self.assertEqual(p['product']['strike'],c.strike)
                self.assertEqual(p['risk']['vega_kt']['maturity_nodes'],[d.isoformat() for d in s.basis.maturity_dates])
                self.assertEqual(p['engine']['independent_sampling_units'],2048)

    def test_timing_records_native_failure_and_does_not_return_fake_risk(self):
        class Inner:
            def evaluate(self,*args):raise BacktestError('native_test_failure','intentional')
        policy=json.loads(benchmark.POLICY.read_text());policy['sessions']=2
        history,contracts,_=benchmark.build_workload(policy)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl';provider=benchmark.TimedProvider(Inner(),path)
            with self.assertRaisesRegex(BacktestError,'intentional'):provider.evaluate(history[0],contracts[0])
            event=json.loads(path.read_text());self.assertEqual(event['status'],'failed')
            self.assertEqual(event['instrument'],contracts[0].contract_id)
            self.assertGreaterEqual(event['seconds'],0.)


if __name__=='__main__':unittest.main()
