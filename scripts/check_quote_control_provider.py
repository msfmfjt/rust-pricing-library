"""Normal-provider two-date pilot with a fresh provider at the saved checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from benchmark_accepted_replay import TimedProvider
from check_portfolio_holdout import sha
from check_portfolio_market import sources, write
from rust_pricing_backtest import RustPricingProvider, run_backtest, BacktestError
from rust_pricing_backtest.domain import fingerprint, require
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.validation import validate_document
from rust_pricing_backtest.portfolio_market_risk import implementation_identity
from rust_pricing_backtest.quote_control import POLICY, CONTROL_ID


def provider():
    return RustPricingProvider(numerical_scope='multi_step_extension_gamma', variance_extension='all_rows',
                               sampling_policy=POLICY, worker_threads=4)


def inputs(path, seed):
    args = load_input(path.read_text()); history = []
    args['config'] = replace(args['config'], solver_iterations=100_000)
    for snapshot in args['history']:
        requests = {}
        for cid, raw in snapshot.pricing_requests.items():
            p = json.loads(raw)
            p['engine'] = dict(type='randomized_quasi_monte_carlo', points_per_scramble=512, scramble_count=24,
                master_scramble_seed=seed, variance_reduction=dict(antithetic=True, brownian_bridge=False))
            requests[cid] = json.dumps(p, allow_nan=False)
        history.append(replace(snapshot, pricing_requests=requests))
    args['history'] = history
    return args


class RetainedProvider(TimedProvider):
    def joint_market_risk(self, snapshot, contracts, risks):
        before = len(self.inner._quote_control_cache)
        report = super().joint_market_risk(snapshot, contracts, risks)
        require(len(self.inner._quote_control_cache) == before, 'control_native_reuse', 'joint calculation repeated native audits')
        write(self.path.parent/(snapshot.snapshot_id+'-joint.json'), report)
        print(json.dumps(dict(date=snapshot.at.isoformat(), joint='passed', native_cache_entries=before)), flush=True)
        return report

    def evaluate(self, snapshot, contract):
        result = super().evaluate(snapshot, contract)
        print(json.dumps(dict(date=snapshot.at.isoformat(), instrument=contract.contract_id, status='passed')), flush=True)
        return result


def prefill(inner, snapshot, contracts, output):
    """Independent contracts in parallel; normal provider gates still run for each.

    A provider is used for one date only. Warm the shared control before workers
    start, then keep each original contract's complete Risk and native audit.
    Replay calls evaluate again, checking the same dated requests before reuse.
    """
    events = []
    def one(contract):
        start = time.perf_counter()
        try:
            inner.evaluate(snapshot, contract)
            event = dict(instrument=contract.contract_id, status='passed')
        except BacktestError as error:
            event = dict(instrument=contract.contract_id, status='failed', code=error.code, message=str(error))
        event['seconds'] = time.perf_counter()-start
        print(json.dumps(dict(prefill=snapshot.snapshot_id, **event)), flush=True)
        return event
    events.append(one(contracts[0]))
    with ThreadPoolExecutor(max_workers=3) as executor:
        events.extend(executor.map(one, contracts[1:]))
    # Diagnostic audit export only; the provider never reloads this cache.
    r = dict(document_kind='quote_control_native_prefill_audit', schema_version=1,
        status='passed' if all(e['status']=='passed' for e in events) else 'failed',
        snapshot_hash=fingerprint(snapshot), events=events, implementation_identity=implementation_identity(),
        native_audits={key:value['audit'] for key,value in inner._quote_control_cache.items()},
        source_sha256=sources(), parallel_contracts=3, reusable_provider_input=False)
    r['report_hash']=fingerprint(r);write(output,r)


def run(original, output, seed):
    require(not output.exists(), 'control_output', 'fresh evidence directory required')
    frozen = sources(); identity = implementation_identity(); script_hash = sha(__file__)
    args = inputs(original, seed); output.mkdir(parents=True)
    started = time.perf_counter()
    first_provider = RetainedProvider(provider(), output/'first-events.jsonl')
    prefill(first_provider.inner, args['history'][0], args['contracts'], output/'prefill-0.json')
    first = run_backtest(**args, provider=first_provider, stop_after=1, on_error='return')
    first.write(output/'first')
    require(first.status == 'partial' and len(first.ledger) == 1, 'control_pilot', str(first.failure))
    # Load the actual serialized input and checkpoint, with an empty native cache.
    second_provider = RetainedProvider(provider(), output/'second-events.jsonl')
    restored = load_input((output/'first/input.json').read_text())
    checkpoint = json.loads((output/'first/checkpoint.json').read_text())
    prefill(second_provider.inner, restored['history'][1], restored['contracts'], output/'prefill-1.json')
    result = run_backtest(**restored, provider=second_provider, checkpoint=checkpoint, stop_after=2, on_error='return')
    result.write(output/'resumed')
    require(result.status == 'partial' and len(result.ledger) == 2, 'control_pilot', str(result.failure))
    require(result.ledger[:1] == first.ledger and result.orders[:len(first.orders)] == first.orders,
            'control_resume', 'committed prefix changed')
    schemas = {part: {name:validate_document(json.loads((output/part/name).read_text()))['status']
        for name in ('input.json','result.json','checkpoint.json')} for part in ('first','resumed')}
    controls = [r['joint_market']['report']['control_variate'] for r in result.risks]
    require(controls[0]['request_hash'] != controls[1]['request_hash'], 'control_roll', 'dated control must change')
    for i, row in enumerate(result.risks):
        report = row['joint_market']['report']
        require(len(report['instrument_order']) == 30 and CONTROL_ID not in report['instrument_order']
                and controls[i]['request']['product']['strike'] == args['history'][i].spot.mid,
                'control_book', 'virtual control must match the current quote and stay outside the book')
        require(len(result.decisions[i]['candidates']) == 20, 'control_book', 'all hedge candidates required')
    # Failure of the underlying native operation is atomic, including on resume.
    with patch('rust_pricing_backtest.quote_control.evaluate_portfolio', side_effect=BacktestError('portfolio_gamma_failure','injected audit failure')):
        rejected = run_backtest(**restored, provider=provider(), checkpoint=checkpoint, stop_after=2, on_error='return')
    require(rejected.status == 'failed' and rejected.failure['code'] == 'portfolio_gamma_failure'
            and rejected.checkpoint == first.checkpoint and rejected.orders == first.orders and rejected.ledger == first.ledger,
            'control_rollback', 'failed native operation changed committed state')
    require(sources() == frozen and implementation_identity() == identity and sha(__file__) == script_hash,
            'control_source', 'source changed during native replay')
    r = dict(document_kind='market_quote_control_provider_pilot', schema_version=1, status='passed', failed=0,
        seed=seed, source_sha256=frozen, implementation_identity=identity, checker_sha256=script_hash,
        original_input_sha256=sha(original), processed_snapshots=2, actual_contracts=30, hedge_candidates=20,
        annual_completed=False, production_provider=True, fresh_provider_resume=True, unchanged_committed_prefix=True,
        failed_native_resume_atomic=True, joint_did_not_repeat_native_audits=True,
        control_requests=[c['request_hash'] for c in controls], control_strikes=[c['request']['product']['strike'] for c in controls],
        native_unique_requests=[len(first_provider.inner._quote_control_cache),len(second_provider.inner._quote_control_cache)],
        held=[row['joint_market']['held']['max_bucket_standard_error'] for row in result.risks],
        proposed=[d['joint_market']['max_bucket_standard_error'] for d in result.decisions],
        schemas=schemas, elapsed_seconds=time.perf_counter()-started,
        native_gates='all raw Gamma recovery and AD checks retained', sampling_error_limit=10.,
        evaluation_schedule='three independent contracts prefetched through the same normal provider; no saved audits imported',
        annual_throughput_assessed=False)
    r['report_hash'] = fingerprint(r); write(output/'report.json',r)
    print(json.dumps(r|{'source_sha256':'retained'}), flush=True)
    return r


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=100129)
    a=p.parse_args();run(a.input,a.output,a.seed)
