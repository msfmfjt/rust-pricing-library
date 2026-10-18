"""Replay the saved annual workload with explicit RQMC; retain failed joint evidence."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from benchmark_accepted_replay import TimedProvider
from check_portfolio_holdout import sealed,sha
from check_portfolio_market import sources,write
from rust_pricing_backtest import RustPricingProvider,run_backtest
from rust_pricing_backtest.domain import require,fingerprint,data
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.validation import validate_document
from rust_pricing_backtest.portfolio_market_risk import implementation_identity,reweight_portfolio

POLICY=ROOT/'fixtures/backtesting/rqmc-provider-policy-v0.1.json'


class RetainedProvider(TimedProvider):
    def joint_market_risk(self,snapshot,contracts,risks):
        report=super().joint_market_risk(snapshot,contracts,risks)
        # Keep a successful native report even if the subsequent portfolio SE gate rolls back the day.
        write(self.path.parent/(snapshot.snapshot_id+'-joint.json'),report)
        return report


def run(original,output):
    require(not output.exists(),'rqmc_annual_output','fresh output directory required')
    prior=sealed(original/'report.json');policy=json.loads(POLICY.read_text())
    args=load_input((original/'backtest/input.json').read_text())
    order=[f'target-{i:02d}' for i in range(prior['policy']['target_contracts'])]
    order += [f'hedge-{i:02d}' for i in range(prior['policy']['hedge_candidates'])]
    master={c.contract_id:c for c in args['contracts']};require(set(master)==set(order),'rqmc_annual_input','universe changed')
    args['contracts']=[master[cid] for cid in order]
    require(fingerprint({k:args[k] for k in ('history','contracts','portfolio')})==prior['workload_hash'],
            'rqmc_annual_input','original market/workload changed')
    args['config']=replace(args['config'],solver_iterations=100_000)
    history=[]
    for snapshot in args['history']:
        requests={}
        for cid,raw in snapshot.pricing_requests.items():
            p=json.loads(raw);p['engine']=dict(deepcopy(policy['engine']),master_scramble_seed=policy['seeds'][0])
            requests[cid]=json.dumps(p,allow_nan=False)
        history.append(replace(snapshot,pricing_requests=requests))
    args['history']=history
    identity=implementation_identity();frozen=sources();script_hash=sha(__file__);policy_hash=sha(POLICY)
    output.mkdir(parents=True)
    provider=RetainedProvider(RustPricingProvider(numerical_scope='multi_step_extension_put_gamma',
        variance_extension='all_rows',sampling_policy='rqmc_256x16',worker_threads=4),output/'events.jsonl')
    before=time.perf_counter()
    result=run_backtest(**args,provider=provider,on_error='return',stop_after=prior['policy']['pilot_sessions'])
    ran=time.perf_counter();result.write(output/'backtest')
    require(identity==implementation_identity() and frozen==sources() and sha(__file__)==script_hash and sha(POLICY)==policy_hash,
            'rqmc_annual_source','source or policy changed during replay')
    schema={name:validate_document(json.loads((output/'backtest'/name).read_text()))['status']
            for name in ('input.json','result.json','checkpoint.json')}
    held={}
    for path in sorted(output.glob('*-joint.json')):
        report=sealed(path);snapshot=next(s for s in history if s.snapshot_id==path.name[:-len('-joint.json')])
        requests={cid:json.loads(snapshot.pricing_requests[cid]) for cid in report['instrument_order']}
        weights={cid:sum(p.quantity for p in args['portfolio'] if p.contract_id==cid)*master[cid].multiplier for cid in requests}
        net=reweight_portfolio(report,requests,weights)
        held[snapshot.snapshot_id]=max(net['net_bucket_standard_errors'])
    coverage=all(len(d['candidates'])==prior['policy']['hedge_candidates'] for d in result.decisions)
    report=dict(document_kind='rqmc_annual_pilot',schema_version=1,status=result.status,failure=result.failure,
        scope='runtime gates and synthetic two-day pilot; not independent all_rows market-Greek acceptance',
        source_sha256=frozen,implementation_identity=identity,checker_sha256=script_hash,policy_sha256=policy_hash,
        original_report_sha256=sha(original/'report.json'),original_input_sha256=sha(original/'backtest/input.json'),
        policy=policy,config=data(args['config']),provider=provider.identity,risk_validation=provider.validation,
        planned_snapshots=len(history),processed_snapshots=len(result.ledger),pilot_sessions=prior['policy']['pilot_sessions'],
        annual_workload_completed=result.status=='completed' and len(result.ledger)==len(history),
        initial_target_book_standard_errors=held,full_candidate_coverage=coverage,schemas=schema,
        committed_held_gates=[r['joint_market']['held'] for r in result.risks],
        committed_proposed_gates=[d['joint_market'] for d in result.decisions],
        replay_seconds=ran-before,individual_risk_calls=sum(e['kind']=='individual_risk' for e in provider.events),
        joint_risk_calls=sum(e['kind']=='joint_risk' for e in provider.events),summary=result.summary,
        performance_acceptance='not_assessed_no_approved_budget',timing_environment='concurrent numerical validation; not isolated throughput')
    report['report_hash']=fingerprint(report);write(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','failure','processed_snapshots','initial_target_book_standard_errors','replay_seconds')}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.original,a.output);raise SystemExit(r['status']=='failed')
