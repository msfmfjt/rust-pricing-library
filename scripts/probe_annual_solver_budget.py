"""Repeat the saved annual pilot with only the solver iteration budget increased."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from benchmark_accepted_replay import TimedProvider
from check_portfolio_holdout import sealed,sha
from rust_pricing_backtest import RustPricingProvider,run_backtest
from rust_pricing_backtest.domain import require,fingerprint,data
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.portfolio_market_risk import implementation_identity


def run(original,output):
    require(not output.exists(),'solver_probe_output','fresh output required')
    prior=sealed(original/'report.json')
    require(prior['status']=='failed' and prior['failure']['code']=='solver_not_converged'
            and prior['implementation_identity']==implementation_identity(),
            'solver_probe_input','current implementation and saved convergence failure required')
    raw=(original/'backtest/input.json').read_text();args=load_input(raw)
    # Wire input sorts the master by id; restore the benchmark's original list
    # order solely for its workload identity (the engine canonicalizes both).
    order=[f'target-{i:02d}' for i in range(prior['policy']['target_contracts'])]
    order += [f'hedge-{i:02d}' for i in range(prior['policy']['hedge_candidates'])]
    master={c.contract_id:c for c in args['contracts']}
    require(set(master)==set(order),'solver_probe_input','saved contract universe changed')
    args['contracts']=[master[cid] for cid in order]
    workload=fingerprint({k:args[k] for k in ('history','contracts','portfolio')})
    require(workload==prior['workload_hash'],'solver_probe_input','saved annual workload changed')
    old=args['config'];require(old.solver_iterations==10_000,'solver_probe_input','expected original solver budget')
    args['config']=replace(old,solver_iterations=100_000)
    difference=[key for key,value in data(old).items() if data(args['config'])[key]!=value]
    require(difference==['solver_iterations'],'solver_probe_input','only iteration budget may change')
    script_hash=sha(__file__);provider=RustPricingProvider(numerical_scope='multi_step_extension_put_gamma',
        variance_extension='all_rows',worker_threads=prior['policy']['worker_threads'])
    output.mkdir(parents=True);timed=TimedProvider(provider,output/'events.jsonl');before=time.perf_counter()
    result=run_backtest(**args,provider=timed,on_error='return',stop_after=prior['policy']['pilot_sessions'])
    ran=time.perf_counter();result.write(output/'backtest');saved=time.perf_counter()
    require(prior['implementation_identity']==implementation_identity() and script_hash==sha(__file__),
            'solver_probe_source','implementation changed during replay')
    require(all(len(d['candidates'])==prior['policy']['hedge_candidates'] for d in result.decisions),
            'solver_probe_coverage','candidate universe changed')
    report=dict(document_kind='annual_solver_budget_probe',schema_version=1,status=result.status,failure=result.failure,
        role='same_objective_same_tolerance_increased_iteration_budget',original_report_sha256=sha(original/'report.json'),
        original_input_sha256=sha(original/'backtest/input.json'),implementation_identity=implementation_identity(),
        checker_sha256=script_hash,timing_provider_sha256=sha(ROOT/'scripts/benchmark_accepted_replay.py'),
        workload_hash=workload,changed_config_fields=difference,old_iterations=old.solver_iterations,
        new_iterations=args['config'].solver_iterations,solver_tolerance=old.solver_tolerance,
        ridge=old.ridge,provider=timed.identity,planned_snapshots=len(args['history']),processed_snapshots=len(result.ledger),
        completed_decisions=len(result.decisions),annual_workload_completed=result.status=='completed',
        replay_seconds=ran-before,write_seconds=saved-ran,total_seconds=saved-before,
        individual_risk_seconds=sum(e['seconds'] for e in timed.events if e['kind']=='individual_risk'),
        joint_risk_seconds=sum(e['seconds'] for e in timed.events if e['kind']=='joint_risk'),
        individual_risk_calls=sum(e['kind']=='individual_risk' for e in timed.events),joint_risk_calls=sum(e['kind']=='joint_risk' for e in timed.events),
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        output_bytes=sum(p.stat().st_size for p in (output/'backtest').rglob('*') if p.is_file()),
        timing_environment_note='Other native validation ran concurrently; not isolated throughput.',
        performance_acceptance='not_assessed_no_approved_budget',annual_runtime_extrapolated=False,
        solver_reports=[d.get('solver') for d in result.decisions],summary=result.summary)
    report['report_hash']=fingerprint(report)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','processed_snapshots','replay_seconds','failure')}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();r=run(a.original,a.output);raise SystemExit(r['status']=='failed')
