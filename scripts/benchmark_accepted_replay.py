"""Measure genuine replay with unchanged numerical gates and joint covariance."""
import argparse
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from multi_step_cases import model
from rust_pricing_backtest import BacktestConfig, Contract, Position, Quote, RiskBasis, Snapshot, RustPricingProvider, run_backtest
from rust_pricing_backtest.domain import fingerprint, require
from rust_pricing_backtest.numerics import black_scholes_call
from rust_pricing_backtest.portfolio_market_risk import implied_surface, implementation_identity

POLICY = ROOT/'fixtures/backtesting/accepted-performance-policy-v0.1.json'


def build_workload(policy):
    start = date.fromisoformat(policy['valuation_date']); expiry = start+timedelta(days=policy['expiry_days'])
    targets = [Contract(f'target-{i:02d}',expiry,95.25+i,lot=.1) for i in range(policy['target_contracts'])]
    hedges = [Contract(f'hedge-{i:02d}',expiry,80.+2*i,lot=.1) for i in range(policy['hedge_candidates'])]
    template = model(policy['intervals'],samples=policy['independent_sampling_units'],seed=policy['seed'])
    history=[]; day=start
    while len(history)<policy['sessions']:
        if day.weekday()<5:
            i=len(history); at=datetime.combine(day,datetime.min.time(),timezone.utc)+timedelta(hours=16)
            spot=100*math.exp(.03*math.sin(i/23)); sigma=.2+.01*math.sin(i/31)
            remaining=(expiry-day).days
            dates=[day+timedelta(days=remaining*j//policy['intervals']) for j in range(1,policy['intervals']+1)]
            times=[0.,*[(d-day).days/365 for d in dates]]
            p=deepcopy(template);p['valuation_date']=day.isoformat();p['market']['spot']=spot
            for name in ('discount_curve','dividend_curve'):
                p['market'][name].update(times=[0.,2.],discount_factors=[1.,1.])
            grid=p['model']['local_variance_grid'];grid['time_nodes']=times
            grid['values']=(np.array(grid['values'])*(sigma/.2)**2).tolist()
            report=p['model']['reporting_iv_basis'];report['maturity_nodes']=times[2::2]
            p['risk']['vega_kt']['maturity_nodes']=[d.isoformat() for d in dates[1::2]]
            p['product'].update(expiry=expiry.isoformat())
            report['implied_volatilities']=implied_surface(p).ravel().tolist()
            basis=RiskBasis(day,tuple(dates[1::2]),tuple(report['log_forward_moneyness_nodes']))
            requests={};quotes={}
            for c in targets+hedges:
                request=deepcopy(p);request['product']['strike']=c.strike
                requests[c.contract_id]=json.dumps(request,allow_nan=False)
                # Synthetic executable quotes are not an independent pricing oracle.
                mark=black_scholes_call(spot,c.strike,remaining/365,sigma)['price']
                quotes[c.contract_id]=Quote(max(0.,mark-.01),mark+.01,at,at)
            history.append(Snapshot(f'accepted-benchmark-{i:03d}',at,Quote(spot-.005,spot+.005,at,at),
                quotes,basis,at,'synthetic_accepted_precision_workload_v1',requests))
        day+=timedelta(days=1)
    positions=[Position(c.contract_id,-1.,history[0].option_quotes[c.contract_id].mid) for c in targets]
    return history,targets+hedges,positions


class TimedProvider:
    def __init__(self,inner,path):
        self.inner=inner;self.path=path;self.events=[]

    @property
    def identity(self):return self.inner.identity

    @property
    def validation(self):return self.inner.validation

    def _time(self,kind,snapshot,call,**context):
        before=time.perf_counter();event=dict(kind=kind,snapshot_id=snapshot.snapshot_id,**context)
        try:
            result=call();event['status']='passed';return result
        except Exception as error:
            event.update(status='failed',error_type=type(error).__name__,error=str(error));raise
        finally:
            event['seconds']=time.perf_counter()-before;self.events.append(event)
            with self.path.open('a') as output:output.write(json.dumps(event,allow_nan=False)+'\n')
            print(json.dumps(event),flush=True)

    def evaluate(self,snapshot,contract):
        return self._time('individual_risk',snapshot,lambda:self.inner.evaluate(snapshot,contract),instrument=contract.contract_id)

    def joint_market_risk(self,snapshot,contracts,risks):
        return self._time('joint_risk',snapshot,lambda:self.inner.joint_market_risk(snapshot,contracts,risks),instruments=len(risks))


def run(output,*,pilot=False,checkpoint=None,timing_note='No isolation from other workstation activity was enforced.'):
    require(not output.exists(),'benchmark_output','fresh output directory required')
    policy=json.loads(POLICY.read_text());script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity=implementation_identity();policy_hash=fingerprint(policy);before=time.perf_counter()
    history,contracts,portfolio=build_workload(policy);built=time.perf_counter()
    output.mkdir(parents=True)
    provider=TimedProvider(RustPricingProvider(numerical_scope='multi_step_extension_put_gamma',
        variance_extension='all_rows',worker_threads=policy['worker_threads'],reduction_block_size=policy['reduction_block_size']),output/'events.jsonl')
    config=BacktestConfig(strategy=policy['strategy'],ridge=policy['ridge'],solver_tolerance=policy['solver_tolerance'],
        option_position_limit=20,option_trade_limit=20,max_relative_spread=policy['max_relative_spread'],min_vega=policy['min_vega'],
        max_joint_bucket_standard_error=policy['max_joint_bucket_standard_error'])
    result=run_backtest(config,history,contracts,portfolio,provider,on_error='return',
        checkpoint=json.loads(checkpoint.read_text()) if checkpoint else None,
        stop_after=policy['pilot_sessions'] if pilot else None)
    ran=time.perf_counter();result.write(output/'backtest');saved=time.perf_counter()
    require(identity==implementation_identity() and policy_hash==fingerprint(json.loads(POLICY.read_text()))
            and script_hash==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'benchmark_source', 'implementation changed during run')
    require(all(len(d['candidates'])==policy['hedge_candidates'] for d in result.decisions),
            'benchmark_workload','every completed decision must cover all hedge candidates')
    completed=len(result.ledger);annual_complete=result.status=='completed' and completed==policy['sessions']
    report=dict(document_kind='accepted_precision_replay_performance',schema_version=1,
        status=result.status,failure=result.failure,mode='pilot' if pilot else 'annual',policy=policy,policy_hash=policy_hash,
        implementation_identity=identity,checker_sha256=script_hash,provider=provider.identity,
        risk_validation=provider.validation,python=sys.version.split()[0],platform=platform.platform(),timing_environment_note=timing_note,
        processed_snapshots=completed,planned_snapshots=len(history),annual_workload_completed=annual_complete,
        workload_hash=fingerprint(dict(history=history,contracts=contracts,portfolio=portfolio)),
        completed_decisions=len(result.decisions),full_candidate_coverage_on_completed_decisions=True,
        individual_risk_calls=sum(e['kind']=='individual_risk' for e in provider.events),
        joint_risk_calls=sum(e['kind']=='joint_risk' for e in provider.events),
        workload_build_seconds=built-before,replay_seconds=ran-built,write_seconds=saved-ran,total_seconds=saved-before,
        individual_risk_seconds=sum(e['seconds'] for e in provider.events if e['kind']=='individual_risk'),
        joint_risk_seconds=sum(e['seconds'] for e in provider.events if e['kind']=='joint_risk'),
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        output_bytes=sum(p.stat().st_size for p in (output/'backtest').rglob('*') if p.is_file()),
        performance_acceptance='not_assessed_no_approved_budget',annual_runtime_extrapolated=False,
        summary=result.summary)
    report['report_hash']=fingerprint(report)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','mode','processed_snapshots','annual_workload_completed','replay_seconds','failure')}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--timing-note',default='No isolation from other workstation activity was enforced.')
    args=parser.parse_args();result=run(args.output,pilot=args.pilot,checkpoint=args.checkpoint,timing_note=args.timing_note)
    raise SystemExit(result['status']=='failed')
