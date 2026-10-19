"""Seal bounded precision evidence without changing the current provider acceptance."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed,sha
from check_portfolio_market import sources,write
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.portfolio_market_risk import implementation_identity


def run(directory,output):
    directory=directory.resolve()
    require(not output.exists(),'precision_seal_output','fresh output required')
    prior_path=ROOT/'design/validation/vega-kt-rqmc-provider-v0.1.json'
    prior=sealed(prior_path);identity=implementation_identity()
    require(prior['status']=='passed' and prior['failed']==0 and prior['implementation_identity']==identity
        and all(sha(ROOT/n)==v for n,v in prior['source_sha256'].items()),
        'precision_seal_source','production and previous accepted dependencies must be unchanged')
    policy_path=ROOT/'fixtures/backtesting/rqmc-portfolio-precision-holdout-v0.1.json'
    policy=json.loads(policy_path.read_text());seeds=policy['seeds']
    require(len(seeds)==len(set(seeds))==2 and not set(seeds)&set(policy['excluded_prior_seeds']),
        'precision_seal_seeds','two distinct unused seeds required')
    evidence={};summaries=[];all_passed=True
    for seed in seeds:
        path=directory/f'holdout-{seed}.json';panel=sealed(path)
        require(panel['policy']==policy and panel['seed']==seed and panel['universe']=='full'
            and panel['instruments']==30 and panel['source_sha256']==sources()
            and panel['implementation_identity']==identity
            and panel['dependencies_sha256'][str(policy_path.resolve())]==sha(policy_path),
            'precision_seal_panel','holdout coverage, source or frozen policy differs')
        require(all(sha(n)==v for n,v in panel['dependencies_sha256'].items()),
            'precision_seal_source','checker/input/reference changed')
        passed=panel['status']=='passed' and panel['failed']==0 and len(panel['checks'])==993
        decision=None
        decision_path=directory/f'decision-{seed}/report.json'
        if decision_path.exists():
            decision=sealed(decision_path)
            require(decision['panel_sha256']==sha(path) and decision['implementation_identity']==identity,
                'precision_seal_decision','decision belongs to another native panel')
            passed=passed and decision['status']=='passed' and decision['processed_snapshots']==1 and decision['orders']==1
            require(decision['schemas']=={'input.json':'valid','result.json':'valid','checkpoint.json':'valid'},
                'precision_seal_schema','saved decision documents failed validation')
            for key in ('held','proposed'):
                if decision[key] is not None:
                    require(decision[key]['limit']==10 and decision[key]['max_bucket_standard_error']<=10,
                        'precision_seal_gate','unrelaxed portfolio gate required')
        else:
            passed=False
        summaries.append(dict(seed=seed,status='passed' if passed else 'failed',
            native_panel_status=panel['status'],checks=len(panel['checks']),failed=panel['failed'],
            error=panel.get('error'),held_standard_error=panel.get('max_bucket_standard_error'),
            proposed_standard_error=None if decision is None or decision['proposed'] is None else decision['proposed']['max_bucket_standard_error'],
            solver=None if decision is None else decision['solver'],elapsed_seconds=panel['elapsed_seconds']))
        all_passed=all_passed and passed
    exploratory=[]
    for name in ('bridge-held.json','direct-held.json'):
        r=sealed(directory/name)
        require(r['source_sha256']==sources() and r['implementation_identity']==identity,
            'precision_seal_source','exploratory source differs')
        exploratory.append({k:r.get(k) for k in ('setting','seed','status','failed','max_bucket_standard_error','elapsed_seconds','error')})
    budget=sealed(directory/'budget.json');require(budget['status']=='passed','precision_seal_budget','budget check required')
    # Bind only completed numerical inputs; a concurrently written log or this
    # summary's own output must never be part of its evidence manifest.
    paths=[directory/name for name in ('bridge-held.json','direct-held.json','budget.json')]
    paths += [directory/f'holdout-{seed}.json' for seed in seeds]
    paths += [directory/f'decision-{seed}/report.json' for seed in seeds if (directory/f'decision-{seed}/report.json').exists()]
    evidence={str(p.relative_to(ROOT)):sha(p) for p in paths}
    added=[Path(__file__),ROOT/'scripts/probe_rqmc_portfolio_precision.py',ROOT/'scripts/check_rqmc_precision_decision.py',
        ROOT/'scripts/check_rqmc_precision_budget.py',policy_path,ROOT/'fixtures/backtesting/rqmc-portfolio-precision-exploration-v0.1.json']
    report=dict(document_kind='rqmc_initial_annual_portfolio_precision',schema_version=1,date='2026-10-13',
        status='passed' if all_passed else 'failed',scope='first annual snapshot, all_rows, thirty European calls, two predeclared seeds',
        implementation_identity=identity,native_binary_sha256=prior['native_binary_sha256'],
        prior_provider_acceptance_sha256=sha(prior_path),production_unchanged=True,
        current_provider_policy_promoted=False,annual_completed=False,
        source_sha256=dict(sources(),**{str(p.relative_to(ROOT)):sha(p) for p in added}),
        evidence_sha256=evidence,policy=policy,holdouts=summaries,exploratory=exploratory,budget=budget['workload_estimates'],
        historical_failures_preserved=True,
        limitations=['The usual provider still accepts the earlier 256/128-point policies only.',
            'Decision replay uses sealed, exact-request native panels and is explicitly distinguished from a normal-provider run.',
            'Only the initial annual market is covered; no subsequent-day, path-product or annual-throughput acceptance.',
            'Standard errors use independent scramble means and exclude calibration/discretization error.'])
    report['report_hash']=fingerprint(report);write(output,report)
    print(json.dumps(dict(status=report['status'],holdouts=summaries)))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.directory,a.output);raise SystemExit(r['status']!='passed')
