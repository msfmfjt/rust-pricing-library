"""Seal complete held-out results without turning exploratory probes into acceptance."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import POLICY,sealed,sha
from check_portfolio_market import write,sources
from rust_pricing_backtest.domain import fingerprint,require


def failure_detail(panel):
    audit=panel['audit'];tol=audit['functional_tolerance'];nodes=[]
    for row in audit['rows']:
        for index,value in enumerate(row['conditional_node_differences']):
            limit=6*value['standard_error']+tol
            if abs(value['mean'])>limit:
                nodes.append(dict(variance_row=row['variance_row'],node=index,mean=value['mean'],
                    standard_error=value['standard_error'],limit=limit,ratio=abs(value['mean'])/limit))
    return dict(instrument=panel['instrument'],seed=panel['seed'],status=audit['status'],
        recovery_passed=audit['recovery_passed'],validation_passed=audit['validation_passed'],
        direction_relative_standard_error=audit['direction_relative_standard_error'],
        failing_conditional_AD_nodes=nodes)


def summarize(directory,annual_report=None,solver_probe=None):
    policy=json.loads(POLICY.read_text());base=json.loads((ROOT/policy['base_policy']).read_text())
    current=sealed(ROOT/policy['current_acceptance']);cases=[];evidence={};probes=[]
    original=sealed(ROOT/policy['reference_acceptance'])
    require(current['prior_acceptance_sha256']==sha(ROOT/policy['reference_acceptance'])
            and all(sha(ROOT/n)==v for n,v in original['shard_sha256'].items())
            and all(sha(ROOT/n)==v for n,v in current['source_sha256'].items()),
            'holdout_summary','historical acceptance or current sources changed')
    frozen=sources()
    for case in base['cases']:
        path=directory/(case+'.json');r=sealed(path)
        require(r['status'] in ('passed','failed') and r['policy']==policy
                and r['provenance']['source_sha256']==frozen
                and r['provenance']['checker_sha256']==sha(ROOT/'scripts/check_portfolio_holdout.py')
                and r['provenance']['policy_sha256']==sha(POLICY)
                and r['provenance']['native_binary_sha256']==current['native_binary_sha256'],
                'holdout_summary','incomplete or changed held-out implementation')
        expected={(seed,cid) for seed in policy['seeds'] for cid in r['instrument_order']}
        actual=[(p['seed'],p['instrument']) for p in r['panels']]
        require(len(expected)==20 and len(actual)==20 and set(actual)==expected,'holdout_summary','incomplete or duplicate panels')
        failed=sum(not c['passed'] for c in r['checks'])
        require(failed==r['failed'] and r['status']==('failed' if failed else 'passed'),
                'holdout_summary','inconsistent status')
        reference=sealed(ROOT/f'benchmark-results/vega-kt-portfolio-market/{case}.json')
        derivatives=np.array(reference['price_record']['differences']);means=derivatives.mean(axis=1)
        se=derivatives.std(axis=1,ddof=1)/np.sqrt(r['precision']['sobol_scrambles'])
        failures=[]
        for panel in r['panels']:
            if panel['accepted']:continue
            detail=failure_detail(panel);j=r['instrument_order'].index(panel['instrument']);ratios=[]
            for k,estimate in enumerate(panel['audit']['recovered_directions']):
                for b in range(len(r['precision']['iv_bumps'])):
                    limit=policy['sigma_multiplier']*np.hypot(estimate['standard_error'],se[b,j,k])+r['precision']['absolute_tolerance']
                    ratios.append(float(abs(estimate['mean']-means[b,j,k])/limit))
            detail['diagnostic_only_max_independent_price_difference_ratio']=max(ratios)
            detail['risk_released']=False;failures.append(detail)
        cases.append(dict(case=case,status=r['status'],panels=len(actual),native_passed=sum(p['accepted'] for p in r['panels']),
            checks=len(r['checks']),failed=failed,failures=failures,
            joint_portfolios_available=sum(p['status']=='available' for p in r['portfolios']),
            pnl_scenarios_checked=sum(len(p.get('pnl',[])) for p in r['portfolios'])))
        evidence[str(path.relative_to(ROOT))]=sha(path)
        probe_path=directory/(case+'-probes.json')
        if probe_path.exists():
            probe=sealed(probe_path)
            require(probe['status']=='complete' and probe['original_holdout_sha256']==sha(path)
                    and probe['checker_sha256']==sha(ROOT/'scripts/probe_portfolio_holdout_failures.py'),
                    'holdout_summary','incomplete or stale exploratory probes')
            probes.append(dict(case=case,role=probe['role'],results=[dict(instrument=p['instrument'],seed=p['seed'],
                setting=p['setting'],passed=p['probe_passed'],native_error=p['native_error'],
                native_status=p['audit']['status'] if p['audit'] else None,
                independent_price_checks_failed=sum(not c['passed'] for c in p['checks'])) for p in probe['probes']]))
            evidence[str(probe_path.relative_to(ROOT))]=sha(probe_path)
    annual=sealed(annual_report) if annual_report else None
    if annual_report:
        require(annual['implementation_identity']==current['implementation_identity']
                and annual['checker_sha256']==sha(ROOT/'scripts/benchmark_accepted_replay.py'),
                'holdout_summary','annual pilot implementation changed')
        evidence[str(annual_report.relative_to(ROOT))]=sha(annual_report)
    solver=sealed(solver_probe) if solver_probe else None
    if solver_probe:
        require(annual_report is not None and solver['original_report_sha256']==sha(annual_report)
                and solver['workload_hash']==annual['workload_hash']
                and solver['changed_config_fields']==['solver_iterations']
                and solver['checker_sha256']==sha(ROOT/'scripts/probe_annual_solver_budget.py')
                and solver['implementation_identity']==current['implementation_identity'],
                'holdout_summary','solver probe changed more than the declared budget')
        evidence[str(solver_probe.relative_to(ROOT))]=sha(solver_probe)
    extra=['scripts/check_portfolio_holdout.py','scripts/probe_portfolio_holdout_failures.py',
           'scripts/summarize_portfolio_holdout.py','scripts/benchmark_accepted_replay.py','scripts/probe_annual_solver_budget.py','scripts/test_portfolio_holdout.py',
           'fixtures/backtesting/portfolio-holdout-policy-v0.1.json','fixtures/backtesting/accepted-performance-policy-v0.1.json']
    result=dict(document_kind='portfolio_unused_seed_summary',schema_version=1,date='2026-10-13',
        status='failed' if any(c['failed'] for c in cases) else 'passed',
        cases=cases,seeds=policy['seeds'],panels=sum(c['panels'] for c in cases),
        native_passed=sum(c['native_passed'] for c in cases),checks=sum(c['checks'] for c in cases),
        failed=sum(c['failed'] for c in cases),pnl_scenarios_checked=sum(c['pnl_scenarios_checked'] for c in cases),
        native_binary_sha256=current['native_binary_sha256'],implementation_identity=current['implementation_identity'],
        original_acceptance_4038_checks_unchanged=True,independent_oracle_reused=True,independent_oracle_rerun=False,
        exploratory_probes=probes,annual_performance=annual,annual_solver_budget_probe=solver,
        source_sha256=dict(frozen,**{n:sha(ROOT/n) for n in extra}),evidence_sha256=evidence,
        limitations=['Existing six markets only, no additional market acceptance',
            'Failed native gates suppress joint risk and P&L for that entire seed/market portfolio',
            'Failure probes are in-sample diagnosis; passing probes do not promote a policy',
            'No production algorithm, minimum sampling setting or threshold changed',
            'Estimation covariance excludes calibration/discretization bias and Delta sampling error',
            'Annual completion and timing environment are reported separately; no extrapolated annual measurement'])
    result['report_hash']=fingerprint(result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--annual-report',type=Path)
    parser.add_argument('--solver-probe',type=Path)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    require(not a.output.exists() and not a.output.with_suffix(a.output.suffix+'.tmp').exists(),'holdout_output','fresh output required')
    r=summarize(a.directory.resolve(),a.annual_report.resolve() if a.annual_report else None,
                a.solver_probe.resolve() if a.solver_probe else None);write(a.output,r)
    print(json.dumps({k:r[k] for k in ('status','panels','native_passed','checks','failed','pnl_scenarios_checked')}))
