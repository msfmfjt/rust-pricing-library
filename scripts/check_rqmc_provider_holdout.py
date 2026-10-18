"""New sampling eligibility; reuse the byte-unchanged held-out numerical checker."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sha,sealed
from check_portfolio_market import sources,policy_for
from market_acceptance_cases import cases,requests
from rust_pricing_backtest.domain import require,fingerprint

POLICY=ROOT/'fixtures/backtesting/rqmc-provider-policy-v0.1.json'
KERNEL=ROOT/'scripts/check_portfolio_holdout.py'
CHANGED={f'backtesting/rust_pricing_backtest/{n}.py' for n in ('rust_adapter','multi_step_provider','__main__')}


def inputs(case_id):
    policy=json.loads(POLICY.read_text())
    require(case_id in policy['cases'],'rqmc_holdout_case','undeclared case')
    seeds=policy['seeds']
    require(len(seeds)>=2 and len(set(seeds))==len(seeds)
            and all(type(s) is int and 0<=s<2**64 for s in seeds)
            and not set(seeds)&set(policy['excluded_seeds']), 'rqmc_holdout_seed','unused seeds required')
    old=sealed(ROOT/policy['reference_acceptance']);current=sealed(ROOT/policy['current_acceptance'])
    require(old['status']==current['status']=='passed' and current['prior_acceptance_sha256']==sha(ROOT/policy['reference_acceptance']),
            'rqmc_holdout_reference','passing reference lineage required')
    changed={n for n,v in current['source_sha256'].items() if sha(ROOT/n)!=v}
    require(changed in (CHANGED, CHANGED|{'scripts/variance_basis_runtime.py'}),
            'rqmc_holdout_source','only provider/CLI eligibility and the diagnostic record pointer may change: '+str(changed))
    import rust_pricing as rp
    binary=sha(getattr(rp,'rust_pricing',rp).__file__)
    require(binary==old['native_binary_sha256']==current['native_binary_sha256'], 'rqmc_holdout_source','native binary changed')
    reference=ROOT/f'benchmark-results/vega-kt-portfolio-market/{case_id}.json'
    require(sha(reference)==old['shard_sha256'][str(reference.relative_to(ROOT))], 'rqmc_holdout_reference','reference shard changed')
    previous=sealed(reference);effective=policy_for(case_id)
    require(previous['status']=='passed' and previous['policy']==effective,'rqmc_holdout_reference','precision changed')
    instruments=requests(next(c for c in cases() if c['id']==case_id))
    for p in instruments.values():p['engine']['independent_sampling_units']=effective['samples']
    require({cid:fingerprint(p) for cid,p in instruments.items()}==previous['request_hashes'],
            'rqmc_holdout_reference','base market/payoffs differ from independent oracle')
    for p in instruments.values():p['engine']=dict(deepcopy(policy['engine']),master_scramble_seed=seeds[0])
    provenance=dict(native_binary_sha256=binary,source_sha256=sources(),
        reference_shard_sha256=sha(reference),reference_acceptance_sha256=sha(ROOT/policy['reference_acceptance']),
        current_acceptance_sha256=sha(ROOT/policy['current_acceptance']),policy_sha256=sha(POLICY),
        checker_sha256=sha(KERNEL),orchestration_sha256=sha(__file__),
        changed_source_sha256={n:sha(ROOT/n) for n in sorted(CHANGED)},numerical_dependencies_unchanged=True)
    return policy,effective,instruments,previous,provenance


def run(case_id,output):
    # Isolate globals so historical callers keep their original seed policy.
    spec=importlib.util.spec_from_file_location('_rqmc_holdout_kernel',KERNEL)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.POLICY=POLICY;module.inputs=inputs
    script_hash=sha(__file__)
    result=module.run(case_id,output)
    require(sha(__file__)==script_hash,'rqmc_holdout_source','orchestration changed during run')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();r=run(a.case,a.output)
    print(json.dumps(dict(status=r['status'],failed=r['failed'],panels=r['completed_panels'])))
    raise SystemExit(r['status']!='passed')
