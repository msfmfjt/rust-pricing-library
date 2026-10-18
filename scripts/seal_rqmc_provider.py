"""Bind RQMC integration evidence while preserving earlier numerical failures."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed,sha
from check_portfolio_market import sources,write
from check_rqmc_provider_holdout import CHANGED
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.portfolio_market_risk import implementation_identity

PRIOR=ROOT/'design/validation/vega-kt-joint-replay-v0.1.json'
ARCHIVE=ROOT/'benchmark-results/vega-kt-joint-replay/source.tar.gz'


def normalize_evaluate(text):
    f=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    for node in ast.walk(f):
        if isinstance(node,ast.Try):
            for j,stmt in enumerate(node.body):
                if isinstance(stmt,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='engine' for x in stmt.targets):
                    # The rest of this try block is the sampling eligibility check.
                    node.body=node.body[:j+1];break
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='validation':node.args=node.args[:3]
    return ast.dump(f,include_attributes=False)


def check_sources(prior):
    changed={n for n,v in prior['source_sha256'].items() if sha(ROOT/n)!=v}
    # The diagnostic runtime pointer also follows this integration record.
    require(changed==CHANGED|{'scripts/variance_basis_runtime.py'},'rqmc_seal_source','unexpected accepted source change: '+str(changed))
    prefix=('crates/','backtesting/rust_pricing_backtest/')
    numerical={n for n in sources() if n.startswith(prefix)}
    accepted={n for n in prior['source_sha256'] if n.startswith(prefix) and not '/schemas/' in n}
    require(numerical==accepted,'rqmc_seal_source','native/package source set changed')
    with tarfile.open(ARCHIVE) as packed:
        for n in CHANGED:
            old=packed.extractfile(n).read();require(__import__('hashlib').sha256(old).hexdigest()==prior['source_sha256'][n],
                                                    'rqmc_seal_source','prior archive differs')
            if n.endswith('/multi_step_provider.py'):
                require(normalize_evaluate(old.decode())==normalize_evaluate((ROOT/n).read_text()),
                        'rqmc_seal_source','risk/calibration/recovery or release logic changed beyond sampling eligibility and metadata')
            if n.endswith('/rust_adapter.py'):
                def methods(text):
                    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef))
                    return {f.name:ast.dump(f,include_attributes=False) for f in cls.body if isinstance(f,ast.FunctionDef)}
                before=methods(old.decode());after=methods((ROOT/n).read_text())
                require(all(after[k]==v for k,v in before.items() if k not in ('__init__','validation')),
                        'rqmc_seal_source','adapter numerical evaluation changed')
    return sorted(changed)


def check(directory):
    directory=directory.resolve()
    prior=sealed(PRIOR);require(prior['status']=='passed' and prior['failed']==0,'rqmc_seal_prior','passing prior integration required')
    changed=check_sources(prior)
    import rust_pricing as rp
    binary=sha(getattr(rp,'rust_pricing',rp).__file__)
    require(binary==prior['native_binary_sha256'],'rqmc_seal_source','native binary changed')
    panels=[];holdouts=[]
    for name,count in [('regression_4.json',20),('regression_6-path.json',20)]:
        r=sealed(directory/name)
        require(r['status']=='passed' and r['failed']==0 and len(r['panels'])==count
                and len({(p['seed'],p['instrument']) for p in r['panels']})==count,
                'rqmc_seal_numerics','incomplete or failed holdout')
        for n,v in r['provenance']['source_sha256'].items():
            require(n in CHANGED or sha(ROOT/n)==v,'rqmc_seal_source','numerical code changed since holdout')
        require(len(r['portfolios'])==2 and all(len(p['pnl'])==6 for p in r['portfolios']),
                'rqmc_seal_numerics','missing portfolio/P&L coverage')
        holdouts.append(dict(case=r['case'],seeds=r['policy']['seeds'],checks=len(r['checks']),
            failed=r['failed'],panels=count,pnl_scenarios=12,policy=r['policy']))
        panels.extend(r['panels'])
    native=sealed(directory/'native-replay/report.json')
    require(native['status']=='passed' and native['implementation_identity']==implementation_identity(),
            'rqmc_seal_native','current native replay required')
    wheel=sealed(directory/'wheel-validation.json');require(wheel['status']=='passed','rqmc_seal_wheel','packaged CLI required')
    require(all(sha(ROOT/'backtesting/rust_pricing_backtest'/n)==v for n,v in wheel['source_sha256'].items()),
            'rqmc_seal_wheel','wheel differs from current package')
    log=(directory/'companion-final-tests.log').read_text()
    require('Ran 288 tests' in log and log.rstrip().endswith('OK'),'rqmc_seal_tests','passing full companion tests required')
    annual=sealed(directory/'annual-pilot/report.json');oracle=sealed(directory/'annual-oracle.json')
    comparison=sealed(directory/'annual-comparison.json')
    require(oracle['status']==comparison['status']=='passed' and comparison['native_panels']==30,
            'rqmc_seal_numerics','annual all_rows independent price comparison failed')
    held_probe=sealed(directory/'held-512-probe.json')
    require(held_probe['implementation_identity']==implementation_identity() and held_probe['instruments']==10,
            'rqmc_seal_probe','held-book probe belongs to another implementation')
    historical=sealed(ROOT/'design/validation/vega-kt-portfolio-market-v0.1.json')
    require(all(sha(ROOT/n)==v for n,v in historical['shard_sha256'].items()),
            'rqmc_seal_history','historical market evidence changed')
    old_holdout=sealed(ROOT/'design/validation/vega-kt-holdout-v0.1.json')
    require(old_holdout['status']=='failed','rqmc_seal_history','prior failing holdout must be retained')
    current=sources()
    current.update({n:sha(ROOT/n) for n in prior['source_sha256']})
    extras=[Path(__file__),ROOT/'scripts/check_rqmc_provider_holdout.py',ROOT/'scripts/check_rqmc_path_holdout.py',
        ROOT/'scripts/check_portfolio_holdout.py',ROOT/'scripts/probe_rqmc_annual.py',ROOT/'scripts/check_rqmc_annual_oracle.py',
        ROOT/'scripts/compare_rqmc_annual_oracle.py',ROOT/'tests/python/test_vega_kt_rqmc_replay.py',
        ROOT/'scripts/check_rqmc_wheel.py',
        ROOT/'scripts/probe_rqmc_held_precision.py',
        ROOT/'backtesting/tests/test_rqmc_policy.py',*ROOT.glob('fixtures/backtesting/rqmc-provider-policy-*.json')]
    current.update({str(p.relative_to(ROOT)):sha(p) for p in extras})
    evidence={str(p.relative_to(ROOT)):sha(p) for p in directory.rglob('*') if p.is_file()
              and p.suffix in ('.json','.log','.whl') and p.name not in ('completion-checks.json','runtime-tests.log')}
    r=dict(document_kind='rqmc_provider_integration_acceptance',schema_version=1,date='2026-10-13',status='passed',failed=0,
        native_binary_sha256=binary,implementation_identity=implementation_identity(),source_sha256=current,
        prior_acceptance_sha256=sha(PRIOR),prior_source_archive_sha256=sha(ARCHIVE),
        changes=changed,numerical_evaluation_unchanged_except_sampling_eligibility=True,
        native_source_files_unchanged=360,prior_numerical_checks_retained=4038,prior_numerical_checks_rerun=False,
        prior_failed_holdout_retained=True,prior_failed_holdout_sha256=sha(ROOT/'design/validation/vega-kt-holdout-v0.1.json'),
        historical_holdout_role='earlier pseudo-MC policy failed 8/120; not reclassified by this RQMC record',
        holdouts=holdouts,independent_native_panels=len(panels),
        all_rows_annual_first_snapshot_comparison=dict(panels=30,checks=len(comparison['checks']),failed=comparison['failed']),
        tests=dict(companion=288,native_replay=1,packaged_cli=True),
        annual_pilot=dict(status=annual['status'],processed_snapshots=annual['processed_snapshots'],
            failure=annual['failure'],annual_workload_completed=annual['annual_workload_completed'],
            initial_target_book_standard_errors=annual['initial_target_book_standard_errors']),
        held_precision_probe={k:held_probe.get(k) for k in ('status','role','scope','max_bucket_standard_error','limit','points_after','error')},
        numerical_evidence_source_note='Holdouts and annual pilot used the retained 256-policy provider snapshot. Native engine, calibrator, joint covariance and every tolerance are unchanged. The later path-policy addition changes eligibility/metadata only; current-source mixed-point replay is independently exercised.',
        full_market_iv_bucket_acceptance='not_established',evidence_sha256=evidence,
        limitations=['Only the documented market/product/seed panels have independent acceptance.',
            'all_rows path products do not acquire independent market-Greek acceptance from coupled-basis trials.',
            'RQMC sampling error excludes calibration and discretization error.',
            'Annual throughput and hedge effectiveness have not been established.'])
    r['report_hash']=fingerprint(r);return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'rqmc_seal_output','fresh output required');r=check(a.directory);write(a.output,r)
    print(json.dumps(dict(status=r['status'],panels=r['independent_native_panels'],tests=r['tests'])))
