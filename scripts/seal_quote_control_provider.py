"""Seal bounded normal-provider quote-control acceptance and unchanged legacy math."""
import argparse
import ast
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sha,sealed
from check_portfolio_market import sources,write
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.portfolio_market_risk import implementation_identity

PRIOR=ROOT/'design/validation/vega-kt-rqmc-provider-v0.1.json'
ARCHIVE=ROOT/'benchmark-results/vega-kt-portfolio-precision/source.tar.gz'
PACKAGE='backtesting/rust_pricing_backtest/'
CHANGED={PACKAGE+n for n in ('__main__.py','multi_step_provider.py','portfolio_market_risk.py','rust_adapter.py')}


def dump(node):return ast.dump(node,include_attributes=False)


def functions(text):
    return {node.name:node for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}


def check_legacy(prior):
    changed={n for n,v in prior['source_sha256'].items() if sha(ROOT/n)!=v}
    require(changed==CHANGED|{'scripts/variance_basis_runtime.py','scripts/test_variance_basis_runtime.py'},
            'control_seal_source','unexpected prior source change: '+str(changed))
    with tarfile.open(ARCHIVE) as packed:
        old={n:packed.extractfile(n).read() for n in CHANGED}
        old_test=packed.extractfile('scripts/test_variance_basis_runtime.py').read().decode()
    new_test=(ROOT/'scripts/test_variance_basis_runtime.py').read_text()
    normalized_test=new_test.replace("]), 64)", "]), 63)").replace(
        "'backtesting/rust_pricing_backtest/multi_step_extension.py',\n                     'backtesting/rust_pricing_backtest/quote_control.py', runtime.RUNNER,",
        "'backtesting/rust_pricing_backtest/multi_step_extension.py', runtime.RUNNER,")
    require(ast.dump(ast.parse(normalized_test))==ast.dump(ast.parse(old_test)),
            'control_seal_legacy','runtime tests changed beyond the new module binding')
    import hashlib
    require(all(hashlib.sha256(v).hexdigest()==prior['source_sha256'][n] for n,v in old.items()),
            'control_seal_archive','archived production source differs')
    before=functions(old[PACKAGE+'multi_step_provider.py'].decode())['evaluate']
    after=functions((ROOT/(PACKAGE+'multi_step_provider.py')).read_text())['evaluate']
    for node in ast.walk(after):
        if isinstance(node,ast.Try):
            node.body=[s for s in node.body if not(isinstance(s,ast.If) and 'rqmc_quote_control_512x24' in ast.unparse(s.test))]
    require(dump(before)==dump(after),'control_seal_legacy','legacy individual-risk calculation changed')
    def methods(text):
        return {f.name:f for f in functions(text)['RustPricingProvider'].body if isinstance(f,ast.FunctionDef)}
    before=methods(old[PACKAGE+'rust_adapter.py'].decode());after=methods((ROOT/(PACKAGE+'rust_adapter.py')).read_text())
    require(all(dump(after[n])==dump(f) for n,f in before.items() if n not in ('__init__','joint_market_risk')),
            'control_seal_legacy','adapter validation/cache boundary changed')
    for node in ast.walk(after['joint_market_risk']):
        if isinstance(node,ast.Try):
            node.body=[s for s in node.body if not(isinstance(s,ast.If) and 'rqmc_quote_control_512x24' in ast.unparse(s.test))]
    require(dump(before['joint_market_risk'])==dump(after['joint_market_risk']),
            'control_seal_legacy','legacy joint entry path changed')
    before=functions(old[PACKAGE+'portfolio_market_risk.py'].decode());after=functions((ROOT/(PACKAGE+'portfolio_market_risk.py')).read_text())
    require(all(dump(after[n])==dump(f) for n,f in before.items() if n!='evaluate_portfolio'),
            'control_seal_legacy','calculation helpers changed')
    def native_calls(f):
        return sorted(dump(node) for node in ast.walk(f) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                      and node.func.attr in ('compile','evaluate_vega_kt_multi_step_gamma_chain_audit','evaluate'))
    require(native_calls(before['evaluate_portfolio'])==native_calls(after['evaluate_portfolio']),
            'control_seal_legacy','native arguments or gates loosened')
    def joint_tail(f):
        start=next(i for i,n in enumerate(f.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cube' for t in n.targets))
        return [dump(n) for n in f.body[start:]]
    require(joint_tail(before['evaluate_portfolio'])==joint_tail(after['evaluate_portfolio']),
            'control_seal_legacy','legacy joint covariance/net-risk math changed')
    return sorted(changed)


def run(directory,output):
    require(not output.exists(),'control_seal_output','fresh record required')
    prior=sealed(PRIOR);require(prior['status']=='passed' and prior['failed']==0,'control_seal_prior','prior acceptance required')
    changed=check_legacy(prior)
    import rust_pricing as rp
    binary=sha(getattr(rp,'rust_pricing',rp).__file__)
    require(binary==prior['native_binary_sha256'],'control_seal_native','native binary changed')
    pilot=sealed(directory/'pilot/report.json');comparison=sealed(directory/'comparison.json');wheel=sealed(directory/'packaged/report.json')
    require(all(r['status']=='passed' and r['failed']==0 and r['source_sha256']==sources() for r in (pilot,comparison,wheel)),
            'control_seal_evidence','passing current-source evidence required')
    require(pilot['processed_snapshots']==2 and pilot['production_provider'] and pilot['fresh_provider_resume']
            and pilot['failed_native_resume_atomic'] and pilot['joint_did_not_repeat_native_audits']
            and pilot['native_unique_requests']==[30,31],
            'control_seal_pilot','native moving-market resume/rollback/reuse not established')
    if pilot.get('storage_recovery'):
        import rust_pricing as rp
        from copy import deepcopy
        from rust_pricing_backtest.quote_control import virtual_request,CONTROL_ID
        saved_input=json.loads((directory/'pilot/first/input.json').read_text())
        requests={cid:json.loads(raw) for cid,raw in saved_input['market_history'][1]['pricing_requests'].items()}
        requests[CONTROL_ID]=virtual_request(requests[min(requests)])
        native=sealed(directory/'pilot/prefill-1.json')
        audits={a['request_fingerprint']:a for a in native['native_audits'].values()}
        for request in requests.values():
            ordinary=deepcopy(request);ordinary['risk'].pop('vega_kt',None);ordinary['risk']['vega']=False
            compiled=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(ordinary)),worker_threads=4,reduction_block_size=256)
            require(compiled.request_fingerprint in audits and compiled.plan_fingerprint==audits[compiled.request_fingerprint]['plan_fingerprint'],
                    'control_seal_recovery','retained native plan differs from the actual compiled plan')
        require(pilot['storage_recovery']['complete_joint_report_identical']
                and pilot['storage_recovery']['additional_native_gamma_evaluations']==0
                and pilot['recovery_checker_sha256']==sha(ROOT/'scripts/recover_quote_control_pilot.py'),
                'control_seal_recovery','storage reconstruction is not strictly source/evidence bound')
    require(comparison['independent_native_panels']==60 and comparison['absolute_tolerances_unchanged'],
            'control_seal_comparison','sixty independent price comparisons required')
    require(all(wheel[k] for k in ('packaged_full_run_byte_identical','packaged_partial_resume_byte_identical',
        'packaged_complete_resume_byte_identical','changed_estimator_resume_rejected')),'control_seal_wheel','packaged execution/resume failed')
    log=(directory/'companion-final-tests.log').read_text()
    require('Ran 294 tests' in log and log.rstrip().endswith('OK'),'control_seal_tests','passing companion tests required')
    runtime_log=(directory/'runtime-tests.log').read_text()
    require('Ran 9 tests' in runtime_log and runtime_log.rstrip().endswith('OK'),
            'control_seal_tests','passing current runtime dependency/continuation tests required')
    failed_path=ROOT/'design/validation/vega-kt-holdout-v0.1.json'
    require(sha(failed_path)==prior['prior_failed_holdout_sha256'] and sealed(failed_path)['status']=='failed',
            'control_seal_history','earlier failed holdout was altered')
    raw_path=ROOT/'benchmark-results/vega-kt-portfolio-precision/raw-summary.json'
    require(sealed(raw_path)['status']=='failed','control_seal_history','raw sampling design failure must remain explicit')
    failure_root=ROOT/'benchmark-results/vega-kt-quote-control-provider'
    failed_pilot=sealed(failure_root/'failure.json')
    repaired_probe=sealed(failure_root/'failure-512x24-probe.json')
    require(failed_pilot['status']=='failed' and repaired_probe['status']=='path_gamma_chain_passed'
            and sha(failure_root/'failure-512x24-probe.json')==failed_pilot['diagnostic_512x24_probe_sha256'],
            'control_seal_repair','same-seed failed-case repair evidence required')
    old_input=json.loads((failure_root/'pilot/first/input.json').read_text())
    request=json.loads(old_input['market_history'][0]['pricing_requests']['hedge-08'])
    request['risk'].pop('vega_kt',None);request['risk']['vega']=False
    request['engine']['points_per_scramble']=512;request['engine']['scramble_count']=24
    require(repaired_probe['request']==request,'control_seal_repair','failed request changed beyond the sampling allocation')
    old_control=sealed(ROOT/'design/validation/vega-kt-quote-control-v0.1.json')
    require(old_control['status']=='passed' and old_control['failed']==0,'control_seal_history','prior control candidate missing')
    current=sources();current.update({n:sha(ROOT/n) for n in prior['source_sha256']})
    extras=[Path(__file__),ROOT/'scripts/check_quote_control_provider.py',ROOT/'scripts/check_quote_control_wheel.py',
        ROOT/'scripts/compare_quote_control_provider.py',ROOT/'scripts/recover_quote_control_pilot.py',ROOT/'backtesting/tests/test_quote_control.py',ROOT/'backtesting/README.md']
    current.update({str(p.relative_to(ROOT)):sha(p) for p in extras})
    evidence={str(p.relative_to(ROOT)):sha(p) for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.whl','.gz')
        and p.name not in ('completion-checks.json','runtime-tests.log','seal.log','source.tar.gz')}
    r=dict(document_kind='market_quote_control_provider_acceptance',schema_version=1,date='2026-10-13',status='passed',failed=0,
        native_binary_sha256=binary,implementation_identity=implementation_identity(),source_sha256=current,evidence_sha256=evidence,
        prior_acceptance_sha256=sha(PRIOR),prior_source_archive_sha256=sha(ARCHIVE),
        prior_control_candidate_sha256=sha(ROOT/'design/validation/vega-kt-quote-control-v0.1.json'),changes=changed,
        new_production_module=PACKAGE+'quote_control.py',native_source_unchanged=True,
        legacy_individual_risk_and_joint_covariance_math_unchanged=True,
        legacy_tests_retained=288,companion_tests=294,runtime_tests=9,independent_native_panels=60,
        independent_checks=len(comparison['checks']),oracle_refinement_checks=1202,
        sampling_policy='rqmc_quote_control_512x24',normal_provider_integrated=True,
        full_book_seed=pilot['seed'],packaged_replay_seed=wheel['seed'],
        tested_dates=2,actual_contracts=30,hedge_candidates=20,held=pilot['held'],proposed=pilot['proposed'],limit=10.,
        control_strikes=pilot['control_strikes'],native_unique_requests=pilot['native_unique_requests'],
        fresh_provider_resume=True,failed_native_resume_atomic=True,packaged_results_byte_identical=True,
        native_replay_elapsed_seconds=None,annual_completed=False,storage_recovery=pilot.get('storage_recovery'),
        timing='not reported as throughput; original native and reconstruction timings are separate',
        earlier_failures_retained=True,earlier_failed_holdout_sha256=sha(failed_path),raw_sampling_failure_sha256=sha(raw_path),
        failed_1024x12_pilot_sha256=sha(failure_root/'failure.json'),
        same_seed_failed_case_repaired=True,failed_case_repair_sha256=sha(failure_root/'failure-512x24-probe.json'),
        full_market_iv_bucket_acceptance='not_established',
        limitations=['Only six-interval European unit calls with zero rates/dividends and all_rows calibration are eligible.',
            'The bounded two-date synthetic pilot is not annual hedge effectiveness or throughput acceptance.',
            'Sampling uncertainty excludes calibration/discretization bias and Delta uncertainty.',
            'Prior raw sampling-policy failures and earlier 8/120 pseudo-MC failures remain unchanged.'])
    r['report_hash']=fingerprint(r);write(output,r);return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.directory.resolve(),a.output);print(json.dumps(dict(status=r['status'],checks=r['independent_checks'],panels=r['independent_native_panels'])))
