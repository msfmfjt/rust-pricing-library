"""Accept fixed Lookback observations on valuation dates and replay prior paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]


def legacy_replay(directory,output):
    from multi_step_cases import model
    from datetime import date,timedelta
    import rust_pricing as rp
    requests=[]
    for intervals in [4,6]:
        for kind in ['call','asian','fixed_asian','lookback','lookback_past','lookback_unfixed_today']:
            p=model(intervals,'lookback' if kind.startswith('lookback') else kind,samples=4096)
            p['risk'].pop('vega_kt')
            if kind=='lookback_past':
                p['product']['monitoring_dates'].insert(0,str(date.fromisoformat(p['valuation_date'])-timedelta(days=1)))
                p['product']['historical_extremum']=110.
            if kind=='lookback_unfixed_today':p['product']['monitoring_dates'].insert(0,p['valuation_date'])
            requests.append(dict(intervals=intervals,kind=kind,payload=p))
    output.mkdir(parents=True,exist_ok=True)
    path=output/'legacy-inputs.json';path.write_text(json.dumps(requests,indent=2)+'\n')
    code='''import json,sys
sys.path.insert(0,sys.argv[1])
import rust_pricing as rp
inputs=json.load(open(sys.argv[2]));results=[]
for case in inputs:
 request=rp.PricingRequest.from_json(json.dumps(case['payload']))
 result=rp.PricingPlan.compile(request,worker_threads=4,reduction_block_size=256).evaluate()
 results.append(dict(request=request.to_json(),result=result.to_json()))
print(json.dumps(results,allow_nan=False))
'''
    env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
    old=subprocess.run([sys.executable,'-c',code,str(directory),str(path)],env=env,capture_output=True,text=True,check=True)
    previous=json.loads(old.stdout)
    for case,expected in zip(requests,previous,strict=True):
        request=rp.PricingRequest.from_json(json.dumps(case['payload']))
        result=rp.PricingPlan.compile(request,worker_threads=4,reduction_block_size=256).evaluate()
        if dict(request=request.to_json(),result=result.to_json())!=expected:
            raise AssertionError('legacy request/result changed: '+case['kind'])
    (output/'legacy-results.json').write_text(json.dumps(previous,indent=2)+'\n')
    binaries=list(directory.rglob('*.so'))+list(directory.rglob('*.pyd'))
    return dict(status='passed',cases=len(requests),full_request_result_json_exact=True,
        reference_binary_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-native',type=Path)
    args=parser.parse_args();sys.path.insert(0,str(ROOT/'tests/python'))
    import test_vega_kt_lookback_fixing_day as panel
    import test_vega_kt_path_ledger as prior
    import rust_pricing as rp
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=list((ROOT/'crates').rglob('*.rs'))+list((ROOT/'backtesting/rust_pricing_backtest').glob('*.py'))
    paths+=list((ROOT/'backtesting/rust_pricing_backtest/schemas').glob('*.json'))
    paths+=list((ROOT/'schemas').rglob('*.json'))
    paths+=list((ROOT/'scripts/wheel_validation').rglob('*.py'))
    paths += [Path(__file__),Path(panel.__file__),Path(prior.__file__),ROOT/'rust_pricing.pyi',ROOT/'Cargo.lock',
        ROOT/'backtesting/tests/test_path_history.py',ROOT/'scripts/multi_step_cases.py',
        ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json',ROOT/'scripts/smoke_test_wheel.py']
    previous=json.loads((ROOT/'design/validation/local-vol-conditional-ad-v0.1.json').read_text())
    gamma_names=[n for n in previous['source_sha256'] if n.startswith('crates/')]
    gamma_unchanged=all(sha(ROOT/n)==previous['source_sha256'][n] for n in gamma_names)
    if not gamma_unchanged:raise ValueError('Gamma implementation changed; prior evidence cannot be reused')
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in [panel.LookbackFixingDayTests,prior.PathLedgerTests])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    legacy=legacy_replay(args.legacy_native,args.output.parent) if args.legacy_native else dict(status='not_run')
    unchanged=all(sha(ROOT/n)==h for n,h in frozen.items())
    report=dict(document_kind='lookback_fixing_day_validation',schema_version=1,
        status='passed' if result.wasSuccessful() and unchanged else 'failed',
        tests=result.testsRun,failed=len(result.failures)+len(result.errors),skipped=len(result.skipped),
        source_unchanged=unchanged,source_sha256=frozen,
        native_binary_sha256=sha(Path(getattr(rp,'rust_pricing',rp).__file__)),
        previous_Gamma_source_unchanged=gamma_unchanged,previous_5904_check_matrix_reexecuted=False,
        previous_native_acceptance_sha256=sha(ROOT/'design/validation/local-vol-conditional-ad-v0.1.json'),
        evidence=panel.LookbackFixingDayTests.evidence,prior_path_evidence=prior.PathLedgerTests.evidence,
        legacy_replay=legacy,full_market_iv_bucket_acceptance='not_established')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if report['status']=='passed':args.output.with_name('input.json').write_text(json.dumps(report['evidence']['sample_request'],indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','tests','failed','source_unchanged']}))
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
