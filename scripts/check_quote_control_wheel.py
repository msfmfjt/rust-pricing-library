"""Two moving dates: API continuation versus packaged CLI full run and resume."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_quote_control_provider import provider,inputs
from check_portfolio_holdout import sha
from check_portfolio_market import sources,write
from rust_pricing_backtest import run_backtest,BacktestError,RustPricingProvider
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.validation import validate_document


def run(original,output,wheel):
    require(not output.exists(),'control_wheel_output','fresh output required')
    frozen=sources();own=sha(__file__);helper=sha(ROOT/'scripts/check_quote_control_provider.py');output.mkdir(parents=True)
    args=inputs(original,100151);ids={'target-00','hedge-10'}
    args['contracts']=[c for c in args['contracts'] if c.contract_id in ids]
    args['portfolio']=[p for p in args['portfolio'] if p.contract_id in ids]
    args['history']=[replace(s,pricing_requests={cid:raw for cid,raw in s.pricing_requests.items() if cid in ids},
        option_quotes={cid:q for cid,q in s.option_quotes.items() if cid in ids}) for s in args['history'][:2]]
    p=provider();first=run_backtest(**args,provider=p,stop_after=1,on_error='return');first.write(output/'partial')
    require(first.status=='partial','control_wheel_api',str(first.failure))
    api=run_backtest(**load_input((output/'partial/input.json').read_text()),provider=p,checkpoint=first.checkpoint,on_error='return')
    api.write(output/'api');require(api.status=='completed','control_wheel_api',str(api.failure))
    require(api.risks[1]['joint_market']['report']['control_variate']['request']['product']['strike']!=100.,
            'control_wheel_roll','next-day control should not equal the existing hedge strike')
    changed=RustPricingProvider(numerical_scope='multi_step_extension_gamma',variance_extension='all_rows',
                               sampling_policy='rqmc_256x16',worker_threads=4)
    try:run_backtest(**args,provider=changed,checkpoint=first.checkpoint)
    except BacktestError as e:require(e.code=='checkpoint_mismatch','control_wheel_identity',str(e))
    else:raise AssertionError('changed estimator resumed a checkpoint')
    package=ROOT/'backtesting/rust_pricing_backtest'
    files=[f for f in package.rglob('*') if f.is_file() and f.suffix in ('.py','.json')]
    with tempfile.TemporaryDirectory(prefix='quote-control-wheel-') as tmp,zipfile.ZipFile(wheel) as z:
        for f in files:require(z.read(str(f.relative_to(package.parent)))==f.read_bytes(),'control_wheel_source','stale wheel')
        z.extractall(tmp);env=dict(os.environ,PYTHONPATH=tmp)
        for label,checkpoint in [('full',None),('resume',output/'partial/checkpoint.json'),('completed',output/'api/checkpoint.json')]:
            command=[sys.executable,'-m','rust_pricing_backtest',str((output/'api/input.json').resolve()),
                     '--output',str((output/label).resolve()),'--validated-extension','all_rows','--rqmc-quote-control','--worker-threads','4']
            if checkpoint:command+=['--checkpoint',str(checkpoint.resolve())]
            with (output/(label+'.log')).open('w') as log:
                done=subprocess.run(command,cwd=tmp,env=env,stdout=log,stderr=subprocess.STDOUT)
            require(done.returncode==0,'control_wheel_cli','packaged CLI failed: '+label)
            require((output/label/'result.json').read_bytes()==(output/'api/result.json').read_bytes(),
                    'control_wheel_cli','packaged CLI differs from API: '+label)
            for name in ('input.json','result.json','checkpoint.json'):
                require(validate_document(json.loads((output/label/name).read_text()))['status']=='valid','control_wheel_schema',name)
            print(json.dumps(dict(mode=label,status='passed')),flush=True)
    require(sources()==frozen and sha(__file__)==own and sha(ROOT/'scripts/check_quote_control_provider.py')==helper,'control_wheel_source','source changed during tests')
    r=dict(document_kind='market_quote_control_wheel_acceptance',schema_version=1,status='passed',failed=0,
        wheel_sha256=sha(wheel),source_sha256=frozen,checker_sha256=own,helper_sha256=helper,seed=100151,
        packaged_full_run_byte_identical=True,packaged_partial_resume_byte_identical=True,
        packaged_complete_resume_byte_identical=True,changed_estimator_resume_rejected=True,
        source_members_checked=len(files),python_members=sum(f.suffix=='.py' for f in files),schema_members=sum(f.suffix=='.json' for f in files),
        api_result_sha256=sha(output/'api/result.json'),original_input_sha256=sha(original))
    r['report_hash']=fingerprint(r);write(output/'report.json',r)
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('input','output','wheel'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.input,a.output,a.wheel)
