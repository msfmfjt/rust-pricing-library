"""Compare an extracted wheel's CLI resume against current native API evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backtesting'))
from rust_pricing_backtest.domain import fingerprint,require
from rust_pricing_backtest.validation import validate_document


def run(wheel,directory,package):
    source=[p for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
    with tempfile.TemporaryDirectory(prefix='rqmc-wheel-') as tmp,zipfile.ZipFile(wheel) as z:
        for p in source:
            name=str(p.relative_to(package.parent))
            require(z.read(name)==p.read_bytes(),'rqmc_wheel_source','stale wheel member: '+name)
        z.extractall(tmp)
        env=dict(os.environ,PYTHONPATH=tmp)
        api=directory/'native-replay/api';partial=directory/'native-replay/partial'
        for label,checkpoint in [('wheel-resume',partial/'checkpoint.json'),('wheel-complete',api/'checkpoint.json')]:
            output=directory/label;require(not output.exists(),'rqmc_wheel_output','fresh output required')
            command=[sys.executable,'-m','rust_pricing_backtest',str((api/'input.json').resolve()),
                     '--output',str(output.resolve()),'--checkpoint',str(checkpoint.resolve()),
                     '--validated-custom-basis','coupled_row_hats','--rqmc-path-16-scrambles','--worker-threads','4']
            with (directory/(label+'.log')).open('w') as log:
                completed=subprocess.run(command,cwd=tmp,env=env,stdout=log,stderr=subprocess.STDOUT)
            require(completed.returncode==0,'rqmc_wheel_cli','CLI failed: '+label)
            require((output/'result.json').read_bytes()==(api/'result.json').read_bytes(),
                    'rqmc_wheel_cli','CLI result differs from API: '+label)
            for name in ('input.json','result.json','checkpoint.json'):
                require(validate_document(json.loads((output/name).read_text()))['status']=='valid','rqmc_wheel_schema',name)
    r=dict(status='passed',wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        source_members_checked=len(source),python_members=sum(p.suffix=='.py' for p in source),
        schema_members=sum(p.suffix=='.json' for p in source),
        packaged_partial_resume_byte_identical=True,packaged_complete_resume_byte_identical=True,
        worker_threads=4,api_result_sha256=hashlib.sha256((api/'result.json').read_bytes()).hexdigest())
    r['report_hash']=fingerprint(r);(directory/'wheel-validation.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r));return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wheel',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--source-package',type=Path,default=ROOT/'backtesting/rust_pricing_backtest')
    a=p.parse_args();run(a.wheel,a.directory,a.source_package)
