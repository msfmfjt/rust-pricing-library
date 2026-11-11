"""Run the pinned PR's original LLDB runner, with recording confined to its inferior."""
import argparse
import concurrent.futures
import json
import mmap
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
from records import HEADER, MAGIC, monitor, unpack_range, utc

def check_calibration(here,out,env):
    probe=out/'calibration'
    probe.mkdir()
    exe=probe/'known-calls'
    subprocess.run(['clang','-O0','-fno-builtin',str(here/'calibration.c'),'-o',str(exe)],check=True)
    subprocess.run(['codesign','--force','--sign','-',str(exe)],check=True)
    tag='pr206-known-calls'
    settings=env | {'PR206_EVENT_DIR':str(probe),'PR206_EVENT_ARG':tag}
    code=('from scripts.wheel_validation.native_debugger import run; from pathlib import Path; import sys; '
          'sys.exit(run(sys.argv[2:],Path(sys.argv[1]),"calibration")[0])')
    subprocess.run([sys.executable,'-c',code,str(probe),str(exe),tag],env=settings,check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    state=json.loads((probe/'native-debugger/calibration.json').read_text())
    file,=probe.glob(f'events-{state["pid"]}-*.bin')
    with file.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as buf:
        header=HEADER.unpack_from(buf)
        assert header[0]==MAGIC and header[2]==state['pid'] and state['exit_code']==0
        rows=unpack_range(buf,0,1,header[3])
    calls=[tuple(map(int,r)) for r in rows]
    alloc=next(r for r in calls if r[2]==1 and r[6]==6817)
    req=next(r for r in calls if r[2]==6 and r[4]==alloc[4] and r[6]==6841)
    returned=next(r for r in calls if r[2]==7 and r[5]==req[1] and r[4] and r[6]==6841)
    aligned=next(r for r in calls if r[2]==3 and r[4] and r[4]%256==0 and r[6]==6833)
    calloc=next(r for r in calls if r[2]==2 and r[5]==37 and r[6]==37*197 and r[4])
    for item in (returned,aligned,calloc):
        freed=next(r for r in calls if r[2]==4 and r[4]==item[4] and r[1]>item[1])
        assert any(r[2]==5 and r[4]==freed[4] and r[5]==freed[1] for r in calls)
    print('Verified malloc, calloc, aligned allocation, realloc and completed frees under LLDB',flush=True)

def run(args):
    root=Path.cwd()
    here=Path(__file__).resolve().parent
    out=Path(args.output).resolve()
    out.mkdir(exist_ok=False, parents=True)
    raw=out/'raw'
    raw.mkdir(mode=0o700)
    shim=out/'shim'
    shim.mkdir()
    binary=out/'history.dylib'
    real_lldb=shutil.which('lldb')
    subprocess.run(['clang','-Wall','-Wextra','-Werror','-arch','arm64','-mmacosx-version-min=15.0',
                    '-O2','-g','-fno-builtin','-dynamiclib',str(here/'history.c'),'-o',str(binary)],check=True)
    subprocess.run(['codesign','--force','--sign','-',str(binary)],check=True)
    lldb=shim/'lldb'
    # Set this in LLDB's embedded Python after LLDB starts. The repository's launch()
    # transports the environment to the inferior. Neither pip nor LLDB loads the dylib.
    setup="script import os; os.environ['DYLD_INSERT_LIBRARIES']=os.environ['PR206_RECORDER']"
    lldb.write_text('#!/bin/sh\nexec '+shlex.quote(real_lldb)+' --one-line-before-file '+shlex.quote(setup)+' "$@"\n')
    lldb.chmod(0o700)
    env=dict(os.environ, PATH=str(shim)+os.pathsep+os.environ['PATH'],
             PR206_RECORDER=str(binary), PR206_EVENT_DIR=str(raw),
             PR206_EVENT_ARG='scripts.wheel_validation.provider_shards',RUST_BACKTRACE='full',
             PYTHONUNBUFFERED='1')
    # Only the original test process handles test input. All recordings are metadata-only.
    command=[sys.executable,'scripts/smoke_test_wheel.py','--phase','validation',
             '--shard',args.selection,'--report-directory',str(out/'reports'),'--native-debugger','lldb']
    if args.selection=='preflight':
        check_calibration(here,out,env)
        command=[sys.executable,'-c',
                 'from pathlib import Path; import sys; from scripts.wheel_validation.native_debugger import run; '
                 'sys.exit(run(sys.argv[2:], Path(sys.argv[1]), "preflight")[0])',
                 str(out/'reports'),sys.executable,'examples/python/european_bs.py']
        env['PR206_EVENT_ARG']='examples/python/european_bs.py'
    done=threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        observer=pool.submit(monitor,raw,done)
        try:
            with (out/'pricing.log').open('w') as log:
                proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        finally:
            done.set()
        histories=observer.result()
    result_path=out/'reports/native-debugger'/f'{args.selection}.json'
    result=json.loads(result_path.read_text()) if result_path.is_file() else {}
    matching=[r for r in histories if r['pid']==result.get('pid') and r['wheel_mapped'] and r['streams']['pricing']['events']>0]
    valid=bool(len(matching)==1 and matching[0]['wheel_allocation_free_pair'] and result.get('valid_process'))
    stored={'selection':args.selection,'checked_at':utc(),'test_returncode':proc.returncode,'lldb_pid':result.get('pid'),
            'lldb_state':result.get('state'),'inferior_exit':result.get('exit_code'),'history_verified':valid,
            'histories':histories,'image_version':os.environ.get('ImageVersion'),
            'os_build':subprocess.check_output(['sw_vers','-buildVersion'],text=True).strip(),
            'lldb':subprocess.check_output([real_lldb,'--version'],text=True).strip()}
    (out/'summary.json').write_text(json.dumps(stored,indent=2)+'\n')
    print(json.dumps({k:v for k,v in stored.items() if k!='histories'}),flush=True)
    if matching: print(json.dumps(matching[0]['streams']),flush=True)
    if not valid: print('Individual pricing allocation / completed free evidence missing.',flush=True)
    return bool(proc.returncode or not valid or result.get('state')!='exited' or result.get('exit_code')!=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('selection')
    parser.add_argument('output')
    sys.exit(run(parser.parse_args()))
