"""Run/resume the declared custom-basis matrix and retain coarse-step failures."""
import argparse
import json
from pathlib import Path
from check_custom_variance_basis import ROOT,KINDS,run,aggregate
from check_custom_variance_basis_refinement import price_run,merge


def write(path,report):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-directory',type=Path,required=True)
    a=p.parse_args();out=a.output_directory
    bp=ROOT/'fixtures/backtesting/custom-variance-basis-policy-v0.1.json'
    pp=ROOT/'fixtures/backtesting/custom-variance-basis-refinement-policy-v0.1.json'
    policy=json.loads(bp.read_text());initial=[];refined=[]
    for n in policy['intervals']:
        for kind in KINDS:
            b=out/f'{n}-{kind}.json';q=out/f'{n}-{kind}-prices.json'
            if not b.exists():write(b,run(policy,bp,b.with_suffix('.partial.json'),n,kind))
            if not q.exists():write(q,price_run(pp,q.with_suffix('.partial.json'),n,kind))
            initial.append(b);refined.append(q)
    # Full source/binary/policy and matrix checks also cover resumed reports.
    baseline=aggregate(initial,policy,bp);write(out/'initial-report.json',baseline)
    result=merge(pp,out/'initial-report.json',refined);write(out/'report.json',result)
    print(json.dumps(dict(status=result['status'],panels=len(result['observations']),checks=len(result['checks']),
        initial_failed=sum(not x['passed'] for x in baseline['checks']),failed=sum(not x['passed'] for x in result['checks']))))
    return int(result['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
