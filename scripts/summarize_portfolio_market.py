"""Fail-closed coverage and legacy numerical comparison for the market matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_market import sources,POLICY,write,policy_for
from rust_pricing_backtest.domain import fingerprint,require


def aggregate(directory):
    policy=json.loads(POLICY.read_text());frozen=sources();reports=[];comparisons=[]
    for case in policy['cases']:
        p=directory/(case+'.json');r=json.loads(p.read_text())
        require(r['report_hash']==fingerprint({k:v for k,v in r.items() if k!='report_hash'}),'market_integrity','changed shard')
        require(r['status']=='passed' and all(c['passed'] for c in r['checks']) and r['case']==case
                and r['policy']==policy_for(case) and r['source_sha256']==frozen,'market_coverage','passing current matching shards required')
        q=len(r['observations'][0]['report']['coordinate_order'])
        require(len(r['checks'])==43+60*q and len(r['calibration_audits'])==4*q+2,
                'market_coverage','incomplete numerical coverage')
        require([o['seed'] for o in r['observations']]==policy['seeds'] and len(r['pnl'])==6
                and {(x['mode'],x['scale']) for x in r['pnl']}=={(m,s) for m in ('spot','iv','combined') for s in (1.,.5)},
                'market_coverage','incomplete seeds/P&L coverage')
        require(np.asarray(r['price_record']['differences']).shape==(2,8,10,q),'market_coverage','incomplete independent prices')
        for o in r['observations']:
            joint=o['report']
            require(joint['report_hash']==fingerprint({k:v for k,v in joint.items() if k!='report_hash'})
                    and len(joint['instrument_order'])==10 and set(joint['audits'])==set(joint['instrument_order']),
                    'market_integrity','incomplete joint report')
        if case.startswith('regression_'):
            n=int(case.rsplit('_',1)[1])
            old_path=ROOT/f'benchmark-results/vega-kt-current-raw-basis/{n}.json'
            old=json.loads(old_path.read_text())
            require(old['status']=='passed','market_reference','passing old raw matrix report required')
            for o in r['observations']:
                for cid,a in o['report']['audits'].items():
                    before=next(x['audit'] for x in old['observations'] if (x['product'],x['seed'])==(cid,o['seed']))
                    excluded={'request_fingerprint','plan_fingerprint','sampling_unit_values_layout'}
                    left={k:v for k,v in before.items() if k not in excluded};right={k:v for k,v in a.items() if k not in excluded}
                    comparison=dict(case=case,seed=o['seed'],instrument=cid,numerical_audit_exact=left==right)
                    comparisons.append(comparison)
                    require(left==right,'market_reference','legacy native numerical audit changed: '+str(comparison))
        reports.append(r)
    require(len({r['native_binary_sha256'] for r in reports})==1,'market_integrity','native binaries differ')
    import rust_pricing as rp
    binary=hashlib.sha256(Path(getattr(rp,'rust_pricing',rp).__file__).read_bytes()).hexdigest()
    require(binary==reports[0]['native_binary_sha256'],'market_integrity','current native binary differs')
    return dict(document_kind='portfolio_market_acceptance_matrix',schema_version=1,status='passed',failed=0,
        cases=policy['cases'],panels=sum(len(o['report']['instrument_order']) for r in reports for o in r['observations']),
        checks=sum(len(r['checks']) for r in reports),pnl_scenarios=sum(len(r['pnl']) for r in reports),
        source_sha256=frozen,native_binary_sha256=reports[0]['native_binary_sha256'],legacy_comparisons=comparisons,
        shard_sha256={str((directory/(r['case']+'.json')).relative_to(ROOT)):hashlib.sha256((directory/(r['case']+'.json')).read_bytes()).hexdigest() for r in reports},
        full_market_iv_bucket_acceptance='not_established',policy=policy)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protected=[a.directory/(case+'.json') for case in json.loads(POLICY.read_text())['cases']]
    protected += [ROOT/n for n in sources()]+[POLICY,Path(__file__)]
    for target in (a.output,a.output.with_suffix(a.output.suffix+'.tmp')):
        require(all(target.resolve()!=p.resolve() and not (target.exists() and p.exists() and target.samefile(p)) for p in protected),
                'market_output','aggregate output aliases an input or implementation')
    r=aggregate(a.directory.resolve());write(a.output,r)
    print(json.dumps({k:r[k] for k in ('status','panels','checks','pnl_scenarios','failed')}))
if __name__=='__main__':main()
