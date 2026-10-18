"""Compare current normal-provider quote-controlled books to independent prices."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed,sha
from check_portfolio_market import sources,write
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.portfolio_market_risk import check_report,reweight_portfolio,sampling_covariance
from rust_pricing_backtest.numerics import threshold_check


def run(directory,output,*,prefix=False):
    directory=directory.resolve()
    require(not output.exists(),'control_comparison_output','fresh output required')
    evidence_path=directory/('pilot/prefill-0.json' if prefix else 'pilot/report.json')
    pilot=sealed(evidence_path)
    require(pilot['status']=='passed' and pilot['source_sha256']==sources(),'control_comparison_source','current completed pilot required')
    saved_path=directory/('pilot/first/result.json' if prefix else 'pilot/resumed/result.json')
    saved=json.loads(saved_path.read_text() if saved_path.exists() else gzip.decompress(saved_path.with_suffix(saved_path.suffix+'.gz').read_bytes()).decode())
    require(saved['status']=='partial' and len(saved['risks'])==(1 if prefix else 2),
            'control_comparison_input','exact committed one/two-day prefix required')
    if prefix:
        pilot['seed']=saved['risks'][0]['joint_market']['report']['instrument_engines']['target-00']['master_scramble_seed']
    # JSON results store committed artifacts at the document root.
    checks=[];evidence={};panels=[]
    def check(name,observed,limit,**ctx):checks.append(threshold_check(name,float(observed),float(limit),**ctx))
    for day,row in enumerate(saved['risks']):
        oracle_path=directory/f'oracle-{day}.json';oracle=sealed(oracle_path)
        require(oracle['status']=='passed' and oracle['source_sha256']==sources(), 'control_comparison_source','current independent oracle required')
        report=row['joint_market']['report'];ids=report['instrument_order'];n=report['independent_sampling_units']
        snapshot=saved['manifest']['market_history'][day]
        requests={cid:json.loads(snapshot['pricing_requests'][cid]) for cid in ids}
        check_report(report,requests)
        require(ids==oracle['instrument_order'] and {cid:fingerprint({k:v for k,v in p.items() if k!='engine'}) for cid,p in requests.items()}
            ==oracle['engine_independent_request_hashes'],'control_comparison_request','independent oracle must use the exact dated market/payoffs')
        diff=np.array(oracle['differences']);bp=np.array(oracle['prices']);precision=oracle['precision']
        means=diff.mean(axis=1);se=diff.std(axis=1,ddof=1)/np.sqrt(diff.shape[1])
        for j,cid in enumerate(ids):
            a=report['audits'][cid];item=report['instruments'][cid]
            check('raw_native_gate',not(a['status']=='path_gamma_chain_passed' and a['recovery_passed'] and a['validation_passed']),0,day=day,instrument=cid)
            check('ordinary_price_vs_independent',abs(item['price']-bp[:,j].mean()),
                6*np.hypot(a['native_price']['standard_error'],bp[:,j].std(ddof=1)/np.sqrt(len(bp)))+precision['absolute_tolerance'],day=day,instrument=cid)
            for b,bump in enumerate(precision['iv_bumps']):
                for k,value in enumerate(item['buckets']):
                    check('controlled_Gamma_vs_independent_price_difference',abs(value-means[b,j,k]),
                        6*np.hypot(item['bucket_standard_errors'][k],se[b,j,k])+precision['absolute_tolerance'],day=day,instrument=cid,coordinate=k,bump=bump)
            panels.append(dict(day=day,instrument=cid,seed=pilot['seed']))
        cube=np.array(report['sampling_unit_values'])
        for label,gate in [('held',row['joint_market']['held']),('proposed',saved['decisions'][day]['joint_market'])]:
            quantities=gate['quantities'];weights=np.array([quantities[cid] for cid in ids])
            net=reweight_portfolio(report,requests,quantities);reference=np.einsum('bsjq,j->bsq',diff,weights)
            raw=np.einsum('niq,i->nq',cube,weights)
            check('net_covariance_direct_samples',np.max(abs(sampling_covariance(raw)/n-np.array(net['net_price_and_market_mean_covariance']))),1e-9,day=day,book=label)
            for b,bump in enumerate(precision['iv_bumps']):
                for k,value in enumerate(net['net_buckets']):
                    samples=reference[b,:,k]
                    check('net_controlled_Gamma_vs_independent',abs(value-samples.mean()),
                        6*np.hypot(net['net_bucket_standard_errors'][k],samples.std(ddof=1)/np.sqrt(len(samples)))+precision['absolute_tolerance']*sum(abs(weights)),
                        day=day,book=label,coordinate=k,bump=bump)
            check('unchanged_net_sampling_error_gate',max(net['net_bucket_standard_errors']),10.,day=day,book=label)
        evidence[str(oracle_path.relative_to(ROOT))]=sha(oracle_path)
    failed=sum(not c['passed'] for c in checks)
    r=dict(document_kind='market_quote_control_provider_comparison',schema_version=1,status='failed' if failed else 'passed',failed=failed,
        checks=checks,panels=panels,independent_native_panels=len(panels),source_sha256=sources(),checker_sha256=sha(__file__),
        pilot_sha256=sha(evidence_path),oracle_sha256=evidence,comparison_scope='first_committed_date' if prefix else 'two_committed_dates',
        absolute_tolerances_unchanged=True,annual_completed=False)
    r['report_hash']=fingerprint(r);write(output,r)
    print(json.dumps(dict(status=r['status'],failed=failed,checks=len(checks),panels=len(panels))))
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix',action='store_true')
    a=p.parse_args();r=run(a.directory,a.output,prefix=a.prefix);raise SystemExit(r['status']!='passed')
