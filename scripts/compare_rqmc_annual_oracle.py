"""Compare a retained native annual panel to independent recalibrated prices."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed,sha
from check_portfolio_market import write
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.numerics import threshold_check
from rust_pricing_backtest.serialization import load_input


def run(oracle_path,pilot,output):
    require(not output.exists(),'annual_compare_output','fresh output required')
    oracle=sealed(oracle_path);require(oracle['status']=='passed','annual_compare_oracle','passing independent oracle required')
    joint_path=pilot/'accepted-benchmark-000-joint.json';joint=sealed(joint_path)
    history=load_input((pilot/'backtest/input.json').read_text())['history'];snapshot=history[0]
    requests={cid:json.loads(raw) for cid,raw in snapshot.pricing_requests.items()}
    require(joint['instrument_order']==oracle['instrument_order'] and
            {cid:fingerprint({k:v for k,v in p.items() if k!='engine'}) for cid,p in requests.items()}==oracle['engine_independent_request_hashes'],
            'annual_compare_input','market, payoff or risk requests differ')
    require(joint['request_hashes']=={cid:fingerprint(p) for cid,p in requests.items()},'annual_compare_input','native report request mismatch')
    differences=np.array(oracle['differences']);prices=np.array(oracle['prices']);precision=oracle['precision'];checks=[]
    means=differences.mean(axis=1);se=differences.std(axis=1,ddof=1)/np.sqrt(len(prices))
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    for j,cid in enumerate(oracle['instrument_order']):
        native=joint['audits'][cid]
        check('native_runtime_gate',native['status']!='path_gamma_chain_passed',0,instrument=cid)
        check('price_vs_independent',abs(native['native_price']['mean']-prices[:,j].mean()),
              6*np.hypot(native['native_price']['standard_error'],prices[:,j].std(ddof=1)/np.sqrt(len(prices)))+precision['absolute_tolerance'],instrument=cid)
        for k,value in enumerate(native['estimate']):
            for b,bump in enumerate(precision['iv_bumps']):
                check('Gamma_vs_independent_price_difference',abs(value['mean']-means[b,j,k]),
                      6*np.hypot(value['standard_error'],se[b,j,k])+precision['absolute_tolerance'],instrument=cid,coordinate=k,bump=bump)
    r=dict(document_kind='rqmc_annual_independent_comparison',schema_version=1,
        status='passed' if all(c['passed'] for c in checks) else 'failed',checks=checks,
        failed=sum(not c['passed'] for c in checks),native_panels=len(prices[0]),
        oracle_sha256=sha(oracle_path),joint_sha256=sha(joint_path),checker_sha256=sha(__file__),
        native_implementation_identity=joint['implementation_identity'],
        scope='first synthetic annual snapshot only; all_rows, 30 European calls, 6 intervals')
    r['report_hash']=fingerprint(r);write(output,r)
    print(json.dumps(dict(status=r['status'],failed=r['failed'],checks=len(checks))))
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.oracle,a.pilot,a.output);raise SystemExit(r['status']!='passed')
