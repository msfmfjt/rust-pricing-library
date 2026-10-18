"""Exploratory 512-point held-book test; no provider or annual acceptance promotion."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed,sha
from check_portfolio_market import sources,write
from rust_pricing_backtest.domain import require,fingerprint,BacktestError
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.portfolio_market_risk import evaluate_portfolio,implementation_identity


def run(pilot,output):
    require(not output.exists(),'held_probe_output','fresh output required')
    prior=sealed(pilot/'report.json');require(prior['failure']['code']=='joint_risk_quality','held_probe_input','expected portfolio SE failure')
    args=load_input((pilot/'backtest/input.json').read_text());snapshot=args['history'][0]
    master={c.contract_id:c for c in args['contracts']};weights={}
    for p in args['portfolio']:weights[p.contract_id]=weights.get(p.contract_id,0.)+p.quantity*master[p.contract_id].multiplier
    requests={cid:json.loads(snapshot.pricing_requests[cid]) for cid in weights}
    for p in requests.values():
        require(p['product']['type']=='european_vanilla' and p['engine']['points_per_scramble']==256,'held_probe_input','six-interval European input required')
        p['engine']['points_per_scramble']=512
    frozen=sources();identity=implementation_identity();script_hash=sha(__file__);before=time.perf_counter()
    r=dict(document_kind='rqmc_held_precision_probe',schema_version=1,
        role='exploratory_in_sample_not_independent_acceptance_no_provider_promotion',
        original_sha256=sha(pilot/'report.json'),source_sha256=frozen,implementation_identity=identity,checker_sha256=script_hash,
        changed_sampling_field='points_per_scramble',points_before=256,points_after=512,scramble_count=16,
        instruments=len(weights),quantities=weights,scope='initial held target book only; proposed hedge and annual replay untested')
    try:
        native=evaluate_portfolio(requests,weights,extension='all_rows',worker_threads=4)
        value=max(native['net_bucket_standard_errors']);limit=args['config'].max_joint_bucket_standard_error
        r.update(status='passed' if value<=limit else 'failed',native_status=native['status'],
            max_bucket_standard_error=value,limit=limit,native_report=native)
    except Exception as error:
        r.update(status='failed',error_type=type(error).__name__,error=str(error))
    require(sources()==frozen and identity==implementation_identity() and sha(__file__)==script_hash,
            'held_probe_source','source changed during probe')
    r['elapsed_seconds']=time.perf_counter()-before;r['report_hash']=fingerprint(r);write(output,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_sha256','native_report')}))
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.pilot,a.output);raise SystemExit(r['status']!='passed')
