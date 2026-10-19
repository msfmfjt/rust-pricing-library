"""Independently check the frozen quote-control design on unobserved seed panels."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed,sha
from check_portfolio_market import sources,write
from market_quote_control import apply_control
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.numerics import threshold_check
from rust_pricing_backtest.portfolio_market_risk import implementation_identity,sampling_covariance


def run(panel_path,pilot,oracle_path,policy_path,freeze_path,output):
    require(not output.exists(),'quote_control_output','fresh output required')
    raw=sealed(panel_path);policy=json.loads(policy_path.read_text());freeze=sealed(freeze_path)
    require(raw['universe']=='full' and raw['instruments']==30 and 'native_report' in raw
        and raw['seed'] in policy['holdout_seeds'] and raw['setting']==policy['native_setting'],
        'quote_control_panel','declared full native holdout required')
    require(all(c['passed'] for c in raw['checks'] if c['check']!='held_bucket_standard_error'),
        'quote_control_native','control cannot hide a native or independent-price failure')
    require(raw['source_sha256']==sources() and raw['implementation_identity']==implementation_identity()
        and all(sha(path)==digest for path,digest in freeze['frozen_sha256'].items())
        and freeze['panels_not_yet_written'] is True,
        'quote_control_source','frozen design/source or unobserved-panel declaration differs')
    args=load_input((pilot/'backtest/input.json').read_text());snapshot=args['history'][0]
    requests={}
    for cid,engine in raw['native_report']['instrument_engines'].items():
        p=json.loads(snapshot.pricing_requests[cid]);p['engine']=engine;requests[cid]=p
    native=apply_control(raw['native_report'],requests,policy)
    oracle=sealed(oracle_path);ids=native['instrument_order'];count=native['independent_sampling_units']
    require(oracle['status']=='passed' and ids==oracle['instrument_order'] and all(
        fingerprint({k:v for k,v in p.items() if k!='engine'})==oracle['engine_independent_request_hashes'][cid]
        for cid,p in requests.items()),'quote_control_oracle','exact independent market/payoff reference required')
    prices=np.array(oracle['prices']);diff=np.array(oracle['differences']);oracle_count=len(prices)
    precision=oracle['precision'];checks=[]
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    for j,cid in enumerate(ids):
        audit=native['audits'][cid];item=native['instruments'][cid]
        check('raw_native_runtime_gate',audit['status']!='path_gamma_chain_passed',0,instrument=cid)
        check('unchanged_price_vs_independent',abs(item['price']-prices[:,j].mean()),
            6*np.hypot(audit['native_price']['standard_error'],prices[:,j].std(ddof=1)/np.sqrt(oracle_count))+precision['absolute_tolerance'],instrument=cid)
        for b,bump in enumerate(precision['iv_bumps']):
            for k,value in enumerate(item['buckets']):
                reference=diff[b,:,j,k]
                check('controlled_Gamma_vs_independent_price_difference',abs(value-reference.mean()),
                    6*np.hypot(item['bucket_standard_errors'][k],reference.std(ddof=1)/np.sqrt(oracle_count))+precision['absolute_tolerance'],
                    instrument=cid,coordinate=k,bump=bump)
    weights=np.array([native['quantities'][cid] for cid in ids]);control=ids.index(policy['control_instrument'])
    known=np.array(native['control_variate']['expected_gradient'])
    for b,bump in enumerate(precision['iv_bumps']):
        reference=np.einsum('niq,i->nq',diff[b],weights)
        for k,value in enumerate(native['net_buckets']):
            check('controlled_held_vs_independent_price_difference',abs(value-reference[:,k].mean()),
                6*np.hypot(native['net_bucket_standard_errors'][k],reference[:,k].std(ddof=1)/np.sqrt(oracle_count))
                +sum(abs(weights))*precision['absolute_tolerance'],coordinate=k,bump=bump)
            quote=diff[b,:,control,k]
            check('known_control_vs_independent_price_difference',abs(known[k]-quote.mean()),
                6*quote.std(ddof=1)/np.sqrt(oracle_count)+precision['absolute_tolerance'],coordinate=k,bump=bump)
    cube=np.array(native['sampling_unit_values']);net=np.einsum('niq,i->nq',cube,weights)
    check('controlled_net_covariance_vs_samples',np.max(abs(sampling_covariance(net)/count-np.array(native['net_price_and_market_mean_covariance']))),1e-9)
    check('controlled_covariance_vs_projection',native['control_variate']['covariance_projection_error'],1e-8)
    value=max(native['net_bucket_standard_errors']);check('held_bucket_standard_error',value,policy['precision_limit'])
    result=dict(document_kind='rqmc_quote_control_precision',schema_version=1,
        status='passed' if all(c['passed'] for c in checks) else 'failed',failed=sum(not c['passed'] for c in checks),checks=checks,
        role='independent_holdout_for_frozen_control_design_not_normal_provider_acceptance',universe='full',instruments=30,
        seed=raw['seed'],policy=policy,native_report=native,max_bucket_standard_error=value,
        raw_max_bucket_standard_error=raw['max_bucket_standard_error'],source_sha256=sources(),implementation_identity=implementation_identity(),
        dependencies_sha256={str(p.resolve()):sha(p) for p in (Path(__file__),ROOT/'scripts/market_quote_control.py',panel_path,
            pilot/'backtest/input.json',oracle_path,policy_path,freeze_path)})
    result['report_hash']=fingerprint(result);write(output,result)
    print(json.dumps({k:result[k] for k in ('status','failed','seed','max_bucket_standard_error','raw_max_bucket_standard_error')}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','pilot','oracle','policy','freeze','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=run(a.panel,a.pilot,a.oracle,a.policy,a.freeze,a.output);raise SystemExit(r['status']!='passed')
