"""Accept conditional native AD and replay all prior Gamma/covariance estimates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import rust_pricing as rp
from multi_step_cases import ROOT, model
from rust_pricing_backtest.numerics import threshold_check


def run(covariance_path, precision_path, price_path, checkpoint):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    covariance=json.loads(covariance_path.read_text())
    precision=json.loads(precision_path.read_text())
    price=json.loads(price_path.read_text())
    if covariance['status']!='passed' or price['status']!='passed':
        raise ValueError('passing covariance and recalibrated-price reports required')
    if sha(price_path)!=covariance['previous_report_sha256'] or sha(covariance_path)!=precision['covariance_report_sha256']:
        raise ValueError('reference provenance mismatch')
    allowed={
        'crates/pricing/src/engine/risk/vegakt_path_chain.rs',
        'crates/pricing/src/risk/vegakt_path_chain.rs',
        'crates/pricing-python/src/vegakt_path_chain.rs',
        'crates/pricing/src/engine/risk/vegakt_terminal.rs',
    }
    for name,digest in covariance['source_sha256'].items():
        if name not in allowed and sha(ROOT/name)!=digest:
            raise ValueError('price or Gamma implementation changed: '+name)
    names=set(covariance['source_sha256'])|allowed|{
        'crates/pricing/src/engine/risk/vegakt_path_chain/conditional_ad.rs',
        'scripts/check_path_gamma_conditional_ad.py',
    }
    frozen={name:sha(ROOT/name) for name in sorted(names)}
    checks=[];observations=[]
    def check(name,value,limit,**ctx):
        checks.append(threshold_check(name,float(value),float(limit),**ctx))
    def unchanged(before,after):
        # Preserve old raw statistics, including covariance. Only validation
        # decisions/estimate exposure can change, and plan workers differ.
        for name,value in before.items():
            if name in {'plan_fingerprint','validation_passed','status','estimate'}:
                continue
            if name=='rows':
                for old,new in zip(value,after[name]):
                    validation_key = 'validation_passed' if 'conditional_native_variance_nodes' in old else 'raw_node_validation_passed'
                    if old['validation_passed']!=new[validation_key]:
                        return False
                    if any(v!=new[k] for k,v in old.items() if k!='validation_passed'):
                        return False
            elif value!=after[name]:
                return False
        if before['estimate'] is not None and before['estimate']!=after['estimate']:
            return False
        return True
    for o in covariance['observations']:
        ctx0={k:o[k] for k in ['intervals','profile','product','seed']}
        refs=next(x for x in price['price_records'] if all(x[k]==o[k] for k in ['intervals','profile','product']))
        for panel in ['all_quotes','composed']:
            before=o['audit'] if panel=='all_quotes' else o['direct_combinations']
            if panel=='composed' and before['estimate'] is None:
                before=next(x['audit'] for x in precision['records'] if all(x[k]==o[k] for k in ctx0))
            ctx=dict(**ctx0,panel=panel)
            p=model(o['intervals'],o['product'],samples=before['independent_sampling_units'],seed=o['seed'])
            a=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256
                ).evaluate_vega_kt_multi_step_gamma_chain_audit([x['variance_row'] for x in before['rows']],before['gamma_bin_boundaries'],
                    variance_risk_directions=before['variance_risk_directions'],subdivisions=before['subdivisions'],
                    relative_tolerance=before['relative_tolerance'],gamma_relative_standard_error_limit=before['gamma_relative_standard_error_limit'])
            replay=unchanged(before,a)
            check('prior_raw_statistics_exact',not replay,0,**ctx)
            if not replay: raise ValueError('Gamma/raw covariance changed; cannot reuse price evidence')
            check('conditional_AD_native_acceptance',a['status']!='path_gamma_chain_passed',0,**ctx)
            for row in a['rows']:
                rctx=dict(**ctx,row=row['variance_row'])
                for j,(g,raw,se) in enumerate(zip(row['conditional_node_differences'],
                        row['raw_conditional_node_differences'],row['raw_conditional_standard_errors'])):
                    check('Gamma_vs_conditional_native_AD',abs(g['mean']),6*g['standard_error']+a['functional_tolerance'],node=j,**rctx)
                    check('raw_vs_conditional_native_AD',abs(raw['mean']),6*se+a['functional_tolerance'],node=j,**rctx)
                if row['variance_row']==o['intervals']-1:
                    check('terminal_conditional_AD_equals_exact_Gamma',
                        max(abs(x['mean']-y['mean']) for x,y in zip(row['conditional_native_variance_nodes'],row['recovered_variance_nodes'])),
                        1e-9,**rctx)
            weights=np.eye(len(a['recovered_directions'])) if panel=='all_quotes' else np.array(o['combinations'])
            mean=np.array([x['mean'] for x in a['recovered_directions']])
            stderr=np.array([x['standard_error'] for x in a['recovered_directions']])
            for label in ['native','independent']:
                batches=np.array(refs[label+'_price_difference_batches'])[1]@weights.T
                for j,(error,limit) in enumerate(zip(abs(mean-batches.mean(axis=0)),
                        6*np.hypot(stderr,batches.std(axis=0,ddof=1)/np.sqrt(len(batches)))+
                        price['policy']['absolute_tolerance']*np.sum(abs(weights),axis=1))):
                    check('Gamma_vs_'+label+'_recalibrated_price_difference',error,limit,coordinate=j,**ctx)
            observations.append(dict(**ctx,previous_status=before['status'],prior_raw_statistics_exact=replay,audit=a))
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            checkpoint.write_text(json.dumps(dict(status='in_progress',checks=checks,observations=observations),allow_nan=False)+'\n')
            print(json.dumps(dict(**ctx,status=a['status'],checks=len(checks),failed=sum(not x['passed'] for x in checks))),flush=True)
    if frozen!={name:sha(ROOT/name) for name in frozen}:raise RuntimeError('source changed during validation')
    return dict(document_kind='path_gamma_conditional_native_AD_validation',schema_version=1,
        status='passed' if all(x['passed'] for x in checks) else 'failed',checks=checks,observations=observations,
        source_sha256=frozen,native_binary_sha256=sha(Path(getattr(rp,'rust_pricing',rp).__file__)),
        previous_report_sha256={str(p):sha(p) for p in [covariance_path,precision_path,price_path]},
        price_reference_reuse='unchanged price/calibration implementation and exact Gamma/native/covariance replay',
        full_market_iv_bucket_acceptance='not_established')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--covariance-report',type=Path,required=True)
    p.add_argument('--precision-report',type=Path,required=True)
    p.add_argument('--price-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(a.covariance_report,a.precision_report,a.price_report,a.output.with_suffix('.partial.json'))
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not c['passed'] for c in r['checks']))))
    return int(r['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
