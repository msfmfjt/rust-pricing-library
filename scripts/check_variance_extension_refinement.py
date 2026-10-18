"""Retain initial failures, refine calibration differences and increase 4-step MC.

The reference price pairs and six-step Gamma are reused only with unchanged
pricing/calibration sources. No failed native estimate is promoted: every
four-step panel is freshly evaluated with 4096 independent units.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backtesting'))
from multi_step_cases import models,with_product
from extension_reference import PRODUCTS
from check_multi_step_gamma_chain import independent_quote_prices
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.domain import fingerprint
from rust_pricing_backtest.numerics import threshold_check


def run(policy,policy_path,baseline_path,checkpoint):
    import rust_pricing as rp
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    baseline=json.loads(baseline_path.read_text());base_policy=baseline['policy'];binary=Path(getattr(rp,'rust_pricing',rp).__file__)
    assert baseline['document_kind']=='variance_extension_validation' and len(baseline['observations'])==32
    assert baseline['policy']==json.loads((ROOT/policy['baseline_policy']).read_text())
    assert sha(binary)==baseline['native_binary_sha256']
    for name,digest in baseline['source_sha256'].items():
        if name!='tests/python/test_vega_kt_extension.py' and sha(ROOT/name)!=digest:raise ValueError('retained numerical source changed: '+name)
    failures=[x for x in baseline['checks'] if not x['passed']]
    for x in failures:
        if x['check']=='tangent_vs_actual_recalibration':continue
        if x['check'] in ['all_row_market_Gamma_acceptance','row_conditional_AD_validation'] and x['intervals']==4:continue
        raise ValueError('unexpected baseline failure cannot be superseded: '+str(x))
    checks=[deepcopy(x) for x in baseline['checks'] if x['check']!='tangent_vs_actual_recalibration' and not ('seed' in x and x['intervals']==4)]
    observations=[deepcopy(x) for x in baseline['observations'] if x['intervals']==6]
    refinement=[];calibrations=[]
    paths=[ROOT/n for n in baseline['source_sha256']]+[Path(__file__),policy_path,ROOT/policy['baseline_policy']]
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    def check(name,observed,limit,**ctx):checks.append(threshold_check(name,float(observed),float(limit),**ctx))
    def save(status):
        result=dict(document_kind='variance_extension_refined_validation',schema_version=1,status=status,
            checks=checks,observations=observations,tangent_refinement=refinement,calibration_audits=calibrations,
            source_sha256=frozen,native_binary_sha256=sha(binary),policy=policy,native_policy=base_policy,
            baseline_report=dict(path=str(baseline_path),sha256=sha(baseline_path),failed_checks=failures,
                supersession='smaller-step and Richardson tangent checks; fresh 4096-unit four-step native Gamma; all other checks retained'),
            price_records=baseline['price_records'],tangents=baseline['tangents'],
            full_market_iv_bucket_acceptance='not_established')
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_text(json.dumps(result,allow_nan=False)+'\n')
        return result
    for case in models():
        n=case['intervals'];base=case['request']
        for extension in base_policy['extensions']:
            ctx=dict(intervals=n,profile=case['profile'],extension=extension);c=MultiStepExtensionCalibrator(base,extension=extension)
            old=next(x for x in baseline['tangents'] if all(x[k]==v for k,v in ctx.items()));t=np.array(old['calibration_tangent']);count=c.iv.size
            actual,ta=c.all_maturity_iv_tangent();check('retained_calibration_tangent_exact',np.max(abs(actual-t)),0,**ctx)
            def restored_variance(audit):
                p=deepcopy(base);rows=c.variance.copy();rows[:-1]+=(c.basis@audit['parameter_increments']).reshape(n,33)
                p['model']['local_variance_grid']['values']=rows.ravel().tolist()
                p['model']['reporting_iv_basis']['implied_volatilities']=(c.iv+audit['iv_shifts']).ravel().tolist()
                if fingerprint(p)!=audit['recalibrated_request_hash']:raise ValueError('retained calibrated request reconstruction failed')
                return rows[:-1].ravel()
            for coordinate in range(count):
                prior_differences=[]
                for bump in base_policy['iv_bumps']:
                    pair=[next(x for x in baseline['calibration_audits'] if all(x[k]==v for k,v in ctx.items())
                        and x.get('coordinate')==coordinate and x.get('bump')==bump and x.get('sign')==sign)['audit'] for sign in [1.,-1.]]
                    prior_differences.append((restored_variance(pair[0])-restored_variance(pair[1]))/(2*bump))
                h=policy['tangent_iv_bump'];values=[]
                for sign in [1.,-1.]:
                    shift=np.zeros(count);shift[coordinate]=sign*h
                    p,a=c.calibrate_all_maturities(shift.reshape(c.iv.shape));values.append(np.array(p['model']['local_variance_grid']['values']).reshape(c.variance.shape)[:-1].ravel())
                    error=np.max(abs(independent_quote_prices(p).ravel()-np.array(a['target_prices'])))
                    check('fine_step_independent_recalibrated_quote_fit',error,base_policy['quote_price_tolerance'],coordinate=coordinate,sign=sign,**ctx)
                    calibrations.append(dict(**ctx,coordinate=coordinate,bump=h,sign=sign,audit=a))
                fine=(values[0]-values[1])/(2*h)
                ratio=base_policy['iv_bumps'][0]/base_policy['iv_bumps'][1]
                richardson=(ratio**2*prior_differences[1]-prior_differences[0])/(ratio**2-1)
                fine_error=np.max(abs(fine-t[:,coordinate]));rich_error=np.max(abs(richardson-t[:,coordinate]))
                check('smaller_step_tangent_vs_recalibration',fine_error,policy['tangent_absolute_tolerance'],coordinate=coordinate,**ctx)
                check('Richardson_tangent_vs_recalibration',rich_error,policy['richardson_absolute_tolerance'],coordinate=coordinate,**ctx)
                refinement.append(dict(**ctx,coordinate=coordinate,coarse_raw_error=float(np.max(abs(prior_differences[0]-t[:,coordinate]))),
                    half_step_raw_error=float(np.max(abs(prior_differences[1]-t[:,coordinate]))),fine_raw_error=float(fine_error),Richardson_error=float(rich_error)))
            if n==6:
                check('retained_six_step_sampling_policy',int(policy['gamma_samples_by_intervals'][str(n)]!=base_policy['gamma_samples']),0,**ctx)
                save('in_progress');continue
            price=next(x for x in baseline['price_records'] if all(x[k]==v for k,v in ctx.items()))
            means=np.array(price['means']);se=np.array(price['standard_errors']);samples=policy['gamma_samples_by_intervals'][str(n)]
            for j,kind in enumerate(PRODUCTS):
                for seed in base_policy['gamma_seeds']:
                    sctx=dict(**ctx,product=kind,seed=seed);p=with_product(base,kind);p['engine'].update(master_seed=seed,independent_sampling_units=samples)
                    a=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate_vega_kt_multi_step_gamma_chain_audit(
                        list(range(n)),base_policy['gamma_bins'],variance_risk_directions=t.T.reshape(count,n,33).tolist(),subdivisions=base_policy['subdivisions'],
                        relative_tolerance=base_policy['relative_tolerance'],gamma_relative_standard_error_limit=base_policy['gamma_relative_standard_error_limit'])
                    check('refined_all_row_market_Gamma_acceptance',int(a['status']!='path_gamma_chain_passed'),0,**sctx)
                    for row in a['rows']:
                        check('row_recovery',not row['recovery_passed'],0,row=row['variance_row'],**sctx)
                        check('row_conditional_AD_validation',not row['validation_passed'],0,row=row['variance_row'],**sctx)
                    joint=np.array(a['joint_sample_covariance']).reshape(3*count,3*count);cov=joint[count:2*count,count:2*count]
                    check('market_covariance_symmetry',np.max(abs(cov-cov.T)),1e-10,**sctx)
                    check('market_covariance_PSD',max(0.,-np.linalg.eigvalsh(cov).min()),1e-8,**sctx)
                    for coordinate,v in enumerate(a['recovered_directions']):
                        check('covariance_diagonal_matches_SE',abs(cov[coordinate,coordinate]/samples-v['standard_error']**2),1e-10,coordinate=coordinate,**sctx)
                        for b,bump in enumerate(base_policy['iv_bumps']):
                            check('Gamma_vs_independent_recalibrated_price_difference',abs(v['mean']-means[b,j,coordinate]),
                                6*np.hypot(v['standard_error'],se[b,j,coordinate])+base_policy['absolute_tolerance'],coordinate=coordinate,bump=bump,**sctx)
                        if kind=='call':
                            expected=c._vegas(c.iv)[-3] if coordinate==count-3 else 0.
                            check('European_market_Vega_analytic',abs(v['mean']-expected),6*v['standard_error']+base_policy['absolute_tolerance'],coordinate=coordinate,**sctx)
                    observations.append(dict(**sctx,audit=a));save('in_progress')
                    print(json.dumps(dict(**sctx,status=a['status'],checks=len(checks),failed=sum(not c['passed'] for c in checks))),flush=True)
    if any(sha(ROOT/n)!=h for n,h in frozen.items()):raise RuntimeError('source changed during refinement')
    return save('passed' if all(c['passed'] for c in checks) else 'failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--baseline-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    policy_path=args.policy.resolve();policy=json.loads(policy_path.read_text())
    if not args.baseline_report.exists():
        from check_variance_extension import run as baseline_run
        bp=ROOT/policy['baseline_policy'];base=baseline_run(json.loads(bp.read_text()),bp,args.baseline_report.with_suffix('.partial.json'))
        args.baseline_report.write_text(json.dumps(base,indent=2,allow_nan=False)+'\n')
    r=run(policy,policy_path,args.baseline_report,args.output.with_suffix('.partial.json'))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not c['passed'] for c in r['checks']))))
    return int(r['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
