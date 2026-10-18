"""Retain initial Put failures and resolve known Lookback kinks at fixed precision."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backtesting'))
from multi_step_cases import models
from extension_put_reference import PRODUCTS,request
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.multi_step_provider import gamma_boundaries
from rust_pricing_backtest.numerics import threshold_check


def run(policy,policy_path,baseline_path,checkpoint):
    import rust_pricing as rp
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    baseline=json.loads(baseline_path.read_text());base_policy=baseline['policy']
    assert baseline['document_kind']=='extension_put_validation' and len(baseline['observations'])==48
    assert base_policy==json.loads((ROOT/policy['baseline_policy']).read_text())
    for name,h in baseline['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('retained numerical source changed: '+name)
    binary=Path(getattr(rp,'rust_pricing',rp).__file__)
    assert sha(binary)==baseline['native_binary_sha256']
    for key in ['subdivisions','relative_tolerance','gamma_relative_standard_error_limit']:assert policy[key]==base_policy[key]
    failed=[x for x in baseline['checks'] if not x['passed']]
    for x in failed:
        if x.get('product')!='lookback_past' or x['check'] not in ['put_market_Gamma_acceptance','row_recovery']:
            raise ValueError('unexpected failure cannot be superseded: '+str(x))
    observations=[deepcopy(x) for x in baseline['observations'] if x['product']!='lookback_past']
    checks=[deepcopy(x) for x in baseline['checks'] if not (x.get('product')=='lookback_past' and 'seed' in x)]
    for o in observations:
        base=next(c['request'] for c in models() if c['intervals']==o['intervals'])
        bins=gamma_boundaries(request(base,o['product']),base['market']['spot'],put_knots=True)
        # Only one-ULP input-label construction differences are reusable. A new
        # effective strike must always trigger a fresh native recovery below.
        old=o['audit']['gamma_bin_boundaries']
        assert len(bins)==len(old) and np.all(abs(np.array(bins)-old)<=np.spacing(old))
        assert o['audit']['status']=='path_gamma_chain_passed'
    paths=[ROOT/n for n in baseline['source_sha256']]+[Path(__file__),policy_path,ROOT/policy['baseline_policy'],
        ROOT/'backtesting/rust_pricing_backtest/multi_step_provider.py',ROOT/'backtesting/tests/test_put_gamma_boundaries.py']
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    def save(status):
        report=dict(document_kind='extension_put_refined_validation',schema_version=1,status=status,policy=policy,native_policy=base_policy,
            checks=checks,observations=observations,price_records=baseline['price_records'],retained_calibrations=baseline['retained_calibrations'],
            source_sha256=frozen,native_binary_sha256=sha(binary),initial_failed_checks=failed,
            baseline_report_sha256=sha(baseline_path),reference_report_sha256=baseline['reference_report_sha256'],
            fresh_native_panels=8,retained_native_panels=40,full_market_iv_bucket_acceptance='not_established')
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_text(json.dumps(report,allow_nan=False)+'\n')
        return report
    for case in models():
        n=case['intervals'];base=case['request']
        for extension in base_policy['extensions']:
            ctx=dict(intervals=n,profile=case['profile'],extension=extension,product='lookback_past')
            price=next(x for x in baseline['price_records'] if all(x[k]==v for k,v in ctx.items() if k!='product'))
            j=PRODUCTS.index('lookback_past');means=np.array(price['means']);se=np.array(price['standard_errors']);base_prices=np.array(price['price_batches'])
            c=MultiStepExtensionCalibrator(base,extension=extension);t,_=c.all_maturity_iv_tangent();count=c.iv.size
            for seed in base_policy['gamma_seeds']:
                sctx=dict(**ctx,seed=seed);p=request(base,'lookback_past');samples=base_policy['gamma_samples_by_intervals'][str(n)]
                p['engine'].update(master_seed=seed,independent_sampling_units=samples)
                a=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate_vega_kt_multi_step_gamma_chain_audit(
                    list(range(n)),gamma_boundaries(p,c.spot,put_knots=True),variance_risk_directions=t.T.reshape(count,n,33).tolist(),
                    subdivisions=policy['subdivisions'],relative_tolerance=policy['relative_tolerance'],
                    gamma_relative_standard_error_limit=policy['gamma_relative_standard_error_limit'])
                check('put_market_Gamma_acceptance',a['status']!='path_gamma_chain_passed',0,**sctx)
                for row in a['rows']:
                    check('row_recovery',not row['recovery_passed'],0,row=row['variance_row'],**sctx)
                    check('row_conditional_AD_validation',not row['validation_passed'],0,row=row['variance_row'],**sctx)
                cov=np.array(a['joint_sample_covariance']).reshape(3*count,3*count)[count:2*count,count:2*count]
                check('covariance_symmetry',np.max(abs(cov-cov.T)),1e-10,**sctx)
                check('covariance_PSD',max(0.,-np.linalg.eigvalsh(cov).min()),1e-8,**sctx)
                check('Put_price_vs_independent_payoff',abs(a['native_price']['mean']-base_prices[:,j].mean()),
                    6*np.hypot(a['native_price']['standard_error'],base_prices[:,j].std(ddof=1)/np.sqrt(base_policy['sobol_scrambles']))+base_policy['absolute_tolerance'],**sctx)
                for coordinate,value in enumerate(a['recovered_directions']):
                    check('covariance_diagonal_matches_SE',abs(cov[coordinate,coordinate]/samples-value['standard_error']**2),1e-10,coordinate=coordinate,**sctx)
                    for b,bump in enumerate(base_policy['iv_bumps']):
                        check('Put_Gamma_vs_independent_price_difference',abs(value['mean']-means[b,j,coordinate]),
                            6*np.hypot(value['standard_error'],se[b,j,coordinate])+base_policy['absolute_tolerance'],coordinate=coordinate,bump=bump,**sctx)
                observations.append(dict(**sctx,audit=a));save('in_progress')
                print(json.dumps(dict(**sctx,status=a['status'],failed=sum(not c['passed'] for c in checks))),flush=True)
    if any(sha(ROOT/n)!=h for n,h in frozen.items()):raise RuntimeError('refinement source changed during validation')
    return save('passed' if all(c['passed'] for c in checks) else 'failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['policy','baseline-report','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--call-reference-report',type=Path,default=ROOT/'benchmark-results/vega-kt-extension/refined-report.json')
    parser.add_argument('--calibration-report',type=Path,default=ROOT/'benchmark-results/vega-kt-extension/report.json')
    args=parser.parse_args();policy=args.policy.resolve()
    if not args.baseline_report.exists():
        from check_extension_put import run as baseline_run
        bp=ROOT/json.loads(policy.read_text())['baseline_policy']
        baseline=baseline_run(json.loads(bp.read_text()),bp,args.call_reference_report,args.calibration_report,args.baseline_report.with_suffix('.partial.json'))
        args.baseline_report.write_text(json.dumps(baseline,indent=2,allow_nan=False)+'\n')
    r=run(json.loads(policy.read_text()),policy,args.baseline_report,args.output.with_suffix('.partial.json'))
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not c['passed'] for c in r['checks']))))
    return int(r['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
