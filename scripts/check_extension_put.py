"""Put-side full-row Gamma versus independent recalibrated payoff differences."""
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
from extension_put_reference import path_prices,PRODUCTS,request
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.domain import fingerprint
from rust_pricing_backtest.numerics import threshold_check


def run(policy,policy_path,reference_path,baseline_path,checkpoint):
    import rust_pricing as rp
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    reference=json.loads(reference_path.read_text());baseline=json.loads(baseline_path.read_text())
    if reference['status']!='passed' or sha(baseline_path)!=reference['baseline_report']['sha256']:
        raise ValueError('accepted extension reference and its original calibration evidence required')
    numerical=['backtesting/rust_pricing_backtest/'+n for n in ['multi_step_extension.py','multi_step_smile.py','discrete_smile.py','domain.py']]
    numerical += [n for n in reference['source_sha256'] if n.startswith('crates/')]
    numerical += ['scripts/multi_step_cases.py','fixtures/local-vol/multi-step-gamma-v0.1.json']
    for name in numerical:
        if sha(ROOT/name)!=reference['source_sha256'][name]:raise ValueError('accepted numerical source changed: '+name)
    binary=Path(getattr(rp,'rust_pricing',rp).__file__)
    if sha(binary)!=reference['native_binary_sha256']:raise ValueError('same native binary as the reference is required')
    assert policy['products']==list(PRODUCTS)
    assert policy['iv_bumps']==baseline['policy']['iv_bumps']
    paths=[ROOT/n for n in numerical]+[Path(__file__),policy_path,ROOT/'scripts/extension_put_reference.py',ROOT/'scripts/test_extension_put_reference.py']
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    checks=[];observations=[];prices=[];calibrations=[]
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    def save(status):
        report=dict(document_kind='extension_put_validation',schema_version=1,status=status,policy=policy,checks=checks,
            observations=observations,price_records=prices,retained_calibrations=calibrations,source_sha256=frozen,
            native_binary_sha256=sha(binary),reference_report_sha256=sha(reference_path),baseline_report_sha256=sha(baseline_path),
            retained_calibration_reason='identical model inputs reconstructed from accepted parameter increments; payoff changes do not alter vanilla quote calibration',
            full_market_iv_bucket_acceptance='not_established')
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_text(json.dumps(report,allow_nan=False)+'\n')
        return report
    for case in models():
        base=case['request'];n=case['intervals']
        base_prices=path_prices(base,power=policy['sobol_power'],scrambles=policy['sobol_scrambles'])
        for extension in policy['extensions']:
            ctx=dict(intervals=n,profile=case['profile'],extension=extension)
            c=MultiStepExtensionCalibrator(base,extension=extension);t,ta=c.all_maturity_iv_tangent();count=c.iv.size
            old=next(x for x in reference['tangents'] if all(x[k]==v for k,v in ctx.items()))
            check('accepted_calibration_tangent_exact',np.max(abs(t-np.array(old['calibration_tangent']))),0,**ctx)
            independent=[]
            for bump in policy['iv_bumps']:
                columns=[]
                for coordinate in range(count):
                    pair=[]
                    for sign in [1.,-1.]:
                        item=next(x for x in baseline['calibration_audits'] if all(x[k]==v for k,v in ctx.items()) and
                            x.get('coordinate')==coordinate and x.get('bump')==bump and x.get('sign')==sign)
                        a=item['audit'];p=deepcopy(base);rows=c.variance.copy()
                        rows[:-1]+=(c.basis@a['parameter_increments']).reshape(n,len(c.nodes))
                        p['model']['local_variance_grid']['values']=rows.ravel().tolist()
                        p['model']['reporting_iv_basis']['implied_volatilities']=(c.iv+a['iv_shifts']).ravel().tolist()
                        if fingerprint(p)!=a['recalibrated_request_hash']:raise ValueError('retained recalibration reconstruction differs')
                        check('retained_independent_quote_fit',item['independent_price_error'],baseline['policy']['quote_price_tolerance'],
                            coordinate=coordinate,bump=bump,sign=sign,**ctx)
                        calibrations.append(dict(**ctx,coordinate=coordinate,bump=bump,sign=sign,request_hash=fingerprint(p)))
                        pair.append(path_prices(p,power=policy['sobol_power'],scrambles=policy['sobol_scrambles']))
                    columns.append((pair[0]-pair[1])/(2*bump))
                independent.append(np.stack(columns,axis=2))
                print(json.dumps(dict(**ctx,bump=bump,stage='independent_put_price_differences')),flush=True);save('in_progress')
            independent=np.array(independent);means=independent.mean(axis=1);se=independent.std(axis=1,ddof=1)/np.sqrt(policy['sobol_scrambles'])
            prices.append(dict(**ctx,products=list(PRODUCTS),price_batches=base_prices.tolist(),difference_batches=independent.tolist(),
                means=means.tolist(),standard_errors=se.tolist()))
            for j,kind in enumerate(PRODUCTS):
                for coordinate in range(count):
                    diff=independent[0,:,j,coordinate]-independent[1,:,j,coordinate]
                    check('paired_independent_bump_refinement',abs(diff.mean()),6*diff.std(ddof=1)/np.sqrt(len(diff))+policy['absolute_tolerance'],
                        product=kind,coordinate=coordinate,**ctx)
                for seed in policy['gamma_seeds']:
                    p=request(base,kind);samples=policy['gamma_samples_by_intervals'][str(n)]
                    p['engine'].update(master_seed=seed,independent_sampling_units=samples)
                    sctx=dict(**ctx,product=kind,seed=seed)
                    a=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate_vega_kt_multi_step_gamma_chain_audit(
                        list(range(n)),policy['gamma_bins'],variance_risk_directions=t.T.reshape(count,n,33).tolist(),
                        subdivisions=policy['subdivisions'],relative_tolerance=policy['relative_tolerance'],
                        gamma_relative_standard_error_limit=policy['gamma_relative_standard_error_limit'])
                    check('put_market_Gamma_acceptance',int(a['status']!='path_gamma_chain_passed'),0,**sctx)
                    for row in a['rows']:
                        check('row_recovery',not row['recovery_passed'],0,row=row['variance_row'],**sctx)
                        check('row_conditional_AD_validation',not row['validation_passed'],0,row=row['variance_row'],**sctx)
                    covariance=np.array(a['joint_sample_covariance']).reshape(3*count,3*count)[count:2*count,count:2*count]
                    check('covariance_symmetry',np.max(abs(covariance-covariance.T)),1e-10,**sctx)
                    check('covariance_PSD',max(0.,-np.linalg.eigvalsh(covariance).min()),1e-8,**sctx)
                    native_price=a['native_price']
                    check('Put_price_vs_independent_payoff',abs(native_price['mean']-base_prices[:,j].mean()),
                        6*np.hypot(native_price['standard_error'],base_prices[:,j].std(ddof=1)/np.sqrt(policy['sobol_scrambles']))+policy['absolute_tolerance'],**sctx)
                    for coordinate,value in enumerate(a['recovered_directions']):
                        check('covariance_diagonal_matches_SE',abs(covariance[coordinate,coordinate]/samples-value['standard_error']**2),1e-10,coordinate=coordinate,**sctx)
                        for b,bump in enumerate(policy['iv_bumps']):
                            check('Put_Gamma_vs_independent_price_difference',abs(value['mean']-means[b,j,coordinate]),
                                6*np.hypot(value['standard_error'],se[b,j,coordinate])+policy['absolute_tolerance'],coordinate=coordinate,bump=bump,**sctx)
                        if kind=='put':
                            expected=c._vegas(c.iv)[-3] if coordinate==count-3 else 0.
                            check('European_Put_market_Vega_analytic',abs(value['mean']-expected),6*value['standard_error']+policy['absolute_tolerance'],coordinate=coordinate,**sctx)
                    observations.append(dict(**sctx,audit=a));save('in_progress')
                    print(json.dumps(dict(**sctx,status=a['status'],checks=len(checks),failed=sum(not x['passed'] for x in checks))),flush=True)
    if any(sha(ROOT/n)!=h for n,h in frozen.items()):raise RuntimeError('numerical source changed during validation')
    return save('passed' if all(x['passed'] for x in checks) else 'failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['policy','reference-report','baseline-report','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();policy_path=args.policy.resolve()
    r=run(json.loads(policy_path.read_text()),policy_path,args.reference_report,args.baseline_report,args.output.with_suffix('.partial.json'))
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not x['passed'] for x in r['checks']))))
    return int(r['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
