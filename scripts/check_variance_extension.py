"""All-row calibration/Gamma against independent recalibrated path prices."""
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
from multi_step_reference import black,path_reference
from check_multi_step_gamma_chain import independent_quote_prices
from extension_reference import path_prices,PRODUCTS
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.numerics import threshold_check


def run(policy,policy_path,checkpoint):
    import rust_pricing as rp
    previous=ROOT/'design/validation/local-vol-lookback-fixing-day-v0.1.json'
    prior=json.loads(previous.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    binary=Path(getattr(rp,'rust_pricing',rp).__file__)
    for name,digest in prior['source_sha256'].items():
        if name.startswith('crates/') and sha(ROOT/name)!=digest:raise ValueError('native implementation changed: '+name)
    paths=set((ROOT/'crates').rglob('*.rs'))|set((ROOT/'backtesting/rust_pricing_backtest').glob('*.py'))
    paths|={Path(__file__),policy_path,ROOT/'scripts/extension_reference.py',ROOT/'scripts/multi_step_reference.py',
        ROOT/'scripts/multi_step_cases.py',ROOT/'scripts/check_multi_step_gamma_chain.py',
        ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json',ROOT/'backtesting/tests/test_multi_step_extension.py',
        ROOT/'tests/python/test_vega_kt_extension.py'}
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    checks=[];records=[];calibrations=[];observations=[];tangents=[]
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    def save(status):
        report=dict(document_kind='variance_extension_validation',schema_version=1,status=status,policy=policy,
            checks=checks,observations=observations,price_records=records,calibration_audits=calibrations,tangents=tangents,
            source_sha256=frozen,native_binary_sha256=sha(binary),prior_acceptance_sha256=sha(previous),
            native_source_unchanged=True,native_binary_matches_prior_platform=sha(binary)==prior['native_binary_sha256'],
            full_market_iv_bucket_acceptance='not_established')
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_text(json.dumps(report,allow_nan=False)+'\n')
        return report
    for case in models():
        base=case['request'];n=case['intervals']
        # The vectorized oracle must preserve the independently maintained payoff formulas.
        ref=path_prices(base,power=10,scrambles=2)
        for j,kind in enumerate(PRODUCTS):
            original=path_reference(base,kind,power=10,scrambles=2)
            check('independent_oracle_equivalence',np.max(abs(ref[:,j]-np.array(original['batches'])[:,0])),1e-12,intervals=n,product=kind)
        for extension in policy['extensions']:
            ctx=dict(intervals=n,profile=case['profile'],extension=extension)
            c=MultiStepExtensionCalibrator(base,extension=extension);t,ta=c.all_maturity_iv_tangent();count=c.iv.size
            check('full_row_IV_response',ta['iv_response_residual'],1e-9,**ctx)
            check('all_transition_rows_have_market_response',int(np.any(np.max(abs(t.reshape(n,33,count)),axis=(1,2))==0)),0,**ctx)
            tangents.append(dict(**ctx,audit=ta,calibration_tangent=t.tolist()))
            independent=[]
            for bump in policy['iv_bumps']:
                columns=[]
                for coordinate in range(count):
                    pair=[];variances=[]
                    for sign in [1.,-1.]:
                        shifts=np.zeros(count);shifts[coordinate]=sign*bump
                        q,a=c.calibrate_all_maturities(shifts.reshape(c.iv.shape));variances.append(np.array(q['model']['local_variance_grid']['values']).reshape(c.variance.shape)[:-1].ravel())
                        target=np.array([black(c.spot,c.strikes,tau*iv**2)[0] for tau,iv in zip(c.maturities,c.iv+shifts.reshape(c.iv.shape))])
                        error=np.max(abs(independent_quote_prices(q)-target))
                        check('independent_recalibrated_quote_fit',error,policy['quote_price_tolerance'],coordinate=coordinate,bump=bump,sign=sign,**ctx)
                        check('unused_terminal_row_fixed',np.max(abs(np.array(q['model']['local_variance_grid']['values'])[-33:]-c.variance[-1])),0,coordinate=coordinate,bump=bump,sign=sign,**ctx)
                        pair.append(path_prices(q,power=policy['sobol_power'],scrambles=policy['sobol_scrambles']))
                        calibrations.append(dict(**ctx,coordinate=coordinate,bump=bump,sign=sign,audit=a,independent_price_error=float(error)))
                    check('tangent_vs_actual_recalibration',np.max(abs((variances[0]-variances[1])/(2*bump)-t[:,coordinate])),2e-5,coordinate=coordinate,bump=bump,**ctx)
                    columns.append((pair[0]-pair[1])/(2*bump))
                independent.append(np.stack(columns,axis=2)) # scramble, product, quote
                print(json.dumps(dict(**ctx,bump=bump,stage='independent_price_differences',checks=len(checks),failed=sum(not c['passed'] for c in checks))),flush=True)
                save('in_progress')
            for sign in [-1,1]:
                shifts=np.zeros_like(c.iv);shifts[0]=sign*policy['stress_iv_bump'];q,a=c.calibrate_all_maturities(shifts)
                target=np.array([black(c.spot,c.strikes,tau*iv**2)[0] for tau,iv in zip(c.maturities,c.iv+shifts)])
                check('finite_first_smile_stress_holds_later_quotes',np.max(abs(independent_quote_prices(q)-target)),policy['quote_price_tolerance'],sign=sign,**ctx)
                calibrations.append(dict(**ctx,stress=sign*policy['stress_iv_bump'],audit=a))
            independent=np.array(independent);means=independent.mean(axis=1);se=independent.std(axis=1,ddof=1)/np.sqrt(policy['sobol_scrambles'])
            records.append(dict(**ctx,products=list(PRODUCTS),independent_price_difference_batches=independent.tolist(),means=means.tolist(),standard_errors=se.tolist()))
            for j,kind in enumerate(PRODUCTS):
                for coordinate in range(count):
                    diff=independent[0,:,j,coordinate]-independent[1,:,j,coordinate]
                    check('paired_independent_bump_refinement',abs(diff.mean()),6*diff.std(ddof=1)/np.sqrt(len(diff))+policy['absolute_tolerance'],product=kind,coordinate=coordinate,**ctx)
                for seed in policy['gamma_seeds']:
                    p=with_product(base,kind);p['engine'].update(master_seed=seed,independent_sampling_units=policy['gamma_samples'])
                    sctx=dict(**ctx,product=kind,seed=seed)
                    audit=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate_vega_kt_multi_step_gamma_chain_audit(
                        c.calibrated_rows,policy['gamma_bins'],variance_risk_directions=t.T.reshape(count,n,33).tolist(),subdivisions=policy['subdivisions'],
                        relative_tolerance=policy['relative_tolerance'],gamma_relative_standard_error_limit=policy['gamma_relative_standard_error_limit'])
                    check('all_row_market_Gamma_acceptance',int(audit['status']!='path_gamma_chain_passed'),0,**sctx)
                    for row in audit['rows']:
                        check('row_recovery',not row['recovery_passed'],0,row=row['variance_row'],**sctx)
                        check('row_conditional_AD_validation',not row['validation_passed'],0,row=row['variance_row'],**sctx)
                    covariance=np.array(audit['joint_sample_covariance']).reshape(3*count,3*count)
                    gamma_cov=covariance[count:2*count,count:2*count]
                    check('market_covariance_symmetry',np.max(abs(gamma_cov-gamma_cov.T)),1e-10,**sctx)
                    check('market_covariance_PSD',max(0.,-np.linalg.eigvalsh(gamma_cov).min()),1e-8,**sctx)
                    for coordinate,value in enumerate(audit['recovered_directions']):
                        check('covariance_diagonal_matches_SE',abs(gamma_cov[coordinate,coordinate]/policy['gamma_samples']-value['standard_error']**2),1e-10,coordinate=coordinate,**sctx)
                        for b,bump in enumerate(policy['iv_bumps']):
                            check('Gamma_vs_independent_recalibrated_price_difference',abs(value['mean']-means[b,j,coordinate]),
                                6*np.hypot(value['standard_error'],se[b,j,coordinate])+policy['absolute_tolerance'],coordinate=coordinate,bump=bump,**sctx)
                        if kind=='call':
                            expected=c._vegas(c.iv)[-3] if coordinate==count-3 else 0.
                            check('European_market_Vega_analytic',abs(value['mean']-expected),6*value['standard_error']+policy['absolute_tolerance'],coordinate=coordinate,**sctx)
                    observations.append(dict(**sctx,audit=audit));save('in_progress')
                    print(json.dumps(dict(**sctx,status=audit['status'],checks=len(checks),failed=sum(not c['passed'] for c in checks))),flush=True)
    if any(sha(ROOT/name)!=digest for name,digest in frozen.items()):raise RuntimeError('validation source changed during run')
    return save('passed' if all(c['passed'] for c in checks) else 'failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    policy_path=args.policy.resolve();r=run(json.loads(policy_path.read_text()),policy_path,args.output.with_suffix('.partial.json'))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not x['passed'] for x in r['checks']))))
    return int(r['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
