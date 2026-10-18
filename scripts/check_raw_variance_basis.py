"""Check a supplied matrix against independent prices on a bound market fixture.

Acceptance belongs to this matrix/request/product/seed matrix only. It never
promotes a raw matrix to an accepted backtest-provider policy.
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
from raw_variance_basis import load_matrix
from check_custom_variance_basis import PRODUCTS,models,product_request,prices,black,independent_quote_prices
from rust_pricing_backtest.domain import BacktestError,fingerprint,require
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.multi_step_provider import gamma_boundaries
from rust_pricing_backtest.numerics import threshold_check

POLICY_PATH=ROOT/'fixtures/backtesting/raw-variance-basis-policy-v0.1.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def provenance():
    import rust_pricing as rp
    prior_path=ROOT/'design/validation/local-vol-custom-basis-provider-v0.1.json'
    prior=json.loads(prior_path.read_text());require(prior['status']=='passed','basis_reference','passing prior acceptance required')
    names={n for n in prior['source_sha256'] if n.startswith(('crates/','backtesting/rust_pricing_backtest/'))}
    names.update(['scripts/check_custom_variance_basis.py','scripts/custom_basis_cases.py','scripts/extension_reference.py',
        'scripts/extension_put_reference.py','scripts/multi_step_cases.py','scripts/multi_step_reference.py',
        'scripts/check_multi_step_gamma_chain.py','fixtures/local-vol/multi-step-gamma-v0.1.json'])
    for name in names:
        require(sha(ROOT/name)==prior['source_sha256'][name],'basis_reference','numerical source changed: '+name)
    binary=sha(Path(getattr(rp,'rust_pricing',rp).__file__))
    require(binary==prior['native_binary_sha256'],'basis_reference','same accepted native binary required')
    names.update(['scripts/raw_variance_basis.py','scripts/test_raw_variance_basis.py','scripts/check_raw_variance_basis.py',
        str(POLICY_PATH.relative_to(ROOT))])
    return {name:sha(ROOT/name) for name in sorted(names)},binary,sha(prior_path)


def write(path,report):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verify_complete(report):
    n=report['intervals'];count=5*(n//2);policy=report['policy']
    expected={(p,s) for p in PRODUCTS for s in policy['gamma_seeds']}
    keys=[(p['product'],p['seed']) for p in report['observations']]
    require(len(keys)==len(set(keys)) and set(keys)==expected,'basis_report_coverage','complete product/seed coverage required')
    require(len(report['checks'])==85+40*n+88*count and len(report['calibration_audits'])==4*count+2,
            'basis_report_coverage','complete calibration/price/native checks required')
    require(np.asarray(report['price_record']['difference_batches']).shape==(2,8,10,count),
            'basis_report_coverage','complete independent price differences required')


def run(intervals,basis_path,output,*,resume=None):
    import rust_pricing as rp
    policy=json.loads(POLICY_PATH.read_text());assert policy['products']==list(PRODUCTS)
    case=next(x for x in models() if x['intervals']==intervals);base=case['request']
    b,ba=load_matrix(basis_path,base);source,binary,prior=provenance()
    common=dict(document_kind='raw_variance_basis_validation',schema_version=1,policy=policy,
        intervals=intervals,profile=case['profile'],basis_input=ba,basis_document=json.loads(basis_path.read_text()),
        basis_file_sha256=sha(basis_path),source_sha256=source,native_binary_sha256=binary,prior_acceptance_sha256=prior,
        diagnostic_only=True,provider_promotion=False,full_market_iv_bucket_acceptance='not_established')
    c=MultiStepExtensionCalibrator(base,extension='custom',variance_basis=b)
    tangent,ta=c.all_maturity_iv_tangent();count=c.iv.size
    if resume:
        report=json.loads(Path(resume).read_text())
        require(all(report.get(k)==v for k,v in common.items()),'basis_resume_mismatch','input, sources, binary or policy changed')
        require(report.get('phase') in ('native','complete') and report['calibration_tangent']==tangent.tolist()
                and report['tangent_audit']==ta,'basis_resume_mismatch','only complete independent-price stages can resume')
        require(len(report['calibration_audits'])==4*count+2 and np.asarray(report['price_record']['difference_batches']).shape==(2,8,10,count),
                'basis_resume_mismatch','incomplete independent prices')
        seen=[(p['product'],p['seed']) for p in report['observations']]
        require(len(seen)==len(set(seen)) and set(seen)<={(p,s) for p in PRODUCTS for s in policy['gamma_seeds']},
                'basis_resume_mismatch','invalid retained native panel keys')
    else:
        report=dict(common,status='in_progress',phase='prices',checks=[],observations=[],calibration_audits=[],
            calibration_tangent=tangent.tolist(),tangent_audit=ta)
    def check(name,observed,limit,**ctx):
        report['checks'].append(threshold_check(name,float(observed),float(limit),**ctx))
    def save():
        require(all(sha(ROOT/n)==h for n,h in source.items()) and sha(basis_path)==common['basis_file_sha256'],
                'basis_source_changed','source or input changed during validation')
        write(output.with_suffix('.partial.json'),report)
    if not resume:
        base_price=prices(base,policy)
        target=np.array([black(c.spot,c.strikes,t*v*v)[0] for t,v in zip(c.maturities,c.iv)])
        check('initial_independent_quote_fit',np.max(abs(independent_quote_prices(base)-target)),policy['quote_price_tolerance'])
        check('basis_hash_matches_calibration',ba['basis_hash']!=ta['basis_hash'],0)
        check('full_quote_IV_response',ta['iv_response_residual'],1e-9)
        independent=[]
        for bump in policy['iv_bumps']:
            columns=[]
            for coordinate in range(count):
                pair=[];variances=[]
                for sign in (1.,-1.):
                    shift=np.zeros(count);shift[coordinate]=sign*bump
                    q,a=c.calibrate_all_maturities(shift.reshape(c.iv.shape));v=np.array(q['model']['local_variance_grid']['values']).reshape(c.variance.shape)
                    target=np.array([black(c.spot,c.strikes,t*iv*iv)[0] for t,iv in zip(c.maturities,c.iv+shift.reshape(c.iv.shape))])
                    error=float(np.max(abs(independent_quote_prices(q)-target)))
                    ctx=dict(coordinate=coordinate,bump=bump,sign=sign)
                    check('independent_recalibrated_quote_fit',error,policy['quote_price_tolerance'],**ctx)
                    check('unused_terminal_row_fixed',np.max(abs(v[-1]-c.variance[-1])),0,**ctx)
                    check('explicit_basis_parameter_increments',np.max(abs((v[:-1]-c.variance[:-1]).ravel()-b@a['parameter_increments'])),1e-17,**ctx)
                    report['calibration_audits'].append(dict(**ctx,audit=a,independent_price_error=error))
                    pair.append(prices(q,policy));variances.append(v[:-1].ravel())
                check('tangent_vs_actual_recalibration',np.max(abs((variances[0]-variances[1])/(2*bump)-tangent[:,coordinate])),
                    policy['tangent_tolerance'],coordinate=coordinate,bump=bump)
                columns.append((pair[0]-pair[1])/(2*bump))
            independent.append(np.stack(columns,axis=2));save()
            print(json.dumps(dict(intervals=intervals,bump=bump,phase='independent_prices',failed=sum(not x['passed'] for x in report['checks']))),flush=True)
        for sign in (-1,1):
            shift=np.zeros_like(c.iv);shift[0]=sign*policy['stress_iv_bump'];q,a=c.calibrate_all_maturities(shift)
            target=np.array([black(c.spot,c.strikes,t*iv*iv)[0] for t,iv in zip(c.maturities,c.iv+shift)])
            check('finite_first_smile_stress_holds_later_quotes',np.max(abs(independent_quote_prices(q)-target)),policy['quote_price_tolerance'],sign=sign)
            report['calibration_audits'].append(dict(stress=sign*policy['stress_iv_bump'],audit=a))
        diff=np.array(independent)
        report['price_record']=dict(products=list(PRODUCTS),price_batches=base_price.tolist(),difference_batches=diff.tolist())
        for j,product in enumerate(PRODUCTS):
            for coordinate in range(count):
                d=diff[0,:,j,coordinate]-diff[1,:,j,coordinate]
                check('paired_independent_bump_refinement',abs(d.mean()),6*d.std(ddof=1)/np.sqrt(len(d))+policy['absolute_tolerance'],product=product,coordinate=coordinate)
        report['phase']='native';save()
    diff=np.array(report['price_record']['difference_batches']);base_price=np.array(report['price_record']['price_batches'])
    means=diff.mean(axis=1);se=diff.std(axis=1,ddof=1)/np.sqrt(policy['sobol_scrambles'])
    seen={(x['product'],x['seed']) for x in report['observations']}
    for j,product in enumerate(PRODUCTS):
        for seed in policy['gamma_seeds']:
            if (product,seed) in seen:continue
            p=product_request(base,j);samples=policy['gamma_samples_by_intervals'][str(intervals)]
            p['engine'].update(master_seed=seed,independent_sampling_units=samples);ctx=dict(product=product,seed=seed)
            a=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate_vega_kt_multi_step_gamma_chain_audit(
                list(range(intervals)),gamma_boundaries(p,c.spot,put_knots=True),variance_risk_directions=tangent.T.reshape(count,intervals,33).tolist(),
                subdivisions=policy['subdivisions'],relative_tolerance=policy['relative_tolerance'],gamma_relative_standard_error_limit=policy['gamma_relative_standard_error_limit'])
            check('raw_basis_market_Gamma_acceptance',a['status']!='path_gamma_chain_passed',0,**ctx)
            for row in a['rows']:
                check('row_recovery',not row['recovery_passed'],0,row=row['variance_row'],**ctx)
                check('row_conditional_AD_validation',not row['validation_passed'],0,row=row['variance_row'],**ctx)
            cov=np.array(a['joint_sample_covariance']).reshape(3*count,3*count)[count:2*count,count:2*count]
            check('covariance_symmetry',np.max(abs(cov-cov.T)),1e-10,**ctx)
            check('covariance_PSD',max(0.,-np.linalg.eigvalsh(cov).min()),1e-8,**ctx)
            check('price_vs_independent_payoff',abs(a['native_price']['mean']-base_price[:,j].mean()),6*np.hypot(a['native_price']['standard_error'],base_price[:,j].std(ddof=1)/np.sqrt(policy['sobol_scrambles']))+policy['absolute_tolerance'],**ctx)
            for coordinate,value in enumerate(a['recovered_directions']):
                check('covariance_diagonal_matches_SE',abs(cov[coordinate,coordinate]/samples-value['standard_error']**2),1e-10,coordinate=coordinate,**ctx)
                for k,bump in enumerate(policy['iv_bumps']):
                    check('Gamma_vs_independent_price_difference',abs(value['mean']-means[k,j,coordinate]),6*np.hypot(value['standard_error'],se[k,j,coordinate])+policy['absolute_tolerance'],coordinate=coordinate,bump=bump,**ctx)
                if product in ('call','put'):
                    expected=c._vegas(c.iv)[-3] if coordinate==count-3 else 0.
                    check('European_market_Vega_analytic',abs(value['mean']-expected),6*value['standard_error']+policy['absolute_tolerance'],coordinate=coordinate,**ctx)
            report['observations'].append(dict(**ctx,audit=a));save()
            print(json.dumps(dict(intervals=intervals,**ctx,status=a['status'],failed=sum(not x['passed'] for x in report['checks']))),flush=True)
    verify_complete(report)
    report.update(phase='complete',status='passed' if all(x['passed'] for x in report['checks']) else 'failed');save()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals',type=int,choices=(4,6),required=True)
    parser.add_argument('--basis-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    try:r=run(args.intervals,args.basis_file,args.output,resume=args.resume)
    except BacktestError as error:
        r=dict(document_kind='raw_variance_basis_validation',schema_version=1,status='rejected',diagnostic_only=True,
            error=dict(code=error.code,message=str(error)),provider_promotion=False,full_market_iv_bucket_acceptance='not_established')
    write(args.output,r)
    print(json.dumps(dict(status=r['status'],panels=len(r.get('observations',[])),checks=len(r.get('checks',[])),
        failed=sum(not c['passed'] for c in r.get('checks',[])),error=r.get('error'))),flush=True)
    return int(r['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
