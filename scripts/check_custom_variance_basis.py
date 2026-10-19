"""Explicit custom metrics versus independently recalibrated Call/Put prices."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backtesting'))
from custom_basis_cases import KINDS,basis_for
from multi_step_cases import models,with_product
from extension_reference import path_prices as call_prices,PRODUCTS as CALLS
from extension_put_reference import path_prices as put_prices,PRODUCTS as PUTS,request as put_request
from multi_step_reference import black
from check_multi_step_gamma_chain import independent_quote_prices
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.multi_step_provider import gamma_boundaries
from rust_pricing_backtest.numerics import threshold_check

PRODUCTS=('call','asian_call','fixed_asian_call','lookback_call','put','asian_put','fixed_asian_put','lookback_put','lookback_past_put','lookback_today_put')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def product_request(base,index):
    return with_product(base,CALLS[index]) if index<4 else put_request(base,PUTS[index-4])


def prices(base,policy):
    kwargs=dict(power=policy['sobol_power'],scrambles=policy['sobol_scrambles'])
    return np.concatenate((call_prices(base,**kwargs),put_prices(base,**kwargs)),axis=1)


def provenance(policy_path):
    import rust_pricing as rp
    prior_path=ROOT/'design/validation/local-vol-extension-put-v0.1.json'
    prior=json.loads(prior_path.read_text())
    # Existing calibration, native sources and acceptance helpers are retained.
    for name,h in prior['source_sha256'].items():
        if name.startswith(('crates/','backtesting/rust_pricing_backtest/')):
            if sha(ROOT/name)!=h:raise ValueError('previous accepted implementation changed: '+name)
    binary=sha(Path(getattr(rp,'rust_pricing',rp).__file__))
    if binary!=prior['native_binary_sha256']:raise ValueError('same native binary required')
    paths=list((ROOT/'crates').rglob('*.rs'))+list((ROOT/'backtesting/rust_pricing_backtest').glob('*.py'))
    paths += [Path(__file__),policy_path,ROOT/'scripts/custom_basis_cases.py',ROOT/'scripts/extension_reference.py',
        ROOT/'scripts/extension_put_reference.py',ROOT/'scripts/multi_step_cases.py',ROOT/'scripts/multi_step_reference.py',
        ROOT/'scripts/check_multi_step_gamma_chain.py',ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json',
        ROOT/'backtesting/tests/test_variance_basis.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))},binary,sha(prior_path)


def run(policy,policy_path,checkpoint,intervals=None,basis=None):
    import rust_pricing as rp
    assert policy['products']==list(PRODUCTS) and policy['bases']==list(KINDS)
    frozen,binary,prior=provenance(policy_path)
    checks=[];observations=[];records=[];calibrations=[];cases=[]
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    def save(status):
        r=dict(document_kind='custom_variance_basis_validation',schema_version=1,status=status,policy=policy,cases=cases,
            checks=checks,observations=observations,price_records=records,calibration_audits=calibrations,
            source_sha256=frozen,native_binary_sha256=binary,prior_acceptance_sha256=prior,
            diagnostic_only=True,full_market_iv_bucket_acceptance='not_established')
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_text(json.dumps(r,allow_nan=False)+'\n');return r
    for case in models():
        n=case['intervals'];base=case['request']
        if intervals is not None and n!=intervals:continue
        base_price=prices(base,policy)
        for kind in KINDS:
            if basis is not None and kind!=basis:continue
            ctx=dict(intervals=n,profile=case['profile'],basis=kind)
            b,ba=basis_for(base,kind);c=MultiStepExtensionCalibrator(base,extension='custom',variance_basis=b)
            tangent,ta=c.all_maturity_iv_tangent();count=c.iv.size
            check('custom_basis_hash_matches_calibration',ba['basis_hash']!=ta['basis_hash'],0,**ctx)
            check('full_row_IV_response',ta['iv_response_residual'],1e-9,**ctx)
            check('all_transition_rows_have_market_response',np.any(np.max(abs(tangent.reshape(n,33,count)),axis=(1,2))==0),0,**ctx)
            cases.append(dict(**ctx,basis_audit=ba,tangent_audit=ta,calibration_tangent=tangent.tolist()))
            independent=[]
            for bump in policy['iv_bumps']:
                columns=[]
                for coordinate in range(count):
                    pair=[];variances=[]
                    for sign in [1.,-1.]:
                        shifts=np.zeros(count);shifts[coordinate]=sign*bump
                        q,a=c.calibrate_all_maturities(shifts.reshape(c.iv.shape))
                        variance=np.array(q['model']['local_variance_grid']['values']).reshape(c.variance.shape)
                        variances.append(variance[:-1].ravel())
                        target=np.array([black(c.spot,c.strikes,t*v*v)[0] for t,v in zip(c.maturities,c.iv+shifts.reshape(c.iv.shape))])
                        error=float(np.max(abs(independent_quote_prices(q)-target)))
                        check('independent_recalibrated_quote_fit',error,policy['quote_price_tolerance'],coordinate=coordinate,bump=bump,sign=sign,**ctx)
                        check('unused_terminal_row_fixed',np.max(abs(variance[-1]-c.variance[-1])),0,coordinate=coordinate,bump=bump,sign=sign,**ctx)
                        check('explicit_basis_parameter_increments',np.max(abs((variance[:-1]-c.variance[:-1]).ravel()-b@a['parameter_increments'])),1e-17,coordinate=coordinate,bump=bump,sign=sign,**ctx)
                        calibrations.append(dict(**ctx,coordinate=coordinate,bump=bump,sign=sign,audit=a,independent_price_error=error))
                        pair.append(prices(q,policy))
                    check('tangent_vs_actual_recalibration',np.max(abs((variances[0]-variances[1])/(2*bump)-tangent[:,coordinate])),2e-5,coordinate=coordinate,bump=bump,**ctx)
                    columns.append((pair[0]-pair[1])/(2*bump))
                independent.append(np.stack(columns,axis=2))
                print(json.dumps(dict(**ctx,bump=bump,stage='independent_Call_Put_prices',failed=sum(not x['passed'] for x in checks))),flush=True);save('in_progress')
            for sign in [-1,1]:
                shifts=np.zeros_like(c.iv);shifts[0]=sign*policy['stress_iv_bump'];q,a=c.calibrate_all_maturities(shifts)
                target=np.array([black(c.spot,c.strikes,t*v*v)[0] for t,v in zip(c.maturities,c.iv+shifts)])
                check('finite_first_smile_stress_holds_later_quotes',np.max(abs(independent_quote_prices(q)-target)),policy['quote_price_tolerance'],sign=sign,**ctx)
                calibrations.append(dict(**ctx,stress=sign*policy['stress_iv_bump'],audit=a))
            independent=np.array(independent);means=independent.mean(axis=1);se=independent.std(axis=1,ddof=1)/np.sqrt(policy['sobol_scrambles'])
            records.append(dict(**ctx,products=list(PRODUCTS),price_batches=base_price.tolist(),difference_batches=independent.tolist(),means=means.tolist(),standard_errors=se.tolist()))
            for j,product in enumerate(PRODUCTS):
                for coordinate in range(count):
                    d=independent[0,:,j,coordinate]-independent[1,:,j,coordinate]
                    check('paired_independent_bump_refinement',abs(d.mean()),6*d.std(ddof=1)/np.sqrt(len(d))+policy['absolute_tolerance'],product=product,coordinate=coordinate,**ctx)
                for seed in policy['gamma_seeds']:
                    p=product_request(base,j);samples=policy['gamma_samples_by_intervals'][str(n)]
                    p['engine'].update(master_seed=seed,independent_sampling_units=samples);sctx=dict(**ctx,product=product,seed=seed)
                    a=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256).evaluate_vega_kt_multi_step_gamma_chain_audit(
                        list(range(n)),gamma_boundaries(p,c.spot,put_knots=True),variance_risk_directions=tangent.T.reshape(count,n,33).tolist(),
                        subdivisions=policy['subdivisions'],relative_tolerance=policy['relative_tolerance'],gamma_relative_standard_error_limit=policy['gamma_relative_standard_error_limit'])
                    check('custom_market_Gamma_acceptance',a['status']!='path_gamma_chain_passed',0,**sctx)
                    for row in a['rows']:
                        check('row_recovery',not row['recovery_passed'],0,row=row['variance_row'],**sctx)
                        check('row_conditional_AD_validation',not row['validation_passed'],0,row=row['variance_row'],**sctx)
                    cov=np.array(a['joint_sample_covariance']).reshape(3*count,3*count)[count:2*count,count:2*count]
                    check('covariance_symmetry',np.max(abs(cov-cov.T)),1e-10,**sctx)
                    check('covariance_PSD',max(0.,-np.linalg.eigvalsh(cov).min()),1e-8,**sctx)
                    check('price_vs_independent_payoff',abs(a['native_price']['mean']-base_price[:,j].mean()),6*np.hypot(a['native_price']['standard_error'],base_price[:,j].std(ddof=1)/np.sqrt(policy['sobol_scrambles']))+policy['absolute_tolerance'],**sctx)
                    for coordinate,value in enumerate(a['recovered_directions']):
                        check('covariance_diagonal_matches_SE',abs(cov[coordinate,coordinate]/samples-value['standard_error']**2),1e-10,coordinate=coordinate,**sctx)
                        for k,bump in enumerate(policy['iv_bumps']):
                            check('Gamma_vs_independent_price_difference',abs(value['mean']-means[k,j,coordinate]),6*np.hypot(value['standard_error'],se[k,j,coordinate])+policy['absolute_tolerance'],coordinate=coordinate,bump=bump,**sctx)
                        if product in ('call','put'):
                            expected=c._vegas(c.iv)[-3] if coordinate==count-3 else 0.
                            check('European_market_Vega_analytic',abs(value['mean']-expected),6*value['standard_error']+policy['absolute_tolerance'],coordinate=coordinate,**sctx)
                    observations.append(dict(**sctx,audit=a));save('in_progress')
                    print(json.dumps(dict(**sctx,status=a['status'],checks=len(checks),failed=sum(not x['passed'] for x in checks))),flush=True)
    if any(sha(ROOT/n)!=h for n,h in frozen.items()):raise RuntimeError('source changed during validation')
    return save('passed' if all(x['passed'] for x in checks) else 'failed')


def aggregate(paths,policy,policy_path):
    frozen,binary,prior=provenance(policy_path);reports=[json.loads(p.read_text()) for p in paths]
    expected={(n,b) for n in policy['intervals'] for b in policy['bases']};seen=set()
    for r in reports:
        if r['source_sha256']!=frozen or r['native_binary_sha256']!=binary or r['policy']!=policy:
            raise ValueError('shards must use identical frozen sources, native binary and policy')
        for c in r['cases']:
            key=(c['intervals'],c['basis'])
            if key not in expected or key in seen:raise ValueError('unexpected or duplicated custom basis case')
            seen.add(key)
        for c in r['cases']:
            obs=[x for x in r['observations'] if (x['intervals'],x['basis'])==(c['intervals'],c['basis'])]
            keys=[(x['product'],x['seed']) for x in obs]
            if len(keys)!=len(set(keys)) or set(keys)!={(k,s) for k in PRODUCTS for s in policy['gamma_seeds']}:
                raise ValueError('incomplete or duplicated product/seed matrix')
    if seen!=expected:raise ValueError('incomplete custom basis matrix')
    result={k:v for k,v in reports[0].items() if k not in ['cases','checks','observations','price_records','calibration_audits','status']}
    for key in ['cases','checks','observations','price_records','calibration_audits']:result[key]=[x for r in reports for x in r[key]]
    result['status']='passed' if all(r['status']=='passed' for r in reports) and all(x['passed'] for x in result['checks']) else 'failed'
    result['shard_sha256']={str(p):sha(p) for p in paths};return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--intervals',type=int,choices=[4,6]);parser.add_argument('--basis',choices=KINDS)
    parser.add_argument('--aggregate',type=Path,nargs='+')
    a=parser.parse_args();pp=a.policy.resolve();policy=json.loads(pp.read_text())
    if a.aggregate and (a.intervals is not None or a.basis is not None):parser.error('aggregate cannot filter cases')
    r=aggregate(a.aggregate,policy,pp) if a.aggregate else run(policy,pp,a.output.with_suffix('.partial.json'),a.intervals,a.basis)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],panels=len(r['observations']),checks=len(r['checks']),failed=sum(not x['passed'] for x in r['checks']))))
    return int(r['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
