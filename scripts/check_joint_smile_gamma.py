"""Coupled first-parallel/terminal-quote IV Gamma versus recalibrated prices.

The unchanged second-Gamma policy is frozen before exotic prices. Initial
Gamma, second Gamma, calibration tangent and joint covariance are independently
checked. All failed gates remain in the report; there is no fallback or retuning.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from check_discrete_smile_calibration import ROOT, base_request, reference_price
from check_path_gamma_recovery import request
from check_initial_smile_gamma import reference as initial_reference
from check_nonflat_gamma_recovery import variance_reference, variance_operator, constrained_inverse, own_quality
from rust_pricing_backtest.discrete_smile import TwoStepSmileCalibrator
from rust_pricing_backtest.joint_smile import evaluate_joint_market_gamma
from rust_pricing_backtest.numerics import threshold_check


def run(policy):
    import rust_pricing as rp
    paths = [Path(__file__), ROOT/'scripts/check_initial_smile_gamma.py', ROOT/'scripts/check_nonflat_gamma_recovery.py',
        ROOT/'scripts/check_gamma_quote_conservation.py', ROOT/'scripts/check_discrete_smile_calibration.py',
        ROOT/'scripts/check_path_gamma_recovery.py', ROOT/'backtesting/rust_pricing_backtest/discrete_smile.py',
        ROOT/'backtesting/rust_pricing_backtest/joint_smile.py', ROOT/'crates/pricing-python/src/lib.rs']
    for folder, pattern in [('crates/pricing/src/risk','vegakt*.rs'),('crates/pricing/src/risk/vegakt_convolution','*.rs'),
        ('crates/pricing/src/engine/risk','vegakt*.rs'),('crates/pricing/src/engine/risk/vegakt_initial_smile','*.rs'),
        ('crates/pricing-python/src','vegakt*.rs')]: paths += sorted((ROOT/folder).glob(pattern))
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    checks, observations, price_records, calibrations = [], [], [], []
    def check(name, value, limit, **ctx): checks.append(threshold_check(name,float(value),float(limit),**ctx))
    for profile in policy['nonflat_profiles']:
        p=base_request(dict(policy,**profile),'nonflat');c=TwoStepSmileCalibrator(p)
        tangent,ta=c.joint_iv_tangent();directions=tangent.T;n=len(directions);pc=dict(profile=profile['name'])
        check('joint_quote_tangent_identity',ta['maximum_joint_iv_response_residual'],1e-9,**pc)
        second=c.variance[1];B=variance_operator(c.spot,c.v0,c.dt,c.nodes,second,c.spot*np.exp(c.nodes))
        inverses={lam:constrained_inverse(B,lam,tangent[1:,1:].T,False) for lam in policy['selection']['candidate_regularizations']}
        # Freeze every bumped model before evaluating any exotic or selecting Gamma.
        pairs=[]
        for bump in policy['iv_bumps']:
            row=[]
            for index,v in enumerate(np.eye(n)):
                pair=[]
                for sign in [1.,-1.]:
                    q,ca=c.calibrate_joint(sign*bump*v[0],sign*bump*v[1:]);pair.append(q)
                    calibrations.append(dict(**pc,bump=bump,coordinate=index,sign=sign,audit=ca))
                    check('bumped_quote_calibration',ca['maximum_verified_price_error_bound'],policy['quote_price_tolerance'],bump=bump,coordinate=index,sign=sign,**pc)
                row.append(pair)
            pairs.append(row)
        for sign in [-1.,1.]:
            q,ca=c.calibrate_joint(sign*policy['stress_iv_bump'],np.zeros(n-1))
            check('finite_first_iv_stress_calibration',ca['maximum_verified_price_error_bound'],policy['quote_price_tolerance'],sign=sign,**pc)
            check('finite_first_iv_stress_preserves_terminal_quotes',np.max(abs(np.array(q['model']['reporting_iv_basis']['implied_volatilities'])[n-1:]-c.iv[1])),0.,sign=sign,**pc)
            calibrations.append(dict(**pc,stress=sign*policy['stress_iv_bump'],audit=ca))
        for kind in policy['products']:
            product=request(kind,samples=16)['product'];q=deepcopy(p);q['product']=product;ctx=dict(**pc,product=kind)
            initial=initial_reference(q,kind,policy['gamma_bin_boundaries'],192)['recovered_variance_sensitivity']
            b=variance_reference(kind,c.spot,100.,c.v0,c.dt,c.nodes,second);population=directions@np.r_[initial,b]
            refs=[];batches=[]
            for bump,models in zip(policy['iv_bumps'],pairs):
                ref=np.array([(reference_price(up,kind,100.,192)-reference_price(down,kind,100.,192))/(2*bump) for up,down in models]);refs.append(ref)
                batch=[]
                for seed in policy['batch_seeds']:
                    row=[]
                    for pair in models:
                        values=[]
                        for model in pair:
                            shifted=deepcopy(model);shifted['product']=product
                            shifted['engine'].update(master_seed=seed,independent_sampling_units=policy['batch_samples'])
                            shifted['risk']=dict(delta=False,vega=False,smile_dynamics=dict(type='sticky_log_moneyness'))
                            values.append(rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(shifted)),worker_threads=1,reduction_block_size=256).evaluate().value)
                        row.append((values[0]-values[1])/(2*bump))
                    batch.append(row)
                batches.append(batch)
            refs=np.array(refs);batches=np.array(batches);mean=batches.mean(axis=1);se=batches.std(axis=1,ddof=1)/np.sqrt(len(policy['batch_seeds']))
            for j,bump in enumerate(policy['iv_bumps']):
                check('joint_population_vs_recalibrated_price',max(abs(population-refs[j])),policy['bump_refinement_tolerance'],bump=bump,**ctx)
                check('native_recalibrated_price_vs_reference',max(abs(mean[j]-refs[j])/(6*se[j]+policy['derivative_absolute_tolerance'])),1.,bump=bump,**ctx)
            check('independent_bump_refinement',max(abs(refs[0]-refs[1])),policy['bump_refinement_tolerance'],**ctx)
            diff=batches[0]-batches[1]
            check('paired_native_bump_refinement',max(abs(diff.mean(axis=0))/(6*diff.std(axis=0,ddof=1)/np.sqrt(len(diff))+policy['bump_refinement_tolerance'])),1.,**ctx)
            if kind=='call':check('atm_call_first_iv_compensation',abs(population[0]),policy['algebra_tolerance'],**ctx)
            price_records.append(dict(**ctx,reference_derivatives=refs.tolist(),batch_derivatives=batches.tolist(),derivative_means=mean.tolist(),derivative_standard_errors=se.tolist(),population=population.tolist()))
            for seed in policy['gamma_seeds']:
                q['engine']['master_seed']=seed
                result=evaluate_joint_market_gamma(q,policy['gamma_bin_boundaries'],policy=policy['selection'],subdivisions=policy['subdivisions'],relative_tolerance=policy['relative_tolerance'])
                audit=result['audit'];sctx=dict(**ctx,seed=seed)
                check('joint_component_and_validation_gates',int(result['status']!='joint_direction_policy_passed'),0,**sctx)
                check('same_frozen_tangent',np.max(abs(np.array(result['calibration_tangent'])-tangent)),0.,**sctx)
                chosen=None
                for row in audit['second']['pilot_candidates']:
                    check('independent_unchanged_pilot_quality',int(own_quality(row,policy['selection'])!=row['passed']),0,regularization=row['relative_regularization'],**sctx)
                    if chosen is None and row['passed']:chosen=row['relative_regularization']
                check('strongest_pilot_feasible_selection',int(chosen!=audit['second']['selected_regularization']),0,**sctx)
                if result['estimate'] is not None:
                    s=result['estimate'];N=s['independent_sampling_units'];C=np.array(s['joint_sample_covariance']).reshape(3*n,3*n)
                    moments=s['native_directions']+s['recovered_directions']+s['differences'];variances=np.array([m['sample_variance'] for m in moments])
                    check('joint_covariance_diagonal',np.max(abs(np.diag(C)-variances))/max(1.,max(variances)),1e-12,**sctx)
                    check('joint_covariance_symmetry',np.max(abs(C-C.T)),0.,**sctx)
                    check('joint_covariance_psd',max(0.,-np.linalg.eigvalsh(C)[0])/max(1.,np.linalg.norm(C)),1e-12,**sctx)
                    transform=np.column_stack([-np.eye(n),np.eye(n)])
                    check('same_path_difference_covariance',np.max(abs(transform@C[:2*n,:2*n]@transform.T-C[2*n:,2*n:]))/max(1.,np.linalg.norm(C)),1e-12,**sctx)
                    R=inverses[chosen];popgamma=directions[:,0]*initial+directions[:,1:]@B@R@b
                    gamma=np.array([m['mean'] for m in s['recovered_directions']]);gse=np.array([m['standard_error'] for m in s['recovered_directions']])
                    for i in range(n):
                        check('joint_population_gamma_bias',abs(popgamma[i]-population[i]),policy['gamma_quote_absolute_target']+policy['gamma_quote_relative_target']*abs(population[i]),coordinate=i,**sctx)
                        check('joint_gamma_vs_recalibrated_native_price',abs(gamma[i]-mean[1,i]),6*np.hypot(gse[i],se[1,i])+policy['derivative_absolute_tolerance'],coordinate=i,**sctx)
                        check('joint_gamma_vs_independent_price',abs(gamma[i]-refs[1,i]),6*gse[i]+policy['derivative_absolute_tolerance'],coordinate=i,**sctx)
                    for i,combination in enumerate([np.ones(n),np.r_[1.,0.,0.,-1.,0.,0.],np.array([.4,.1,-.2,.3,-.2,.1])]):
                        gv=float(combination@C[n:2*n,n:2*n]@combination/N)
                        sample=batches[1]@combination;pv=float(sample.var(ddof=1)/len(sample))
                        check('combined_risk_full_covariance_vs_native_price',abs(combination@gamma-sample.mean()),6*np.sqrt(max(0.,gv)+pv)+policy['derivative_absolute_tolerance']*sum(abs(combination)),combination=i,**sctx)
                    result['population_gamma']=popgamma.tolist()
                observations.append(dict(**sctx,result=result))
                print(json.dumps(dict(**sctx,status=result['status'])),flush=True)
    if hashes!={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:raise RuntimeError('validation sources changed during execution')
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='joint_smile_gamma_validation',schema_version=1,status='passed' if all(c['passed'] for c in checks) else 'failed',policy=policy,checks=checks,observations=observations,price_records=price_records,calibration_audits=calibrations,source_sha256=hashes,native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),full_market_iv_bucket_acceptance='not_established')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(json.loads(args.policy.read_text()));args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],checks=len(result['checks']),failed=sum(not c['passed'] for c in result['checks']))));return int(result['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
