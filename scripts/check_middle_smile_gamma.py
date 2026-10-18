"""Three-step conditional middle Gamma and coupled middle/terminal market IVs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from check_three_step_gamma import ROOT, request, with_product, independent_operator
from middle_smile_reference import reference
from check_gamma_quote_conservation import constrained_inverse
from check_nonflat_gamma_recovery import own_quality
from rust_pricing_backtest.numerics import threshold_check
from rust_pricing_backtest.three_step_smile import ThreeStepTerminalSmileCalibrator
from rust_pricing_backtest.middle_smile import evaluate_middle_terminal_market_gamma


def run(policy):
    import rust_pricing as rp
    sources = {Path(__file__),ROOT/'scripts/middle_smile_reference.py',ROOT/'scripts/check_three_step_gamma.py',
        ROOT/'scripts/check_gamma_quote_conservation.py',ROOT/'scripts/check_nonflat_gamma_recovery.py',
        ROOT/'backtesting/rust_pricing_backtest/three_step_smile.py',ROOT/'backtesting/rust_pricing_backtest/discrete_smile.py',
        ROOT/'backtesting/rust_pricing_backtest/middle_smile.py',ROOT/'crates/pricing-python/src/lib.rs',
        ROOT/'crates/pricing/src/engine/risk/pricing.rs',ROOT/'fixtures/backtesting/middle-smile-gamma-policy-v0.1.json'}
    for directory in ['crates/pricing/src/risk','crates/pricing/src/engine/risk','crates/pricing-python/src']:
        sources.update(p for p in (ROOT/directory).rglob('*.rs') if 'vegakt' in str(p))
    def hashes():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    frozen=hashes();checks=[];observations=[];price_records=[];calibrations=[];models=[]
    def check(name,value,limit,**context):checks.append(threshold_check(name,float(value),float(limit),**context))
    for profile in policy['nonflat_profiles']:
        base=request(policy,profile);models.append(dict(profile=profile['name'],request=base))
        c=ThreeStepTerminalSmileCalibrator(base);T,ta=c.joint_iv_tangent();m=len(c.nodes);k=len(c.quote_nodes)
        D=np.array(ta['terminal_conservation_basis']);readouts=T[m:].T;pc=dict(profile=profile['name'])
        check('joint_market_tangent_identity',ta['iv_response_residual'],1e-9,**pc)
        check('cross_maturity_jacobian_refinement_and_tail',ta['cross_jacobian_refinement_with_tail'],1e-9,**pc)
        B=independent_operator(base);inverses={lam:constrained_inverse(B,lam,D,False) for lam in policy['selection']['candidate_regularizations']}
        pairs=[]
        for bump in policy['iv_bumps']:
            row=[]
            for i,direction in enumerate(np.eye(2*k)):
                pair=[]
                for sign in [1.,-1.]:
                    q,a=c.calibrate_joint(sign*bump*direction[:k],sign*bump*direction[k:]);pair.append(q)
                    calibrations.append(dict(**pc,bump=bump,coordinate=i,sign=sign,audit=a))
                    for stage in ['middle_calibration','terminal_calibration']:
                        check('joint_quote_fit',a[stage]['maximum_verified_price_error_bound'],policy['quote_price_tolerance'],stage=stage,bump=bump,coordinate=i,sign=sign,**pc)
                    check('first_variance_row_unchanged',max(abs(np.array(q['model']['local_variance_grid']['values'])[:m]-c.variance[0])),0.,bump=bump,coordinate=i,sign=sign,**pc)
                row.append(pair)
            pairs.append(row)
        for sign in [-1.,1.]:
            q,a=c.calibrate_joint(np.full(k,sign*policy['stress_iv_bump']),np.zeros(k));calibrations.append(dict(**pc,stress=sign*policy['stress_iv_bump'],audit=a))
            actual=[reference(q,'call',96,strike)['price'] for strike in c.strikes]
            check('finite_middle_shift_holds_terminal_quotes_independent_prices',max(abs(np.array(actual)-a['terminal_calibration']['target_prices'])),policy['quote_price_tolerance'],sign=sign,**pc)
        for kind in policy['products']:
            ctx=dict(**pc,product=kind);p=with_product(base,kind)
            coarse=reference(p,kind,64);ref=reference(p,kind,96)
            b2=np.array(ref['middle_variance_sensitivities']);b3=np.array(ref['terminal_variance_sensitivities']);population=T.T@np.r_[b2,b3]
            check('independent_continuation_price_refinement',abs(coarse['price']-ref['price']),1e-8,**ctx)
            for key in ['middle_variance_sensitivities','terminal_variance_sensitivities']:
                check('independent_Gamma_functional_refinement',max(abs(np.array(coarse[key])-ref[key])),policy['quadrature_derivative_tolerance'],component=key,**ctx)
            refs=[];batches=[]
            for bump,model_pairs in zip(policy['iv_bumps'],pairs):
                refs.append([(reference(up,kind,96)['price']-reference(down,kind,96)['price'])/(2*bump) for up,down in model_pairs])
                batch=[]
                for seed in policy['batch_seeds']:
                    row=[]
                    for pair in model_pairs:
                        values=[]
                        for model in pair:
                            q=with_product(model,kind);q['engine'].update(master_seed=seed,independent_sampling_units=policy['batch_samples'])
                            q['risk']=dict(delta=False,vega=False,smile_dynamics=dict(type='sticky_log_moneyness'))
                            values.append(rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(q)),worker_threads=1,reduction_block_size=256).evaluate().value)
                        row.append((values[0]-values[1])/(2*bump))
                    batch.append(row)
                batches.append(batch)
            refs=np.array(refs);batches=np.array(batches);mean=batches.mean(axis=1);se=batches.std(axis=1,ddof=1)/np.sqrt(len(policy['batch_seeds']))
            for i,bump in enumerate(policy['iv_bumps']):
                check('joint_tangent_vs_independent_recalibrated_price',max(abs(population-refs[i])),policy['bump_refinement_tolerance'],bump=bump,**ctx)
                check('native_recalibrated_price_vs_independent_price',max(abs(mean[i]-refs[i])/(6*se[i]+policy['derivative_absolute_tolerance'])),1.,bump=bump,**ctx)
            check('independent_bump_refinement',max(abs(refs[0]-refs[1])),policy['bump_refinement_tolerance'],**ctx)
            diff=batches[0]-batches[1]
            check('paired_native_bump_refinement',max(abs(diff.mean(axis=0))/(6*diff.std(axis=0,ddof=1)/np.sqrt(len(diff))+policy['bump_refinement_tolerance'])),1.,**ctx)
            price_records.append(dict(**ctx,reference=ref,population_quote_risks=population.tolist(),reference_price_differences=refs.tolist(),native_batch_differences=batches.tolist(),native_means=mean.tolist(),native_standard_errors=se.tolist()))
            for seed in policy['gamma_seeds']:
                p['engine']['master_seed']=seed;sctx=dict(**ctx,seed=seed)
                audit=evaluate_middle_terminal_market_gamma(p,policy['gamma_bin_boundaries'],policy=policy['selection'],subdivisions=policy['subdivisions'],relative_tolerance=policy['relative_tolerance'])
                check('joint_Gamma_acceptance',int(audit['status']!='middle_terminal_market_gamma_passed'),0.,**sctx)
                middle=audit['middle_gamma_audit'];terminal=audit['terminal_gamma_audit'];f=middle['fine_history']
                check('middle_price_vs_independent_continuation',abs(f['recovered_price']-ref['price']),middle['functional_tolerance'],**sctx)
                check('middle_Gamma_vs_independent_integration_by_parts',max(abs(np.array(f['variance_node_sensitivities'])-b2)),policy['quadrature_derivative_tolerance'],**sctx)
                check('Gamma_bins_sum_to_parallel_middle_variance',abs(sum(f['gamma_bin_variance_sensitivities'])-sum(f['variance_node_sensitivities'])),1e-10,**sctx)
                for name in ['inner_price_refinement','inner_variance_refinement','score_price_refinement','score_variance_refinement','price_tail_bound','variance_tail_bound']:
                    check('middle_'+name,f[name],middle['functional_tolerance'],**sctx)
                check('middle_bin_mass_refinement',f['inner_bin_mass_refinement'],middle['mass_tolerance'],**sctx)
                for name in ['outer_price_refinement','outer_variance_refinement','outer_bin_mass_refinement']:
                    check(name,middle[name],middle['mass_tolerance'] if 'mass' in name else middle['functional_tolerance'],**sctx)
                check('terminal_operator_independent_non_gaussian_source',np.max(abs(np.array(terminal['forward_matrix'])-B)),1e-10,**sctx)
                choice=None
                for index,row in enumerate(terminal['pilot_candidates']+([terminal['validation']] if terminal['validation'] else [])):
                    lam=row['relative_regularization'];R=inverses[lam];rctx=dict(**sctx,sample='pilot' if index<len(terminal['pilot_candidates']) else 'validation',regularization=lam)
                    check('terminal_original_quality_verdict',int(own_quality(row,policy['selection'])!=row['passed']),0.,**rctx)
                    if index<len(terminal['pilot_candidates']) and choice is None and row['passed']:choice=lam
                    native=np.array([v['mean'] for v in row['variance_sensitivities']]);gamma=np.array([v['mean'] for v in row['gamma_masses']])
                    check('terminal_independent_primal_KKT',max(abs(gamma-R@native)),policy['solver_absolute_tolerance'],**rctx)
                    check('all_ten_same_path_readouts_conserved',max(abs(v['mean'])+6*v['standard_error'] for v in row['direction_differences']),policy['identity_absolute_tolerance'],**rctx)
                check('frozen_terminal_pilot_selection',int(choice!=terminal['selected_regularization']),0.,**sctx)
                if audit['estimate'] is not None:
                    popgamma=T[:m].T@np.array(f['variance_node_sensitivities'])+readouts@B@inverses[choice]@b3
                    gm=np.array([v['mean'] for v in audit['estimate']]);gs=np.array([v['standard_error'] for v in audit['estimate']]);allowance=np.array(audit['deterministic_middle_functional_allowance'])
                    for j in range(2*k):
                        check('joint_population_Gamma_bias',abs(popgamma[j]-population[j]),policy['gamma_quote_absolute_target']+policy['gamma_quote_relative_target']*abs(population[j]),quote=j,**sctx)
                        check('joint_Gamma_vs_independent_recalibrated_price',abs(gm[j]-refs[1,j]),6*gs[j]+allowance[j]+policy['derivative_absolute_tolerance'],quote=j,**sctx)
                        check('joint_Gamma_vs_native_recalibrated_price',abs(gm[j]-mean[1,j]),6*np.hypot(gs[j],se[1,j])+allowance[j]+policy['derivative_absolute_tolerance'],quote=j,**sctx)
                observations.append(dict(**sctx,audit=audit))
                print(json.dumps(dict(**sctx,status=audit['status'],selected=terminal['selected_regularization'])),flush=True)
    if frozen!=hashes():raise RuntimeError('validation sources changed during run')
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='three_step_middle_terminal_gamma_validation',schema_version=1,
        status='passed' if all(c['passed'] for c in checks) else 'failed',checks=checks,policy=policy,
        models=models,observations=observations,price_records=price_records,calibration_audits=calibrations,
        source_sha256=frozen,native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        full_market_iv_bucket_acceptance='not_established')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=run(json.loads(args.policy.read_text()));args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(report['checks']),failed=sum(not c['passed'] for c in report['checks']))))
    return int(report['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
