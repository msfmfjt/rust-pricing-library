"""Three-step initial Gamma and the first-to-middle-to-terminal IV chain.

The preceding middle/terminal panel is an explicit input: its ten price-risk
coordinates are retained only after exact model/calibration/component replay.
The additional first parallel IV is checked with fresh independent price bumps.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from check_three_step_gamma import ROOT,with_product
from initial_three_reference import reference
from rust_pricing_backtest.numerics import threshold_check
from rust_pricing_backtest.three_step_smile import ThreeStepTerminalSmileCalibrator
from rust_pricing_backtest.three_step_full_smile import evaluate_three_step_market_gamma


def run(policy, previous_path):
    import rust_pricing as rp
    previous_bytes=previous_path.read_bytes();previous=json.loads(previous_bytes)
    if previous['document_kind']!='three_step_middle_terminal_gamma_validation' or previous['status']!='passed' or any(not x['passed'] for x in previous['checks']):
        raise ValueError('passed middle/terminal panel required')
    for key,value in previous['policy'].items():
        if key not in ['policy_id','scope'] and policy.get(key)!=value:raise ValueError('baseline policy mismatch: '+key)
    for name in ['scripts/middle_smile_reference.py','scripts/check_three_step_gamma.py','backtesting/rust_pricing_backtest/discrete_smile.py']:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=previous['source_sha256'][name]:raise ValueError('retained reference source mismatch: '+name)
    name='backtesting/rust_pricing_backtest/three_step_smile.py'
    current_calibrator=(ROOT/name).read_bytes()
    retained_calibrator=current_calibrator.split(b'    def calibrate_all_maturities(',1)[0].rstrip()+b'\n'
    if previous['source_sha256'][name] not in {hashlib.sha256(x).hexdigest() for x in [current_calibrator,retained_calibrator]}:
        raise ValueError('retained calibrator implementation changed; regenerate the ten-coordinate baseline')
    sources={Path(__file__),ROOT/'scripts/initial_three_reference.py',ROOT/'scripts/middle_smile_reference.py',ROOT/'scripts/check_three_step_gamma.py',ROOT/'backtesting/rust_pricing_backtest/three_step_full_smile.py',ROOT/'backtesting/rust_pricing_backtest/three_step_smile.py',ROOT/'backtesting/rust_pricing_backtest/discrete_smile.py',ROOT/'crates/pricing-python/src/lib.rs',ROOT/'crates/pricing/src/engine/risk/pricing.rs',ROOT/'fixtures/backtesting/initial-three-gamma-policy-v0.1.json'}
    for directory in ['crates/pricing/src/risk','crates/pricing/src/engine/risk','crates/pricing-python/src']:
        sources.update(p for p in (ROOT/directory).rglob('*.rs') if 'vegakt' in str(p))
    def hashes():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    frozen=hashes();checks=[];observations=[];prices=[];calibrations=[];models=[]
    old_prices={(p['profile'],p['product']):p for p in previous['price_records']}
    old_audits={(o['profile'],o['product'],o['seed']):o['audit'] for o in previous['observations']}
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    for source in previous['models']:
        base=deepcopy(source['request']);profile=source['profile'];ctx_profile=dict(profile=profile);models.append(source)
        c=ThreeStepTerminalSmileCalibrator(base);T,ta=c.all_maturity_iv_tangent();m=len(c.nodes);k=len(c.quote_nodes)
        check('first_terminal_IV_response_identity',ta['first_terminal_iv_response_residual'],1e-9,**ctx_profile)
        check('first_response_refinement_with_tail',ta['first_response_refinement_with_tail'],1e-9,**ctx_profile)
        # All bumped requests for the retained ten directions must be exactly
        # the same as the calibrated requests used by the preceding protocol.
        for bump in policy['iv_bumps']:
            for index,direction in enumerate(np.eye(2*k)):
                for sign in [-1.,1.]:
                    old,_=c.calibrate_joint(sign*bump*direction[:k],sign*bump*direction[k:])
                    new,_=c.calibrate_all_maturities(0.,sign*bump*direction[:k],sign*bump*direction[k:])
                    check('retained_ten_calibrated_requests_identical',int(old!=new),0.,bump=bump,coordinate=index,sign=sign,**ctx_profile)
        pairs=[]
        for bump in policy['iv_bumps']:
            pair=[]
            for sign in [1.,-1.]:
                q,a=c.calibrate_all_maturities(sign*bump,np.zeros(k),np.zeros(k));pair.append(q);calibrations.append(dict(**ctx_profile,bump=bump,sign=sign,audit=a))
                for stage in ['prefix_calibration','terminal_calibration']:
                    check('first_shift_recalibrated_quote_fit',a[stage]['maximum_verified_price_error_bound'],policy['quote_price_tolerance'],stage=stage,bump=bump,sign=sign,**ctx_profile)
                check('later_market_quotes_held_fixed',int(q['model']['reporting_iv_basis']['implied_volatilities'][k:]!=base['model']['reporting_iv_basis']['implied_volatilities'][k:]),0.,bump=bump,sign=sign,**ctx_profile)
            pairs.append(pair)
        for sign in [-1.,1.]:
            q,a=c.calibrate_all_maturities(sign*policy['stress_iv_bump'],np.zeros(k),np.zeros(k));calibrations.append(dict(**ctx_profile,stress=sign*policy['stress_iv_bump'],audit=a))
            actual=[reference(q,'call',96,strike)['price'] for strike in c.strikes]
            check('finite_first_IV_stress_holds_terminal_quotes_independent_prices',max(abs(np.array(actual)-a['terminal_calibration']['target_prices'])),policy['quote_price_tolerance'],sign=sign,**ctx_profile)
        for kind in policy['products']:
            ctx=dict(**ctx_profile,product=kind);p=with_product(base,kind)
            coarse=reference(p,kind,64);ref=reference(p,kind,96)
            b=np.r_[ref['initial_variance_sensitivity'],ref['middle_variance_sensitivities'],ref['terminal_variance_sensitivities']];population=T.T@b
            check('independent_initial_Delta_integral_refinement',abs(coarse['initial_variance_sensitivity']-ref['initial_variance_sensitivity']),policy['quadrature_derivative_tolerance'],**ctx)
            check('independent_three_step_price_refinement',abs(coarse['price']-ref['price']),1e-8,**ctx)
            retained=old_prices[(profile,kind)];oldref=retained['reference']
            check('retained_payoff_reference_price',abs(ref['price']-oldref['price']),1e-12,**ctx)
            for name in ['middle_variance_sensitivities','terminal_variance_sensitivities']:
                check('retained_payoff_reference_variance',max(abs(np.array(ref[name])-oldref[name])),1e-12,component=name,**ctx)
            refs=[];batches=[]
            for bump,pair in zip(policy['iv_bumps'],pairs):
                refs.append((reference(pair[0],kind,96)['price']-reference(pair[1],kind,96)['price'])/(2*bump))
                batch=[]
                for seed in policy['batch_seeds']:
                    values=[]
                    for model in pair:
                        q=with_product(model,kind);q['engine'].update(master_seed=seed,independent_sampling_units=policy['batch_samples']);q['risk']=dict(delta=False,vega=False,smile_dynamics=dict(type='sticky_log_moneyness'))
                        values.append(rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(q)),worker_threads=1,reduction_block_size=256).evaluate().value)
                    batch.append((values[0]-values[1])/(2*bump))
                batches.append(batch)
            refs=np.column_stack([refs,retained['reference_price_differences']])
            batches=np.concatenate([np.array(batches)[:,:,None],np.array(retained['native_batch_differences'])],axis=2)
            mean=batches.mean(axis=1);se=batches.std(axis=1,ddof=1)/np.sqrt(len(policy['batch_seeds']))
            for index,bump in enumerate(policy['iv_bumps']):
                check('all_tangent_directions_vs_recalibrated_reference_prices',max(abs(population-refs[index])),policy['bump_refinement_tolerance'],bump=bump,**ctx)
                check('new_first_direction_native_price_vs_reference',abs(mean[index,0]-refs[index,0]),6*se[index,0]+policy['derivative_absolute_tolerance'],bump=bump,**ctx)
            check('first_direction_independent_bump_refinement',abs(refs[0,0]-refs[1,0]),policy['bump_refinement_tolerance'],**ctx)
            diff=batches[0,:,0]-batches[1,:,0]
            check('first_direction_paired_native_bump_refinement',abs(diff.mean()),6*diff.std(ddof=1)/np.sqrt(len(diff))+policy['bump_refinement_tolerance'],**ctx)
            prices.append(dict(**ctx,reference=ref,population_quote_risks=population.tolist(),reference_price_differences=refs.tolist(),native_batch_differences=batches.tolist(),native_means=mean.tolist(),native_standard_errors=se.tolist(),fresh_coordinates=[0],retained_coordinates=list(range(1,1+2*k))))
            for seed in policy['gamma_seeds']:
                p['engine']['master_seed']=seed;sctx=dict(**ctx,seed=seed)
                d=evaluate_three_step_market_gamma(p,policy['gamma_bin_boundaries'],policy=policy['selection'],subdivisions=policy['subdivisions'],relative_tolerance=policy['relative_tolerance'])
                check('all_reachable_market_Gamma_acceptance',int(d['status']!='all_reachable_market_gamma_passed'),0.,**sctx)
                initial=d['initial_gamma_audit'];middle=d['middle_gamma_audit'];terminal=d['terminal_gamma_audit'];old=old_audits[(profile,kind,seed)]
                check('initial_Gamma_vs_independent_Delta_integral',abs(initial['fine']['recovered_variance_sensitivity']-ref['initial_variance_sensitivity']),policy['quadrature_derivative_tolerance'],**sctx)
                check('initial_price_vs_independent_price',abs(initial['fine']['interpolated_price']-ref['price']),initial['functional_tolerance'],**sctx)
                check('initial_Gamma_bin_sum',abs(sum(initial['fine']['gamma_bin_variance_sensitivities'])-initial['fine']['recovered_variance_sensitivity']),1e-10,**sctx)
                for name in ['price_tail_bound','variance_tail_bound','quadrature_price_difference','quadrature_variance_difference','price_refinement_difference','variance_refinement_difference','continuation_price_refinement_allowance','continuation_variance_refinement_allowance']:
                    check('initial_'+name,initial[name],initial['functional_tolerance'],**sctx)
                check('initial_bin_mass_refinement',initial['bin_mass_refinement_difference'],initial['mass_tolerance'],**sctx)
                check('retained_middle_audit_identical',int(middle!=old['middle_gamma_audit']),0.,**sctx)
                check('retained_ten_tangent_columns_identical',int(T[1:,1:].tolist()!=old['calibration_tangent']),0.,**sctx)
                ot=old['terminal_gamma_audit']
                for index,(newrow,oldrow) in enumerate(zip(terminal['pilot_candidates']+[terminal['validation']],ot['pilot_candidates']+[ot['validation']])):
                    if newrow is None:
                        check('retained_terminal_statistics_identical',1.,0.,sample=index,component='missing_validation',**sctx)
                        continue
                    for name in ['gamma_masses','variance_sensitivities','reconstruction_differences','quality']:
                        check('retained_terminal_statistics_identical',int(newrow[name]!=oldrow[name]),0.,sample=index,component=name,**sctx)
                    check('all_eleven_same_path_directions_conserved',max(abs(v['mean'])+6*v['standard_error'] for v in newrow['direction_differences']),policy['identity_absolute_tolerance'],sample=index,**sctx)
                check('retained_terminal_pilot_choice',int(terminal['selected_regularization']!=ot['selected_regularization']),0.,**sctx)
                if d['estimate'] is not None:
                    gm=np.array([x['mean'] for x in d['estimate']]);gs=np.array([x['standard_error'] for x in d['estimate']]);allowance=np.array(d['deterministic_functional_allowance'])
                    # Conserved terminal readouts preserve their population;
                    # deterministic first/middle Gamma errors remain explicit.
                    pop=T[0]*initial['estimate']+T[1:1+m].T@middle['estimate']+T[1+m:].T@b[1+m:]
                    for j in range(1+2*k):
                        check('all_Gamma_population_functional_bias',abs(pop[j]-population[j]),policy['gamma_quote_absolute_target']+policy['gamma_quote_relative_target']*abs(population[j]),coordinate=j,**sctx)
                        check('all_Gamma_vs_independent_recalibrated_prices',abs(gm[j]-refs[1,j]),6*gs[j]+allowance[j]+policy['derivative_absolute_tolerance'],coordinate=j,**sctx)
                        check('all_Gamma_vs_native_recalibrated_prices',abs(gm[j]-mean[1,j]),6*np.hypot(gs[j],se[1,j])+allowance[j]+policy['derivative_absolute_tolerance'],coordinate=j,**sctx)
                    check('retained_ten_market_Gamma_means',max(abs(gm[1:]-[x['mean'] for x in old['estimate']])),1e-10,**sctx)
                observations.append(dict(**sctx,audit=d));print(json.dumps(dict(**sctx,status=d['status'],selected=terminal['selected_regularization'],checks_so_far=len(checks),failed_so_far=sum(not c['passed'] for c in checks))),flush=True)
    if frozen!=hashes() or previous_path.read_bytes()!=previous_bytes:raise RuntimeError('validation source or retained evidence changed during run')
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='three_step_initial_and_all_reachable_market_Gamma_validation',schema_version=1,status='passed' if all(x['passed'] for x in checks) else 'failed',checks=checks,policy=policy,observations=observations,price_records=prices,calibration_audits=calibrations,models=models,source_sha256=frozen,native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),retained_evidence=dict(path=str(previous_path),sha256=hashlib.sha256(previous_bytes).hexdigest(),checks=len(previous['checks']),fresh_price_difference_coordinates=[0],retained_price_difference_coordinates=list(range(1,11))),full_market_iv_bucket_acceptance='not_established')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--previous-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=run(json.loads(args.policy.read_text()),args.previous_report);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(report['checks']),failed=sum(not c['passed'] for c in report['checks']))));return int(report['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
