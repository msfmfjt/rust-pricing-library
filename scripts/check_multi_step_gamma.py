"""Variable-interval terminal Gamma, independently checked in source and payoff space."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from multi_step_cases import ROOT,models,with_product
from multi_step_reference import density_reference,path_reference
from check_gamma_quote_conservation import constrained_inverse
from check_nonflat_gamma_recovery import own_quality
from rust_pricing_backtest.multi_step_smile import MultiStepTerminalSmileCalibrator
from rust_pricing_backtest.numerics import threshold_check


def run(policy):
    import rust_pricing as rp
    sources={Path(__file__),ROOT/'scripts/multi_step_cases.py',ROOT/'scripts/multi_step_reference.py',ROOT/'scripts/check_gamma_quote_conservation.py',ROOT/'scripts/check_nonflat_gamma_recovery.py',ROOT/'backtesting/rust_pricing_backtest/multi_step_smile.py',ROOT/'backtesting/rust_pricing_backtest/discrete_smile.py',ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json',ROOT/'fixtures/backtesting/multi-step-gamma-policy-v0.1.json',ROOT/'crates/pricing-python/src/lib.rs',ROOT/'crates/pricing/src/engine/risk/pricing.rs'}
    for directory in ['crates/pricing/src/risk','crates/pricing/src/engine/risk','crates/pricing-python/src']:
        sources.update(p for p in (ROOT/directory).rglob('*.rs') if 'vegakt' in str(p))
    def hashes():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    frozen=hashes();checks=[];observations=[];price_records=[];calibrations=[]
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    for case in models():
        base=case['request'];ctx0=dict(intervals=case['intervals'],profile=case['profile']);c=MultiStepTerminalSmileCalibrator(base);T,ta=c.terminal_iv_tangent();D=T.T
        check('nonflat_first_reported_maturity',float(np.ptp(c.iv[0])<1e-4 or c.indices[0]<2),0.,**ctx0)
        check('terminal_quote_tangent_identity',ta['iv_response_residual'],1e-9,**ctx0)
        B=density_reference(base,96)['operator'];coarse=density_reference(base,64)['operator']
        check('independent_operator_refinement',np.max(abs(B-coarse)),1e-10,**ctx0)
        inverses={lam:constrained_inverse(B,lam,D,preserve_total=False) for lam in policy['selection']['candidate_regularizations']}
        pairs=[]
        for bump in policy['iv_bumps']:
            row=[]
            for j,direction in enumerate(np.eye(len(c.quote_nodes))):
                pair=[]
                for sign in [1.,-1.]:
                    q,a=c.calibrate(sign*bump*direction);pair.append(q);calibrations.append(dict(**ctx0,bump=bump,quote=j,sign=sign,audit=a))
                    check('recalibrated_quote_fit',a['maximum_verified_price_error_bound'],policy['quote_price_tolerance'],bump=bump,quote=j,sign=sign,**ctx0)
                    check('earlier_market_quotes_preserved',int(q['model']['reporting_iv_basis']['implied_volatilities'][:-5]!=base['model']['reporting_iv_basis']['implied_volatilities'][:-5]),0.,**ctx0)
                    check('earlier_variance_rows_preserved',int(q['model']['local_variance_grid']['values'][:-66]!=base['model']['local_variance_grid']['values'][:-66]),0.,**ctx0)
                row.append(pair)
            pairs.append(row)
        for sign in [-1.,1.]:
            q,a=c.calibrate(sign*policy['stress_iv_bump']*np.ones(5));MultiStepTerminalSmileCalibrator(q)
            check('finite_terminal_stress_quote_fit',a['maximum_verified_price_error_bound'],policy['quote_price_tolerance'],sign=sign,**ctx0)
        for kind in policy['products']:
            ctx=dict(**ctx0,product=kind);p=with_product(base,kind)
            ref=path_reference(p,kind,power=policy['sobol_power'],scrambles=policy['sobol_scrambles']);rb=np.array(ref['batches']);quote_batches=rb[:,1:]@T
            qm=quote_batches.mean(axis=0);qs=quote_batches.std(axis=0,ddof=1)/np.sqrt(len(rb));price_batches=[]
            for bump,rows in zip(policy['iv_bumps'],pairs):
                values=[]
                for seed in policy['batch_seeds']:
                    batch=[]
                    for pair in rows:
                        prices=[]
                        for model in pair:
                            q=with_product(model,kind);q['engine'].update(master_seed=seed,independent_sampling_units=policy['batch_samples']);q['risk']=dict(delta=False,vega=False,smile_dynamics=dict(type='sticky_log_moneyness'))
                            prices.append(rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(q)),worker_threads=1,reduction_block_size=256).evaluate().value)
                        batch.append((prices[0]-prices[1])/(2*bump))
                    values.append(batch)
                price_batches.append(values)
            batches=np.array(price_batches);pm=batches.mean(axis=1);ps=batches.std(axis=1,ddof=1)/np.sqrt(len(policy['batch_seeds']))
            for i,bump in enumerate(policy['iv_bumps']):
                for j in range(5):check('native_recalibrated_price_vs_independent_conditional_risk',abs(pm[i,j]-qm[j]),6*np.hypot(ps[i,j],qs[j])+policy['derivative_absolute_tolerance'],quote=j,bump=bump,**ctx)
            delta=batches[0]-batches[1]
            for j in range(5):check('paired_native_bump_refinement',abs(delta[:,j].mean()),6*delta[:,j].std(ddof=1)/np.sqrt(len(delta))+policy['bump_refinement_tolerance'],quote=j,**ctx)
            price_records.append(dict(**ctx,reference=ref,reference_quote_means=qm.tolist(),reference_quote_standard_errors=qs.tolist(),native_batch_differences=batches.tolist(),native_means=pm.tolist(),native_standard_errors=ps.tolist()))
            for seed in policy['gamma_seeds']:
                p['engine'].update(master_seed=seed,independent_sampling_units=policy['gamma_samples']);sctx=dict(**ctx,seed=seed)
                d=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=256).evaluate_vega_kt_multi_step_terminal_gamma_audit((c.spot*np.exp(c.nodes)).tolist(),policy=rp.VegaKtGammaRecoveryPolicy(**policy['selection']),variance_conservation_directions=D.tolist())
                check('terminal_Gamma_acceptance',int(d['status']!='positive_time_policy_passed'),0.,**sctx)
                check('native_operator_vs_independent_density_integration',np.max(abs(np.array(d['forward_matrix'])-B)),1e-10,**sctx)
                for stage in d['operator_accuracy']['steps']:
                    check('source_probability',abs(stage['probability']-1),1e-10+stage['probability_tail_bound'],stage=stage['transition_count'],**sctx)
                    check('source_martingale',abs(stage['first_spot_moment']-100),1e-8+stage['spot_tail_bound'],stage=stage['transition_count'],**sctx)
                rows=d['pilot_candidates']+([d['validation']] if d['validation'] else [])
                for i,row in enumerate(rows):
                    check('unchanged_quality_verdict',int(own_quality(row,policy['selection'])!=row['quality']['passed']),0.,sample=i,**sctx)
                    b=np.array([v['mean'] for v in row['variance_sensitivities']]);actual=np.array([v['mean'] for v in row['gamma_masses']]);inv=inverses[row['relative_regularization']]
                    check('independent_primal_KKT',np.max(abs(inv@b-actual)),policy['solver_absolute_tolerance'],sample=i,**sctx)
                    check('all_same_path_quote_directions_conserved',max(abs(v['mean'])+6*v['standard_error'] for v in row['direction_differences']),policy['identity_absolute_tolerance'],sample=i,**sctx)
                if d['validation']:
                    row=d['validation'];npv=row['native_price'];check('native_price_vs_independent_prefix_paths',abs(npv['mean']-ref['mean'][0]),6*np.hypot(npv['standard_error'],ref['standard_error'][0])+policy['derivative_absolute_tolerance'],**sctx)
                    for j,v in enumerate(row['variance_sensitivities']):check('conditional_native_variance_vs_independent_prefix_paths',abs(v['mean']-ref['mean'][j+1]),6*np.hypot(v['standard_error'],ref['standard_error'][j+1])+policy['derivative_absolute_tolerance'],node=j,**sctx)
                if d['estimate']:
                    for j,v in enumerate(d['estimate']):
                        check('Gamma_vs_independent_conditional_quote_risk',abs(v['mean']-qm[j]),6*np.hypot(v['standard_error'],qs[j])+policy['derivative_absolute_tolerance'],quote=j,**sctx)
                        check('Gamma_vs_native_recalibrated_prices',abs(v['mean']-pm[1,j]),6*np.hypot(v['standard_error'],ps[1,j])+policy['derivative_absolute_tolerance'],quote=j,**sctx)
                observations.append(dict(**sctx,audit=d));print(json.dumps(dict(**sctx,status=d['status'],checks=len(checks),failed=sum(not x['passed'] for x in checks))),flush=True)
    if frozen!=hashes():raise RuntimeError('validation implementation changed during run')
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='multi_step_terminal_Gamma_validation',schema_version=1,status='passed' if all(x['passed'] for x in checks) else 'failed',checks=checks,policy=policy,models=models(),observations=observations,price_records=price_records,calibration_audits=calibrations,source_sha256=frozen,native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),full_market_iv_bucket_acceptance='not_established')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    r=run(json.loads(a.policy.read_text()));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not x['passed'] for x in r['checks']))));return int(r['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
