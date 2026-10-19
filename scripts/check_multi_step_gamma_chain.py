"""Fresh all-reported-maturity price bumps for conditional path Gamma chains."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from multi_step_cases import ROOT, models, with_product
from multi_step_reference import density_reference, path_reference, black
from rust_pricing_backtest.multi_step_full_smile import MultiStepSmileCalibrator
from rust_pricing_backtest.numerics import threshold_check


def independent_quote_prices(p):
    states=density_reference(p,96)['states']
    g=p['model']['local_variance_grid'];basis=p['model']['reporting_iv_basis'];spot=p['market']['spot']
    times=np.array(g['time_nodes']);nodes=np.array(g['log_forward_moneyness_nodes']);v=np.array(g['values']).reshape(len(times),-1)
    strikes=spot*np.exp(basis['log_forward_moneyness_nodes']);out=[]
    for t in basis['maturity_nodes']:
        i=int(np.flatnonzero(times==t)[0]);x,pdf=states[i-2]
        calls,_=black((spot*np.exp(x))[:,None],strikes[None,:],(np.interp(x,nodes,v[i-1])*(times[i]-times[i-1]))[:,None])
        out.append(pdf@calls)
    return np.array(out)


def run(policy, checkpoint=None):
    import rust_pricing as rp
    paths={Path(__file__),ROOT/'scripts/multi_step_cases.py',ROOT/'scripts/multi_step_reference.py',
        ROOT/'backtesting/rust_pricing_backtest/multi_step_full_smile.py',ROOT/'backtesting/rust_pricing_backtest/multi_step_smile.py',
        ROOT/'backtesting/rust_pricing_backtest/discrete_smile.py',ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json',
        ROOT/'fixtures/backtesting/path-gamma-chain-policy-v0.1.json',ROOT/'crates/pricing-python/src/lib.rs',
        ROOT/'crates/pricing/src/engine/risk/pricing.rs',ROOT/'rust_pricing.pyi'}
    for directory in ['crates/pricing/src/risk','crates/pricing/src/engine/risk','crates/pricing-python/src']:
        paths.update(p for p in (ROOT/directory).rglob('*.rs') if 'vegakt' in str(p))
    def hashes():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    frozen=hashes();checks=[];observations=[];prices=[];calibration=[]
    def check(name,observed,limit,**ctx):checks.append(threshold_check(name,float(observed),float(limit),**ctx))
    for case in models():
        base=case['request'];n=case['intervals'];ctx0=dict(intervals=n,profile=case['profile']);c=MultiStepSmileCalibrator(base)
        tangent,ta=c.all_maturity_iv_tangent();count=c.iv.size;directions=tangent.T.reshape(count,len(c.indices),len(c.nodes)).tolist()
        check('all_market_tangent_refinement',ta['iv_response_residual'],1e-9,**ctx0)
        check('first_market_is_nonflat',int(np.ptp(c.iv[0])<=1e-4),0,**ctx0)
        pairs=[]
        for bump in policy['iv_bumps']:
            qlist=[]
            for coordinate in range(count):
                pair=[]
                for sign in [1.,-1.]:
                    shifts=np.zeros(count);shifts[coordinate]=sign*bump
                    q,a=c.calibrate_all_maturities(shifts.reshape(c.iv.shape));pair.append(q)
                    target=np.array([black(c.spot,c.strikes,t*iv**2)[0] for t,iv in zip(c.maturities,c.iv+shifts.reshape(c.iv.shape))])
                    error=float(np.max(abs(independent_quote_prices(q)-target)))
                    check('independent_all_maturity_recalibrated_quote_fit',error,policy['quote_price_tolerance'],coordinate=coordinate,bump=bump,sign=sign,**ctx0)
                    v=np.array(q['model']['local_variance_grid']['values']).reshape(c.variance.shape)
                    fixed=[i for i in range(n) if i not in c.calibrated_rows]
                    check('unreported_variance_rows_fixed',np.max(abs(v[fixed]-c.variance[fixed])),0,**ctx0)
                    calibration.append(dict(**ctx0,coordinate=coordinate,bump=bump,sign=sign,audit=a,independent_price_error=error))
                qlist.append(pair)
            pairs.append(qlist)
        for sign in [-1,1]:
            shifts=np.zeros_like(c.iv);shifts[0]=sign*policy['stress_iv_bump']
            q,a=c.calibrate_all_maturities(shifts)
            target=np.array([black(c.spot,c.strikes,t*iv**2)[0] for t,iv in zip(c.maturities,c.iv+shifts)])
            check('first_market_stress_later_quotes_fixed',np.max(abs(independent_quote_prices(q)-target)),policy['quote_price_tolerance'],sign=sign,**ctx0)
        for kind in policy['products']:
            ctx=dict(**ctx0,product=kind);p=with_product(base,kind);independent=[];native=[]
            for bump,qlist in zip(policy['iv_bumps'],pairs):
                refcols=[];nativecols=[]
                for pair in qlist:
                    refs=[];values=[]
                    for q in pair:
                        q=with_product(q,kind)
                        ref=path_reference(q,kind,power=policy['sobol_power'],scrambles=policy['sobol_scrambles'])
                        refs.append(np.array(ref['batches'])[:,0]);batches=[]
                        for seed in policy['price_seeds']:
                            q['engine'].update(master_seed=seed,independent_sampling_units=policy['price_samples'])
                            q['risk']=dict(delta=False,vega=False,smile_dynamics=dict(type='sticky_log_moneyness'))
                            batches.append(rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(q)),worker_threads=1,reduction_block_size=256).evaluate().value)
                        values.append(np.array(batches))
                    refcols.append((refs[0]-refs[1])/(2*bump));nativecols.append((values[0]-values[1])/(2*bump))
                independent.append(np.array(refcols).T);native.append(np.array(nativecols).T)
            independent=np.array(independent);native=np.array(native)
            rm=independent.mean(axis=1);rs=independent.std(axis=1,ddof=1)/np.sqrt(policy['sobol_scrambles'])
            pm=native.mean(axis=1);ps=native.std(axis=1,ddof=1)/np.sqrt(len(policy['price_seeds']))
            for j in range(count):
                for i,bump in enumerate(policy['iv_bumps']):
                    check('native_vs_independent_recalibrated_price_Greek',abs(pm[i,j]-rm[i,j]),6*np.hypot(ps[i,j],rs[i,j])+policy['absolute_tolerance'],coordinate=j,bump=bump,**ctx)
                for label,batches in [('independent',independent),('native',native)]:
                    diff=batches[0,:,j]-batches[1,:,j]
                    check(label+'_paired_bump_refinement',abs(diff.mean()),6*diff.std(ddof=1)/np.sqrt(len(diff))+policy['absolute_tolerance'],coordinate=j,**ctx)
            prices.append(dict(**ctx,independent_price_difference_batches=independent.tolist(),native_price_difference_batches=native.tolist()))
            for seed in policy['gamma_seeds']:
                p['engine'].update(master_seed=seed,independent_sampling_units=policy['gamma_samples']);sctx=dict(**ctx,seed=seed)
                a=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=256).evaluate_vega_kt_multi_step_gamma_chain_audit(c.calibrated_rows,policy['gamma_bins'],variance_risk_directions=directions,subdivisions=policy['subdivisions'],relative_tolerance=policy['relative_tolerance'],gamma_relative_standard_error_limit=policy['gamma_relative_standard_error_limit'])
                check('all_reported_market_Gamma_acceptance',int(a['status']!='path_gamma_chain_passed'),0,**sctx)
                def upper(v):return abs(v['mean'])+6*v['standard_error']
                for row in a['rows']:
                    rctx=dict(**sctx,row=row['variance_row'])
                    check('conditional_variance_refinement',upper(row['variance_refinement'])+upper(row['variance_tail_bound']),a['functional_tolerance'],**rctx)
                    check('conditional_price_refinement',upper(row['price_refinement'])+upper(row['price_tail_bound']),a['functional_tolerance'],**rctx)
                    check('Gamma_bin_relative_SE',row['gamma_relative_standard_error'] if row['gamma_relative_standard_error'] is not None else 1e100,policy['gamma_relative_standard_error_limit'],**rctx)
                    check('bin_risk_equals_node_risk',abs(sum(x['mean'] for x in row['gamma_bin_variance_sensitivities'])-sum(x['mean'] for x in row['recovered_variance_nodes'])),1e-9,**rctx)
                    for j,v in enumerate(row['node_differences']):
                        cond=row['terminal_ad_conditional_variances'][j]
                        stderr=max(v['standard_error'],np.sqrt(upper(cond)/policy['gamma_samples']))
                        check('native_node_validation',abs(v['mean']),6*stderr+a['functional_tolerance'],node=j,**rctx)
                for j,v in enumerate(a['recovered_directions']):
                    means=np.array([[x['mean'] for x in row['recovered_variance_nodes']] for row in a['rows']]).ravel()
                    check('joint_direction_composition',abs(v['mean']-means@tangent[:,j]),1e-9,coordinate=j,**sctx)
                    check('Gamma_vs_independent_recalibrated_price_Greek',abs(v['mean']-rm[1,j]),6*np.hypot(v['standard_error'],rs[1,j])+policy['absolute_tolerance'],coordinate=j,**sctx)
                    check('Gamma_vs_native_recalibrated_price_Greek',abs(v['mean']-pm[1,j]),6*np.hypot(v['standard_error'],ps[1,j])+policy['absolute_tolerance'],coordinate=j,**sctx)
                observations.append(dict(**sctx,tangent_audit=ta,calibration_tangent=tangent.tolist(),audit=a))
                if checkpoint is not None:
                    checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    checkpoint.write_text(json.dumps(dict(status='in_progress',policy=policy,checks=checks,
                        observations=observations,price_records=prices,calibration_audits=calibration,
                        source_sha256=frozen),allow_nan=False)+'\n')
                print(json.dumps(dict(**sctx,status=a['status'],checks=len(checks),failed=sum(not x['passed'] for x in checks))),flush=True)
    if frozen!=hashes():raise RuntimeError('validation source changed during run')
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='multi_step_reported_market_gamma_validation',schema_version=1,policy=policy,
        status='passed' if all(x['passed'] for x in checks) else 'failed',checks=checks,observations=observations,
        price_records=prices,calibration_audits=calibration,source_sha256=frozen,
        native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        full_market_iv_bucket_acceptance='not_established')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    r=run(json.loads(a.policy.read_text()),a.output.with_suffix('.partial.json'));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not x['passed'] for x in r['checks']))));return int(r['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
