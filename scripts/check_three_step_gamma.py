"""Three-step terminal Gamma against actual non-Gaussian source and prices."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from check_path_gamma_recovery import ROOT
from check_gamma_quote_conservation import constrained_inverse
from check_nonflat_gamma_recovery import own_quality
from rust_pricing_backtest.numerics import flat_request,threshold_check
from rust_pricing_backtest.discrete_smile import TwoStepSmileCalibrator
from rust_pricing_backtest.three_step_smile import ThreeStepTerminalSmileCalibrator


def rule(cuts,order):
    z,w=np.polynomial.legendre.leggauss(order);cuts=np.array(sorted(set(cuts)))
    return ((cuts[:-1]+cuts[1:])[:,None]/2+np.diff(cuts)[:,None]*z/2).ravel(),(np.diff(cuts)[:,None]*w/2).ravel()


def reference(p,kind,order=64,strike=100.):
    """Independent nested integration over conditional, rather than global, windows.

    Split on variance knots and each history-dependent payoff kink. No source
    density builder or calibration Jacobian is used in this payoff oracle.
    """
    g=p['model']['local_variance_grid'];t=np.array(g['time_nodes']);nodes=np.array(g['log_forward_moneyness_nodes']);v=np.array(g['values']).reshape(4,-1)
    spot=p['market']['spot'];v0=v[0,0]*t[1];sd0=math.sqrt(v0);lo=-v0/2-12*sd0;hi=-v0/2+12*sd0
    x1,w1=rule([lo,hi]+[x for x in nodes if lo<x<hi]+([math.log(strike/spot)] if lo<math.log(strike/spot)<hi else []),order)
    w1*=np.exp(-.5*((x1+v0/2)/sd0)**2)/(sd0*math.sqrt(2*math.pi));price=0.;b=np.zeros(len(nodes));gl=np.polynomial.legendre.leggauss(order)
    for x,a in zip(x1,w1):
        s1=spot*math.exp(x);v2=np.interp(x,nodes,v[1])*(t[2]-t[1]);sd2=math.sqrt(v2);mu=x-v2/2;l=mu-12*sd2;h=mu+12*sd2
        extra=[]
        if kind=='lookback':extra=[math.log(max(strike,s1)/spot)]
        elif kind in ['asian','fixed_asian']:
            threshold=(3*strike-s1) if kind=='asian' else (4*strike-98-s1)
            if threshold>0:extra=[math.log(threshold/spot)]
        cuts=np.array(sorted(set([l,h]+[y for y in list(nodes)+extra if l<y<h])))
        z,w=gl;y=((cuts[:-1]+cuts[1:])[:,None]/2+np.diff(cuts)[:,None]*z/2).ravel();weights=(np.diff(cuts)[:,None]*w/2).ravel()
        weights*=a*np.exp(-.5*((y-mu)/sd2)**2)/(sd2*math.sqrt(2*math.pi));s2=spot*np.exp(y);sd3=np.sqrt(np.interp(y,nodes,v[2])*(t[3]-t[2]))
        if kind=='call':k=np.full_like(s2,strike);q=1.;constant=0.
        elif kind=='asian':k=3*strike-s1-s2;q=1/3;constant=0.
        elif kind=='fixed_asian':k=4*strike-98-s1-s2;q=.25;constant=0.
        else:k=np.maximum(max(strike,s1),s2);q=1.;constant=np.maximum(np.maximum(s1,s2)-strike,0.)
        valid=k>0;d1=np.zeros_like(k);d1[valid]=np.log(s2[valid]/k[valid])/sd3[valid]+sd3[valid]/2
        calls=np.where(valid,s2*ndtr(d1)-k*ndtr(d1-sd3),s2-k)
        price+=float(weights@(q*calls+constant))
        derivative=np.where(valid,q*s2*np.exp(-d1*d1/2)/math.sqrt(2*math.pi)*(t[3]-t[2])/(2*sd3),0.)
        left=np.clip(np.searchsorted(nodes,y,side='right')-1,0,len(nodes)-2);right_weight=np.clip((y-nodes[left])/(nodes[left+1]-nodes[left]),0,1)
        b+=np.bincount(left,weights*derivative*(1-right_weight),minlength=len(nodes))+np.bincount(left+1,weights*derivative*right_weight,minlength=len(nodes))
    return dict(price=price,variance_sensitivities=b.tolist())


def request(policy,profile,kind='call'):
    default=json.loads((ROOT/'fixtures/backtesting/numerical-policy-v0.1.json').read_text());default['expiry_days']=192
    p=json.loads(flat_request(default,seed=policy['gamma_seeds'][0],samples=policy['gamma_samples'],steps=3,nodes=policy['local_nodes']).to_json())
    g=p['model']['local_variance_grid'];b=p['model']['reporting_iv_basis'];nodes=np.array(g['log_forward_moneyness_nodes']);qnodes=np.array(policy['quote_log_nodes'])
    b.update(log_forward_moneyness_nodes=qnodes.tolist(),implied_volatilities=[.2]*(3*len(qnodes)),shape=[3,len(qnodes)])
    p['risk']['vega_kt']['log_forward_moneyness_nodes']=qnodes.tolist()
    prefix=deepcopy(p);pg=prefix['model']['local_variance_grid'];pb=prefix['model']['reporting_iv_basis']
    pg.update(time_nodes=g['time_nodes'][:3],values=g['values'][:3*len(nodes)],shape=[3,len(nodes)])
    pb.update(maturity_nodes=b['maturity_nodes'][:2],implied_volatilities=[.2]*(2*len(qnodes)),shape=[2,len(qnodes)])
    fitted,_=TwoStepSmileCalibrator(prefix).calibrate(profile['base_skew']*qnodes+profile['base_curvature']*qnodes*qnodes)
    third=.04+profile['third_skew']*nodes+profile['third_curvature']*nodes*nodes
    g['values']=fitted['model']['local_variance_grid']['values'][:2*len(nodes)]+third.tolist()*2
    b['implied_volatilities'][:2*len(qnodes)]=fitted['model']['reporting_iv_basis']['implied_volatilities']
    # Synthetic terminal market quotes are defined by the independent price
    # oracle; no recovered Gamma or exotic enters their construction.
    iv=[];T=g['time_nodes'][-1];spot=p['market']['spot']
    for strike in spot*np.exp(qnodes):
        price=reference(p,'call',96,float(strike))['price'];lo,hi=.001,2.
        for _ in range(60):
            mid=(lo+hi)/2;sd=mid*math.sqrt(T);d1=math.log(spot/strike)/sd+sd/2
            if spot*ndtr(d1)-strike*ndtr(d1-sd)>price:hi=mid
            else:lo=mid
        iv.append((lo+hi)/2)
    b['implied_volatilities'][2*len(qnodes):]=iv
    return with_product(p,kind)


def with_product(p,kind):
    p=deepcopy(p);dates=p['risk']['vega_kt']['maturity_nodes'];expiry=dates[-1]
    if kind in ['asian','fixed_asian']:
        obs=[dict(date=d,weight=1/3 if kind=='asian' else .25,value=dict(type='unknown')) for d in dates]
        if kind=='fixed_asian':
            from datetime import date,timedelta
            obs.insert(0,dict(date=str(date.fromisoformat(p['valuation_date'])-timedelta(days=1)),weight=.25,value=dict(type='known',fixing=98.)))
        p['product']=dict(type='arithmetic_asian',underlying_id=1,currency_id=2,strike=100.,notional=1.,side=dict(type='call'),observations=obs,payment_date=expiry)
    elif kind=='lookback':p['product']=dict(type='fixed_lookback',underlying_id=1,currency_id=2,strike=100.,notional=1.,side=dict(type='call'),monitoring_dates=dates,payment_date=expiry)
    return p


def audit(p,policy,directions,workers=1):
    import rust_pricing as rp
    nodes=p['model']['local_variance_grid']['log_forward_moneyness_nodes'];spot=p['market']['spot']
    return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256).evaluate_vega_kt_three_step_terminal_gamma_audit(
        (spot*np.exp(nodes)).tolist(),policy=rp.VegaKtGammaRecoveryPolicy(**policy),variance_conservation_directions=directions.tolist())


def independent_operator(p,order=96):
    g=p['model']['local_variance_grid'];nodes=np.array(g['log_forward_moneyness_nodes']);v=np.array(g['values']).reshape(4,-1);t=np.array(g['time_nodes']);spot=p['market']['spot']
    v0=v[0,0]*t[1];sd0=math.sqrt(v0);lo=-v0/2-12*sd0;hi=-v0/2+12*sd0
    x1,w1=rule([lo,hi]+[x for x in nodes if lo<x<hi],order);w1*=np.exp(-.5*((x1+v0/2)/sd0)**2)/(sd0*math.sqrt(2*math.pi))
    # A deliberately wider, symmetric window than the native Chernoff window.
    vmax=v0+max(v[1])*(t[2]-t[1]);bound=14*math.sqrt(vmax)+vmax
    x2,w2=rule([-bound,bound]+[x for x in nodes if -bound<x<bound],order)
    v2=np.interp(x1,nodes,v[1])*(t[2]-t[1]);density=np.zeros_like(x2)
    for a in range(0,len(x1),128):
        variance=v2[a:a+128];mu=x1[a:a+128]-variance/2
        density+=(np.exp(-.5*(x2[:,None]-mu)**2/variance)/np.sqrt(2*math.pi*variance))@w1[a:a+128]
    hats=np.array([np.interp(x2,nodes,unit) for unit in np.eye(len(nodes))]).T
    sd3=np.sqrt(np.interp(x2,nodes,v[2])*(t[3]-t[2]));d1=(x2[:,None]-nodes)/sd3[:,None]+sd3[:,None]/2
    B=hats.T@((w2*density*spot*np.exp(x2)*(t[3]-t[2])/(2*sd3))[:,None]*np.exp(-d1*d1/2)/math.sqrt(2*math.pi))
    return B


def run(policy):
    import rust_pricing as rp
    paths=[Path(__file__),ROOT/'backtesting/rust_pricing_backtest/three_step_smile.py',ROOT/'backtesting/rust_pricing_backtest/discrete_smile.py',
        ROOT/'scripts/check_gamma_quote_conservation.py',ROOT/'scripts/check_nonflat_gamma_recovery.py',ROOT/'crates/pricing-python/src/lib.rs',ROOT/'crates/pricing/src/engine/risk/pricing.rs']
    for folder,pattern in [('crates/pricing/src/risk','vegakt*.rs'),('crates/pricing/src/risk/vegakt_convolution','*.rs'),('crates/pricing/src/engine/risk','vegakt*.rs'),('crates/pricing-python/src','vegakt*.rs')]:paths+=sorted((ROOT/folder).glob(pattern))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};checks=[];observations=[];prices=[];calibrations=[];models=[]
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    for profile in policy['nonflat_profiles']:
        base=request(policy,profile);models.append(dict(profile=profile['name'],request=base));c=ThreeStepTerminalSmileCalibrator(base);U,ta=c.terminal_iv_tangent();D=U.T;pc=dict(profile=profile['name'])
        B=independent_operator(base);Bcoarse=independent_operator(base,64)
        check('independent_operator_refinement',np.max(abs(B-Bcoarse)),1e-10,**pc)
        check('terminal_iv_tangent_identity',ta['iv_response_residual'],1e-9,**pc)
        inverses={lam:constrained_inverse(B,lam,D,False) for lam in policy['selection']['candidate_regularizations']}
        pairs=[]
        for bump in policy['iv_bumps']:
            row=[]
            for index,direction in enumerate(np.eye(len(D))):
                pair=[]
                for sign in [1.,-1.]:
                    q,ca=c.calibrate(sign*bump*direction);pair.append(q);calibrations.append(dict(**pc,bump=bump,coordinate=index,sign=sign,audit=ca))
                    check('terminal_quote_fit',ca['maximum_verified_price_error_bound'],policy['quote_price_tolerance'],bump=bump,coordinate=index,sign=sign,**pc)
                    check('earlier_variance_rows_unchanged',np.max(abs(np.array(q['model']['local_variance_grid']['values'])[:2*len(c.nodes)]-c.variance[:2].ravel())),0.,bump=bump,coordinate=index,sign=sign,**pc)
                row.append(pair)
            pairs.append(row)
        for sign in [-1.,1.]:
            q,ca=c.calibrate(sign*policy['stress_iv_bump']*np.ones(len(D)));calibrations.append(dict(**pc,stress=sign*policy['stress_iv_bump'],audit=ca))
            target=ca['target_prices'];actual=[reference(q,'call',96,k)['price'] for k in c.strikes]
            check('finite_terminal_iv_stress_independent_prices',max(abs(np.array(actual)-target)),policy['quote_price_tolerance'],sign=sign,**pc)
        for kind in policy['products']:
            p=with_product(base,kind);ctx=dict(**pc,product=kind)
            ref=reference(p,kind,64);refined=reference(p,kind,96);b=np.array(refined['variance_sensitivities']);population=D@b
            check('independent_payoff_price_refinement',abs(ref['price']-refined['price']),1e-8,**ctx)
            check('independent_payoff_variance_refinement',max(abs(np.array(ref['variance_sensitivities'])-b)),2e-6,**ctx)
            refs=[];batches=[]
            for bump,models_for_bump in zip(policy['iv_bumps'],pairs):
                refs.append([(reference(up,kind,96)['price']-reference(down,kind,96)['price'])/(2*bump) for up,down in models_for_bump])
                batch=[]
                for seed in policy['batch_seeds']:
                    row=[]
                    for pair in models_for_bump:
                        values=[]
                        for model in pair:
                            q=with_product(model,kind);q['engine'].update(master_seed=seed,independent_sampling_units=policy['batch_samples']);q['risk']=dict(delta=False,vega=False,smile_dynamics=dict(type='sticky_log_moneyness'))
                            values.append(rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(q)),worker_threads=1,reduction_block_size=256).evaluate().value)
                        row.append((values[0]-values[1])/(2*bump))
                    batch.append(row)
                batches.append(batch)
            refs=np.array(refs);batches=np.array(batches);mean=batches.mean(axis=1);se=batches.std(axis=1,ddof=1)/np.sqrt(len(policy['batch_seeds']))
            for i,bump in enumerate(policy['iv_bumps']):
                check('calibration_tangent_vs_independent_price',max(abs(population-refs[i])),policy['bump_refinement_tolerance'],bump=bump,**ctx)
                check('native_recalibrated_price_vs_reference',max(abs(mean[i]-refs[i])/(6*se[i]+policy['derivative_absolute_tolerance'])),1.,bump=bump,**ctx)
            check('independent_bump_refinement',max(abs(refs[0]-refs[1])),policy['bump_refinement_tolerance'],**ctx)
            diff=batches[0]-batches[1];check('paired_native_bump_refinement',max(abs(diff.mean(axis=0))/(6*diff.std(axis=0,ddof=1)/np.sqrt(len(diff))+policy['bump_refinement_tolerance'])),1.,**ctx)
            prices.append(dict(**ctx,reference=refined,population_quote_risks=population.tolist(),reference_price_differences=refs.tolist(),native_batch_differences=batches.tolist(),native_means=mean.tolist(),native_standard_errors=se.tolist()))
            for seed in policy['gamma_seeds']:
                p['engine']['master_seed']=seed;result=audit(p,policy['selection'],D);sctx=dict(**ctx,seed=seed)
                check('heldout_recovery_gate',int(result['status']!='positive_time_policy_passed'),0,**sctx)
                check('actual_non_gaussian_operator',np.max(abs(np.array(result['forward_matrix'])-B)),1e-10,**sctx)
                accuracy=result['operator_accuracy'];check('native_operator_refinement',accuracy['operator_refinement_difference'],accuracy['operator_entry_tolerance'],**sctx)
                check('omitted_operator_tail',accuracy['omitted_operator_entry_bound'],accuracy['operator_entry_tolerance'],**sctx)
                check('source_probability',abs(accuracy['source_probability']-1.),1e-10+accuracy['source_probability_tail_bound'],**sctx)
                check('source_martingale',abs(accuracy['source_first_spot_moment']-c.spot),1e-10*c.spot+accuracy['source_spot_tail_bound'],**sctx)
                choice=None
                for i,row in enumerate(result['pilot_candidates']+([result['validation']] if result['validation'] else [])):
                    lam=row['relative_regularization'];R=inverses[lam];rctx=dict(**sctx,sample='pilot' if i<len(result['pilot_candidates']) else 'validation',regularization=lam)
                    check('unchanged_quality_verdict',int(own_quality(row,policy['selection'])!=row['passed']),0,**rctx)
                    if i<len(result['pilot_candidates']) and choice is None and row['passed']:choice=lam
                    native=np.array([v['mean'] for v in row['variance_sensitivities']]);nse=np.array([v['standard_error'] for v in row['variance_sensitivities']]);gamma=np.array([v['mean'] for v in row['gamma_masses']])
                    check('native_variance_AD_vs_independent_payoff',max(abs(native-b)/(6*nse+policy['derivative_absolute_tolerance'])),1.,**rctx)
                    check('independent_primal_KKT',max(abs(gamma-R@native)),policy['solver_absolute_tolerance'],**rctx)
                    check('all_spatial_residuals_retained',max(abs(B@gamma-native-[v['mean'] for v in row['reconstruction_differences']])),policy['solver_absolute_tolerance'],**rctx)
                    check('same_path_quote_direction_conservation',max(abs(v['mean'])+6*v['standard_error'] for v in row['direction_differences']),policy['identity_absolute_tolerance'],**rctx)
                check('frozen_strongest_pilot_selection',int(choice!=result['selected_regularization']),0,**sctx)
                record=dict(**sctx,audit=result)
                if result['estimate'] is not None:
                    popgamma=D@B@inverses[choice]@b;gm=np.array([v['mean'] for v in result['estimate']]);gs=np.array([v['standard_error'] for v in result['estimate']])
                    for j in range(len(D)):
                        check('population_quote_gamma_bias',abs(popgamma[j]-population[j]),policy['gamma_quote_absolute_target']+policy['gamma_quote_relative_target']*abs(population[j]),quote=j,**sctx)
                        check('gamma_vs_independent_recalibrated_price',abs(gm[j]-refs[1,j]),6*gs[j]+policy['derivative_absolute_tolerance'],quote=j,**sctx)
                        check('gamma_vs_native_recalibrated_price',abs(gm[j]-mean[1,j]),6*np.hypot(gs[j],se[1,j])+policy['derivative_absolute_tolerance'],quote=j,**sctx)
                    record['population_gamma']=popgamma.tolist()
                observations.append(record);print(json.dumps(dict(**sctx,status=result['status'],selected=result['selected_regularization'])),flush=True)
    if hashes!={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:raise RuntimeError('validation sources changed during run')
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='three_step_terminal_gamma_validation',schema_version=1,status='passed' if all(c['passed'] for c in checks) else 'failed',checks=checks,policy=policy,models=models,observations=observations,price_records=prices,calibration_audits=calibrations,source_sha256=hashes,native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),full_market_iv_bucket_acceptance='not_established')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    r=run(json.loads(args.policy.read_text()));args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not c['passed'] for c in r['checks']))));return int(r['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
