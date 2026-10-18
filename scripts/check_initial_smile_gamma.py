"""Initial continuation Gamma: independent analytic slopes, integration by parts,
initial-variance price bumps and native AD. Future local variance is held fixed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from check_discrete_smile_calibration import ROOT, base_request, reference_price
from check_path_gamma_recovery import request
from rust_pricing_backtest.numerics import threshold_check


def reference(p, kind, boundaries, order=96):
    g=p['model']['local_variance_grid'];nodes=np.array(g['log_forward_moneyness_nodes'])
    t=np.array(g['time_nodes']);v=np.array(g['values']).reshape(3,-1)[1];s0=p['market']['spot'];strike=p['product']['strike']
    v0=g['values'][0]*t[1];sd0=math.sqrt(v0);lo,hi=-v0/2-12*sd0,-v0/2+12*sd0
    cdf=np.vectorize(lambda z:.5*math.erfc(-z/math.sqrt(2)))
    def slope(x, side='right'):
        x=np.atleast_1d(x);s=s0*np.exp(x);iv=np.interp(x,nodes,v);sd=np.sqrt(iv*(t[2]-t[1]))
        # One-sided derivative includes slope jumps at every native variance knot.
        j=np.searchsorted(nodes,x,side=side)-1;dv=np.zeros_like(x);valid=(j>=0)&(j<len(nodes)-1)
        dv[valid]=np.diff(v)[j[valid]]/np.diff(nodes)[j[valid]]
        is_upper=s>strike if side=='left' else s>=strike
        if kind=='call':k=np.full_like(s,strike);dk=np.zeros_like(s);q=1.
        elif kind=='asian':k=2*strike-s;dk=np.full_like(s,-1.);q=.5
        elif kind=='fixed_asian':k=2*strike-49.-.5*s;dk=np.full_like(s,-.5);q=.5
        else:k=np.maximum(strike,s);dk=is_upper.astype(float);q=1.
        valid=k>0;d1=np.zeros_like(s);d1[valid]=np.log(s[valid]/k[valid])/sd[valid]+sd[valid]/2
        result=q*np.where(valid,cdf(d1)-dk*cdf(d1-sd)+np.exp(-d1*d1/2)/np.sqrt(2*np.pi)*(t[2]-t[1])*dv/(2*sd),1-dk)
        if kind=='lookback':result+=is_upper
        return result
    def kernel(x):
        d1=-np.asarray(x)/sd0+sd0/2
        return s0*np.exp(-d1*d1/2)/np.sqrt(2*np.pi)*t[1]/(2*sd0)
    edges=np.r_[-np.inf,np.log(np.array(boundaries)/s0),np.inf];masses=[];risks=[];z,w=np.polynomial.legendre.leggauss(order)
    special=list(nodes)+[math.log(strike/s0)]
    for a,b in zip(edges,edges[1:]):
        a=max(a,lo);b=min(b,hi)
        if b<=a:masses.append(0.);risks.append(0.);continue
        da=float(slope(a,'right' if a==lo else 'left')[0]);db=float(slope(b,'left')[0]);masses.append(db-da)
        cuts=sorted(set([a,b]+[x for x in special if a<x<b]));integral=0.
        for low,high in zip(cuts,cuts[1:]):
            x=(low+high)/2+(high-low)*z/2;d1=-x/sd0+sd0/2
            integral+=float(w@(kernel(x)*d1/sd0*slope(x)))*(high-low)/2
        risks.append(float(kernel(b))*db-float(kernel(a))*da-integral)
    return dict(gamma_bin_masses=masses,gamma_bin_variance_sensitivities=risks,total_gamma_mass=math.fsum(masses),recovered_variance_sensitivity=math.fsum(risks),price=reference_price(p,kind,strike,order))


def evaluate(p, bounds, *, subdivisions=32768, relative_tolerance=1e-8, workers=1):
    import rust_pricing as rp
    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)
    return plan.evaluate_vega_kt_initial_smile_gamma_audit(bounds,subdivisions=subdivisions,relative_tolerance=relative_tolerance)


def run(policy):
    import rust_pricing as rp
    source_paths=[Path(__file__),ROOT/'scripts/check_discrete_smile_calibration.py',ROOT/'scripts/check_path_gamma_recovery.py',ROOT/'backtesting/rust_pricing_backtest/discrete_smile.py']
    source_paths+=list((ROOT/'crates/pricing/src/engine/risk').glob('vegakt*.rs'))+list((ROOT/'crates/pricing/src/engine/risk/vegakt_initial_smile').glob('*.rs'))+list((ROOT/'crates/pricing/src/risk').glob('vegakt*.rs'))+list((ROOT/'crates/pricing/src/risk/vegakt_convolution').glob('*.rs'))+list((ROOT/'crates/pricing-python/src').glob('vegakt*.rs'))
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    checks=[];observations=[];prices=[]
    def check(name,value,limit,**ctx):checks.append(threshold_check(name,float(value),float(limit),**ctx))
    for profile in policy['nonflat_profiles']:
        config=dict(policy,**profile);base=base_request(config,'nonflat')
        for kind in policy['products']:
            p=deepcopy(base);p['product']=request(kind,samples=16)['product'];ctx=dict(profile=profile['name'],product=kind)
            ref=reference(p,kind,policy['gamma_bin_boundaries']);ref2=reference(p,kind,policy['gamma_bin_boundaries'],192)
            for key in ['gamma_bin_masses','gamma_bin_variance_sensitivities','recovered_variance_sensitivity','price']:
                check('independent_reference_refinement_'+key,np.max(abs(np.array(ref[key])-ref2[key])),policy['reference_tolerance'],**ctx)
            derivatives=[];batches=[]
            for bump in policy['variance_bumps']:
                values=[];models=[]
                for sign in [1,-1]:
                    q=deepcopy(p);n=len(q['model']['local_variance_grid']['log_forward_moneyness_nodes']);q['model']['local_variance_grid']['values'][:n]=[q['model']['local_variance_grid']['values'][0]+sign*bump]*n
                    models.append(q);values.append(reference_price(q,kind,p['product']['strike'],192))
                fd=(values[0]-values[1])/(2*bump);derivatives.append(fd)
                check('independent_first_variance_price_difference',abs(fd-ref['recovered_variance_sensitivity']),policy['derivative_refinement_tolerance'],bump=bump,**ctx)
                batch=[]
                for seed in policy['batch_seeds']:
                    values=[]
                    for q in deepcopy(models):
                        q['engine'].update(master_seed=seed,independent_sampling_units=policy['batch_samples']);q['risk']=dict(delta=False,vega=False,smile_dynamics=dict(type='sticky_log_moneyness'))
                        values.append(rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(q)),worker_threads=1,reduction_block_size=256).evaluate().value)
                    batch.append((values[0]-values[1])/(2*bump))
                batches.append(batch)
                se=float(np.std(batch,ddof=1)/np.sqrt(len(batch)))
                check('native_first_variance_price_difference',abs(np.mean(batch)-fd),policy['statistical_multiplier']*se+policy['derivative_absolute_tolerance'],bump=bump,**ctx)
            check('independent_price_bump_refinement',abs(derivatives[0]-derivatives[1]),policy['derivative_refinement_tolerance'],**ctx)
            paired=np.array(batches[0])-batches[1];check('paired_native_bump_refinement',abs(paired.mean()),policy['statistical_multiplier']*paired.std(ddof=1)/np.sqrt(len(paired))+policy['derivative_refinement_tolerance'],**ctx)
            prices.append(dict(**ctx,independent_derivatives=derivatives,native_batch_derivatives=batches))
            for seed in policy['gamma_seeds']:
                p['engine']['master_seed']=seed
                d=evaluate(p,policy['gamma_bin_boundaries'],subdivisions=policy['subdivisions'],relative_tolerance=policy['relative_tolerance']);sctx=dict(**ctx,seed=seed)
                check('initial_recovery_and_native_validation',int(d['status']!='initial_variance_functional_passed'),0,**sctx)
                check('finite_window_gamma_mass_vs_analytic_slope',np.max(abs(np.array(d['fine']['gamma_bin_masses'])-ref['gamma_bin_masses'])),d['mass_tolerance'],**sctx)
                # Same bin error budget as native: maximal first-step variance kernel.
                bin_limit=d['mass_tolerance']*p['market']['spot']*d['maturity']/(2*math.sqrt(d['first_step_variance']))/math.sqrt(2*math.pi)
                check('bin_gamma_weighted_variance_vs_integration_by_parts',np.max(abs(np.array(d['fine']['gamma_bin_variance_sensitivities'])-ref['gamma_bin_variance_sensitivities'])),bin_limit,**sctx)
                check('initial_gamma_functional_vs_independent_derivative',abs(d['fine']['recovered_variance_sensitivity']-ref['recovered_variance_sensitivity']),d['functional_tolerance'],**sctx)
                check('continuation_price_vs_independent_payoff',abs(d['fine']['interpolated_price']-ref['price']),d['functional_tolerance'],**sctx)
                check('score_vs_independent_integration_by_parts',abs(d['quadrature_variance_sensitivity']-ref['recovered_variance_sensitivity']),policy['reference_tolerance'],**sctx)
                for key,total in [('gamma_bin_masses','total_gamma_mass'),('gamma_bin_variance_sensitivities','recovered_variance_sensitivity')]:
                    check('retained_signed_bins_'+key,abs(math.fsum(d['fine'][key])-d['fine'][total]),1e-12,**sctx)
                for key in ['price_difference','variance_difference']:
                    check('native_'+key,abs(d[key]['mean']),policy['statistical_multiplier']*d[key]['standard_error']+d['functional_tolerance'],**sctx)
                # Gamma has no MC SE: combine only the independent native-price batch uncertainty.
                check('gamma_vs_native_first_variance_price_bump',abs(d['fine']['recovered_variance_sensitivity']-np.mean(batches[1])),policy['statistical_multiplier']*np.std(batches[1],ddof=1)/np.sqrt(len(batches[1]))+policy['derivative_absolute_tolerance'],**sctx)
                observations.append(dict(**sctx,reference=ref,audit=d))
                print(json.dumps(dict(**sctx,status=d['status'],refinement=d['variance_refinement_difference'])),flush=True)
    if sources!={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}:raise RuntimeError('validation sources changed during execution')
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='initial_nonflat_smile_gamma_validation',schema_version=1,status='passed' if all(c['passed'] for c in checks) else 'failed',policy=policy,checks=checks,observations=observations,price_records=prices,source_sha256=sources,native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),full_market_iv_bucket_acceptance='not_established',limitations=['Two intervals and a spatially flat first variance row only.','First annual variance is varied with the second variance row fixed, not a first-market-IV bucket with terminal quotes held fixed.','Gamma bins cover only the declared finite window; weighted price and variance tails are separately bounded.','Refinement differences are numerical convergence checks, not a proof of a rigorous global Gamma approximation bound.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    r=run(json.loads(args.policy.read_text()));args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not c['passed'] for c in r['checks']))));return int(r['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
