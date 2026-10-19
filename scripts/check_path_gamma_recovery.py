"""Independent quadrature/SVD and path-payoff validation of Gamma recovery.

This is component validation, not acceptance of the full paper IV hedge.
No thresholds, regularization, input rows or recovered masses are fitted to pass.
"""
import argparse
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backtesting'))
from rust_pricing_backtest.numerics import flat_request


def integrate_hats(nodes, source_variance, integrand, order=96):
    # Direct integration in the source log state, independently of the native
    # conditional-hat identity. Tails beyond 12 SD are negligible in these cases.
    mu=-source_variance/2; sd=math.sqrt(source_variance)
    boundaries=sorted({mu-12*sd,mu+12*sd,*[x for x in nodes if mu-12*sd<x<mu+12*sd]})
    z,w=np.polynomial.legendre.leggauss(order);out=np.zeros(len(nodes))
    for lo,hi in zip(boundaries,boundaries[1:]):
        x=(lo+hi)/2+(hi-lo)/2*z
        density=np.exp(-.5*((x-mu)/sd)**2)/(sd*math.sqrt(2*math.pi))
        v=integrand(x)*density*(hi-lo)/2*w
        for j in range(len(nodes)):
            unit=np.zeros(len(nodes));unit[j]=1
            out[j]+=np.dot(v,np.interp(x,nodes,unit))
    return out


def operator(spot, source_variance, step_variance, sigma_dt, nodes, strikes):
    sd=math.sqrt(step_variance)
    return np.array([integrate_hats(nodes,source_variance,
        lambda x: spot*np.exp(x)*np.exp(-.5*((np.log(spot/k)+x+step_variance/2)/sd)**2)
            /math.sqrt(2*math.pi)*sigma_dt/sd) for k in strikes]).T


def inverse(b, ridge):
    scale=np.linalg.norm(b);u,s,vh=np.linalg.svd(b/scale,full_matrices=False)
    return (vh.T*(s/(s*s+ridge*ridge)))@u.T/scale


def source_reference(kind, spot, strike, v0, v1, sigma, nodes, sigma_bump=0.):
    dt=v1/(sigma*sigma);vol=sigma+sigma_bump;sd=vol*math.sqrt(dt)
    def conditional(x):
        s=spot*np.exp(x)
        if kind=='asian': q=.5; k=2*strike-s
        elif kind=='fixed_asian': q=.5;k=(strike-.25*98.-.25*s)/q
        elif kind=='lookback':q=1.;k=np.maximum(strike,s)
        else: q=1.;k=np.full_like(s,strike)
        valid=k>0
        d1=np.zeros_like(s);d1[valid]=(np.log(s[valid]/k[valid])+.5*sd*sd)/sd
        return np.where(valid,q*s*np.exp(-.5*d1*d1)/math.sqrt(2*math.pi)*math.sqrt(dt),0.)
    return integrate_hats(nodes,v0,conditional)


def request(kind, *, samples, seed=7, nodes=17):
    import rust_pricing as rp
    policy=json.loads((ROOT/'fixtures/backtesting/numerical-policy-v0.1.json').read_text())
    p=json.loads(flat_request(policy,seed=seed,samples=samples,steps=2,nodes=nodes).to_json())
    dates=p['risk']['vega_kt']['maturity_nodes'];expiry=dates[-1]
    if kind in ('asian','fixed_asian'):
        obs=[{'date':d,'weight':.5,'value':{'type':'unknown'}} for d in dates]
        if kind=='fixed_asian':
            obs[0]['weight']=.25
            obs.insert(0,{'date':(date.fromisoformat(p['valuation_date'])-timedelta(days=1)).isoformat(),
                          'weight':.25,'value':{'type':'known','fixing':98.}})
        p['product']={'type':'arithmetic_asian','underlying_id':1,'currency_id':2,'strike':100.,'notional':1.,
            'side':{'type':'call'},'observations':obs,'payment_date':expiry}
    elif kind=='lookback':
        p['product']={'type':'fixed_lookback','underlying_id':1,'currency_id':2,'strike':100.,'notional':1.,
            'side':{'type':'call'},'monitoring_dates':dates,'payment_date':expiry}
    return p


def evaluate(p, strikes, ridge, workers=1):
    import rust_pricing as rp
    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)
    return plan.evaluate_vega_kt_gamma_recovery_diagnostic(strikes,relative_regularization=ridge)


def make_fixture():
    nodes=np.linspace(-.6,.6,17);strikes=100*np.exp(nodes);ridge=1e-3
    b=operator(100.,.02,.01,.05,nodes,strikes);r=inverse(b,ridge)
    q=np.zeros(17);q[8]=1.;q[10]=-.4;inputs=b@q
    return dict(spot=100.,source_variance=.02,step_variance=.01,sigma_dt=.05,
        source_log_nodes=nodes.tolist(),gamma_strikes=strikes.tolist(),relative_regularization=ridge,
        forward_matrix=b.tolist(),local_vega=inputs.tolist(),recovered_masses=(r@inputs).tolist(),
        reference='96-point Gauss-Legendre integration in each source-hat cell; NumPy SVD')


def run(policy):
    checks=[];observations=[];sweeps=[]
    def check(name,value,limit,**context):
        checks.append(dict(check=name,observed=float(value),limit=float(limit),passed=bool(math.isfinite(value) and value<=limit),**context))
    for count in policy['node_counts']:
        for kind in policy['products']:
            for seed in policy['seeds']:
                p=request(kind,samples=policy['samples'],seed=seed,nodes=count)
                grid=p['model']['local_variance_grid'];nodes=np.array(grid['log_forward_moneyness_nodes']);strikes=100*np.exp(nodes)
                dt=np.diff(grid['time_nodes']);sigma=math.sqrt(grid['values'][0]);v0=sigma*sigma*dt[0];v1=sigma*sigma*dt[1]
                b=operator(100.,v0,v1,sigma*dt[1],nodes,strikes)
                reference=source_reference(kind,100.,100.,v0,v1,sigma,nodes)
                for ridge in policy['regularizations']:
                    d=evaluate(p,strikes,ridge);row=d['rows'][0];r=inverse(b,ridge)
                    values=np.array([v['mean'] for v in row['local_vega']]);errors=np.array([v['standard_error'] for v in row['local_vega']])
                    q=np.array([v['mean'] for v in row['gamma_masses']]);qse=np.array([v['standard_error'] for v in row['gamma_masses']])
                    context=dict(product=kind,nodes=count,seed=seed,regularization=ridge)
                    check('native_ad_vs_conditional_price_vega',np.max(abs(values-reference)/(policy['statistical_multiplier']*errors+policy['local_vega_absolute_tolerance'])),1.,**context)
                    check('native_recovery_vs_independent_svd',np.max(abs(q-r@values)),policy['solver_absolute_tolerance'],**context)
                    qref=r@reference
                    check('recovered_masses_vs_independent_reference',np.max(abs(q-qref)/(policy['statistical_multiplier']*qse+policy['mass_absolute_tolerance'])),1.,**context)
                    check('native_forward_residual_vs_independent_operator',np.max(abs(b@q-values-np.array([x['mean'] for x in row['reconstruction_difference']]))),policy['solver_absolute_tolerance'],**context)
                    check('reported_noise_amplification',abs(row['inverse_noise_gain_bound']-np.linalg.norm(r))/np.linalg.norm(r),policy['operator_relative_tolerance'],**context)
                    forward_error=np.linalg.norm(b@q-values)/max(np.linalg.norm(values),1e-12)
                    sweeps.append(dict(**context,forward_relative_residual=forward_error,
                        gamma_mass_l2=float(np.linalg.norm(q)),gamma_marginal_se_l2=float(np.linalg.norm(qse)),
                        inverse_noise_gain_bound=row['inverse_noise_gain_bound'],resolution_trace=row['resolution_trace'],
                        # Passing this alone does not identify Gamma or IV risk.
                        reconstruction_within_three_percent=bool(forward_error<=.03)))
                observations.append(dict(product=kind,nodes=count,seed=seed,reference_local_vega=reference.tolist(),diagnostic=d))
    # Paired full native price differences validate each source hat independently.
    # Rebuild the original LV grid at +/- bump; no derivative implementation is reused.
    import rust_pricing as rp
    paired=[]
    for kind in policy['products']:
        base=request(kind,samples=policy['batch_samples'],nodes=17)
        width=17;h=policy['vol_bump'];indices=[7,8,9]
        for j in indices:
            derivatives=[];ads=[]
            for seed in range(policy['batch_first_seed'],policy['batch_first_seed']+policy['batches']):
                p=json.loads(json.dumps(base));p['engine']['master_seed']=seed
                d=evaluate(p,(100*np.exp(p['model']['local_variance_grid']['log_forward_moneyness_nodes'])).tolist(),.01)
                prices=[]
                for sign in [1,-1]:
                    q=json.loads(json.dumps(p));grid=q['model']['local_variance_grid'];index=width+j
                    grid['values'][index]=(math.sqrt(grid['values'][index])+sign*h)**2
                    q['risk']={'delta':False,'vega':False,'smile_dynamics':{'type':'sticky_log_moneyness'}}
                    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(q)),worker_threads=1,reduction_block_size=256)
                    prices.append(plan.evaluate().value)
                derivatives.append((prices[0]-prices[1])/(2*h));ads.append(d['rows'][0]['local_vega'][j]['mean'])
            diff=np.array(derivatives)-ads;se=float(np.std(diff,ddof=1)/math.sqrt(len(diff)))
            check('paired_source_hat_price_bump',abs(float(np.mean(diff))),policy['statistical_multiplier']*se+policy['paired_absolute_tolerance'],product=kind,node=j)
            paired.append(dict(product=kind,node=j,price_derivative=float(np.mean(derivatives)),native_ad=float(np.mean(ads)),paired_difference=float(np.mean(diff)),paired_se=se,batches=len(diff)))
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='path_gamma_recovery_component_validation',schema_version=1,
        status='passed' if all(c['passed'] for c in checks) else 'failed',full_vegakt_acceptance='not_established',
        policy=policy,checks=checks,observations=observations,regularization_sweep=sweeps,paired_price_checks=paired,
        native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',action='store_true');args=p.parse_args()
    if args.fixture: r=make_fixture()
    else:
        if args.policy is None: p.error('--policy is required unless --fixture is used')
        r=run(json.loads(args.policy.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    if not args.fixture:
        print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not c['passed'] for c in r['checks']))))
        return int(r['status']!='passed')
    return 0

if __name__=='__main__':raise SystemExit(main())
