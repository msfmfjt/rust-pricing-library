"""Independent flat-model atom IV-chain component validation, not market-IV acceptance.

The population reference uses scalar payoff formulas, direct quadrature of
Gamma(S)*BS_IV_Vega(S)*dS, and a NumPy SVD inverse of an independently integrated
source-hat operator. It intentionally retains the same regularization bias as
the declared estimator. A separate parallel price-bump check measures true risk.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from check_path_gamma_recovery import request, operator, inverse, source_reference


def initial_projection(kind, nodes, v0, v1, t0, order=96):
    a,b,strike={'call':(0.,1.,100.),'asian':(.5,.5,100.),
        'fixed_asian':(.25,.5,75.5),'lookback':(0.,1.,100.)}[kind]
    cuts=[-12.,12.]
    for k in 100*np.exp(nodes):
        ratio=(strike/k-a)/b
        if ratio>0:
            z=(math.log(ratio)+v1/2)/math.sqrt(v1)
            if -12<z<12:cuts.append(z)
    if kind=='lookback':cuts.append(math.sqrt(v1)/2)
    cuts=sorted(set(cuts));z0,w0=np.polynomial.legendre.leggauss(order);out=np.zeros(len(nodes))
    for lo,hi in zip(cuts,cuts[1:]):
        z=(lo+hi)/2+(hi-lo)/2*z0;w=w0*(hi-lo)/2*np.exp(-z*z/2)/math.sqrt(2*math.pi)
        ratio=np.exp(-v1/2+math.sqrt(v1)*z)
        scale=np.maximum(1.,ratio) if kind=='lookback' else a+b*ratio
        k=strike/scale;x=np.log(k/100.);d1=(-x+v0/2)/math.sqrt(v0)
        value=scale*100*np.exp(-d1*d1/2)/math.sqrt(2*math.pi)*math.sqrt(t0)
        for j in range(len(nodes)):
            unit=np.zeros(len(nodes));unit[j]=1.
            out[j]+=np.dot(w,value*np.interp(x,nodes,unit))
    return out


def propagation_matrix(nodes, strikes, v0, v1, t0, order=96):
    # Direct integration in physical strike via S=100 exp(x); no native bridge
    # probabilities, analytical hat expectations, or Gamma propagation reused.
    x0,w0=np.polynomial.legendre.leggauss(order);out=np.zeros((len(nodes),len(strikes)))
    cuts=sorted(set([-6.,6.,*nodes]))
    for lo,hi in zip(cuts,cuts[1:]):
        x=(lo+hi)/2+(hi-lo)/2*x0;w=w0*(hi-lo)/2;s=100*np.exp(x)
        gamma=np.exp(-.5*((np.log(s[:,None]/strikes)+v1/2)/math.sqrt(v1))**2)/(s[:,None]*math.sqrt(2*math.pi*v1))
        vega=100*np.exp(-.5*((-x+v0/2)/math.sqrt(v0))**2)/math.sqrt(2*math.pi)*math.sqrt(t0)
        for j in range(len(nodes)):
            unit=np.zeros(len(nodes));unit[j]=1.
            out[j]+=np.dot(w*s*vega*np.interp(x,nodes,unit),gamma)
    return out


def run(policy):
    import rust_pricing as rp
    checks=[];observations=[];paired=[]
    def check(name,value,limit,**ctx):
        checks.append(dict(check=name,observed=float(value),limit=float(limit),passed=bool(math.isfinite(value) and value<=limit),**ctx))
    def compile_plan(p):
        return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=256)
    for kind in policy['products']:
        for sigmas in policy['interval_volatilities']:
            for seed in policy['seeds']:
                n=policy['node_count'];p=request(kind,samples=policy['samples'],nodes=n,seed=seed)
                grid=p['model']['local_variance_grid'];nodes=np.array(grid['log_forward_moneyness_nodes']);dt=np.diff(grid['time_nodes']);times=np.cumsum(dt)
                grid['values']=[sigmas[min(i,1)]**2 for i in range(3) for _ in range(n)]
                variances=np.array(sigmas)**2*dt;v0,v1=variances;vols=np.sqrt(np.cumsum(variances)/times)
                p['model']['reporting_iv_basis']['implied_volatilities']=[v for v in vols for _ in range(n)]
                p['risk']['vega_kt']['full_bucket_covariance']=True
                strikes=100*np.exp(nodes);selection=rp.VegaKtGammaRecoveryPolicy(**policy['selection'])
                d=compile_plan(p).evaluate_vega_kt_atom_chain_diagnostic(strikes,policy=selection)
                ctx=dict(product=kind,interval_volatilities=sigmas,seed=seed)
                check('heldout_recovery_gate',int(d['estimate'] is None),0,**ctx)
                if d['estimate'] is None:
                    observations.append(dict(**ctx,diagnostic=d));continue
                e=d['estimate'];q=d['recovery']['selected_regularization'];B=operator(100,v0,v1,sigmas[1]*dt[1],nodes,strikes)
                gamma=inverse(B,q)@source_reference(kind,100,100,v0,v1,sigmas[1],nodes)
                initial=initial_projection(kind,nodes,v0,v1,times[0]);initial2=initial_projection(kind,nodes,v0,v1,times[0],192)
                propagation=propagation_matrix(nodes,strikes,v0,v1,times[0]);propagation2=propagation_matrix(nodes,strikes,v0,v1,times[0],192)
                check('initial_reference_quadrature',np.max(abs(initial-initial2)),policy['quadrature_absolute_tolerance'],**ctx)
                check('propagation_reference_quadrature',np.max(abs(propagation-propagation2)),policy['quadrature_absolute_tolerance'],**ctx)
                terminal_vegas=100*np.exp(-.5*((-nodes+(v0+v1)/2)/math.sqrt(v0+v1))**2)/math.sqrt(2*math.pi)*math.sqrt(times[1])
                ref=np.concatenate([initial-propagation@gamma,terminal_vegas*gamma])
                means=np.array([m['mean'] for m in e['raw_buckets']]);se=np.array([m['standard_error'] for m in e['raw_buckets']])
                check('regularized_population_iv_buckets',np.max(abs(means-ref)/(policy['statistical_multiplier']*se+policy['vega_absolute_tolerance'])),1.,**ctx)
                # Terminal row is a deterministic transform of held-out Gamma;
                # this isolates chain wiring from MC reference deviations.
                raw=d['recovery']['validation']['diagnostic']['rows'][0]
                native_gamma=np.array([m['mean'] for m in raw['gamma_masses']])
                check('terminal_gamma_projection',np.max(abs(means[n:]-native_gamma*terminal_vegas)),policy['identity_absolute_tolerance'],**ctx)
                check('total_bucket_conservation',abs(means.sum()-e['total_iv_vega']['mean']),policy['identity_absolute_tolerance'],**ctx)
                cov=np.array(e['full_bucket_sample_covariance']).reshape(2*n,2*n)
                check('joint_covariance_total_variance',abs(cov.sum()-e['total_iv_vega']['sample_variance']),policy['covariance_absolute_tolerance'],**ctx)
                check('joint_covariance_psd',max(0.,-np.linalg.eigvalsh(cov).min()),policy['covariance_absolute_tolerance'],**ctx)
                check('joint_covariance_diagonal',np.max(abs(cov.diagonal()-[m['sample_variance'] for m in e['raw_buckets']])),policy['covariance_absolute_tolerance'],**ctx)
                if sigmas[0]==sigmas[1]:
                    residual=d['recovery']['initial']['initial_local_vega_difference']['mean']+raw['total_difference']['mean']
                    check('iv_minus_ad_telescoping_residual',abs(e['iv_minus_local_parallel']['mean']-residual),policy['identity_absolute_tolerance'],**ctx)
                    # Same-shock native price bump, recorded without removing
                    # inverse bias. This only shocks the flat parallel direction.
                    prices=[]
                    for sign in [1,-1]:
                        bumped=json.loads(json.dumps(p));g=bumped['model']['local_variance_grid']
                        g['values']=[(math.sqrt(v)+sign*policy['vol_bump'])**2 for v in g['values']]
                        bumped['risk']={'delta':False,'vega':False,'smile_dynamics':{'type':'sticky_log_moneyness'}}
                        prices.append(compile_plan(bumped).evaluate().value)
                    fd=(prices[0]-prices[1])/(2*policy['vol_bump'])
                    check('parallel_ad_vs_price_bump',abs(e['native_parallel_local_vega']['mean']-fd),policy['price_fd_absolute_tolerance'],**ctx)
                    check('parallel_iv_vs_price_bump',abs(e['total_iv_vega']['mean']-fd),
                        policy['statistical_multiplier']*e['iv_minus_local_parallel']['standard_error']
                        +policy['vega_absolute_tolerance']+policy['price_fd_absolute_tolerance'],**ctx)
                    paired.append(dict(**ctx,price_derivative=fd,total_iv_vega=e['total_iv_vega']['mean'],difference=e['total_iv_vega']['mean']-fd,
                        same_path_iv_minus_ad=e['iv_minus_local_parallel'],regularization=q))
                observations.append(dict(**ctx,reference=ref.tolist(),diagnostic=d))
                print(json.dumps(dict(**ctx,status='passed' if all(c['passed'] for c in checks) else 'failed')),flush=True)
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='signed_atom_iv_chain_component_validation',schema_version=1,
        status='passed' if all(c['passed'] for c in checks) else 'failed',full_market_iv_acceptance='not_established',
        policy=policy,checks=checks,observations=observations,parallel_price_checks=paired,
        native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=run(json.loads(args.policy.read_text()));args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(report['checks']),failed=sum(not c['passed'] for c in report['checks']))))
    return int(report['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
