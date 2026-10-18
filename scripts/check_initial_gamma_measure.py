"""Independent two-observation price/Gamma reference for the initial boundary.

The native compiler uses the entire payoff graph. This reference instead derives
simple two-fixing formulas and integrates the one remaining normal increment.
No native Gamma, AD, or bin interpolation is reused by the reference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from check_path_gamma_recovery import request

ROOT = Path(__file__).resolve().parents[1]


def reference(kind, boundaries, v0, v1, sigma0, dt0, *, order=96):
    # Conditional payoff is (A(R)*S1 - K_eff)+, R=exp(-v1/2+sqrt(v1)*Z).
    a, b, strike = {'call': (0., 1., 100.), 'asian': (.5, .5, 100.),
                    'fixed_asian': (.25, .5, 75.5), 'lookback': (0., 1., 100.)}[kind]
    cuts = [-12., 12.]
    for k in boundaries:
        r = (strike/k-a)/b
        if r > 0:
            z = (math.log(r)+v1/2)/math.sqrt(v1)
            if -12 < z < 12: cuts.append(z)
    if kind == 'lookback': cuts.append(math.sqrt(v1)/2)
    cuts = sorted(set(cuts)); z0, w0 = np.polynomial.legendre.leggauss(order)
    masses = np.zeros(len(boundaries)+1); vegas = np.zeros_like(masses)
    price = 0.
    cdf = np.vectorize(lambda x: .5*math.erfc(-x/math.sqrt(2)))
    for lo, hi in zip(cuts, cuts[1:]):
        z = (hi+lo)/2+(hi-lo)/2*z0
        w = w0*(hi-lo)/2*np.exp(-z*z/2)/math.sqrt(2*math.pi)
        r = np.exp(-v1/2+math.sqrt(v1)*z)
        scale = np.maximum(1.,r) if kind == 'lookback' else a+b*r
        k = strike/scale
        d1 = (np.log(100./k)+v0/2)/math.sqrt(v0); d2=d1-math.sqrt(v0)
        call = 100.*cdf(d1)-k*cdf(d2)
        vega = scale*100.*np.exp(-d1*d1/2)/math.sqrt(2*math.pi)*sigma0*dt0/math.sqrt(v0)
        bins = np.searchsorted(boundaries,k,side='right')
        np.add.at(masses,bins,w*scale);np.add.at(vegas,bins,w*vega)
        price += float(np.dot(w,scale*call))
    return {'gamma_bin_masses':masses.tolist(),'gamma_bin_first_step_vegas':vegas.tolist(),
            'total_gamma_mass':float(masses.sum()),'recovered_initial_local_vega':float(vegas.sum()),
            'conditional_price':price}


def evaluate(p, boundaries, workers=1):
    import rust_pricing as rp
    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)
    return plan.evaluate_vega_kt_initial_gamma_diagnostic(boundaries)


def run(policy):
    import rust_pricing as rp
    checks=[];observations=[]
    def check(name, value, limit, **context):
        checks.append(dict(check=name, observed=float(value),limit=float(limit),
                           passed=bool(math.isfinite(value) and value<=limit),**context))
    bounds=policy['gamma_bin_boundaries'];z=policy['statistical_multiplier']
    for kind in policy['products']:
        for sigmas in policy['interval_volatilities']:
            for seed in policy['seeds']:
                for anti in policy['antithetic']:
                    p=request(kind,samples=policy['samples'],seed=seed,nodes=17)
                    p['engine']['variance_reduction']['antithetic']=anti
                    grid=p['model']['local_variance_grid'];dt=np.diff(grid['time_nodes'])
                    grid['values']=[sigmas[min(i,1)]**2 for i in range(3) for _ in range(17)]
                    # Ordinary IV buckets are deliberately outside this validation.
                    p['risk']={'vega':True,'delta':False,'smile_dynamics':{'type':'sticky_log_moneyness'}}
                    d=evaluate(p,bounds)
                    args=(kind,bounds,sigmas[0]**2*dt[0],sigmas[1]**2*dt[1],sigmas[0],dt[0])
                    ref=reference(*args);ref2=reference(*args,order=192)
                    ctx=dict(product=kind,interval_volatilities=sigmas,seed=seed,antithetic=anti)
                    for key in ref:
                        if isinstance(ref[key],list):
                            errors=np.array([m['mean'] for m in d[key]])-ref[key]
                            limits=z*np.array([m['standard_error'] for m in d[key]])+policy['mass_absolute_tolerance' if key=='gamma_bin_masses' else 'vega_absolute_tolerance']
                            check(key+'_reference',max(abs(errors)/limits),1.,**ctx)
                            quad=max(abs(np.array(ref[key])-ref2[key]))
                        else:
                            check(key+'_reference',abs(d[key]['mean']-ref[key]),z*d[key]['standard_error']+policy['mass_absolute_tolerance' if key=='total_gamma_mass' else 'vega_absolute_tolerance'],**ctx)
                            quad=abs(ref[key]-ref2[key])
                        check(key+'_quadrature_convergence',quad,policy['quadrature_absolute_tolerance'],**ctx)
                    for key in ['initial_local_vega_difference','price_difference']:
                        check(key,abs(d[key]['mean']),z*d[key]['standard_error']+policy['vega_absolute_tolerance'],**ctx)
                    check('compiled_payoff_reconstruction',abs(d['payoff_reconstruction_difference']['mean'])+z*d['payoff_reconstruction_difference']['standard_error'],policy['identity_absolute_tolerance'],**ctx)
                    for key,total in [('gamma_bin_masses','total_gamma_mass'),('gamma_bin_first_step_vegas','recovered_initial_local_vega')]:
                        check(key+'_conservation',abs(math.fsum(m['mean'] for m in d[key])-d[total]['mean']),policy['identity_absolute_tolerance'],**ctx)
                    observations.append(dict(**ctx,reference=ref,diagnostic=d))
    paired=[]
    for kind in policy['products']:
        differences=[];derivatives=[];recovered=[]
        for seed in range(policy['batch_first_seed'],policy['batch_first_seed']+policy['batches']):
            p=request(kind,samples=policy['batch_samples'],seed=seed,nodes=17)
            d=evaluate(p,bounds);prices=[]
            for sign in [1,-1]:
                bumped=json.loads(json.dumps(p));g=bumped['model']['local_variance_grid']
                g['values'][:17]=[(math.sqrt(v)+sign*policy['vol_bump'])**2 for v in g['values'][:17]]
                bumped['risk']={'delta':False,'vega':False,'smile_dynamics':{'type':'sticky_log_moneyness'}}
                plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(bumped)),worker_threads=1,reduction_block_size=256)
                prices.append(plan.evaluate().value)
            fd=(prices[0]-prices[1])/(2*policy['vol_bump']);value=d['recovered_initial_local_vega']['mean']
            derivatives.append(fd);recovered.append(value);differences.append(value-fd)
        se=float(np.std(differences,ddof=1)/math.sqrt(len(differences)))
        check('paired_first_interval_price_bump',abs(float(np.mean(differences))),z*se+policy['vega_absolute_tolerance'],product=kind)
        paired.append(dict(product=kind,price_derivative=float(np.mean(derivatives)),recovered=float(np.mean(recovered)),difference=float(np.mean(differences)),paired_se=se))
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='initial_gamma_measure_component_validation',schema_version=1,
                status='passed' if all(c['passed'] for c in checks) else 'failed',full_vegakt_acceptance='not_established',
                policy=policy,checks=checks,observations=observations,paired_price_checks=paired,
                native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(json.loads(args.policy.read_text()));args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],checks=len(result['checks']),failed=sum(not c['passed'] for c in result['checks']))))
    return int(result['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
