"""Conservative Gamma and independently recalibrated flat maturity-IV validation.

Checks real price differences after changing quote IV and rebuilding forward
variance. Reference prices integrate two-observation payoff formulas and never
use recovered Gamma. The complete market-IV bucket gate remains open.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from check_path_gamma_recovery import request, operator, inverse, source_reference
from check_atom_gamma_chain import initial_projection
from rust_pricing_backtest.flat_iv import recalibrate_flat_market_iv, maturity_risk_statistics


def kkt_inverse(b, ridge):
    """Independent primal KKT solve, not the native rank-one construction."""
    n=b.shape[1];c=b.sum(axis=0);h=b.T@b+(ridge*np.linalg.norm(b))**2*np.eye(n)
    kkt=np.block([[h,c[:,None]],[c[None,:],np.zeros((1,1))]])
    return np.linalg.solve(kkt,np.vstack([b.T,np.ones((1,b.shape[0]))]))[:n]


def reference_price(kind, v0, v1, order=96):
    # Integrate over the first fixing; price the last Gaussian increment in
    # closed form. Split at the product's nonsmooth first-fixing boundaries.
    z0,w0=np.polynomial.legendre.leggauss(order);cuts=[-12.,12.]
    threshold={'call':None,'asian':200.,'fixed_asian':302.,'lookback':100.}[kind]
    if threshold:
        z=(math.log(threshold/100)+v0/2)/math.sqrt(v0)
        if -12<z<12:cuts.append(z)
    cuts=sorted(cuts);out=0.;cdf=np.vectorize(lambda x:.5*math.erfc(-x/math.sqrt(2)))
    for lo,hi in zip(cuts,cuts[1:]):
        z=(lo+hi)/2+(hi-lo)/2*z0;w=w0*(hi-lo)/2*np.exp(-z*z/2)/math.sqrt(2*math.pi)
        s=100*np.exp(-v0/2+math.sqrt(v0)*z)
        k={'call':np.full_like(s,100.),'asian':200-s,'fixed_asian':151-.5*s,'lookback':np.maximum(100,s)}[kind]
        valid=k>0;d1=np.zeros_like(s);d1[valid]=(np.log(s[valid]/k[valid])+v1/2)/math.sqrt(v1)
        calls=np.where(valid,s*cdf(d1)-k*cdf(d1-math.sqrt(v1)),s-k)
        payoff=calls*(.5 if kind in ('asian','fixed_asian') else 1.)
        if kind=='lookback':payoff+=np.maximum(s-100,0.)
        out+=float(w@payoff)
    return out


def make_request(kind, sigmas, policy, seed):
    p=request(kind,samples=policy['samples'],seed=seed,nodes=policy['node_count']);n=policy['node_count']
    g=p['model']['local_variance_grid'];dt=np.diff(g['time_nodes']);times=np.cumsum(dt)
    g['values']=[sigmas[min(i,1)]**2 for i in range(3) for _ in range(n)]
    iv=np.sqrt(np.cumsum(np.array(sigmas)**2*dt)/times)
    p['model']['reporting_iv_basis']['implied_volatilities']=[v for v in iv for _ in range(n)]
    p['risk']['vega_kt']['full_bucket_covariance']=True
    return p


def run(policy):
    import rust_pricing as rp
    checks=[];observations=[];recalibrations=[]
    def check(name,value,limit,**ctx):
        checks.append(dict(check=name,observed=float(value),limit=float(limit),passed=bool(math.isfinite(value) and value<=limit),**ctx))
    def compile_plan(p):
        return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=256)
    for kind in policy['products']:
        for sigmas in policy['interval_volatilities']:
            # Batch price differences use separate seeds from Gamma validation.
            # Within a batch, every sign/size/maturity uses the same native paths.
            base=make_request(kind,sigmas,policy,policy['seeds'][0]);times=np.array(base['model']['reporting_iv_basis']['maturity_nodes'])
            dt=np.diff(np.r_[0.,times]);v=np.array(sigmas)**2*dt
            refs=[];batches=[];audits=[];fine_refs=[]
            for bump in policy['iv_bumps']:
                shifted=[];local_refs=[];local_refs_fine=[]
                for i in range(2):
                    prices=[];fine_prices=[];pair=[]
                    for sign in [1,-1]:
                        shift=np.zeros(2);shift[i]=sign*bump
                        p,a=recalibrate_flat_market_iv(base,shift);pair.append(p);audits.append(a)
                        # Independent cumulative-variance calculation, separate
                        # from recalibration utility and the Gamma estimator.
                        quotes=np.sqrt(np.cumsum(v)/times)+shift
                        variances=np.diff(np.r_[0.,times*quotes*quotes])
                        check('recalibrated_variance_identity',np.max(abs(np.array(a['interval_variances'])-variances)),policy['algebra_tolerance'],product=kind,vols=sigmas,maturity=i,bump=bump,sign=sign)
                        prices.append(reference_price(kind,*variances));fine_prices.append(reference_price(kind,*variances,192))
                    shifted.append(pair);local_refs.append((prices[0]-prices[1])/(2*bump));local_refs_fine.append((fine_prices[0]-fine_prices[1])/(2*bump))
                samples=[]
                for seed in policy['batch_seeds']:
                    row=[]
                    for pair in shifted:
                        prices=[]
                        for p in pair:
                            p=json.loads(json.dumps(p));p['engine']['master_seed']=seed;p['engine']['independent_sampling_units']=policy['batch_samples']
                            p['risk']={'delta':False,'vega':False,'smile_dynamics':{'type':'sticky_log_moneyness'}}
                            prices.append(compile_plan(p).evaluate().value)
                        row.append((prices[0]-prices[1])/(2*bump))
                    samples.append(row)
                refs.append(local_refs);fine_refs.append(local_refs_fine);batches.append(samples)
            refs=np.array(refs);batch=np.array(batches);means=batch.mean(axis=1);se=batch.std(axis=1,ddof=1)/math.sqrt(len(policy['batch_seeds']))
            for h,bump in enumerate(policy['iv_bumps']):
                check('price_reference_quadrature',np.max(abs(refs[h]-fine_refs[h])),policy['quadrature_derivative_tolerance'],product=kind,vols=sigmas,bump=bump)
                check('recalibrated_native_price_derivative',np.max(abs(means[h]-refs[h])/(policy['statistical_multiplier']*se[h]+policy['vega_absolute_tolerance'])),1.,product=kind,vols=sigmas,bump=bump)
            difference=batch[0]-batch[1];dse=difference.std(axis=0,ddof=1)/math.sqrt(len(policy['batch_seeds']))
            check('paired_price_bump_refinement',np.max(abs(difference.mean(axis=0))/(policy['statistical_multiplier']*dse+policy['vega_absolute_tolerance'])),1.,product=kind,vols=sigmas)
            check('reference_bump_refinement',np.max(abs(refs[0]-refs[1])),policy['reference_derivative_tolerance'],product=kind,vols=sigmas)
            recalibrations.append(dict(product=kind,interval_volatilities=sigmas,audits=audits,reference_derivatives=refs.tolist(),batch_derivatives=batch.tolist(),native_derivative_means=means.tolist(),native_derivative_standard_errors=se.tolist()))
            for seed in policy['seeds']:
                p=make_request(kind,sigmas,policy,seed);nodes=np.array(p['model']['local_variance_grid']['log_forward_moneyness_nodes']);strikes=100*np.exp(nodes)
                d=compile_plan(p).evaluate_vega_kt_atom_chain_diagnostic(strikes,policy=rp.VegaKtGammaRecoveryPolicy(**policy['selection']))
                ctx=dict(product=kind,interval_volatilities=sigmas,seed=seed)
                check('conservative_heldout_recovery_gate',int(d['estimate'] is None),0,**ctx)
                if d['estimate'] is None:
                    observations.append(dict(**ctx,diagnostic=d));continue
                maturity=maturity_risk_statistics(d);values=np.array(maturity['means']);errors=np.array(maturity['standard_errors'])
                check('maturity_iv_vs_independent_price_reference',np.max(abs(values-refs[1])/(policy['statistical_multiplier']*errors+policy['vega_absolute_tolerance'])),1.,**ctx)
                check('maturity_iv_vs_recalibrated_native_prices',np.max(abs(values-means[1])/(policy['statistical_multiplier']*np.sqrt(errors**2+se[1]**2)+policy['vega_absolute_tolerance'])),1.,**ctx)
                B=operator(100.,v[0],v[1],sigmas[1]*dt[1],nodes,strikes);lam=d['recovery']['selected_regularization'];R=kkt_inverse(B,lam)
                pop=source_reference(kind,100.,100.,v[0],v[1],sigmas[1],nodes)
                initial=initial_projection(kind,nodes,v[0],v[1],times[0]).sum();iv=np.sqrt(np.cumsum(v)/times)
                def population_risk(inverse_matrix):
                    total=(B@(inverse_matrix@pop)).sum()
                    return np.array([initial-iv[0]*times[0]/(sigmas[1]*dt[1])*total,
                        iv[1]*times[1]/(sigmas[1]*dt[1])*total])
                population=population_risk(R);legacy=population_risk(inverse(B,lam))
                check('population_maturity_bias_removed',np.max(abs(population-refs[1])),policy['reference_derivative_tolerance'],**ctx)
                candidates=d['recovery']['pilot_candidates']+[d['recovery']['validation']]
                for index,c in enumerate(candidates):
                    raw=c['diagnostic'];row=raw['rows'][0];own=kkt_inverse(B,raw['relative_regularization'])
                    b=np.array([m['mean'] for m in row['local_vega']]);q=np.array([m['mean'] for m in row['gamma_masses']])
                    cctx=dict(**ctx,sample='validation' if index==len(candidates)-1 else 'pilot',regularization=raw['relative_regularization'])
                    check('independent_kkt_recovery',np.max(abs(q-own@b)),policy['solver_absolute_tolerance'],**cctx)
                    check('full_spatial_residual_retained',np.max(abs(B@q-b-[m['mean'] for m in row['reconstruction_difference']])),policy['solver_absolute_tolerance'],**cctx)
                    check('pathwise_total_conservation',abs(row['total_difference']['mean'])+policy['statistical_multiplier']*row['total_difference']['standard_error'],policy['algebra_tolerance'],**cctx)
                    check('updated_inverse_noise_gain',abs(np.linalg.norm(own)-row['inverse_noise_gain_bound'])/np.linalg.norm(own),policy['operator_relative_tolerance'],**cctx)
                observations.append(dict(**ctx,diagnostic=d,maturity_risk=maturity,population_maturity_risk=population.tolist(),legacy_same_lambda_population_maturity_risk=legacy.tolist(),reference_maturity_risk=refs[1].tolist()))
                print(json.dumps(dict(**ctx,status='passed' if all(c['passed'] for c in checks) else 'failed')),flush=True)
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='conservative_gamma_maturity_market_iv_validation',schema_version=1,status='passed' if all(c['passed'] for c in checks) else 'failed',
        full_market_iv_bucket_acceptance='not_established',policy=policy,checks=checks,observations=observations,recalibrations=recalibrations,
        native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=run(json.loads(args.policy.read_text()));args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(report['checks']),failed=sum(not c['passed'] for c in report['checks']))));return int(report['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
