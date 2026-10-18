"""Audit native pilot selection on disjoint counters and held-out validation.

No oracle participates in native selection. Independent quadrature and SVD are
used here only after selection. Original fixed-lambda failures are not rewritten.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from check_path_gamma_recovery import request, operator, inverse, source_reference


def quality(row, p):
    def norms(key):
        return (float(np.linalg.norm([m['mean'] for m in row[key]])),
                float(np.linalg.norm([m['standard_error'] for m in row[key]])))
    b,bse=norms('local_vega');r,rse=norms('reconstruction_difference');q,qse=norms('gamma_masses')
    denom=b-p['mc_margin_multiplier']*bse
    residual=(r+p['mc_margin_multiplier']*rse)/denom if denom>0 else None
    noise=qse/q if q>0 else None
    passed=residual is not None and noise is not None and residual<=p['reconstruction_relative_limit'] and noise<=p['gamma_relative_standard_error_limit']
    return dict(reconstruction_margin_ratio=residual,gamma_relative_standard_error=noise,passed=bool(passed))


def evaluate(p, strikes, policy, workers=1):
    import rust_pricing as rp
    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=workers,reduction_block_size=256)
    return plan.evaluate_vega_kt_gamma_recovery_audit(strikes,policy=rp.VegaKtGammaRecoveryPolicy(**policy))


def run(policy):
    import rust_pricing as rp
    checks=[];observations=[]
    def check(name,value,limit,**ctx):
        checks.append(dict(check=name,observed=float(value),limit=float(limit),passed=bool(math.isfinite(value) and value<=limit),**ctx))
    for count in policy['node_counts']:
        for kind in policy['products']:
            for seed in policy['seeds']:
                p=request(kind,samples=policy['validation_sampling_units'],nodes=count,seed=seed)
                grid=p['model']['local_variance_grid'];nodes=grid['log_forward_moneyness_nodes'];strikes=100*np.exp(nodes)
                sigma=math.sqrt(grid['values'][0]);dt=np.diff(grid['time_nodes']);v0=sigma*sigma*dt[0];v1=sigma*sigma*dt[1]
                b=operator(100.,v0,v1,sigma*dt[1],nodes,strikes);ref=source_reference(kind,100.,100.,v0,v1,sigma,nodes)
                d=evaluate(p,strikes,policy['selection']);ctx=dict(product=kind,nodes=count,seed=seed)
                check('positive_time_policy_status',int(d['status']!='positive_time_policy_passed'),0,**ctx)
                check('nonoverlapping_sampling_ranges',int(d['pilot_index_start']!=policy['validation_sampling_units'] or d['pilot_index_end_exclusive']!=policy['validation_sampling_units']+policy['selection']['pilot_sampling_units']),0,**ctx)
                independent_choice=None
                candidates=d['pilot_candidates']+([d['validation']] if d['validation'] else [])
                for i,c in enumerate(candidates):
                    raw=c['diagnostic'];qctx=dict(**ctx,sample='pilot' if i<len(d['pilot_candidates']) else 'validation',regularization=raw['relative_regularization'])
                    r=inverse(b,raw['relative_regularization']);feasible=True
                    for row,reported in zip(raw['rows'],c['quality']):
                        own=quality(row,policy['selection']);feasible=feasible and own['passed']
                        numeric=max(abs((own[k] or 0.)-(reported[k] or 0.)) for k in ['reconstruction_margin_ratio','gamma_relative_standard_error'])
                        check('independent_quality_metrics',numeric,1e-11,**qctx)
                        check('independent_quality_verdict',int(own['passed']!=reported['passed']),0,**qctx)
                        local=np.array([m['mean'] for m in row['local_vega']]);gamma=np.array([m['mean'] for m in row['gamma_masses']])
                        check('native_recovery_vs_svd',np.max(abs(gamma-r@local)),policy['solver_absolute_tolerance'],**qctx)
                        check('residual_vs_independent_forward',np.max(abs(b@gamma-local-np.array([m['mean'] for m in row['reconstruction_difference']]))),policy['solver_absolute_tolerance'],**qctx)
                        check('noise_gain',abs(row['inverse_noise_gain_bound']-np.linalg.norm(r))/np.linalg.norm(r),policy['operator_relative_tolerance'],**qctx)
                    if i<len(d['pilot_candidates']) and feasible and independent_choice is None:
                        independent_choice=raw['relative_regularization']
                check('strongest_pilot_feasible_choice',int(independent_choice!=d['selected_regularization']),0,**ctx)
                if d['validation']:
                    raw=d['validation']['diagnostic'];row=raw['rows'][0];r=inverse(b,raw['relative_regularization'])
                    means=np.array([m['mean'] for m in row['gamma_masses']]);se=np.array([m['standard_error'] for m in row['gamma_masses']])
                    check('heldout_gamma_reference',np.max(abs(means-r@ref)/(policy['statistical_multiplier']*se+policy['mass_absolute_tolerance'])),1.,**ctx)
                    local=np.array([m['mean'] for m in row['local_vega']]);lse=np.array([m['standard_error'] for m in row['local_vega']])
                    check('heldout_input_reference',np.max(abs(local-ref)/(policy['statistical_multiplier']*lse+policy['local_vega_absolute_tolerance'])),1.,**ctx)
                    check('population_reconstruction_unchanged_three_percent',np.linalg.norm(b@r@ref-ref)/np.linalg.norm(ref),policy['selection']['reconstruction_relative_limit'],**ctx)
                    check('initial_ad_connection',abs(d['initial']['native_initial_local_vega']['mean']-raw['unrecovered_initial_local_vega']['mean']),1e-11,**ctx)
                    initial=d['initial']['initial_local_vega_difference'];check('initial_gamma_vs_ad',abs(initial['mean']),policy['statistical_multiplier']*initial['standard_error']+policy['local_vega_absolute_tolerance'],**ctx)
                observations.append(dict(**ctx,audit=d))
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='split_sample_gamma_recovery_component_validation',schema_version=1,
        status='passed' if all(c['passed'] for c in checks) else 'failed',full_vegakt_acceptance='not_established',policy=policy,
        checks=checks,observations=observations,native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(json.loads(a.policy.read_text()));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not c['passed'] for c in r['checks']))))
    return int(r['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
