"""Retest the difficult 33-node products without relaxing reconstruction limits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from check_path_gamma_recovery import request, operator, source_reference
from check_gamma_selection import evaluate, quality
from check_conservative_gamma_market import kkt_inverse


def run(policy):
    import rust_pricing as rp
    checks=[];observations=[]
    def check(name,value,limit,**ctx):
        checks.append(dict(check=name,observed=float(value),limit=float(limit),passed=bool(math.isfinite(value) and value<=limit),**ctx))
    for kind in policy['products']:
        for seed in policy['seeds']:
            p=request(kind,samples=policy['samples'],seed=seed,nodes=policy['nodes']);g=p['model']['local_variance_grid'];nodes=g['log_forward_moneyness_nodes'];strikes=100*np.exp(nodes)
            dt=np.diff(g['time_nodes']);sigma=math.sqrt(g['values'][0]);v=sigma*sigma*dt
            B=operator(100.,v[0],v[1],sigma*dt[1],nodes,strikes);ref=source_reference(kind,100.,100.,v[0],v[1],sigma,nodes)
            d=evaluate(p,strikes,policy['selection']);ctx=dict(product=kind,seed=seed,nodes=policy['nodes'])
            check('heldout_recovery_gate',int(d['status']!='positive_time_policy_passed'),0,**ctx)
            choice=None
            for i,c in enumerate(d['pilot_candidates']+([d['validation']] if d['validation'] else [])):
                row=c['diagnostic']['rows'][0];lam=c['diagnostic']['relative_regularization'];R=kkt_inverse(B,lam)
                b=np.array([m['mean'] for m in row['local_vega']]);q=np.array([m['mean'] for m in row['gamma_masses']]);own=quality(row,policy['selection'])
                pilot=i<len(d['pilot_candidates']);cctx=dict(**ctx,regularization=lam,sample='pilot' if pilot else 'validation')
                check('independent_kkt_recovery',np.max(abs(q-R@b)),policy['solver_absolute_tolerance'],**cctx)
                check('quality_verdict',int(own['passed']!=c['passed']),0,**cctx)
                check('samplewise_conservation',abs(row['total_difference']['mean'])+6*row['total_difference']['standard_error'],policy['identity_absolute_tolerance'],**cctx)
                if pilot and own['passed'] and choice is None:choice=lam
                if not pilot:
                    check('population_spatial_reconstruction',np.linalg.norm(B@R@ref-ref)/np.linalg.norm(ref),policy['selection']['reconstruction_relative_limit'],**ctx)
                    se=np.array([m['standard_error'] for m in row['gamma_masses']])
                    check('population_gamma_reference',np.max(abs(q-R@ref)/(6*se+policy['mass_absolute_tolerance'])),1.,**ctx)
            check('strongest_pilot_feasible_choice',int(choice!=d['selected_regularization']),0,**ctx)
            observations.append(dict(**ctx,audit=d))
            print(json.dumps(dict(**ctx,status=d['status'],selected=d['selected_regularization'])),flush=True)
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='conservative_gamma_33_node_regression',schema_version=1,status='passed' if all(c['passed'] for c in checks) else 'failed',
        full_market_iv_bucket_acceptance='not_established',policy=policy,checks=checks,observations=observations,
        native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(json.loads(a.policy.read_text()));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not c['passed'] for c in r['checks']))));return int(r['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
