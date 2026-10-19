"""Separate population regularization bias from sampled input noise.

This analysis preserves the original quality verdicts and does not choose a
regularization or change acceptance limits. The two norm contributions need not
add: their vectors can align or cancel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from check_path_gamma_recovery import operator, inverse


def analyze(report, source_sha256):
    rows=[]
    for o in report['observations']:
        d=o['diagnostic'];row=d['rows'][0];dt=row['recovered_maturity']-row['source_maturity']
        args=(100.,row['source_total_variance'],row['step_variance'],np.sqrt(row['step_variance']/dt)*dt,d['source_log_nodes'])
        b=operator(*args,d['gamma_strikes']);expected=np.array(o['reference_local_vega'])
        sampled=np.array([m['mean'] for m in row['local_vega']]);denom=np.linalg.norm(expected)
        for ridge in report['policy']['regularizations']:
            r=inverse(b,ridge);p=b@r-np.eye(len(sampled))
            population=p@expected;noise=p@(sampled-expected);actual=p@sampled
            rows.append(dict(product=o['product'],nodes=o['nodes'],seed=o['seed'],regularization=ridge,
                actual_forward_relative_residual=float(np.linalg.norm(actual)/np.linalg.norm(sampled)),
                population_forward_relative_residual=float(np.linalg.norm(population)/denom),
                sampling_residual_component_relative_to_population=float(np.linalg.norm(noise)/denom),
                vector_decomposition_error=float(np.max(abs(actual-population-noise))),
                original_three_percent_quality_passed=bool(np.linalg.norm(actual)/np.linalg.norm(sampled)<=.03)))
    # Deterministic basis refinement at the problematic source grid, independent
    # of seeds. Both range and lambda are fixed during each refinement series.
    o=next(o for o in report['observations'] if o['product']=='asian' and o['nodes']==33)
    d=o['diagnostic'];row=d['rows'][0];dt=row['recovered_maturity']-row['source_maturity'];expected=np.array(o['reference_local_vega'])
    refinements=[]
    for n in [17,33,65,129]:
        strikes=100*np.exp(np.linspace(d['source_log_nodes'][0],d['source_log_nodes'][-1],n))
        b=operator(100.,row['source_total_variance'],row['step_variance'],np.sqrt(row['step_variance']/dt)*dt,d['source_log_nodes'],strikes)
        for ridge in report['policy']['regularizations']:
            r=inverse(b,ridge);q=r@expected
            refinements.append(dict(source_nodes=33,gamma_nodes=n,regularization=ridge,
                population_forward_relative_residual=float(np.linalg.norm(b@q-expected)/np.linalg.norm(expected)),
                recovered_mass_l2=float(np.linalg.norm(q)),inverse_noise_gain_bound=float(np.linalg.norm(r))))
    return dict(document_kind='path_gamma_bias_decomposition',schema_version=1,
        status='diagnostic_only',quality_policy_changed=False,full_vegakt_acceptance='not_established',
        source_report_sha256=source_sha256,rows=rows,basis_refinement=refinements,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=a.report.read_bytes();r=analyze(json.loads(data),hashlib.sha256(data).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(r['rows']),'unchanged_quality_failures':sum(not c['original_three_percent_quality_passed'] for c in r['rows']),
                      'max_decomposition_error':max(c['vector_decomposition_error'] for c in r['rows'])}))

if __name__=='__main__':main()
