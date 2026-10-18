"""Refine custom-basis price differences without weakening acceptance gates."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_custom_variance_basis import (ROOT,PRODUCTS,KINDS,models,basis_for,prices,black,independent_quote_prices,
    MultiStepExtensionCalibrator,threshold_check,provenance,sha,aggregate)


def sources(policy_path):
    policy=json.loads(policy_path.read_text());bp=ROOT/policy['baseline_policy'];base=json.loads(bp.read_text())
    frozen,binary,prior=provenance(bp)
    frozen.update({str(p.relative_to(ROOT)):sha(p) for p in [policy_path,Path(__file__)]})
    return policy,base,frozen,binary,prior


def price_run(policy_path,checkpoint,intervals,kind):
    policy,base,frozen,binary,prior=sources(policy_path);effective=dict(base,iv_bumps=policy['iv_bumps'])
    case=next(x for x in models() if x['intervals']==intervals);payload=case['request']
    ctx=dict(intervals=intervals,profile=case['profile'],basis=kind)
    b,ba=basis_for(payload,kind);c=MultiStepExtensionCalibrator(payload,extension='custom',variance_basis=b)
    tangent,ta=c.all_maturity_iv_tangent();count=c.iv.size;checks=[];calibrations=[];independent=[]
    def check(name,value,limit,**context):checks.append(threshold_check(name,float(value),float(limit),**ctx,**context))
    def save(status):
        r=dict(document_kind='custom_basis_price_refinement',schema_version=1,status=status,policy=policy,baseline_policy=base,
            case=dict(**ctx,basis_audit=ba,tangent_audit=ta,calibration_tangent=tangent.tolist()),checks=checks,calibration_audits=calibrations,
            difference_batches=[x.tolist() for x in independent],source_sha256=frozen,native_binary_sha256=binary,prior_acceptance_sha256=prior)
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_text(json.dumps(r,allow_nan=False)+'\n');return r
    for bump in policy['iv_bumps']:
        columns=[]
        for coordinate in range(count):
            pair=[];variances=[]
            for sign in [1.,-1.]:
                shifts=np.zeros(count);shifts[coordinate]=sign*bump
                q,a=c.calibrate_all_maturities(shifts.reshape(c.iv.shape));v=np.array(q['model']['local_variance_grid']['values']).reshape(c.variance.shape)
                variances.append(v[:-1].ravel());target=np.array([black(c.spot,c.strikes,t*iv*iv)[0] for t,iv in zip(c.maturities,c.iv+shifts.reshape(c.iv.shape))])
                error=float(np.max(abs(independent_quote_prices(q)-target)))
                check('independent_recalibrated_quote_fit',error,base['quote_price_tolerance'],coordinate=coordinate,bump=bump,sign=sign)
                check('unused_terminal_row_fixed',np.max(abs(v[-1]-c.variance[-1])),0,coordinate=coordinate,bump=bump,sign=sign)
                check('explicit_basis_parameter_increments',np.max(abs((v[:-1]-c.variance[:-1]).ravel()-b@a['parameter_increments'])),1e-17,coordinate=coordinate,bump=bump,sign=sign)
                calibrations.append(dict(**ctx,coordinate=coordinate,bump=bump,sign=sign,audit=a,independent_price_error=error))
                pair.append(prices(q,effective))
            check('tangent_vs_actual_recalibration',np.max(abs((variances[0]-variances[1])/(2*bump)-tangent[:,coordinate])),2e-5,coordinate=coordinate,bump=bump)
            columns.append((pair[0]-pair[1])/(2*bump))
        independent.append(np.stack(columns,axis=2));save('in_progress')
        print(json.dumps(dict(**ctx,bump=bump,checks=len(checks),failed=sum(not x['passed'] for x in checks))),flush=True)
    independent=np.array(independent)
    for j,product in enumerate(PRODUCTS):
        for coordinate in range(count):
            d=independent[0,:,j,coordinate]-independent[1,:,j,coordinate]
            check('paired_independent_bump_refinement',abs(d.mean()),6*d.std(ddof=1)/np.sqrt(len(d))+base['absolute_tolerance'],product=product,coordinate=coordinate)
    if any(sha(ROOT/n)!=h for n,h in frozen.items()):raise RuntimeError('price refinement source changed')
    return save('passed' if all(x['passed'] for x in checks) else 'failed')


def merge(policy_path,baseline_path,price_paths):
    policy,base,frozen,binary,prior=sources(policy_path)
    baseline=json.loads(baseline_path.read_text());reports=[json.loads(p.read_text()) for p in price_paths]
    bf,bb,bp=provenance(ROOT/policy['baseline_policy'])
    if baseline['source_sha256']!=bf or baseline['native_binary_sha256']!=bb or baseline['policy']!=base:raise ValueError('baseline provenance differs')
    expected={(n,b) for n in base['intervals'] for b in KINDS};keys=[(r['case']['intervals'],r['case']['basis']) for r in reports]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('incomplete or duplicate refinement cases')
    failed=[x for x in baseline['checks'] if not x['passed']]
    if any(x['check']!='tangent_vs_actual_recalibration' for x in failed):raise ValueError('unexpected baseline failure cannot be superseded')
    replaced={'independent_recalibrated_quote_fit','unused_terminal_row_fixed','explicit_basis_parameter_increments',
        'tangent_vs_actual_recalibration','paired_independent_bump_refinement','Gamma_vs_independent_price_difference'}
    checks=[x for x in baseline['checks'] if x['check'] not in replaced];price_records=[]
    if len(baseline['observations'])!=policy['retained_native_panels']:raise ValueError('missing native panels')
    for r in reports:
        if r['source_sha256']!=frozen or r['native_binary_sha256']!=binary or r['policy']!=policy or r['baseline_policy']!=base:
            raise ValueError('price refinement provenance differs')
        case=r['case'];ctx={k:case[k] for k in ['intervals','profile','basis']}
        old=next(x for x in baseline['cases'] if all(x[k]==v for k,v in ctx.items()))
        if case!=old:raise ValueError('native tangent or basis changed')
        checks+=r['checks'];diff=np.array(r['difference_batches']);means=diff.mean(axis=1);se=diff.std(axis=1,ddof=1)/np.sqrt(base['sobol_scrambles'])
        price_records.append(dict(**ctx,products=list(PRODUCTS),difference_batches=diff.tolist(),means=means.tolist(),standard_errors=se.tolist()))
        for o in baseline['observations']:
            if not all(o[k]==v for k,v in ctx.items()):continue
            if o['audit']['status']!='path_gamma_chain_passed':raise ValueError('failed native panel cannot be retained')
            j=PRODUCTS.index(o['product']);sctx=dict(**ctx,product=o['product'],seed=o['seed'])
            for coordinate,value in enumerate(o['audit']['recovered_directions']):
                for k,bump in enumerate(policy['iv_bumps']):
                    checks.append(threshold_check('Gamma_vs_independent_price_difference',float(abs(value['mean']-means[k,j,coordinate])),
                        float(6*np.hypot(value['standard_error'],se[k,j,coordinate])+base['absolute_tolerance']),coordinate=coordinate,bump=bump,**sctx))
    r=dict(document_kind='custom_variance_basis_refined_validation',schema_version=1,policy=policy,baseline_policy=base,
        status='passed' if all(x['passed'] for x in checks) and all(x['status']=='passed' for x in reports) else 'failed',
        checks=checks,cases=baseline['cases'],observations=baseline['observations'],price_records=price_records,
        calibration_audits=[x for r in reports for x in r['calibration_audits']],retained_stress_audits=[x for x in baseline['calibration_audits'] if 'stress' in x],
        initial_failed_checks=failed,source_sha256=frozen,native_binary_sha256=binary,prior_acceptance_sha256=prior,
        baseline_sha256=sha(baseline_path),price_report_sha256={str(p):sha(p) for p in price_paths},
        diagnostic_only=True,full_market_iv_bucket_acceptance='not_established')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--intervals',type=int,choices=[4,6]);p.add_argument('--basis',choices=KINDS)
    p.add_argument('--baseline-report',type=Path);p.add_argument('--price-reports',type=Path,nargs='+')
    a=p.parse_args();pp=a.policy.resolve()
    if a.baseline_report:
        if not a.price_reports or a.intervals or a.basis:p.error('merge requires price reports without case filters')
        r=merge(pp,a.baseline_report,a.price_reports)
    else:
        if a.intervals is None or a.basis is None or a.price_reports:p.error('price run requires intervals and basis')
        r=price_run(pp,a.output.with_suffix('.partial.json'),a.intervals,a.basis)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not x['passed'] for x in r['checks']))));return int(r['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
