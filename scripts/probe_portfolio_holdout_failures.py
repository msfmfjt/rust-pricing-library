"""Exploratory precision probes; never overwrite or promote the held-out result."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import inputs,sealed,sha
from check_portfolio_market import write,sources
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.multi_step_provider import gamma_boundaries
from rust_pricing_backtest.named_variance_basis import named_variance_basis
from rust_pricing_backtest.numerics import threshold_check

# Declared before running probes. This is failure diagnosis, not new held-out acceptance.
SETTINGS=[dict(id='pseudo_MC_4x',sample_multiplier=4),dict(id='pseudo_MC_2x',sample_multiplier=2),
          dict(id='RQMC_256x16',points_per_scramble=256,scramble_count=16,brownian_bridge=False)]


def run(failure_path,output):
    import rust_pricing as rp
    require(not output.exists() and not output.with_suffix(output.suffix+'.tmp').exists(),
            'probe_output','fresh output required')
    original=sealed(failure_path);case=original['case']
    require(original['status']=='failed','probe_input','completed failed held-out evidence required')
    policy,precision,instruments,previous,provenance=inputs(case)
    require(original['provenance']==provenance,'probe_input','held-out implementation changed')
    ids=sorted(instruments);base=instruments[ids[0]]
    basis=named_variance_basis(base,'coupled_row_hats')[0] if case.startswith('regression_') else None
    args=dict(extension='custom',variance_basis=basis) if basis is not None else dict(extension='all_rows')
    c=MultiStepExtensionCalibrator(base,**args);tangent,_=c.all_maturity_iv_tangent();q=c.iv.size
    diffs=np.asarray(previous['price_record']['differences']);means=diffs.mean(axis=1)
    errors=diffs.std(axis=1,ddof=1)/np.sqrt(precision['sobol_scrambles'])
    result=dict(document_kind='holdout_failure_precision_probe',schema_version=1,status='in_progress',case=case,
        role='exploratory_in_sample_no_acceptance_promotion',original_holdout_status='failed',
        original_holdout_sha256=sha(failure_path),provenance=provenance,checker_sha256=sha(__file__),settings=SETTINGS,probes=[])
    for panel in original['panels']:
        if panel['accepted']:continue
        cid=panel['instrument'];seed=panel['seed'];j=ids.index(cid)
        for setting in SETTINGS:
            p=deepcopy(instruments[cid]);engine=p['engine']
            require(engine['type']=='pseudo_monte_carlo','probe_scope','this diagnosis targets pseudo-MC failures')
            if 'sample_multiplier' in setting:
                engine.update(master_seed=seed,independent_sampling_units=engine['independent_sampling_units']*setting['sample_multiplier'])
            else:
                p['engine']=dict(type='randomized_quasi_monte_carlo',master_scramble_seed=seed,
                    points_per_scramble=setting['points_per_scramble'],scramble_count=setting['scramble_count'],
                    variance_reduction=dict(antithetic=True,brownian_bridge=setting['brownian_bridge']))
            request_hash=fingerprint(p);p['risk'].pop('vega_kt',None);p['risk']['vega']=False
            before=time.perf_counter()
            plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=policy['worker_threads'],reduction_block_size=256)
            native_error=None
            try:
                audit=plan.evaluate_vega_kt_multi_step_gamma_chain_audit(list(range(c.intervals)),gamma_boundaries(p,c.spot,put_knots=True),
                    variance_risk_directions=tangent.T.reshape(q,c.intervals,len(c.nodes)).tolist(),
                    subdivisions=precision.get('subdivisions_by_instrument',{}).get(cid,precision['subdivisions']),
                    relative_tolerance=precision['relative_tolerance'],gamma_relative_standard_error_limit=precision['gamma_relative_standard_error_limit'])
            except rp.PricingError as error:
                native_error=str(error);audit=None
            checks=[]
            for k,estimate in enumerate(audit['recovered_directions'] if audit is not None else []):
                for b,bump in enumerate(precision['iv_bumps']):
                    checks.append(threshold_check('Gamma_vs_independent_price_difference',float(abs(estimate['mean']-means[b,j,k])),
                        float(policy['sigma_multiplier']*np.hypot(estimate['standard_error'],errors[b,j,k])+precision['absolute_tolerance']),coordinate=k,bump=bump))
            probe=dict(instrument=cid,seed=seed,setting=setting['id'],request_hash=request_hash,engine=p['engine'],
                audit=audit,native_error=native_error,checks=checks,elapsed_seconds=time.perf_counter()-before,
                probe_passed=audit is not None and audit['status']=='path_gamma_chain_passed' and all(v['passed'] for v in checks))
            result['probes'].append(probe)
            require(sources()==provenance['source_sha256'] and sha(__file__)==result['checker_sha256'],
                    'probe_source','source changed during probe')
            body={k:v for k,v in result.items() if k!='report_hash'};body['report_hash']=fingerprint(body);write(output,body)
            print(json.dumps({k:probe[k] for k in ('instrument','seed','setting','probe_passed','elapsed_seconds')}),flush=True)
    result['status']='complete';result['report_hash']=fingerprint(result);write(output,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failure',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.failure,a.output)
