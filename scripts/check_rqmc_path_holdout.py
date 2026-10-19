"""Audit unused native seeds against sealed, seed-independent price references."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from check_portfolio_market import sources, policy_for, write
from market_acceptance_cases import cases, requests
from rust_pricing_backtest.domain import fingerprint, require
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.multi_step_provider import gamma_boundaries
from rust_pricing_backtest.named_variance_basis import named_variance_basis
from rust_pricing_backtest.numerics import threshold_check
from rust_pricing_backtest.portfolio_market_risk import sampling_covariance, spot_iv_response

POLICY = ROOT/'fixtures/backtesting/rqmc-provider-policy-v0.2.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    require(value.get('report_hash') == fingerprint({k:v for k,v in value.items() if k!='report_hash'}),
            'holdout_reference', 'corrupted report: '+str(path))
    return value


def inputs(case_id):
    from check_rqmc_provider_holdout import inputs as checked_inputs
    _, precision, instruments, previous, provenance = checked_inputs(case_id)
    policy=json.loads(POLICY.read_text())
    require(case_id in policy['cases'] and len(set(policy['seeds']))==len(policy['seeds'])
            and len(policy['seeds'])>=2 and all(type(s) is int and 0<=s<2**64 for s in policy['seeds'])
            and not set(policy['seeds'])&set(policy['excluded_seeds']), 'rqmc_path_seed', 'fresh declared seeds required')
    for cid,p in instruments.items():
        p['engine']=dict(deepcopy(policy['engine']),master_scramble_seed=policy['seeds'][0])
        p['engine']['points_per_scramble']=policy['points_per_scramble_by_instrument'].get(cid,256)
    provenance.update(policy_sha256=sha(POLICY),checker_sha256=sha(__file__),
        orchestration_sha256=sha(ROOT/'scripts/check_rqmc_provider_holdout.py'))
    return policy,precision,instruments,previous,provenance


def run(case_id, output):
    import rust_pricing as rp
    started = time.perf_counter()
    policy, precision, instruments, previous, provenance = inputs(case_id)
    protected = [ROOT/n for n in provenance['source_sha256']]
    protected += [POLICY, Path(__file__), ROOT/policy['base_policy'], ROOT/policy['reference_acceptance'],
                  ROOT/policy['current_acceptance'], ROOT/f'benchmark-results/vega-kt-portfolio-market/{case_id}.json']
    for target in (output, output.with_suffix(output.suffix+'.tmp')):
        require(not target.exists() and all(target.resolve()!=p.resolve() for p in protected),
                'holdout_output', 'fresh output required; evidence must never be overwritten')
    ids = sorted(instruments); base = instruments[ids[0]]
    basis = named_variance_basis(base, 'coupled_row_hats')[0] if case_id.startswith('regression_') else None
    args = dict(extension='custom', variance_basis=basis) if basis is not None else dict(extension='all_rows')
    c = MultiStepExtensionCalibrator(base, price_tolerance=precision.get('calibration_price_tolerance', 1e-10), **args)
    tangent, tangent_audit = c.all_maturity_iv_tangent(); q = c.iv.size
    oracle = previous['price_record']
    require(oracle['instrument_order']==ids, 'holdout_reference', 'reference instrument order changed')
    bp = np.asarray(oracle['prices']); diffs = np.asarray(oracle['differences'])
    ref_mean = diffs.mean(axis=1); ref_se = diffs.std(axis=1, ddof=1)/np.sqrt(precision['sobol_scrambles'])
    weights = np.array([((-1)**j)*(j+1)/10 for j in range(len(ids))])
    response, response_audit = spot_iv_response(base)
    result = dict(document_kind='portfolio_unused_seed_validation', schema_version=1, case=case_id,
        status='in_progress', policy=policy, precision=precision, provenance=provenance,
        independent_reference_reused=True, independent_price_oracle_rerun=False,
        instrument_order=ids, tangent_audit=tangent_audit, spot_iv_response_audit=response_audit,
        checks=[], panels=[], portfolios=[])

    def check(name, value, limit, **context):
        result['checks'].append(threshold_check(name, float(value), float(limit), **context))

    def save():
        require(sources()==provenance['source_sha256'] and sha(POLICY)==provenance['policy_sha256']
                and sha(__file__)==provenance['checker_sha256'], 'holdout_source', 'inputs changed during validation')
        result['elapsed_seconds'] = time.perf_counter()-started
        body = {k:v for k,v in result.items() if k!='report_hash'}
        body['report_hash'] = fingerprint(body); write(output, body)

    check('tangent_IV_identity', tangent_audit['iv_response_residual'], 1e-9)
    for seed in policy['seeds']:
        cube = []; deltas = []; native_passed = True
        for j,cid in enumerate(ids):
            p = deepcopy(instruments[cid])
            p['engine']['master_seed' if p['engine']['type']=='pseudo_monte_carlo' else 'master_scramble_seed'] = seed
            request_hash = fingerprint(p)
            p['risk'].pop('vega_kt', None); p['risk']['vega'] = False
            plan = rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),
                                        worker_threads=policy['worker_threads'], reduction_block_size=256)
            before = time.perf_counter()
            try:
                audit = plan.evaluate_vega_kt_multi_step_gamma_chain_audit(list(range(c.intervals)),
                    gamma_boundaries(p, c.spot, put_knots=True),
                    variance_risk_directions=tangent.T.reshape(q, c.intervals, len(c.nodes)).tolist(),
                    subdivisions=precision.get('subdivisions_by_instrument', {}).get(cid, precision['subdivisions']),
                    relative_tolerance=precision['relative_tolerance'],
                    gamma_relative_standard_error_limit=precision['gamma_relative_standard_error_limit'], retain_samples=True)
            except rp.PricingError as error:
                native_passed=False
                check('native_runtime_gate', 1, 0, instrument=cid, seed=seed)
                panel=dict(instrument=cid,seed=seed,request_hash=request_hash,accepted=False,
                           native_error=str(error),elapsed_seconds=time.perf_counter()-before)
                result['panels'].append(panel);save()
                print(json.dumps(panel),flush=True)
                continue
            raw = audit.pop('sampling_unit_values', None)
            accepted = audit['status']=='path_gamma_chain_passed' and audit['estimate'] is not None
            native_passed &= accepted
            check('native_runtime_gate', not accepted, 0, instrument=cid, seed=seed)
            panel = dict(instrument=cid, seed=seed, request_hash=request_hash, audit=audit,
                         elapsed_seconds=time.perf_counter()-before, accepted=accepted)
            result['panels'].append(panel)
            if accepted:
                sample = np.asarray(raw); cube.append(sample)
                require(sample.shape==(audit['independent_sampling_units'],q+1), 'holdout_samples', 'unexpected sample layout')
                ordinary = plan.evaluate(); deltas.append(ordinary.delta_raw)
                check('ordinary_price_identity', abs(ordinary.value-audit['native_price']['mean']), 1e-12, instrument=cid, seed=seed)
                check('retained_sample_means', np.max(abs(sample[:,1:].mean(axis=0)-[v['mean'] for v in audit['estimate']])), 1e-10, instrument=cid, seed=seed)
                own = np.array(audit['joint_sample_covariance']).reshape(3*q,3*q)[q:2*q,q:2*q]
                check('retained_sample_covariance', np.max(abs(sampling_covariance(sample[:,1:])-own)), 1e-8, instrument=cid, seed=seed)
                check('price_vs_independent', abs(ordinary.value-bp[:,j].mean()),
                    policy['sigma_multiplier']*np.hypot(audit['native_price']['standard_error'],bp[:,j].std(ddof=1)/np.sqrt(len(bp)))+precision['absolute_tolerance'], instrument=cid, seed=seed)
                for k,estimate in enumerate(audit['estimate']):
                    for b,bump in enumerate(precision['iv_bumps']):
                        check('Gamma_vs_independent_price_difference', abs(estimate['mean']-ref_mean[b,j,k]),
                            policy['sigma_multiplier']*np.hypot(estimate['standard_error'],ref_se[b,j,k])+precision['absolute_tolerance'],
                            instrument=cid, seed=seed, coordinate=k, bump=bump)
            save()
            print(json.dumps(dict(case=case_id, seed=seed, instrument=cid, status=audit['status'],
                                  elapsed_seconds=panel['elapsed_seconds'])), flush=True)
        if native_passed:
            samples = np.stack(cube, axis=1); n = len(samples)
            covariance = sampling_covariance(samples[:,:,1:].reshape(n,-1))
            net = np.einsum('niq,i->nq',samples,weights); net_cov = sampling_covariance(net)/n
            transform = np.kron(weights[None,:],np.eye(q))
            check('joint_covariance_PSD', max(0.,-np.linalg.eigvalsh(covariance).min()), 1e-8, seed=seed)
            check('joint_projection_vs_direct', np.max(abs(transform@covariance@transform.T/n-net_cov[1:,1:])), 1e-10, seed=seed)
            check('off_diagonal_covariance_present', not np.any(abs(covariance[:q,q:2*q])>1e-6), 0, seed=seed)
            portfolio = dict(seed=seed, status='available', net_buckets=net[:,1:].mean(axis=0).tolist(),
                net_bucket_standard_errors=np.sqrt(np.maximum(0,np.diag(net_cov)[1:])).tolist(), pnl=[])
            for scenario in previous['pnl']:
                old = scenario['explanation']; ds = old['spot_change']; div = np.array(old['iv_change'])
                net_vega = net[:,1:].mean(axis=0); delta = float(weights@np.array(deltas))
                explained = (delta-float(net_vega@response))*ds+float(net_vega@div)
                actual = np.array(scenario['independent_pnl_batches'])
                vector = div-response*ds; variance = float(vector@net_cov[1:,1:]@vector)
                residual = float(actual.mean()-explained)
                check('Pnl_vs_independent_revaluation', abs(residual), policy['sigma_multiplier']*
                    np.hypot(np.sqrt(max(0.,variance)),actual.std(ddof=1)/np.sqrt(len(actual)))+precision['absolute_tolerance'],
                    seed=seed, mode=scenario['mode'], scale=scenario['scale'])
                portfolio['pnl'].append(dict(mode=scenario['mode'], scale=scenario['scale'],
                    available_greek_pnl=explained, unexplained_pnl=residual,
                    vega_and_delta_correction_standard_error=np.sqrt(max(0.,variance))))
            result['portfolios'].append(portfolio)
        else:
            result['portfolios'].append(dict(seed=seed, status='unavailable_native_gate_failure'))
        save()
    result['status'] = 'passed' if all(v['passed'] for v in result['checks']) else 'failed'
    result['failed'] = sum(not v['passed'] for v in result['checks'])
    result['completed_panels'] = len(result['panels']); save()
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.case, args.output)
    print(json.dumps(dict(status=result['status'], failed=result['failed'], panels=result['completed_panels'])))
    raise SystemExit(result['status']!='passed')
