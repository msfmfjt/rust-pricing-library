"""Source-bound annual-book precision experiments; never promote provider policy."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed, sha
from check_portfolio_market import sources, write
from rust_pricing_backtest.domain import fingerprint, require
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.portfolio_market_risk import evaluate_portfolio, implementation_identity, sampling_covariance
from rust_pricing_backtest.numerics import threshold_check


def run(pilot, oracle_path, policy_path, setting, seed, output, universe):
    require(not output.exists(), 'precision_output', 'fresh output required')
    policy = json.loads(policy_path.read_text())
    selected = policy['settings'][setting]
    require(seed in policy['seeds'], 'precision_seed', 'seed must be frozen before execution')
    args = load_input((pilot/'backtest/input.json').read_text())
    snapshot = args['history'][0]
    master = {c.contract_id:c for c in args['contracts']}
    weights = {}
    for p in args['portfolio']:
        weights[p.contract_id] = weights.get(p.contract_id, 0.) + p.quantity*master[p.contract_id].multiplier
    if universe == 'full':
        weights = {cid:weights.get(cid, 0.) for cid in snapshot.pricing_requests}
    requests = {cid:json.loads(snapshot.pricing_requests[cid]) for cid in sorted(weights)}
    for p in requests.values():
        require(p['product']['type']=='european_vanilla', 'precision_scope', 'European workload required')
        p['engine'] = dict(deepcopy(selected), master_scramble_seed=seed)
    oracle = sealed(oracle_path)
    require(oracle['status']=='passed' and all(
        fingerprint({k:v for k,v in p.items() if k!='engine'}) == oracle['engine_independent_request_hashes'][cid]
        for cid,p in requests.items()), 'precision_oracle', 'exact independent reference required')
    frozen = sources(); identity = implementation_identity()
    dependencies = {str(p.resolve()):sha(p) for p in (Path(__file__), policy_path, oracle_path, pilot/'backtest/input.json')}
    before = time.perf_counter()
    result = dict(document_kind='rqmc_portfolio_precision_probe', schema_version=1,
        role=policy['role'], setting=setting, seed=seed, policy=policy,
        source_sha256=frozen, implementation_identity=identity, dependencies_sha256=dependencies,
        universe=universe, instruments=len(weights), quantities=weights,
        scope='annual initial snapshot only; no normal-provider or annual completion promotion', checks=[])
    def check(name, value, limit, **ctx):
        result['checks'].append(threshold_check(name, float(value), float(limit), **ctx))
    try:
        native = evaluate_portfolio(requests, weights, extension='all_rows', worker_threads=4, retain_samples=True)
        ids = native['instrument_order']; count = native['independent_sampling_units']
        samples = np.asarray(native['sampling_unit_values']); w = np.array([weights[cid] for cid in ids])
        direct = np.einsum('niq,i->nq', samples, w)
        check('weighted_covariance_vs_direct', np.max(abs(sampling_covariance(direct)/count - np.array(native['net_price_and_market_mean_covariance']))), 1e-9)
        transform = np.kron(w[None,:], np.eye(len(native['coordinate_order'])))
        check('cross_instrument_covariance_projection', np.max(abs(transform@np.array(native['market_mean_covariance'])@transform.T - np.array(native['net_price_and_market_mean_covariance'])[1:,1:])), 1e-8)
        prices = np.asarray(oracle['prices']); diff = np.asarray(oracle['differences'])
        precision = oracle['precision']; oracle_count = len(prices)
        for cid in ids:
            j = oracle['instrument_order'].index(cid); audit = native['audits'][cid]
            check('native_runtime_gate', audit['status']!='path_gamma_chain_passed', 0, instrument=cid)
            check('price_vs_independent', abs(audit['native_price']['mean']-prices[:,j].mean()),
                6*np.hypot(audit['native_price']['standard_error'], prices[:,j].std(ddof=1)/np.sqrt(oracle_count))+precision['absolute_tolerance'], instrument=cid)
            for b,bump in enumerate(precision['iv_bumps']):
                for k,moment in enumerate(audit['estimate']):
                    reference = diff[b,:,j,k]
                    check('Gamma_vs_independent_price_difference', abs(moment['mean']-reference.mean()),
                        6*np.hypot(moment['standard_error'], reference.std(ddof=1)/np.sqrt(oracle_count))+precision['absolute_tolerance'], instrument=cid, coordinate=k, bump=bump)
        # Project the independent oracle with the same signed quantities, keeping
        # its own cross-instrument correlation rather than summing marginal SEs.
        indices = [oracle['instrument_order'].index(cid) for cid in ids]
        for b,bump in enumerate(precision['iv_bumps']):
            net_ref = np.einsum('niq,i->nq', diff[b,:,indices,:].transpose(1,0,2), w)
            for k,value in enumerate(native['net_buckets']):
                check('held_Gamma_vs_independent_price_difference', abs(value-net_ref[:,k].mean()),
                    6*np.hypot(native['net_bucket_standard_errors'][k], net_ref[:,k].std(ddof=1)/np.sqrt(oracle_count))+sum(abs(w))*precision['absolute_tolerance'], coordinate=k, bump=bump)
        value = max(native['net_bucket_standard_errors']); limit = args['config'].max_joint_bucket_standard_error
        check('held_bucket_standard_error', value, limit)
        result.update(native_report=native, max_bucket_standard_error=value, limit=limit,
            worst_coordinate=native['coordinate_order'][int(np.argmax(native['net_bucket_standard_errors']))])
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    require(sources()==frozen and identity==implementation_identity() and all(sha(p)==v for p,v in dependencies.items()),
        'precision_source', 'sources or inputs changed during execution')
    result.update(status='passed' if 'error' not in result and all(c['passed'] for c in result['checks']) else 'failed',
        failed=sum(not c['passed'] for c in result['checks']), elapsed_seconds=time.perf_counter()-before)
    result['report_hash'] = fingerprint(result); write(output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','native_report','checks','policy')}), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','oracle','policy','output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--setting', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--universe', choices=('held','full'), default='held')
    a=p.parse_args(); r=run(a.pilot,a.oracle,a.policy,a.setting,a.seed,a.output,a.universe)
    raise SystemExit(r['status']!='passed')
