"""Independently compare only the newly committed date of a daily continuation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed
from check_portfolio_market import sources, write
from continue_quote_control_replay import read_document, sha, verify_accepted_runtime, ACCEPTANCE
from rust_pricing_backtest.domain import require, fingerprint
from rust_pricing_backtest.portfolio_market_risk import check_report, reweight_portfolio, sampling_covariance
from rust_pricing_backtest.numerics import threshold_check


def run(directory, oracle_path, output):
    directory, oracle_path, output = map(Path, (directory, oracle_path, output))
    require(not output.exists(), 'continuation_comparison', 'fresh output required')
    verify_accepted_runtime()
    stage = sealed(directory/'report.json'); oracle = sealed(oracle_path)
    require(stage['status'] in ('partial', 'completed') and stage['failure'] is None
            and stage['source_sha256'] == oracle['source_sha256'] == sources()
            and stage['acceptance_sha256'] == sha(ACCEPTANCE) and oracle['status'] == 'passed',
            'continuation_comparison', 'successful current-source continuation and independent oracle required')
    for name, info in stage['files'].items():
        require(sha(directory/name) == info['sha256'], 'continuation_comparison', 'saved bundle changed: '+name)
    saved = read_document(directory/'result.json.gz')
    day = stage['preceding_snapshots']
    require(len(saved['risks']) == day+1 and len(saved['decisions']) == day+1
            and stage['processed_snapshots'] == day+1, 'continuation_comparison', 'exactly one new date required')
    row = saved['risks'][day]; report = row['joint_market']['report']; ids = report['instrument_order']
    snapshot = saved['manifest']['market_history'][day]
    require(snapshot['snapshot_id'] == stage['attempted_snapshot_id'], 'continuation_comparison', 'wrong date')
    requests = {cid:json.loads(snapshot['pricing_requests'][cid]) for cid in ids}
    check_report(report, requests)
    require(ids == oracle['instrument_order'] and
            {cid:fingerprint({k:v for k,v in p.items() if k!='engine'}) for cid,p in requests.items()}
            == oracle['engine_independent_request_hashes'], 'continuation_comparison', 'independent request differs')
    diff = np.array(oracle['differences']); prices = np.array(oracle['prices']); precision = oracle['precision']
    # Same two-bump / six-standard-error criterion as the accepted two-day comparison.
    means = diff.mean(axis=1); errors = diff.std(axis=1, ddof=1)/np.sqrt(diff.shape[1])
    checks = []
    def check(name, observed, limit, **context):
        checks.append(threshold_check(name, float(observed), float(limit), day=day, **context))
    for j, cid in enumerate(ids):
        audit, item = report['audits'][cid], report['instruments'][cid]
        check('raw_native_gate', not (audit['status']=='path_gamma_chain_passed'
              and audit['recovery_passed'] and audit['validation_passed']), 0, instrument=cid)
        check('ordinary_price_vs_independent', abs(item['price']-prices[:,j].mean()),
              6*np.hypot(audit['native_price']['standard_error'], prices[:,j].std(ddof=1)/np.sqrt(len(prices)))
              +precision['absolute_tolerance'], instrument=cid)
        for b, bump in enumerate(precision['iv_bumps']):
            for k, value in enumerate(item['buckets']):
                check('controlled_Gamma_vs_independent_price_difference', abs(value-means[b,j,k]),
                      6*np.hypot(item['bucket_standard_errors'][k], errors[b,j,k])+precision['absolute_tolerance'],
                      instrument=cid, coordinate=k, bump=bump)
    cube = np.array(report['sampling_unit_values']); n = report['independent_sampling_units']
    for label, gate in [('held', row['joint_market']['held']), ('proposed', saved['decisions'][day]['joint_market'])]:
        quantities = gate['quantities']; weights = np.array([quantities[cid] for cid in ids])
        net = reweight_portfolio(report, requests, quantities)
        reference = np.einsum('bsjq,j->bsq', diff, weights)
        raw = np.einsum('niq,i->nq', cube, weights)
        check('net_covariance_direct_samples', np.max(abs(sampling_covariance(raw)/n
              -np.array(net['net_price_and_market_mean_covariance']))), 1e-9, book=label)
        for b, bump in enumerate(precision['iv_bumps']):
            for k, value in enumerate(net['net_buckets']):
                samples = reference[b,:,k]
                check('net_controlled_Gamma_vs_independent', abs(value-samples.mean()),
                      6*np.hypot(net['net_bucket_standard_errors'][k], samples.std(ddof=1)/np.sqrt(len(samples)))
                      +precision['absolute_tolerance']*sum(abs(weights)), book=label, coordinate=k, bump=bump)
        check('unchanged_net_sampling_error_gate', max(net['net_bucket_standard_errors']), 10., book=label)
    failed = sum(not c['passed'] for c in checks)
    record = dict(document_kind='quote_control_continuation_comparison', schema_version=1,
        status='failed' if failed else 'passed', failed=failed, checks=checks,
        independent_native_panels=len(ids), independent_checks=len(checks), date=snapshot['at'],
        day_index=day, source_sha256=sources(), checker_sha256=sha(__file__),
        continuation_report_sha256=sha(directory/'report.json'), oracle_sha256=sha(oracle_path),
        prior_acceptance_sha256=sha(ACCEPTANCE), absolute_tolerances_unchanged=True,
        earlier_dates_recounted=False, annual_completed=stage['annual_completed'],
        scope='only the single newly committed date; not full smile/annual acceptance')
    record['report_hash'] = fingerprint(record); write(output, record)
    print(json.dumps({k:record[k] for k in ('status','failed','independent_native_panels','independent_checks')}))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); r = run(a.directory, a.oracle, a.output)
    raise SystemExit(r['status'] != 'passed')
