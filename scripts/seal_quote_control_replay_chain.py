"""Append a verified daily replay to an immutable chain of accepted dates."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from continue_quote_control_replay import ACCEPTANCE, read_document, sha, verify_accepted_runtime
from check_portfolio_holdout import sealed
from check_portfolio_market import sources, write
from rust_pricing_backtest.domain import require, fingerprint
from rust_pricing_backtest.portfolio_market_risk import check_report


def verify_manifest(record):
    for section in ('source_sha256', 'evidence_sha256'):
        for name, expected in record[section].items():
            require(sha(ROOT/name) == expected, 'replay_chain_integrity', 'changed bound file: '+name)


def check_link(prior, previous, current):
    """Reject skipped/recounted dates, changed history, and false annual claims."""
    require(prior['status'] == 'passed' and prior['failed'] == 0,
            'replay_chain_prior', 'passing preceding acceptance required')
    n = prior['committed_snapshots']; total = prior['planned_snapshots']
    require(type(n) is int and type(total) is int and 0 < n < total
            and previous['processed_snapshots'] == current['preceding_snapshots'] == n
            and current['processed_snapshots'] == n+1
            and previous['planned_snapshots'] == current['planned_snapshots'] == total,
            'replay_chain_sequence', 'exactly the next date of the same schedule required')
    require(previous['status'] == 'partial' and previous['failure'] is None
            and current['status'] == ('completed' if n+1 == total else 'partial')
            and current['failure'] is None and current['annual_completed'] == (n+1 == total),
            'replay_chain_status', 'successful continuation with honest completion status required')
    require(current['sampling_policy'] == previous['sampling_policy'] == prior['sampling_policy']
            and current['original_sampling_error_limit'] == prior['original_precision_limit'] == 10.,
            'replay_chain_policy', 'sampling policy and numerical limit must not change')
    for name in ('held', 'proposed'):
        old, new = prior[name], current[name]
        require(len(old) == n and previous[name] == old and len(new) == n+1 and new[:n] == old
                and all(math.isfinite(v) and 0 <= v <= 10 for v in new),
                'replay_chain_history', 'history or precision gate changed: '+name)
    require(current['fresh_native_provider'] and not current['retained_native_audits_imported']
            and current['checkpoint_prefix_preserved'],
            'replay_chain_native', 'fresh native evaluation and preserved checkpoint prefix required')
    require(prior['cumulative_native_product_days'] == 30*n
            and prior['cumulative_independent_checks'] == 1024*n,
            'replay_chain_counts', 'preceding accepted counts are inconsistent')
    return n


def run(prior_path, previous_bundle, directory, output, *, notes=None):
    prior_path, previous_bundle, directory, output = (p.resolve() for p in (prior_path, previous_bundle, directory, output))
    require(not output.exists(), 'replay_chain_output', 'fresh acceptance record required')
    production = verify_accepted_runtime()
    prior = sealed(prior_path); verify_manifest(prior)
    previous_path = previous_bundle/'report.json'
    require(prior['evidence_sha256'].get(str(previous_path.relative_to(ROOT))) == sha(previous_path),
            'replay_chain_prior', 'previous bundle must be bound by the preceding acceptance')
    previous = sealed(previous_path)
    bundle = directory/'replay/committed'
    current = sealed(bundle/'report.json'); comparison = sealed(directory/'comparison.json')
    oracle = sealed(directory/'oracle.json'); native = sealed(directory/'replay/prefill.json')
    day = check_link(prior, previous, current)
    require(current['acceptance_sha256'] == previous['acceptance_sha256'] == sha(ACCEPTANCE)
            and current['native_unique_requests'] == 31, 'replay_chain_native', 'same provider and 31 raw requests required')
    require(current['original_files_sha256'] == {str(previous_bundle/name):sha(previous_bundle/name)
                for name in ('input.json.gz','checkpoint.json.gz')},
            'replay_chain_origin', 'continuation must use the exact preceding input and checkpoint')
    for folder, record in ((previous_bundle, previous), (bundle, current)):
        require(set(record['files']) == {'input.json.gz','checkpoint.json.gz','result.json.gz'},
                'replay_chain_bundle', 'complete committed bundle required')
        for name, info in record['files'].items():
            require(info['sha256'] == sha(folder/name) and info['schema'] == 'valid',
                    'replay_chain_bundle', 'saved document changed: '+name)
    require(current['files']['input.json.gz']['uncompressed_sha256'] == previous['files']['input.json.gz']['uncompressed_sha256'],
            'replay_chain_origin', 'full input changed')
    require(comparison['status'] == oracle['status'] == native['status'] == 'passed'
            and comparison['failed'] == 0 and comparison['independent_native_panels'] == 30
            and comparison['independent_checks'] == len(comparison['checks']) == 1024
            and all(c['passed'] for c in comparison['checks']) and all(c['passed'] for c in oracle['checks'])
            and comparison['day_index'] == day and not comparison['earlier_dates_recounted']
            and comparison['absolute_tolerances_unchanged'], 'replay_chain_comparison', 'passing new-date comparison required')
    require(comparison['continuation_report_sha256'] == sha(bundle/'report.json')
            and comparison['oracle_sha256'] == sha(directory/'oracle.json')
            and comparison['prior_acceptance_sha256'] == sha(ACCEPTANCE),
            'replay_chain_comparison', 'comparison is not bound to this day and oracle')
    require(all(r['source_sha256'] == sources() for r in (current, comparison, oracle, native)),
            'replay_chain_source', 'unchanged current production source required')
    for record, name in ((current,'continue_quote_control_replay.py'),
                         (comparison,'compare_quote_control_continuation.py'),(oracle,'check_rqmc_annual_oracle.py')):
        require(record['checker_sha256'] == sha(ROOT/'scripts'/name), 'replay_chain_source', 'checker changed: '+name)
    audits = list(native['native_audits'].values())
    require(len(audits) == 31 and len(native['events']) == 30 and all(e['status'] == 'passed' for e in native['events'])
            and all(a['status'] == 'path_gamma_chain_passed' and a['recovery_passed'] and a['validation_passed'] for a in audits),
            'replay_chain_native', 'all actual/virtual raw native gates required')
    saved = read_document(bundle/'result.json.gz')
    require(saved['failure'] is None and saved['status'] == current['status']
            and len(saved['ledger']) == len(saved['risks']) == len(saved['decisions']) == day+1,
            'replay_chain_result', 'saved result differs from committed status/counts')
    snapshot = saved['manifest']['market_history'][day]; market = saved['risks'][day]['joint_market']['report']
    require(snapshot['snapshot_id'] == current['attempted_snapshot_id'] and snapshot['at'] == comparison['date']
            and native['snapshot_hash'] == fingerprint(snapshot)
            and datetime.fromisoformat(snapshot['at']) > datetime.fromisoformat(saved['manifest']['market_history'][day-1]['at']),
            'replay_chain_date', 'wrong or non-increasing comparison date')
    require(current['held'] == [r['joint_market']['held']['max_bucket_standard_error'] for r in saved['risks']]
            and current['proposed'] == [r['joint_market']['max_bucket_standard_error'] for r in saved['decisions']],
            'replay_chain_history', 'summary precision differs from saved portfolio reports')
    check_report(market, {cid:json.loads(snapshot['pricing_requests'][cid]) for cid in market['instrument_order']})
    require(len(market['instrument_order']) == 30 and len(saved['decisions'][day]['candidates']) == 20
            and market['control_variate']['request']['product']['strike'] == (snapshot['spot']['bid']+snapshot['spot']['ask'])/2
            and all(e['master_scramble_seed'] == prior['seed'] and e['points_per_scramble'] == 512
                    and e['scramble_count'] == 24 for e in market['instrument_engines'].values()),
            'replay_chain_policy', 'full universe and fixed estimator required')
    by_request = {a['request_fingerprint']:{k:v for k,v in a.items() if k!='sampling_unit_values'} for a in audits}
    reported = [*market['audits'].values(), market['control_variate']['raw_native_audit']]
    require(all(by_request.get(a['request_fingerprint']) == a for a in reported),
            'replay_chain_native', 'reported actual/control audits do not match fresh native evidence')
    source = dict(prior['source_sha256'])
    for name in ('scripts/seal_quote_control_replay_chain.py','scripts/test_quote_control_replay_chain.py'):
        source[name] = sha(ROOT/name)
    if notes is not None:
        source[str(notes.resolve().relative_to(ROOT))] = sha(notes)
    evidence = dict(prior['evidence_sha256'])
    evidence[str(prior_path.relative_to(ROOT))] = sha(prior_path)
    evidence.update({str(p.relative_to(ROOT)):sha(p) for p in directory.rglob('*') if p.is_file()
        and p.suffix in ('.json','.gz','.log','.jsonl') and p.resolve() != output
        and p.name not in ('seal.log','source.tar.gz','completion-checks.json')})
    result = dict(document_kind='quote_control_replay_chain_acceptance', schema_version=1,
        status='passed', failed=0, valuation_date=snapshot['at'][:10],
        verified_at_utc=datetime.now(timezone.utc).isoformat(), production_source_unchanged=True,
        source_sha256=source, evidence_sha256=evidence, native_binary_sha256=production['native_binary_sha256'],
        previous_acceptance_path=str(prior_path.relative_to(ROOT)), previous_acceptance_sha256=sha(prior_path),
        production_acceptance_sha256=sha(ACCEPTANCE), preceding_snapshots=day, committed_snapshots=day+1,
        planned_snapshots=current['planned_snapshots'], sampling_policy=prior['sampling_policy'], seed=prior['seed'],
        held=current['held'], proposed=current['proposed'], original_precision_limit=10.,
        additional_native_product_days=30, cumulative_native_product_days=prior['cumulative_native_product_days']+30,
        additional_independent_checks=1024, cumulative_independent_checks=prior['cumulative_independent_checks']+1024,
        oracle_checks=len(oracle['checks']), fresh_native_continuation=True, saved_native_audits_imported=False,
        prefix_preserved=True, earlier_failures_retained=True, earlier_two_day_storage_recovery_remains_disclosed=True,
        annual_completed=current['annual_completed'], full_market_iv_bucket_acceptance='not_established',
        scope='next consecutive date of the same synthetic European-call workload, same seed; no new seed holdout')
    result['report_hash'] = fingerprint(result); write(output, result)
    print(json.dumps({k:result[k] for k in ('status','committed_snapshots','additional_independent_checks','cumulative_independent_checks')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prior','previous-bundle','directory','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--notes', type=Path)
    a = parser.parse_args(); run(a.prior,a.previous_bundle,a.directory,a.output,notes=a.notes)
