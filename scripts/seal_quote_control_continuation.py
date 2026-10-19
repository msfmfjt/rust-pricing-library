"""Bind one additional independently checked day to the previous acceptance."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from continue_quote_control_replay import read_document, sha, verify_accepted_runtime, ACCEPTANCE
from check_portfolio_holdout import sealed
from check_portfolio_market import sources, write
from rust_pricing_backtest.domain import require, fingerprint
from rust_pricing_backtest.portfolio_market_risk import check_report


def run(directory, output):
    directory, output = directory.resolve(), output.resolve()
    require(not output.exists(), 'continuation_seal', 'fresh acceptance record required')
    prior = verify_accepted_runtime()
    require(prior['independent_native_panels'] == 60 and prior['independent_checks'] == 2048,
            'continuation_seal', 'the accepted two-day baseline is required')
    continuation = sealed(directory/'day-003/committed/report.json')
    comparison = sealed(directory/'comparison-2.json')
    oracle = sealed(directory/'oracle-2.json')
    native = sealed(directory/'day-003/prefill.json')
    schedule = sealed(directory/'schedule-preflight.json')
    storage = sealed(directory/'storage-probe.json')
    require(storage['status'] == 'passed' and storage['prior_checkpoint_unchanged']
            and storage['new_native_evaluations'] == 0
            and storage['checker_sha256'] == sha(ROOT/'scripts/continue_quote_control_replay.py')
            and storage['output']['uncompressed_sha256'] == storage['original_uncompressed_sha256']
            and storage['output']['sha256'] == sha(directory/'storage-probe-checkpoint.json.gz'),
            'continuation_seal', 'actual checkpoint compression/readback proof required')
    require(continuation['status'] == 'partial' and continuation['failure'] is None
            and continuation['preceding_snapshots'] == 2 and continuation['processed_snapshots'] == 3
            and continuation['planned_snapshots'] == 252 and not continuation['annual_completed'],
            'continuation_seal', 'exact third-day partial continuation required')
    require(comparison['status'] == oracle['status'] == native['status'] == 'passed'
            and comparison['failed'] == 0 and comparison['independent_native_panels'] == 30
            and comparison['independent_checks'] == 1024 and not comparison['earlier_dates_recounted'],
            'continuation_seal', 'new-date independent comparison required')
    require(comparison['continuation_report_sha256'] == sha(directory/'day-003/committed/report.json')
            and comparison['oracle_sha256'] == sha(directory/'oracle-2.json')
            and continuation['acceptance_sha256'] == comparison['prior_acceptance_sha256'] == sha(ACCEPTANCE),
            'continuation_seal', 'comparison/continuation/prior binding differs')
    require(all(r['source_sha256'] == sources() for r in (continuation, comparison, oracle, native, schedule)),
            'continuation_seal', 'source differs from evaluated evidence')
    for evidence_record, filename in ((continuation, 'continue_quote_control_replay.py'),
                                      (comparison, 'compare_quote_control_continuation.py'),
                                      (schedule, 'check_quote_control_schedule.py'),
                                      (oracle, 'check_rqmc_annual_oracle.py')):
        require(evidence_record['checker_sha256'] == sha(ROOT/'scripts'/filename),
                'continuation_seal', 'checker changed after evaluation: '+filename)
    for p, expected in continuation['original_files_sha256'].items():
        require(sha(p) == expected, 'continuation_seal', 'original checkpoint/input changed')
    for name, info in continuation['files'].items():
        require(sha(directory/'day-003/committed'/name) == info['sha256'] and info['schema'] == 'valid',
                'continuation_seal', 'saved document differs')
    require(continuation['fresh_native_provider'] and not continuation['retained_native_audits_imported']
            and continuation['checkpoint_prefix_preserved'] and continuation['native_unique_requests'] == 31,
            'continuation_seal', 'fresh native continuation required')
    audits = native['native_audits']
    require(len(audits) == 31 and len(native['events']) == 30
            and all(e['status'] == 'passed' for e in native['events'])
            and all(a['status'] == 'path_gamma_chain_passed' and a['recovery_passed'] and a['validation_passed'] for a in audits.values()),
            'continuation_seal', 'all actual and virtual raw native gates required')
    saved = read_document(directory/'day-003/committed/result.json.gz')
    report = saved['risks'][2]['joint_market']['report']; snapshot = saved['manifest']['market_history'][2]
    check_report(report, {cid:json.loads(snapshot['pricing_requests'][cid]) for cid in report['instrument_order']})
    require(all(e['master_scramble_seed'] == 100129 and e['points_per_scramble'] == 512
                and e['scramble_count'] == 24 for e in report['instrument_engines'].values()),
            'continuation_seal', 'original fixed sampling design/seed required')
    require(len(report['instrument_order']) == 30 and len(saved['decisions'][2]['candidates']) == 20
            and report['control_variate']['request']['product']['strike'] == (snapshot['spot']['bid']+snapshot['spot']['ask'])/2
            and max(continuation['held']+continuation['proposed']) <= 10,
            'continuation_seal', 'full universe and original precision limit required')
    require(schedule['status'] == 'ready_for_numerical_validation' and schedule['failed'] == 0
            and schedule['snapshot_count'] == 252 and schedule['contract_snapshot_count'] == 7560
            and schedule['native_evaluations'] == 0 and schedule['numerical_acceptance'] == 'not_evaluated',
            'continuation_seal', 'schedule preflight must not claim numerical acceptance')
    for name, count in [('storage-runtime-tests.log',16), ('comparison-tests.log',3)]:
        log = (directory/name).read_text()
        require(f'Ran {count} tests' in log and log.rstrip().endswith('OK'), 'continuation_seal', 'passing test log required')
    source = dict(prior['source_sha256'])
    names = ('continue_quote_control_replay.py', 'test_continue_quote_control_replay.py',
             'compare_quote_control_continuation.py', 'test_compare_quote_control_continuation.py',
             'check_quote_control_schedule.py', 'seal_quote_control_continuation.py')
    source.update({'scripts/'+name:sha(ROOT/'scripts'/name) for name in names})
    for name in ('design/README.md', 'design/validation/vega-kt-quote-control-continuation-v0.1.md'):
        source[name] = sha(ROOT/name)
    evidence = {str(p.relative_to(ROOT)):sha(p) for p in directory.rglob('*') if p.is_file()
                and p.suffix in ('.json','.gz','.log','.jsonl')
                and p.name not in ('seal.log','source.tar.gz','completion-checks.json')}
    result = dict(document_kind='quote_control_continuation_acceptance', schema_version=1,
        date='2026-10-15', status='passed', failed=0, prior_acceptance_sha256=sha(ACCEPTANCE),
        native_binary_sha256=prior['native_binary_sha256'], source_sha256=source, evidence_sha256=evidence,
        production_source_unchanged=True, sampling_policy=prior['sampling_policy'], seed=100129,
        additional_native_product_days=30, cumulative_native_product_days=90,
        additional_independent_checks=1024, cumulative_independent_checks=3072,
        oracle_checks=len(oracle['checks']), committed_snapshots=3, planned_snapshots=252,
        held=continuation['held'], proposed=continuation['proposed'], original_precision_limit=10.,
        fresh_native_continuation=True, saved_native_audits_imported=False,
        original_input_and_checkpoint_unchanged=True, compressed_atomic_bundle=True,
        new_storage_and_comparison_tests=10, runtime_tests=9,
        schedule_preflight_contract_days=7560, schedule_preflight_numerical_acceptance='not_evaluated',
        annual_completed=False, full_market_iv_bucket_acceptance='not_established',
        earlier_failures_retained=True, earlier_two_day_storage_recovery_remains_disclosed=True,
        scope='third consecutive date of the same synthetic European-call workload; same seed, no new seed holdout',
        timing='contract prefetch concurrency and local workstation activity; not isolated annual throughput')
    result['report_hash'] = fingerprint(result); write(output, result)
    print(json.dumps({k:result[k] for k in ('status','committed_snapshots','additional_independent_checks','cumulative_independent_checks')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); run(a.directory, a.output)
