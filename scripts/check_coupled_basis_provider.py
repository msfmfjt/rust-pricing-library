"""Check the coupled-basis ledger against the prior independent raw-matrix evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest
from check_extension_provider import compare_audit

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
BOUNDARY = {f'backtesting/rust_pricing_backtest/{n}.py' for n in
            ('named_variance_basis', 'multi_step_provider', 'rust_adapter', '__main__')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals', type=int, choices=(4, 6))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--aggregate', nargs=2, type=Path)
    parser.add_argument('--recheck-report', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'tests/python'))
    import test_vega_kt_coupled_basis_provider as panel
    import rust_pricing as rp
    record_path = ROOT/'design/validation/local-vol-raw-basis-v0.1.json'
    record = json.loads(record_path.read_text())
    binary = sha(Path(getattr(rp, 'rust_pricing', rp).__file__))
    if record['status'] != 'passed' or record['native_binary_sha256'] != binary:
        raise ValueError('passing independent raw-matrix acceptance on this native binary required')
    references = {}
    for n in (4, 6):
        path = ROOT/f'benchmark-results/vega-kt-raw-basis/{n}.json'
        if sha(path) != record['numerical_report_sha256'][str(n)]:
            raise ValueError('independent numerical report changed')
        references[n] = json.loads(path.read_text())
    retained = {n: h for n, h in references[4]['source_sha256'].items() if n not in BOUNDARY}
    if references[4]['source_sha256'] != references[6]['source_sha256']:
        raise ValueError('independent numerical sources differ')
    for n, h in retained.items():
        if sha(ROOT/n) != h: raise ValueError('accepted numerical source changed: '+n)
    paths = {ROOT/n for n in retained | {n: '' for n in BOUNDARY}}
    paths.update(ROOT/n for n in (
        'scripts/check_coupled_basis_provider.py', 'scripts/check_extension_provider.py',
        'scripts/wheel_validation/provider_shards.py', 'tests/python/provider_evidence.py',
        'tests/python/test_vega_kt_coupled_basis_provider.py', 'tests/python/test_vega_kt_custom_basis_provider.py',
        'tests/python/test_vega_kt_extension_put_provider.py', 'tests/python/test_vega_kt_extension_provider.py',
        'tests/python/test_vega_kt_path_ledger.py', 'tests/python/test_vega_kt_multi_step_provider.py',
        'backtesting/tests/test_named_variance_basis.py', 'backtesting/tests/test_coupled_variance_basis.py',
        'fixtures/backtesting/raw-coupled-variance-basis-4-v0.1.json',
        'fixtures/backtesting/raw-coupled-variance-basis-6-v0.1.json'))
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    common = dict(document_kind='coupled_basis_provider_validation', schema_version=1,
                  source_sha256=frozen, native_binary_sha256=binary, prior_acceptance_sha256=sha(record_path),
                  reference_report_sha256=record['numerical_report_sha256'], retained_numerical_source_files=len(retained),
                  changed_boundary_files=sorted(BOUNDARY), full_market_iv_bucket_acceptance='not_established')
    if args.aggregate:
        reports = [json.loads(p.read_text()) for p in args.aggregate]
        if {r['intervals'] for r in reports} != {4, 6}: raise ValueError('both interval shards required')
        for r in reports:
            if r['status'] != 'passed' or any(r[k] != v for k, v in common.items()):
                raise ValueError('passing identical-source shards required')
        report = dict(common, tests=sum(r['tests'] for r in reports), failed=0, skipped=0,
                      evidence={k: sum((r['evidence'][k] for r in reports), []) for k in ('panels', 'replays', 'snapshots', 'failures')},
                      numerical_comparisons=sum((r['numerical_comparisons'] for r in reports), []),
                      shard_sha256={str(p): sha(p) for p in args.aggregate})
        report['evidence']['samples'] = {k: v for r in reports for k, v in r['evidence']['samples'].items()}
        expected = (20, 32, 48, 16, 10)
    else:
        if args.intervals is None: parser.error('--intervals required')
        if args.recheck_report:
            report = json.loads(args.recheck_report.read_text())
            if any(report[k] != v for k, v in common.items()) or report['intervals'] != args.intervals:
                raise ValueError('retained runtime provenance changed')
            if report['failed'] or report['skipped'] or report['tests'] != 5 or not report['source_unchanged']:
                raise ValueError('only passing unchanged runtime evidence can be rechecked')
        else:
            panel.CoupledBasisProviderTests.intervals = (args.intervals,)
            result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(panel.CoupledBasisProviderTests))
            report = dict(common, intervals=args.intervals, tests=result.testsRun,
                          failed=len(result.failures)+len(result.errors), skipped=len(result.skipped),
                          evidence=panel.CoupledBasisProviderTests.evidence, status='pending_reference',
                          source_unchanged=all(sha(ROOT/n) == h for n, h in frozen.items()))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix('.runtime.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        ref = references[args.intervals]; comparisons = []
        for p in report['evidence']['panels']:
            old = next(x for x in ref['observations'] if (x['product'], x['seed']) == (p['product'], p['seed']))
            a = p['audit']; m = a['market_gamma']
            comparisons.append(dict(intervals=p['intervals'], product=p['product'], seed=p['seed'],
                **compare_audit(m['gamma_audit'], old['audit']),
                construction_exact=(a['basis_construction']['basis_hash'] == ref['basis_input']['basis_hash']
                    == a['basis_construction']['accepted_matrix_hash']
                    and a['basis_construction']['basis_shape'] == ref['basis_input']['basis_shape']
                    and a['basis_construction']['parameter_metric'] == ref['basis_input']['parameter_metric']),
                tangent_exact=m['calibration_tangent'] == ref['calibration_tangent'] and m['tangent_audit'] == ref['tangent_audit']))
        report['numerical_comparisons'] = comparisons
        expected = (10, 16, 32 if args.intervals == 4 else 16, 14 if args.intervals == 4 else 2, 5)
    e = report['evidence']; comparisons = report['numerical_comparisons']
    actual = (len(comparisons), len(e['replays']), len(e['snapshots']), len(e['failures']), report['tests'])
    keys = {(p['intervals'], p['product'], p['seed']) for p in comparisons}
    replay_keys = {(p['intervals'], p['side'], p['kind'], p['strategy']) for p in e['replays']}
    report['source_unchanged'] = all(sha(ROOT/n) == h for n, h in frozen.items())
    passed = (not report['failed'] and not report['skipped'] and report['source_unchanged'] and actual == expected
              and len(keys) == actual[0] and len(replay_keys) == actual[1]
              and all(all(c[k] for k in ('numerical_audit_exact', 'gamma_bin_roundoff_passed', 'construction_exact', 'tangent_exact')) for c in comparisons))
    report['status'] = 'passed' if passed else 'failed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if passed:
        for key, sample in e['samples'].items():
            args.output.with_name(key+'-input.json').write_text(json.dumps(sample['request'], indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], counts=actual, expected=expected, source_unchanged=report['source_unchanged'])), flush=True)
    return int(not passed)


if __name__ == '__main__': raise SystemExit(main())
