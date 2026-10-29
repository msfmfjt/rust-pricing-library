"""Accept named-basis providers against frozen independent numerical evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest
from check_extension_provider import compare_audit

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
POLICIES=('row_hats','weighted_time_hats')
CHANGED_BOUNDARY_FILES={f'backtesting/rust_pricing_backtest/{n}.py' for n in
                        ('multi_step_provider','rust_adapter','__main__')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-report',type=Path,default=ROOT/'benchmark-results/vega-kt-custom-basis/report.json')
    parser.add_argument('--intervals',type=int,choices=(4,6))
    parser.add_argument('--basis',choices=POLICIES)
    parser.add_argument('--recheck-report',type=Path)
    parser.add_argument('--aggregate',nargs=4,type=Path)
    args=parser.parse_args();sys.path.insert(0,str(ROOT/'tests/python'))
    import test_vega_kt_custom_basis_provider as panel
    import rust_pricing as rp
    ref=json.loads(args.reference_report.read_text());binary=sha(Path(getattr(rp,'rust_pricing',rp).__file__))
    if ref['status']!='passed' or ref['native_binary_sha256']!=binary:
        raise ValueError('passing independent named-basis numerical acceptance on the same binary required')
    # Only provider/CLI boundary code changes here. Every prior numerical source,
    # native source, basis definition, independent oracle and policy stays frozen.
    retained={n:h for n,h in ref['source_sha256'].items() if n not in CHANGED_BOUNDARY_FILES}
    for n,h in retained.items():
        if sha(ROOT/n)!=h: raise ValueError('accepted numerical source changed: '+n)
    paths={ROOT/n for n in retained}
    paths.update((ROOT/'backtesting/rust_pricing_backtest').glob('*.py'))
    paths.update((ROOT/'backtesting/rust_pricing_backtest/schemas').glob('*.json'))
    paths.update(ROOT/n for n in ('scripts/check_custom_basis_provider.py','scripts/check_extension_provider.py',
        'scripts/wheel_validation/provider_shards.py', 'tests/python/provider_evidence.py',
        'tests/python/test_vega_kt_custom_basis_provider.py','tests/python/test_vega_kt_extension_put_provider.py',
        'tests/python/test_vega_kt_extension_provider.py','tests/python/test_vega_kt_path_ledger.py',
        'tests/python/test_vega_kt_multi_step_provider.py','backtesting/tests/test_named_variance_basis.py'))
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    common=dict(document_kind='custom_basis_provider_validation',schema_version=1,source_sha256=frozen,
        native_binary_sha256=binary,reference_report_sha256=sha(args.reference_report),
        retained_numerical_source_files=len(retained),changed_boundary_files=sorted(CHANGED_BOUNDARY_FILES),
        full_market_iv_bucket_acceptance='not_established')
    if args.aggregate:
        reports=[json.loads(p.read_text()) for p in args.aggregate]
        if {(r['intervals'],r['basis']) for r in reports}!={(n,b) for n in (4,6) for b in POLICIES}:
            raise ValueError('all four unique interval/basis shards required')
        for r in reports:
            if r['status']!='passed' or any(r[k]!=v for k,v in common.items()):
                raise ValueError('aggregate requires passing shards with identical sources and reference')
        report=dict(common,tests=sum(r['tests'] for r in reports),failed=0,skipped=0,
                    evidence={k:sum((r['evidence'][k] for r in reports),[]) for k in ('panels','replays','snapshots','failures')},
                    numerical_comparisons=sum((r['numerical_comparisons'] for r in reports),[]),
                    shard_sha256={str(p):sha(p) for p in args.aggregate})
        report['evidence']['samples']={k:v for r in reports for k,v in r['evidence']['samples'].items()}
        expected_panels=40;expected_replays=64;expected_snapshots=96
    else:
        if args.intervals is None or args.basis is None: parser.error('--intervals and --basis required for a runtime shard')
        if args.recheck_report:
            report=json.loads(args.recheck_report.read_text())
            if any(report[k]!=v for k,v in common.items()) or report['intervals']!=args.intervals or report['basis']!=args.basis:
                raise ValueError('retained runtime source, binary, reference or scope changed')
            if report['failed'] or report['skipped'] or report['tests']!=5 or not report['source_unchanged']:
                raise ValueError('only passing unchanged runtime evidence can be rechecked')
        else:
            panel.CustomBasisProviderTests.intervals=(args.intervals,);panel.CustomBasisProviderTests.policies=(args.basis,)
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(panel.CustomBasisProviderTests))
            report=dict(common,intervals=args.intervals,basis=args.basis,tests=result.testsRun,
                failed=len(result.failures)+len(result.errors),skipped=len(result.skipped),
                evidence=panel.CustomBasisProviderTests.evidence,status='pending_reference',
                source_unchanged=all(sha(ROOT/n)==h for n,h in frozen.items()))
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.with_suffix('.runtime.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        comparisons=[]
        for p in report['evidence']['panels']:
            old=next(x for x in ref['observations'] if all(x[k]==p[k] for k in ('intervals','basis','product','seed')))
            case=next(x for x in ref['cases'] if all(x[k]==p[k] for k in ('intervals','basis')))
            a=p['audit'];m=a['market_gamma']
            comparisons.append({**{k:p[k] for k in ('intervals','basis','product','seed')},
                **compare_audit(m['gamma_audit'],old['audit']),
                'construction_exact':a['basis_construction']==case['basis_audit'],
                'tangent_exact':m['calibration_tangent']==case['calibration_tangent'] and m['tangent_audit']==case['tangent_audit']})
        report['numerical_comparisons']=comparisons
        expected_panels=10;expected_replays=16;expected_snapshots=32 if args.intervals==4 else 16
    report['source_unchanged']=all(sha(ROOT/n)==h for n,h in frozen.items())
    evidence=report['evidence'];comparisons=report['numerical_comparisons']
    panel_keys={(p['intervals'],p['basis'],p['product'],p['seed']) for p in comparisons}
    replay_keys={(p['intervals'],p['basis'],p['side'],p['kind'],p['strategy']) for p in evidence['replays']}
    passed=(not report['failed'] and not report['skipped'] and report['source_unchanged']
        and len(comparisons)==len(panel_keys)==expected_panels
        and len(evidence['replays'])==len(replay_keys)==expected_replays and len(evidence['snapshots'])==expected_snapshots
        and all(all(c[k] for k in ('numerical_audit_exact','gamma_bin_roundoff_passed','construction_exact','tangent_exact')) for c in comparisons))
    report['status']='passed' if passed else 'failed'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if passed:
        for key,sample in evidence['samples'].items():
            args.output.with_name(key+'-input.json').write_text(json.dumps(sample['request'],indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],tests=report['tests'],failed=report['failed'],panels=len(comparisons),
        replays=len(evidence['replays']),snapshots=len(evidence['snapshots']),source_unchanged=report['source_unchanged'])),flush=True)
    return int(not passed)


if __name__=='__main__': raise SystemExit(main())
