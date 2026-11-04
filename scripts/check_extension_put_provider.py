"""Accept Put extension ledgers against independently validated native panels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest
from check_extension_provider import compare_audit
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference-report',type=Path,required=True)
    parser.add_argument('--recheck-report',type=Path)
    args=parser.parse_args();sys.path.insert(0,str(ROOT/'tests/python'))
    import test_vega_kt_extension_put_provider as panel
    import rust_pricing as rp
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=list((ROOT/'crates').rglob('*.rs'))+list((ROOT/'backtesting/rust_pricing_backtest').glob('*.py'))
    paths+=list((ROOT/'backtesting/rust_pricing_backtest/schemas').glob('*.json'))
    paths += [Path(__file__),Path(panel.__file__),ROOT/'scripts/extension_put_reference.py',ROOT/'scripts/multi_step_cases.py',
        ROOT/'scripts/check_extension_provider.py',ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json',
        ROOT/'scripts/wheel_validation/provider_shards.py',ROOT/'tests/python/provider_evidence.py',
        ROOT/'tests/python/test_vega_kt_path_ledger.py',ROOT/'tests/python/test_vega_kt_extension_provider.py',
        ROOT/'tests/python/test_vega_kt_multi_step_provider.py']
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    binary=sha(Path(getattr(rp,'rust_pricing',rp).__file__))
    if args.recheck_report:
        report=json.loads(args.recheck_report.read_text())
        if report['tests']!=5 or report['failed'] or report['skipped'] or not report['source_unchanged']:
            raise ValueError('only passing frozen runtime tests may be rechecked')
        if report['source_sha256']!=frozen or report['native_binary_sha256']!=binary:
            raise ValueError('runtime test/production sources or binary changed')
        report['retained_runtime_sha256']=sha(args.recheck_report)
    else:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(panel.PutExtensionProviderTests))
        unchanged=all(sha(ROOT/n)==h for n,h in frozen.items())
        report=dict(document_kind='extension_put_provider_validation',schema_version=1,status='pending_reference',
            tests=result.testsRun,failed=len(result.failures)+len(result.errors),skipped=len(result.skipped),source_unchanged=unchanged,
            source_sha256=frozen,native_binary_sha256=binary,evidence=panel.PutExtensionProviderTests.evidence,
            full_market_iv_bucket_acceptance='not_established')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_suffix('.runtime.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not args.reference_report.exists():raise ValueError('numerical reference is not finished; recheck the retained runtime report after it completes')
    reference=json.loads(args.reference_report.read_text())
    if reference['status']!='passed' or reference['native_binary_sha256']!=binary:
        raise ValueError('passing Put numerical reference from the same binary required')
    for n,h in reference['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('accepted Put numerical source changed: '+n)
    comparisons=[]
    for a in report['evidence']['panels']:
        b=next(x for x in reference['observations'] if all(x[k]==a[k] for k in ['intervals','extension','product','seed']))
        comparisons.append(dict(intervals=a['intervals'],extension=a['extension'],product=a['product'],
            **compare_audit(a['audit']['market_gamma']['gamma_audit'],b['audit'])))
    passed=report['failed']==0 and report['skipped']==0 and report['source_unchanged'] and len(comparisons)==24 and len(report['evidence']['replays'])==32
    passed=passed and all(x['numerical_audit_exact'] and x['gamma_bin_roundoff_passed'] for x in comparisons)
    report.update(status='passed' if passed else 'failed',reference_report_sha256=sha(args.reference_report),numerical_comparisons=comparisons)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if passed:
        for extension,sample in report['evidence']['samples'].items():
            args.output.with_name(extension+'-input.json').write_text(json.dumps(sample['request'],indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],tests=report['tests'],failed=report['failed'],panels=len(comparisons),
        replays=len(report['evidence']['replays']),source_unchanged=report['source_unchanged'])),flush=True)
    return int(not passed)


if __name__=='__main__':raise SystemExit(main())
