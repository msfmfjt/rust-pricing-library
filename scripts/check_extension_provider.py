"""Accept explicit variance extensions at the immutable backtest boundary."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]


def numerical(value):
    if isinstance(value,dict):
        return {k:numerical(v) for k,v in value.items() if k not in ('request_fingerprint','plan_fingerprint')}
    if isinstance(value,list):return [numerical(v) for v in value]
    return value


def compare_audit(actual, reference):
    # The provider constructs spot * 1.1; the reference policy spells 110.
    # Keep every calculated quantity exact, and report this input-label rounding
    # separately instead of applying an allclose tolerance to Gamma or covariance.
    a,b=deepcopy(actual),deepcopy(reference)
    ab,bb=a.pop('gamma_bin_boundaries'),b.pop('gamma_bin_boundaries')
    if len(ab)!=len(bb) or not all(math.isfinite(x) for x in [*ab,*bb]):
        raise ValueError('invalid Gamma bin comparison')
    ulps=max(abs(x-y)/max(math.ulp(x),math.ulp(y)) for x,y in zip(ab,bb))
    return dict(numerical_audit_exact=numerical(a)==numerical(b),
        gamma_bin_maximum_ulp_difference=ulps,gamma_bin_roundoff_passed=ulps<=1,
        provider_gamma_bins=ab,reference_gamma_bins=bb)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-report',type=Path,required=True)
    parser.add_argument('--recheck-report',type=Path,
        help='recheck retained passing tests after a checker-only correction; all other frozen sources must match')
    args=parser.parse_args();sys.path.insert(0,str(ROOT/'tests/python'))
    import test_vega_kt_extension_provider as panel
    import rust_pricing as rp
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    reference=json.loads(args.reference_report.read_text())
    if reference['status']!='passed':raise ValueError('prior extension price/Gamma acceptance is required')
    numerical_names=['backtesting/rust_pricing_backtest/'+n for n in
        ['multi_step_extension.py','multi_step_smile.py','discrete_smile.py','domain.py']]
    numerical_names += [n for n in reference['source_sha256'] if n.startswith('crates/')]
    numerical_names += ['scripts/extension_reference.py','scripts/multi_step_cases.py',
        'fixtures/local-vol/multi-step-gamma-v0.1.json']
    for n in numerical_names:
        if sha(ROOT/n)!=reference['source_sha256'][n]:raise ValueError('accepted numerical implementation changed: '+n)
    binary=sha(Path(getattr(rp,'rust_pricing',rp).__file__))
    if binary!=reference['native_binary_sha256']:raise ValueError('reference must use this native binary')
    paths=list((ROOT/'backtesting/rust_pricing_backtest').glob('*.py'))
    paths+=list((ROOT/'backtesting/rust_pricing_backtest/schemas').glob('*.json'))
    paths+=[ROOT/n for n in numerical_names]
    paths+=[Path(__file__),Path(panel.__file__),ROOT/'tests/python/test_vega_kt_path_ledger.py',
            ROOT/'scripts/wheel_validation/provider_shards.py',
            ROOT/'tests/python/test_vega_kt_multi_step_provider.py']
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    retained=None
    if args.recheck_report:
        previous=json.loads(args.recheck_report.read_text())
        if (previous['document_kind']!='extension_provider_validation' or previous['tests']!=6 or previous['failed']!=0
                or previous['skipped']!=0 or not previous['source_unchanged'] or previous['native_binary_sha256']!=binary
                or previous['reference_report_sha256']!=sha(args.reference_report)):
            raise ValueError('only passing, unchanged native/test evidence may be rechecked')
        for name,digest in previous['source_sha256'].items():
            if name!=str(Path(__file__).relative_to(ROOT)) and sha(ROOT/name)!=digest:
                raise ValueError('retained test or production source changed: '+name)
        evidence=previous['evidence'];tests,failed,skipped=previous['tests'],0,0
        retained=dict(report_sha256=sha(args.recheck_report),reason='checker-only correction: separate one-ULP bin input labels from exact calculated quantities')
    else:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(panel.ExtensionProviderTests))
        evidence=panel.ExtensionProviderTests.evidence
        tests,failed,skipped=result.testsRun,len(result.failures)+len(result.errors),len(result.skipped)
    comparisons=[]
    for actual in evidence['panels']:
        old=next(o for o in reference['observations'] if o['intervals']==actual['intervals'] and o['seed']==actual['seed']
            and o['extension']==actual['extension'] and o['product']==actual['kind'])
        comparisons.append(dict(intervals=actual['intervals'],seed=actual['seed'],extension=actual['extension'],kind=actual['kind'],
            **compare_audit(actual['audit']['market_gamma']['gamma_audit'],old['audit'])))
    unchanged=all(sha(ROOT/n)==h for n,h in frozen.items())
    passed=failed==0 and skipped==0 and tests==6 and unchanged and len(comparisons)==16 and len(evidence['replays'])==32 and all(
        c['numerical_audit_exact'] and c['gamma_bin_roundoff_passed'] for c in comparisons)
    report=dict(document_kind='extension_provider_validation',schema_version=1,status='passed' if passed else 'failed',
        tests=tests,failed=failed,skipped=skipped,retained_test_run=retained,
        source_unchanged=unchanged,source_sha256=frozen,native_binary_sha256=binary,
        retained_numerical_source_files=len(set(numerical_names)),reference_report_sha256=sha(args.reference_report),
        numerical_comparisons=comparisons,comparison_excluded_fields=['request_fingerprint','plan_fingerprint'],
        gamma_bin_comparison='at_most_one_ulp_in_input_labels_only',
        comparison_reason='ordinary risk request disables the older VegaKT allocation; calculated audit quantities remain exact; bin input labels have a separate one-ULP gate',
        evidence=evidence,full_market_iv_bucket_acceptance='not_established')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if passed:
        for extension,sample in report['evidence']['samples'].items():
            args.output.with_name(extension+'-input.json').write_text(json.dumps(sample['request'],indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],tests=tests,failed=report['failed'],
        panels=len(comparisons),replays=len(report['evidence']['replays']),source_unchanged=unchanged)),flush=True)
    return int(not passed)


if __name__=='__main__':raise SystemExit(main())
