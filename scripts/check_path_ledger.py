"""Accept native path histories, conditional Gamma, settlement and resume."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'tests/python'))
    import test_vega_kt_path_ledger as panel
    import rust_pricing as rp
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=list((ROOT/'backtesting/rust_pricing_backtest').glob('*.py'))+list((ROOT/'backtesting/rust_pricing_backtest/schemas').glob('*.json'))
    paths += [Path(__file__),ROOT/'scripts/generate_backtest_schemas.py',ROOT/'tests/python/test_vega_kt_path_ledger.py',
        ROOT/'backtesting/tests/test_path_history.py',ROOT/'backtesting/tests/test_validation.py',
        ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json',ROOT/'scripts/multi_step_cases.py']
    previous=json.loads((ROOT/'design/validation/local-vol-conditional-ad-v0.1.json').read_text())
    for name,digest in previous['source_sha256'].items():
        if name.startswith('crates/'):
            if sha(ROOT/name)!=digest:raise ValueError('native Gamma source changed: '+name)
            paths.append(ROOT/name)
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(panel.PathLedgerTests))
    unchanged=all(sha(ROOT/n)==h for n,h in frozen.items())
    report=dict(document_kind='path_ledger_validation',schema_version=1,
        status='passed' if result.wasSuccessful() and unchanged else 'failed',
        tests=result.testsRun,failed=len(result.failures)+len(result.errors),skipped=len(result.skipped),
        source_unchanged=unchanged,source_sha256=frozen,
        native_binary_sha256=sha(Path(getattr(rp,'rust_pricing',rp).__file__)),
        previous_native_acceptance_sha256=sha(ROOT/'design/validation/local-vol-conditional-ad-v0.1.json'),
        evidence=panel.PathLedgerTests.evidence,full_market_iv_bucket_acceptance='not_established')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if report['status']=='passed':args.output.with_name('input.json').write_text(json.dumps(report['evidence']['sample_request'],indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','tests','failed','source_unchanged']}))
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
