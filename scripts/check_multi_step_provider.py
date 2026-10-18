"""Validate the bounded multi-step Gamma provider and retain ledger evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'tests/python'))
    import test_vega_kt_multi_step_provider as panel
    import rust_pricing as rp
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    paths = list((ROOT/'backtesting/rust_pricing_backtest').glob('*.py')) + [
        Path(__file__), ROOT/'tests/python/test_vega_kt_multi_step_provider.py',
        ROOT/'scripts/multi_step_cases.py', ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json']
    # Bind the native sources too, so an old binary is never implicit evidence of
    # source acceptance. The binary hash is recorded separately below.
    previous = json.loads((ROOT/'design/validation/local-vol-conditional-ad-v0.1.json').read_text())
    for name, digest in previous['source_sha256'].items():
        if name.startswith('crates/'):
            if sha(ROOT/name) != digest:
                raise ValueError('native source no longer matches accepted Gamma record: '+name)
            paths.append(ROOT/name)
    frozen = {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    binary = sha(Path(getattr(rp,'rust_pricing',rp).__file__))
    if binary != previous['installed_runtime']['native_binary_sha256']:
        # CI builds its own platform binary; its hash need not match macOS.
        # The recorded build/runtime checks establish the platform correspondence.
        platform_rebuild = True
    else:
        platform_rebuild = False
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(panel.MultiStepProviderTests))
    unchanged = all(sha(ROOT/name)==digest for name,digest in frozen.items())
    report = dict(document_kind='multi_step_gamma_provider_validation', schema_version=1,
        status='passed' if result.wasSuccessful() and unchanged else 'failed',
        tests=result.testsRun, failed=len(result.failures)+len(result.errors),
        skipped=len(result.skipped), source_unchanged=unchanged, source_sha256=frozen,
        native_binary_sha256=binary, platform_binary_differs_from_prior_record=platform_rebuild,
        prior_numerical_record_sha256=sha(ROOT/'design/validation/local-vol-conditional-ad-v0.1.json'),
        evidence=panel.MultiStepProviderTests.evidence,
        full_market_iv_bucket_acceptance='not_established')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if report['status'] == 'passed':
        args.output.with_name('input.json').write_text(json.dumps(
            report['evidence']['sample_request'],indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','tests','failed','skipped','source_unchanged']}))
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
