"""Seal joint-replay integration evidence without relabelling old numerical runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from check_portfolio_market import sources
from rust_pricing_backtest.domain import fingerprint, require
from rust_pricing_backtest.portfolio_market_risk import implementation_identity
from variance_basis_runtime import AUXILIARY, RUNTIME


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(directory, archive):
    prior_path = ROOT/'design/validation/vega-kt-portfolio-market-v0.1.json'
    prior = json.loads(prior_path.read_text())
    require(prior['report_hash'] == fingerprint({k:v for k,v in prior.items() if k!='report_hash'})
            and prior['status']=='passed' and prior['failed']==0, 'joint_reference', 'sealed passing prior acceptance required')
    current = sources()
    changed = sorted(n for n,digest in prior['numerical_source_sha256'].items() if sha(ROOT/n)!=digest)
    allowed = ['domain.py','engine.py','ensemble.py','portfolio_market_risk.py','rust_adapter.py','serialization.py','validation.py']
    require(changed == sorted('backtesting/rust_pricing_backtest/'+n for n in allowed),
            'joint_reference', 'unexpected changes to prior numerical dependencies')
    added = set(current)-set(prior['numerical_source_sha256'])
    require(added == {'backtesting/rust_pricing_backtest/joint_replay.py'}, 'joint_reference', 'unexpected added numerical dependency')
    # Only fixed-strike axis alignment changed in the portfolio numerical module.
    # The no-axis numerical matrix and all Gamma/price/calibration code are exact.
    name = 'backtesting/rust_pricing_backtest/portfolio_market_risk.py'
    with tarfile.open(archive) as packed:
        old = packed.extractfile(name).read()
    require(hashlib.sha256(old).hexdigest()==prior['source_sha256'][name], 'joint_reference', 'prior source archive differs')
    expected = old.decode().replace(
        "np.array(b['log_forward_moneyness_nodes'])+math.log(spot)-math.log(forward[i])",
        "np.array(b['log_forward_moneyness_nodes'])+(math.log(spot)-math.log(forward[i]))")
    require(expected==(ROOT/name).read_text(), 'joint_reference', 'unexpected portfolio numerical change')
    for name in AUXILIARY:
        require(sha(ROOT/name)==prior['source_sha256'][name], 'joint_reference', 'independent numerical runner changed')
    for name,digest in prior['shard_sha256'].items():
        require(sha(ROOT/name)==digest, 'joint_reference', 'retained prior numerical shard changed')
    historical_schemas = {n:digest for n,digest in prior['source_sha256'].items() if '/schemas/' in n}
    require(len(historical_schemas)==21 and all(sha(ROOT/n)==h for n,h in historical_schemas.items()),
            'joint_reference', 'historical wire schemas changed')
    import rust_pricing as rp
    binary = sha(Path(getattr(rp,'rust_pricing',rp).__file__))
    require(binary==prior['native_binary_sha256'], 'joint_reference', 'accepted native binary changed')
    for name,count in [('companion-tests.log',283),('native-tests.log',1),('native-portfolio-tests.log',5)]:
        text = (directory/name).read_text()
        require(f'Ran {count} test' in text and text.rstrip().endswith('OK'), 'joint_tests', name)
    for name in ('native-validation.json','wheel-contents.json','wheel-cli-validation.json'):
        require(json.loads((directory/name).read_text())['status']=='passed', 'joint_tests', name)
    api = json.loads((directory/'api/result.json').read_text())
    require(api['manifest']['implementation_hash']==fingerprint({p.name:p.read_text() for p in sorted((ROOT/'backtesting/rust_pricing_backtest').glob('*.py'))}),
            'joint_tests', 'integration evidence belongs to an earlier implementation')
    for row in api['risks']:
        report = row['joint_market']['report']
        require(report is None or report['implementation_identity']==implementation_identity(), 'joint_tests', 'stale joint evidence')
    extra = [ROOT/n for n in (*AUXILIARY,*RUNTIME,'scripts/generate_backtest_schemas.py',
        'scripts/check_joint_replay_compatibility.py','scripts/check_source_archive.py',
        'backtesting/tests/test_joint_replay.py','backtesting/tests/test_validation.py',
        'tests/python/test_vega_kt_joint_replay.py','tests/python/test_vega_kt_joint_portfolio.py',
        '.github/workflows/backtesting.yml','backtesting/pyproject.toml')]
    extra += list((ROOT/'backtesting/rust_pricing_backtest/schemas').glob('*.json'))
    current.update({str(p.relative_to(ROOT)):sha(p) for p in extra})
    paths = [p for p in directory.rglob('*') if p.is_file() and (p.suffix in ('.json','.log','.whl'))
             and p.name not in ('completion-checks.json','runtime-tests.log','preflight-4.json','preflight-6.json')]
    report = dict(document_kind='joint_market_replay_acceptance',schema_version=1,date='2026-10-13',status='passed',failed=0,
        native_binary_sha256=binary,source_sha256=current,implementation_identity=implementation_identity(),
        prior_acceptance_sha256=sha(prior_path),prior_source_archive_sha256=sha(archive),
        prior_numerical_checks_retained=4038,prior_numerical_checks_rerun=False,
        prior_numerical_shards_unchanged=True,native_rust_source_files_unchanged=sum(n.startswith('crates/') for n in current),
        numerical_dependency_changes=changed,portfolio_change='fixed_strike_alignment_log_difference_parenthesization_only',
        historical_schema_files_unchanged=21,new_schema_files=5,
        tests=dict(companion_passed=283,joint_replay_native_passed=1,portfolio_native_passed=5),
        integration=dict(legacy_ledger_orders_fills_exact=True,partial_resume_byte_identical=True,
            packaged_partial_cli_resume=True,packaged_complete_cli_resume_byte_identical=True,
            worker_counts_compared=[1,4],explained_intervals=1,expiry_interval='explicitly_unavailable'),
        full_market_iv_bucket_acceptance='not_established',
        limitations=['Explicit existing four/six-interval extension and named-basis providers only',
            'Joint covariance is estimation error, not market-return covariance',
            'Decision Delta remains frozen-grid; market Delta correction applies to attribution',
            'Delta sampling error, Gamma, Theta and higher-order attribution remain outside this feature',
            'No new-market, year-long performance or historical hedge-effectiveness claim'],
        evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    report['report_hash']=fingerprint(report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--prior-source-archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=check(args.directory.resolve(),args.prior_source_archive.resolve())
    protected=[ROOT/n for n in (*report['source_sha256'],*report['evidence_sha256'])]
    protected += [ROOT/'design/validation/vega-kt-portfolio-market-v0.1.json',args.prior_source_archive]
    require(all(args.output.resolve()!=p.resolve() and not (args.output.exists() and p.exists() and args.output.samefile(p)) for p in protected),
            'joint_output', 'output must not overwrite evidence or implementation')
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],failed=report['failed'],tests=report['tests'],native_unchanged=True)))


if __name__=='__main__':main()
