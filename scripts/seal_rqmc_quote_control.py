"""Seal independent fixed-quote control evidence and its precise release boundary."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed,sha
from check_portfolio_market import sources,write
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.portfolio_market_risk import implementation_identity


def run(directory,output):
    directory=directory.resolve()
    require(not output.exists(),'control_seal_output','fresh output required')
    prior_path=ROOT/'design/validation/vega-kt-rqmc-provider-v0.1.json';prior=sealed(prior_path)
    identity=implementation_identity()
    require(prior['status']=='passed' and prior['failed']==0 and prior['implementation_identity']==identity
        and all(sha(ROOT/n)==v for n,v in prior['source_sha256'].items()),
        'control_seal_source','production and previous accepted dependencies must be unchanged')
    policy_path=ROOT/'fixtures/backtesting/rqmc-quote-control-policy-v0.1.json';policy=json.loads(policy_path.read_text())
    freeze=sealed(directory/'quote-control-freeze.json')
    require(all(sha(path)==digest for path,digest in freeze['frozen_sha256'].items()),
        'control_seal_freeze','unobserved-seed design must stay frozen')
    require('Ran 4 tests' in (directory/'control-tests.log').read_text()
        and (directory/'control-tests.log').read_text().rstrip().endswith('OK'),
        'control_seal_tests','analytical covariance and fail-closed tests required')
    refined=sealed(directory/'oracle-refined.json')
    require(refined['status']=='passed' and len(refined['checks'])==601 and refined['precision']['sobol_power']==18
        and refined['refinement_driver_sha256']==sha(ROOT/'scripts/check_rqmc_precision_oracle.py'),
        'control_seal_oracle','independent fourfold price-reference refinement required')
    summaries=[]
    for seed in policy['holdout_seeds']:
        raw_path=directory/f'holdout-{seed}.json';raw=sealed(raw_path)
        path=directory/f'control-{seed}.json';r=sealed(path)
        coarse=sealed(directory/f'control-coarse-{seed}.json')
        require(coarse['status']=='passed' and coarse['failed']==0 and len(coarse['checks'])==1023
            and coarse['native_report']['report_hash']==r['native_report']['report_hash']
            and all(sha(n)==v for n,v in coarse['dependencies_sha256'].items())
            and str((directory/'oracle-refined.json').resolve()) in r['dependencies_sha256'],
            'control_seal_oracle','both price-reference resolutions must pass on the same controlled estimates')
        require(r['status']=='passed' and r['failed']==0 and len(r['checks'])==1023
            and r['policy']==policy and r['seed']==seed and r['instruments']==30 and r['universe']=='full'
            and r['source_sha256']==sources() and r['implementation_identity']==identity
            and all(sha(n)==v for n,v in r['dependencies_sha256'].items()),
            'control_seal_numerics','complete independent control checks on unchanged inputs required')
        require(r['native_report']['control_variate']['source_native_report_hash']==raw['native_report']['report_hash'],
            'control_seal_lineage','raw native parent differs')
        decision=sealed(directory/f'control-decision-{seed}/report.json')
        require(decision['status']=='passed' and decision['panel_sha256']==sha(path) and decision['implementation_identity']==identity
            and decision['checker_sha256']==sha(ROOT/'scripts/check_rqmc_precision_decision.py')
            and decision['processed_snapshots']==decision['orders']==1,
            'control_seal_decision','actual one-day solver and lot-rounded decision required')
        require(decision['schemas']=={'input.json':'valid','result.json':'valid','checkpoint.json':'valid'}
            and decision['stale_snapshot_rejection']==dict(code='precision_replay',checkpoint_unchanged=True,orders_unchanged=True,ledger_unchanged=True),
            'control_seal_replay','saved schemas and stale-date rollback required')
        for key in ('held','proposed'):
            require(decision[key]['status']=='passed' and decision[key]['limit']==10
                and decision[key]['max_bucket_standard_error']<=10,
                'control_seal_gate','unchanged held and proposed precision gates required')
        summaries.append(dict(seed=seed,native_instruments=30,independent_checks=len(r['checks'])+len(coarse['checks']),failed=0,
            raw_status=raw['status'],raw_standard_error=raw['max_bucket_standard_error'],
            controlled_held_standard_error=decision['held']['max_bucket_standard_error'],
            controlled_proposed_standard_error=decision['proposed']['max_bucket_standard_error'],
            solver_iterations=decision['solver']['iterations'],projected_gradient=decision['solver']['projected_gradient'],
            raw_panel_seconds=raw['elapsed_seconds']))
    added=[Path(__file__),ROOT/'scripts/market_quote_control.py',ROOT/'scripts/test_market_quote_control.py',
        ROOT/'scripts/check_rqmc_quote_control.py',ROOT/'scripts/check_rqmc_precision_decision.py',
        ROOT/'scripts/probe_rqmc_portfolio_precision.py',ROOT/'scripts/check_rqmc_precision_budget.py',
        ROOT/'scripts/check_rqmc_precision_oracle.py',
        ROOT/'scripts/seal_rqmc_portfolio_precision.py',policy_path,
        *ROOT.glob('fixtures/backtesting/rqmc-portfolio-precision-*.json')]
    evidence={str(p.relative_to(ROOT)):sha(p) for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.log')}
    r=dict(document_kind='rqmc_market_quote_control_candidate',schema_version=1,date='2026-10-13',status='passed',failed=0,
        scope='first annual snapshot only; all_rows, thirty European calls, two independent unobserved seed panels',
        implementation_identity=identity,native_binary_sha256=prior['native_binary_sha256'],
        prior_provider_acceptance_sha256=sha(prior_path),production_unchanged=True,
        current_provider_policy_promoted=False,annual_completed=False,
        source_sha256=dict(sources(),**{str(p.relative_to(ROOT)):sha(p) for p in added}),
        evidence_sha256=evidence,policy=policy,holdouts=summaries,tests=4,
        frozen_control_before_observing_holdout_results=True,historical_failures_preserved=True,
        limitations=['This is an explicit diagnostic estimator candidate, not a promoted RustPricingProvider policy.',
            'Sealed-panel replay exercises the existing decision engine, solver, lot rounding and joint gates for one day only.',
            'The control must exactly match a calibrated quote. Subsequent annual dates may need a virtual control instrument.',
            'No path-product, subsequent-date, settlement, full annual or throughput acceptance is inferred.',
            'Independent units are twelve scramble means; SE excludes calibration and discretization uncertainty.'])
    r['report_hash']=fingerprint(r);write(output,r);print(json.dumps(dict(status=r['status'],holdouts=summaries)))
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.directory,a.output)
