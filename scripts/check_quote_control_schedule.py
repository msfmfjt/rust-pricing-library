"""Check all dated input boundaries without claiming native numerical acceptance."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from continue_quote_control_replay import read_document, sha, verify_accepted_runtime, ACCEPTANCE
from check_quote_control_provider import provider
from check_portfolio_market import sources, write
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.domain import BacktestError, fingerprint, require
from rust_pricing_backtest.quote_control import virtual_request, CONTROL_ID


def run(input_path, output):
    require(not output.exists(), 'schedule_output', 'fresh report required')
    verify_accepted_runtime(); frozen = sources(); own = sha(__file__)
    args = load_input(json.dumps(read_document(input_path), allow_nan=False))
    p = provider(); rows = []; calls = 0
    def stop(*unused):
        nonlocal calls
        calls += 1
        raise BacktestError('schedule_preflight_stop', 'native numerical evaluation intentionally not run')
    # All ordinary adapter/date/sampling checks execute. The numerical boundary
    # is stopped before a plan or Gamma estimator can be constructed.
    with patch('rust_pricing_backtest.quote_control.evaluate', side_effect=stop), \
         patch.object(p.rp, 'PricingPlan', side_effect=AssertionError('native plan entered')):
        for snapshot in args['history']:
            failures = []; control = None
            for contract in args['contracts']:
                try:
                    candidate = virtual_request(json.loads(snapshot.pricing_requests[contract.contract_id]))
                    if control is None: control = candidate
                    require(candidate == control, 'quote_control_scope', 'inconsistent virtual control')
                    require(contract.contract_id != CONTROL_ID and len(args['contracts']) <= 31,
                            'quote_control_scope', 'invalid actual universe')
                    p.evaluate(snapshot, contract)
                    raise AssertionError('preflight unexpectedly released risk')
                except BacktestError as error:
                    if error.code != 'schedule_preflight_stop':
                        failures.append(dict(instrument=contract.contract_id, code=error.code, message=error.message))
            rows.append(dict(snapshot_id=snapshot.snapshot_id, date=snapshot.at.isoformat(),
                contracts=len(args['contracts']), failures=failures,
                control_strike=control['product']['strike'] if control else None))
    require(sources() == frozen and sha(__file__) == own, 'schedule_source', 'source changed during check')
    failed = sum(len(row['failures']) for row in rows)
    require(calls+failed == len(args['contracts'])*len(args['history']), 'schedule_coverage', 'missing boundary checks')
    result = dict(document_kind='quote_control_schedule_preflight', schema_version=1,
        status='failed' if failed else 'ready_for_numerical_validation', failed=failed, snapshots=rows,
        snapshot_count=len(rows), contract_snapshot_count=calls+failed, numerical_boundaries_reached=calls,
        numerical_acceptance='not_evaluated', native_evaluations=0, provider_promotion=False,
        annual_completed=False, source_sha256=frozen, checker_sha256=own,
        input_sha256=sha(input_path), acceptance_sha256=sha(ACCEPTANCE))
    result['report_hash'] = fingerprint(result); write(output, result)
    print(json.dumps({k:result[k] for k in ('status','failed','snapshot_count','contract_snapshot_count','native_evaluations')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); r = run(a.input, a.output); raise SystemExit(r['failed'] != 0)
