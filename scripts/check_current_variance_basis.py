"""Validate a raw matrix on the current accepted runtime without promoting it to the ledger."""
import argparse
import json
from pathlib import Path
from variance_basis_runtime import (
    ROOT, ACCEPTANCE, current_provenance, guard_destinations, isolated_runner,
    verify_resume, write_report,
)
from raw_variance_basis import load_matrix
from multi_step_cases import models
from rust_pricing_backtest.domain import BacktestError, require
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator


def preflight(intervals, basis_path, provenance):
    base = next(c['request'] for c in models() if c['intervals'] == intervals)
    basis, audit = load_matrix(basis_path, base)
    _, tangent_audit = MultiStepExtensionCalibrator(base, extension='custom', variance_basis=basis).all_maturity_iv_tangent()
    return dict(document_kind='raw_variance_basis_preflight', schema_version=1,
                status='ready_for_validation', phase='preflight', numerical_acceptance='not_evaluated',
                intervals=intervals, basis_input=audit, tangent_audit=tangent_audit,
                source_sha256=provenance[0], native_binary_sha256=provenance[1],
                prior_acceptance_sha256=provenance[2], acceptance_record=ACCEPTANCE,
                diagnostic_only=True, provider_promotion=False, full_market_iv_bucket_acceptance='not_established')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals', type=int, choices=(4, 6), required=True)
    parser.add_argument('--basis-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='check binding, runtime and quote span; does not establish numerical acceptance')
    mode.add_argument('--resume', type=Path)
    args = parser.parse_args(argv); safe_output = False
    try:
        guard_destinations(args.output, args.basis_file, args.resume); safe_output = True
        provenance = current_provenance()
        if args.preflight:
            report = preflight(args.intervals, args.basis_file, provenance)
        else:
            if args.resume: verify_resume(args.resume)
            runner = isolated_runner(provenance)
            report = runner.run(args.intervals, args.basis_file, args.output, resume=args.resume)
        require(current_provenance() == provenance, 'basis_source_changed', 'accepted runtime changed during validation')
    except (BacktestError, OSError, KeyError, TypeError) as error:
        code = error.code if isinstance(error, BacktestError) else 'basis_runtime_input'
        report = dict(document_kind='raw_variance_basis_validation', schema_version=1, status='rejected',
                      error=dict(code=code, message=str(error)), diagnostic_only=True, provider_promotion=False,
                      full_market_iv_bucket_acceptance='not_established')
    if safe_output: write_report(args.output, report)
    print(json.dumps(dict(status=report['status'], panels=len(report.get('observations', [])),
                          checks=len(report.get('checks', [])), error=report.get('error'))), flush=True)
    return int(report['status'] not in ('passed', 'ready_for_validation'))


if __name__ == '__main__': raise SystemExit(main())
