"""Recheck low-precision composed directions without relaxing acceptance gates."""
import argparse
import hashlib
import json
from pathlib import Path
import rust_pricing as rp
from multi_step_cases import ROOT, model


def run(reference):
    previous = json.loads(reference.read_text())
    if previous['status'] != 'passed':
        raise ValueError('passing covariance report required')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, digest in previous['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('covariance implementation changed: '+name)
    records = []
    for o in previous['observations']:
        before = o['direct_combinations']
        if before['estimate'] is not None:
            continue
        ctx = {k:o[k] for k in ['intervals','profile','product','seed']}
        # Double the same RNG prefix; halve the starting grid to stay within the
        # original resource cap. Both independent precision gates remain active.
        payload = model(o['intervals'], o['product'], samples=8192, seed=o['seed'])
        audit = rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(payload)),
            worker_threads=4, reduction_block_size=256).evaluate_vega_kt_multi_step_gamma_chain_audit(
                [r['variance_row'] for r in before['rows']], before['gamma_bin_boundaries'],
                variance_risk_directions=before['variance_risk_directions'], subdivisions=256,
                relative_tolerance=before['relative_tolerance'],
                gamma_relative_standard_error_limit=before['gamma_relative_standard_error_limit'])
        records.append(dict(**ctx, previous_status=before['status'], audit=audit))
        print(json.dumps(dict(**ctx, status=audit['status'],
            relative_standard_error=audit['direction_relative_standard_error'])), flush=True)
    for name, digest in previous['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('source changed during precision check: '+name)
    return dict(document_kind='path_gamma_composed_precision_validation', schema_version=1,
        status='passed' if all(r['audit']['estimate'] is not None for r in records) else 'failed',
        records=records, passed=sum(r['audit']['estimate'] is not None for r in records),
        failed=sum(r['audit']['estimate'] is None for r in records),
        independent_sampling_units=8192, subdivisions=256,
        relative_tolerance=1e-6, gamma_relative_standard_error_limit=.25,
        covariance_report_sha256=sha(reference),
        source_sha256={**previous['source_sha256'], str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))},
        native_binary_sha256=sha(Path(getattr(rp,'rust_pricing',rp).__file__)),
        full_market_iv_bucket_acceptance='not_established')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--covariance-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=run(a.covariance_report)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','passed','failed']}))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
