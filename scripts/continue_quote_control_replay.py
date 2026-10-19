"""Continue one accepted quote-control replay day and commit a compressed bundle.

This orchestration does not change the provider or import retained native audits.
The output is a fresh directory; a generation is published only after every
compressed document is read back, checked and flushed to durable storage.
"""
import argparse
from dataclasses import fields
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed
from check_portfolio_market import sources, write
from check_quote_control_provider import provider, prefill, RetainedProvider
from rust_pricing_backtest import run_backtest, BacktestError
from rust_pricing_backtest.domain import data, fingerprint, require
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.validation import validate_document
from rust_pricing_backtest.portfolio_market_risk import implementation_identity

ACCEPTANCE = ROOT/'design/validation/vega-kt-quote-control-provider-v0.1.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_document(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'continuation_json', 'duplicate JSON key')
            result[key] = value
        return result
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique,
                         parse_constant=lambda _: require(False, 'continuation_json', 'nonfinite JSON number'))


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def compressed_document(path, value, *, schema=True):
    """Never expose a final filename until compression/readback/schema succeeds."""
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'continuation_output', 'output already exists: '+str(path))
    temporary = path.with_name(path.name+'.tmp')
    require(not temporary.exists() and not temporary.is_symlink(), 'continuation_output', 'temporary output already exists')
    digest = hashlib.sha256()
    created = False
    try:
        # Canonical JSON is byte-equivalent to the existing result writer, but
        # its entire encoded string is never held in memory at once.
        encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
        with temporary.open('xb') as raw:
            created = True
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as zipped:
                chunks = []; size = 0
                for chunk in encoder.iterencode(value):
                    payload = chunk.encode('utf-8'); chunks.append(payload); size += len(payload)
                    if size >= 1024*1024:
                        block = b''.join(chunks); digest.update(block); zipped.write(block)
                        chunks = []; size = 0
                block = b''.join(chunks)+b'\n'; digest.update(block); zipped.write(block)
            raw.flush(); os.fsync(raw.fileno())
        with gzip.open(temporary, 'rb') as stream:
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == digest.hexdigest(),
                    'continuation_storage', 'compressed bytes differ')
        # temporary has .tmp suffix, so explicitly use gzip for this readback.
        with gzip.open(temporary, 'rt', encoding='utf-8') as stream:
            restored = json.load(stream)
        require(restored == value, 'continuation_storage', 'compressed content differs')
        if schema:
            require(validate_document(restored)['status'] == 'valid', 'continuation_schema', str(path))
        del restored
        temporary.replace(path); sync_directory(path.parent)
        return dict(sha256=sha(path), uncompressed_sha256=digest.hexdigest(), bytes=path.stat().st_size,
                    schema='valid' if schema else 'not_applicable')
    finally:
        if created and temporary.exists():
            temporary.unlink()


def commit_bundle(directory, documents, report):
    """An interrupted save cannot be mistaken for a published replay generation."""
    directory = Path(directory)
    pending = directory.with_name(directory.name+'.pending')
    require(not directory.exists() and not pending.exists() and not directory.is_symlink()
            and not pending.is_symlink(), 'continuation_output', 'fresh generation required')
    pending.mkdir()
    files = {}
    for name, value in documents.items():
        require(Path(name).name == name and name.endswith('.json.gz'), 'continuation_output', 'invalid bundle filename')
        files[name] = compressed_document(pending/name, value)
    committed = dict(report, files=files)
    committed['report_hash'] = fingerprint(committed)
    marker = pending/'report.json'
    with marker.open('x', encoding='utf-8') as stream:
        json.dump(committed, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    sync_directory(pending)
    pending.rename(directory); sync_directory(directory.parent)
    return committed


def verify_accepted_runtime():
    accepted = sealed(ACCEPTANCE)
    require(accepted['status'] == 'passed' and accepted['failed'] == 0, 'continuation_source', 'passing acceptance required')
    for name, expected in accepted['source_sha256'].items():
        require(sha(ROOT/name) == expected, 'continuation_source', 'accepted source changed: '+name)
    import rust_pricing as rp
    require(sha(getattr(rp, 'rust_pricing', rp).__file__) == accepted['native_binary_sha256']
            and implementation_identity() == accepted['implementation_identity'],
            'continuation_source', 'accepted native binary/provider required')
    return accepted


def preflight(args, checkpoint):
    """Use real engine validation and rollback before any native evaluation."""
    p = provider()
    class NoPricing:
        identity = p.identity
        validation = p.validation
        def evaluate(self, *unused):
            raise BacktestError('continuation_preflight', 'native evaluation intentionally deferred')
        def joint_market_risk(self, *unused):
            raise AssertionError('preflight reached joint evaluation')
    result = run_backtest(**args, provider=NoPricing(), checkpoint=checkpoint,
                          stop_after=checkpoint['state']['next_index']+1, on_error='return')
    require(result.status == 'failed' and result.failure['code'] == 'continuation_preflight'
            and result.checkpoint == checkpoint, 'continuation_preflight', 'input/checkpoint not ready for unchanged continuation')


def run(input_path, checkpoint_path, output):
    input_path, checkpoint_path, output = (Path(p).resolve() for p in (input_path, checkpoint_path, output))
    require(not output.exists(), 'continuation_output', 'fresh output directory required')
    accepted = verify_accepted_runtime()
    frozen = sources(); own = sha(__file__); acceptance_hash = sha(ACCEPTANCE)
    original_hashes = {str(p):sha(p) for p in (input_path, checkpoint_path)}
    input_doc, checkpoint = read_document(input_path), read_document(checkpoint_path)
    for document in (input_doc, checkpoint):
        require(validate_document(document)['status'] == 'valid', 'continuation_schema', 'invalid input/checkpoint')
    args = load_input(json.dumps(input_doc, allow_nan=False))
    day = checkpoint['state']['next_index']
    require(0 < day < len(args['history']), 'continuation_range', 'checkpoint must have a next snapshot')
    preflight(args, checkpoint)
    output.mkdir(parents=True)
    start = time.perf_counter()
    p = RetainedProvider(provider(), output/'events.jsonl')
    require(not p.inner._quote_control_cache, 'continuation_cache', 'empty native cache required')
    prefill(p.inner, args['history'][day], args['contracts'], output/'prefill.json')
    prefilled = time.perf_counter()
    result = run_backtest(**args, provider=p, checkpoint=checkpoint, stop_after=day+1, on_error='return')
    evaluated = time.perf_counter()
    if result.status == 'failed':
        require(result.checkpoint == checkpoint, 'continuation_rollback', 'failed day changed committed checkpoint')
    else:
        require(result.checkpoint['state']['next_index'] == day+1 and len(result.ledger) == day+1
                and list(result.ledger[:day]) == checkpoint['state']['ledger'],
                'continuation_prefix', 'prior ledger changed or next day missing')
        for name in ('risks', 'decisions', 'fills', 'cashflows', 'settlements', 'pnl_components', 'cost_basis', 'greek_pnl', 'iv_alignment'):
            old = checkpoint['state'][name]
            require(list(getattr(result, name)[:len(old)]) == old, 'continuation_prefix', 'committed table changed: '+name)
        # Only the pending order may change when its next-snapshot fill is applied.
        for i, order in enumerate(checkpoint['state']['orders']):
            if i != checkpoint['state']['pending']:
                require(result.orders[i] == order, 'continuation_prefix', 'old order changed')
    require(sources() == frozen and sha(__file__) == own and sha(ACCEPTANCE) == acceptance_hash,
            'continuation_source', 'implementation changed during replay')
    require(all(sha(p) == h for p, h in original_hashes.items()), 'continuation_input', 'original files changed')
    report = dict(document_kind='quote_control_daily_continuation', schema_version=1,
        status=result.status, failure=result.failure, processed_snapshots=len(result.ledger),
        preceding_snapshots=day, attempted_snapshot_id=args['history'][day].snapshot_id,
        planned_snapshots=len(args['history']), fresh_native_provider=True, retained_native_audits_imported=False,
        native_unique_requests=len(p.inner._quote_control_cache), checkpoint_prefix_preserved=True,
        sampling_policy=accepted['sampling_policy'], acceptance_sha256=acceptance_hash,
        source_sha256=frozen, checker_sha256=own, original_files_sha256=original_hashes,
        prefill_seconds=prefilled-start, replay_after_prefill_seconds=evaluated-prefilled,
        original_sampling_error_limit=args['config'].max_joint_bucket_standard_error,
        held=[r['joint_market']['held']['max_bucket_standard_error'] for r in result.risks],
        proposed=[r['joint_market']['max_bucket_standard_error'] for r in result.decisions],
        annual_completed=result.status == 'completed' and len(result.ledger) == len(args['history']),
        independent_price_comparison='not_yet_evaluated', scope='single additional date of the same synthetic annual workload')
    # Serialize data once per document; the price engine and its identity stay unchanged.
    result_doc = {f.name:data(getattr(result, f.name)) for f in fields(result)}
    committed = commit_bundle(output/'committed', {'input.json.gz':input_doc,
        'checkpoint.json.gz':result.checkpoint, 'result.json.gz':result_doc}, report)
    print(json.dumps({k:committed[k] for k in ('status','failure','processed_snapshots','held','proposed','native_unique_requests')}), flush=True)
    return committed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    r = run(a.input, a.checkpoint, a.output)
    raise SystemExit(r['status'] == 'failed')
