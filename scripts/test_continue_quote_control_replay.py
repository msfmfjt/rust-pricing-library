import errno
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import continue_quote_control_replay as replay
from rust_pricing_backtest.domain import BacktestError, canonical, fingerprint


class CompressedContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_compression_preserves_existing_canonical_format(self):
        value = {'z': [1.2, -0.0, None, True], '日本語': {'a': 'line\ntext'}}
        path = self.root/'output.json.gz'
        record = replay.compressed_document(path, value, schema=False)
        self.assertEqual(gzip.decompress(path.read_bytes()).decode(), canonical(value)+'\n')
        self.assertEqual(replay.read_document(path), value)
        self.assertEqual(record['sha256'], replay.sha(path))
        self.assertFalse(path.with_name(path.name+'.tmp').exists())

    def test_storage_failure_cannot_replace_old_checkpoint_or_publish_partial_file(self):
        old = self.root/'checkpoint.json'; old.write_text('{"prior":"committed"}')
        output = self.root/'new.json.gz'
        with patch.object(replay.gzip.GzipFile, 'write', side_effect=OSError(errno.ENOSPC, 'injected disk full')):
            with self.assertRaises(OSError): replay.compressed_document(output, {'new':'value'}, schema=False)
        self.assertFalse(output.exists()); self.assertFalse(output.with_name(output.name+'.tmp').exists())
        self.assertEqual(old.read_text(), '{"prior":"committed"}')

    def test_existing_outputs_and_temporary_files_are_never_removed(self):
        output = self.root/'new.json.gz'; temporary = output.with_name(output.name+'.tmp')
        for occupied in (output, temporary):
            occupied.write_text('retain')
            with self.assertRaisesRegex(BacktestError, 'continuation_output'):
                replay.compressed_document(output, {}, schema=False)
            self.assertEqual(occupied.read_text(), 'retain'); occupied.unlink()
        output.symlink_to(self.root/'missing')
        with self.assertRaisesRegex(BacktestError, 'continuation_output'):
            replay.compressed_document(output, {}, schema=False)
        self.assertTrue(output.is_symlink())

    def test_schema_rejection_does_not_publish(self):
        output = self.root/'new.json.gz'
        with patch.object(replay, 'validate_document', return_value={'status':'invalid'}):
            with self.assertRaisesRegex(BacktestError, 'continuation_schema'):
                replay.compressed_document(output, {})
        self.assertFalse(output.exists()); self.assertEqual(list(self.root.iterdir()), [])

    def test_bundle_failure_retains_unpublished_evidence_and_original_input(self):
        old = self.root/'old.json'; old.write_text('original')
        destination = self.root/'next'
        actual = replay.compressed_document
        def injected(path, value):
            if Path(path).name == 'checkpoint.json.gz':
                raise OSError(errno.ENOSPC, 'injected second document failure')
            return actual(path, value, schema=False)
        with patch.object(replay, 'compressed_document', side_effect=injected):
            with self.assertRaises(OSError):
                replay.commit_bundle(destination, {'input.json.gz':{},'checkpoint.json.gz':{}}, {'status':'partial'})
        self.assertFalse(destination.exists())
        self.assertTrue((self.root/'next.pending/input.json.gz').is_file())
        self.assertFalse((self.root/'next.pending/report.json').exists())
        self.assertEqual(old.read_text(), 'original')
        with self.assertRaisesRegex(BacktestError, 'continuation_output'):
            replay.commit_bundle(destination, {}, {})

    def test_committed_marker_binds_all_files_after_successful_validation(self):
        destination = self.root/'next'
        with patch.object(replay, 'validate_document', return_value={'status':'valid'}) as validate:
            record = replay.commit_bundle(destination, {'input.json.gz':{'i':1},'checkpoint.json.gz':{'state':2}}, {'status':'partial'})
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(record, json.loads((destination/'report.json').read_text()))
        self.assertEqual(record['report_hash'], fingerprint({k:v for k,v in record.items() if k!='report_hash'}))
        for name, evidence in record['files'].items():
            self.assertEqual(evidence['sha256'], replay.sha(destination/name))
        self.assertFalse((self.root/'next.pending').exists())

    def test_untrusted_plain_and_compressed_json_reject_duplicates_and_nonfinite_numbers(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            for suffix in ('.json', '.json.gz'):
                path = self.root/('input'+suffix)
                path.write_bytes(gzip.compress(raw.encode()) if suffix.endswith('.gz') else raw.encode())
                with self.assertRaisesRegex(BacktestError, 'continuation_json'):
                    replay.read_document(path)


if __name__ == '__main__': unittest.main()
