from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import variance_basis_runtime as runtime
from check_current_variance_basis import main
from rust_pricing_backtest.domain import BacktestError


class VarianceBasisRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.binary = runtime.read_json(runtime.ROOT/runtime.ACCEPTANCE)['native_binary_sha256']
        self.fixture = runtime.ROOT/'fixtures/backtesting/raw-coupled-variance-basis-4-v0.1.json'

    def test_current_runtime_binds_all_numerical_dependencies(self):
        source, binary, accepted = runtime.current_provenance(self.binary)
        self.assertEqual(binary, self.binary)
        self.assertEqual(accepted, runtime.sha(runtime.ROOT/runtime.ACCEPTANCE))
        self.assertEqual(len([n for n in source if n.startswith('crates/')]), 360)
        self.assertEqual(len([n for n in source if n.startswith('backtesting/rust_pricing_backtest/')]), 64)
        for n in (*runtime.AUXILIARY, *runtime.RUNTIME, runtime.ACCEPTANCE): self.assertIn(n, source)

    def test_changed_native_or_numerical_source_is_rejected(self):
        with self.assertRaisesRegex(BacktestError, 'basis_reference'):
            runtime.current_provenance('different_binary')
        original = runtime.sha
        for name in ('backtesting/rust_pricing_backtest/multi_step_extension.py',
                     'backtesting/rust_pricing_backtest/quote_control.py', runtime.RUNNER,
                     'scripts/extension_put_reference.py', 'fixtures/backtesting/raw-variance-basis-policy-v0.1.json'):
            with self.subTest(name=name), patch.object(runtime, 'sha', side_effect=lambda p: 'changed' if Path(p)==runtime.ROOT/name else original(p)):
                with self.assertRaisesRegex(BacktestError, 'basis_reference'): runtime.current_provenance(self.binary)

    def test_numerical_runner_is_unchanged_and_historical_module_is_isolated(self):
        import check_raw_variance_basis as historical
        original_hook, original_write = historical.provenance, historical.write
        evidence = runtime.current_provenance(self.binary); instance = runtime.isolated_runner(evidence)
        self.assertIs(historical.provenance, original_hook); self.assertIs(historical.write, original_write)
        self.assertEqual(instance.run.__code__.co_code, historical.run.__code__.co_code)
        self.assertEqual(instance.provenance(), evidence)
        self.assertIs(instance.write, runtime.write_report)
        changed = deepcopy(evidence); changed[0][runtime.RUNNER] = 'changed'
        with self.assertRaisesRegex(BacktestError, 'basis_reference'): runtime.isolated_runner(changed)

    def test_report_seal_detects_edits_and_does_not_mutate_report(self):
        path = self.folder/'report.json'
        report = dict(document_kind='raw_variance_basis_validation', phase='native', status='in_progress', checks=[])
        original = deepcopy(report); runtime.write_report(path, report)
        self.assertEqual(report, original)
        retained = runtime.verify_resume(path)
        retained['checks'].append(dict(passed=True)); path.write_text(json.dumps(retained))
        with self.assertRaisesRegex(BacktestError, 'basis_resume_integrity'): runtime.verify_resume(path)
        runtime.write_report(path, dict(report, phase='preflight', status='ready_for_validation'))
        with self.assertRaisesRegex(BacktestError, 'basis_resume_integrity'): runtime.verify_resume(path)
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(BacktestError, 'basis_resume_integrity'): runtime.verify_resume(path)

    def test_duplicate_keys_and_nonfinite_saved_numbers_are_rejected(self):
        path = self.folder/'bad.json'
        for text in ('{"status":1,"status":2}', '{"x":NaN}', '{"x":Infinity}', '{'):
            path.write_text(text)
            with self.assertRaisesRegex(BacktestError, 'basis_runtime_json'): runtime.read_json(path)

    def test_output_aliases_cannot_overwrite_input_or_accepted_sources(self):
        basis = self.folder/'basis.json'; basis.write_bytes(self.fixture.read_bytes())
        for output in (basis, runtime.ROOT/runtime.RUNNER, self.folder/'basis.partial.json'):
            with self.subTest(output=output), self.assertRaisesRegex(BacktestError, 'basis_output_collision'):
                runtime.guard_destinations(output, basis, output if output.name=='basis.partial.json' else None)
        alias = self.folder/'alias.json'; alias.hardlink_to(basis)
        with self.assertRaisesRegex(BacktestError, 'basis_output_collision'): runtime.guard_destinations(alias, basis)
        alias.unlink(); alias.symlink_to(basis)
        with self.assertRaisesRegex(BacktestError, 'basis_output_collision'): runtime.guard_destinations(alias, basis)
        runtime.guard_destinations(self.folder/'safe.json', basis)
        self.assertEqual(basis.read_bytes(), self.fixture.read_bytes())

    def invoke(self, *args):
        with patch.object(runtime, 'native_binary_hash', return_value=self.binary), redirect_stdout(io.StringIO()):
            return main(list(args))

    def test_preflight_never_claims_numerical_acceptance_or_runs_native_prices(self):
        for n in (4, 6):
            path = self.folder/f'{n}.json'; basis = runtime.ROOT/f'fixtures/backtesting/raw-coupled-variance-basis-{n}-v0.1.json'
            with patch('check_current_variance_basis.isolated_runner', side_effect=AssertionError('native path entered')):
                code = self.invoke('--intervals', str(n), '--basis-file', str(basis), '--output', str(path), '--preflight')
            self.assertEqual(code, 0); report = runtime.read_json(path)
            self.assertEqual(report['status'], 'ready_for_validation')
            self.assertEqual(report['numerical_acceptance'], 'not_evaluated')
            self.assertFalse(report['provider_promotion']); self.assertTrue(report['diagnostic_only'])
            with self.assertRaisesRegex(BacktestError, 'basis_resume_integrity'): runtime.verify_resume(path)

    def test_rank_deficiency_and_historical_resume_fail_closed(self):
        doc = runtime.read_json(self.fixture); doc['values'] = [[0.]*len(doc['values'][0]) for _ in doc['values']]
        zero = self.folder/'zero.json'; zero.write_text(json.dumps(doc)); out = self.folder/'out.json'
        self.assertEqual(self.invoke('--intervals','4','--basis-file',str(zero),'--output',str(out),'--preflight'), 1)
        self.assertEqual(runtime.read_json(out)['error']['code'], 'extension_rank')
        old = self.folder/'old.json'; old.write_text('{"status":"passed","phase":"complete"}')
        with patch('check_current_variance_basis.isolated_runner', side_effect=AssertionError('native path entered')):
            self.assertEqual(self.invoke('--intervals','4','--basis-file',str(self.fixture),'--output',str(out),'--resume',str(old)), 1)
        self.assertEqual(runtime.read_json(out)['error']['code'], 'basis_resume_integrity')

    def test_cli_refuses_destructive_output_without_writing_error_over_input(self):
        basis = self.folder/'basis.json'; basis.write_bytes(self.fixture.read_bytes()); before = basis.read_bytes()
        self.assertEqual(self.invoke('--intervals','4','--basis-file',str(basis),'--output',str(basis),'--preflight'), 1)
        self.assertEqual(basis.read_bytes(), before)


if __name__ == '__main__': unittest.main()
