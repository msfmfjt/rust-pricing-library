"""Installed-wheel diagnostics, migration and owned Python snapshots."""
import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator
import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def task(name):
    return rp.PricingTask.from_json((ROOT / f'fixtures/task-v1/{name}.golden.json').read_text())


def report(name):
    return task(name).compile().evaluate().report


class CalibrationReportTests(unittest.TestCase):
    def test_fresh_reports_and_v1_migration_distinguish_missing_from_empty(self):
        for name in ('standard_risk', 'multi_asset_bs', 'stochastic_dividend_bs', 'stochastic_dividend_hw'):
            self.assertEqual(report(name).calibration_diagnostics, [])
        text = (ROOT / 'fixtures/report-v1/multi_asset_joint.golden.json').read_text()
        old = rp.ValuationReport.from_json(text)
        self.assertIsNone(old.calibration_diagnostics)
        expected = json.loads(text)
        expected['schema_version'] = 2
        self.assertEqual(json.loads(old.to_json()), expected)
        current = json.loads(report('multi_asset_joint').to_json())
        self.assertTrue(current.pop('calibration_diagnostics'))
        self.assertEqual(current['task_fingerprint'], expected['task_fingerprint'])
        if current['platform'] == expected['platform']:
            self.assertEqual(current, expected)
        self.assertIsNone(rp.ValuationReport.from_json(old.to_json()).calibration_diagnostics)

    def test_typed_snapshots_preserve_every_native_diagnostic_column(self):
        schema = Draft202012Validator(json.loads(rp.valuation_report_json_schema()))
        for name in ('multi_asset_joint', 'stochastic_dividend_lsv'):
            with self.subTest(name=name):
                r = report(name)
                saved = r.to_json()
                value = json.loads(saved)
                self.assertEqual(value['schema_version'], 2)
                schema.validate(value)
                self.assertEqual(rp.ValuationReport.from_json(r.to_pretty_json()).to_json(), saved)
                for b, wire in zip(r.calibration_diagnostics, value['calibration_diagnostics'], strict=True):
                    self.assertIsInstance(b, rp.ReportedCalibration)
                    self.assertEqual((b.kind, b.seed, b.underlying_ids), (wire['kind'], wire['seed'], wire['underlying_ids']))
                    self.assertEqual(b.axes, [(a['name'], a['values']) for a in wire['axes']])
                    for m, metric in zip(b.metrics, wire['metrics'], strict=True):
                        self.assertIsInstance(m, rp.CalibrationMetric)
                        self.assertEqual((m.name, m.unit, m.values, m.convention),
                                         (metric['name'], metric['unit'], metric['values'], metric['convention']))
                        m.values.clear()
                        with self.assertRaises(AttributeError):
                            m.unit = 'dimensionless'
                    b.axes[0][1].clear()
                    b.underlying_ids.clear()
                    b.metrics.clear()
                    with self.assertRaises(AttributeError):
                        b.seed = 1
                r.calibration_diagnostics.clear()
                self.assertEqual(r.to_json(), saved)
                with self.assertRaises(AttributeError):
                    r.calibration_diagnostics = []

    def test_operation_reselection_preserves_calibration_and_seed_provenance(self):
        for name in ('multi_asset_joint', 'stochastic_dividend_lsv'):
            original = task(name)
            compiled = original.compile()
            initial = json.loads(compiled.evaluate().report.to_json())['calibration_diagnostics']
            price = compiled.with_operation('evaluate')
            self.assertEqual(json.loads(price.evaluate().report.to_json())['calibration_diagnostics'], initial)
            self.assertEqual(json.loads(price.task.compile().evaluate().report.to_json())['calibration_diagnostics'], initial)
        source = json.loads(task('stochastic_dividend_lsv').to_json())
        source['source']['equity']['model']['equity']['particles']['seed'] += 1
        changed = rp.PricingTask.from_json(json.dumps(source)).compile().evaluate().report
        self.assertNotEqual(changed.task_fingerprint, compiled.task_fingerprint)
        self.assertEqual(changed.calibration_diagnostics[0].seed, source['source']['equity']['model']['equity']['particles']['seed'])

    def test_invalid_metrics_fail_with_report_pointer_and_schema_rejections(self):
        original = json.loads(report('multi_asset_joint').to_json())
        for path, value in [
            (['calibration_diagnostics'], None),
            (['calibration_diagnostics', 0, 'metrics', 0, 'values'], [1.0]),
            (['calibration_diagnostics', 0, 'metrics', 0, 'values', 0], float('inf')),
            (['calibration_diagnostics', 0, 'metrics', 0, 'convention'], ''),
            (['calibration_diagnostics', 0, 'seed'], -1),
            (['calibration_diagnostics', 0, 'unknown'], 1),
            (['schema_version'], 1),
        ]:
            v = copy.deepcopy(original)
            parent = v
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(rp.ValidationError) as caught:
                rp.ValuationReport.from_json(json.dumps(v))
            self.assertEqual(caught.exception.issues[0].document_kind, 'valuation_report')
        v = copy.deepcopy(original)
        v['calibration_diagnostics'][0]['metrics'][0]['values'] = []
        with self.assertRaises(rp.ValidationError) as caught:
            rp.ValuationReport.from_json(json.dumps(v))
        self.assertEqual(caught.exception.issues[0].pointer, '/calibration_diagnostics/0')
        self.assertEqual(caught.exception.issues[0].schema_version, 2)

    def test_current_golden_and_future_metric_roundtrip(self):
        saved = (ROOT / 'fixtures/report-v2/multi_asset_joint.golden.json').read_text()
        actual = report('multi_asset_joint')
        self.assertEqual(json.loads(rp.ValuationReport.from_json(saved).to_json()), json.loads(saved))
        # Saved numerical snapshots obey the native platform-specific replay contract.
        if actual.platform == json.loads(saved)['platform']:
            self.assertEqual(json.loads(actual.to_json()), json.loads(saved))
        v = json.loads(saved)
        extra = copy.deepcopy(v['calibration_diagnostics'][0]['metrics'][0])
        extra.update(name='future_native_metric', convention='Explicit future convention.')
        v['calibration_diagnostics'][0]['metrics'].append(extra)
        Draft202012Validator(json.loads(rp.valuation_report_json_schema())).validate(v)
        self.assertEqual(json.loads(rp.ValuationReport.from_json(json.dumps(v)).to_json()), v)


if __name__ == '__main__':
    unittest.main()
