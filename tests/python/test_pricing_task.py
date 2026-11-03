"""Complete source replay, capability preflight and common risk report contracts."""
import copy
import json
from pathlib import Path
import unittest
import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def fixture(name):
    return (ROOT / f'fixtures/task-v1/{name}.golden.json').read_text()


class PricingTaskTests(unittest.TestCase):
    def test_exercise_conditioning_and_vegakt_covariance_are_preserved(self):
        value = json.loads((ROOT / 'fixtures/v3/pricing_request_american.golden.json').read_text())
        value['risk']['delta'] = True
        request = rp.PricingRequest.from_json(json.dumps(value))
        task = rp.PricingTask.from_specification(rp.PricingSpecification.from_request(
            request, worker_threads=1, reduction_block_size=16))
        report = task.compile().evaluate().report
        self.assertEqual(report.uncertainty_scope, 'conditional_on_exercise_policy')
        self.assertTrue(report.risks)
        self.assertTrue(all(r.calibration == 'fixed_exercise_policy' for r in report.risks))
        self.assertEqual(report.to_json(), rp.ValuationReport.from_json(report.to_json()).to_json())

        from rust_pricing_backtest.numerics import flat_request
        policy = json.loads((ROOT / 'fixtures/backtesting/numerical-policy-v0.1.json').read_text())
        value = json.loads(flat_request(policy, seed=7, samples=32, steps=2, nodes=9).to_json())
        value['risk']['vega_kt']['full_bucket_covariance'] = True
        request = rp.PricingRequest.from_json(json.dumps(value))
        source = rp.PricingSpecification.from_request(request, worker_threads=1, reduction_block_size=16)
        native = source.compile().evaluate().native_result
        report = rp.PricingTask.from_specification(source).compile().evaluate().report
        self.assertEqual(report.vega_kt.raw_buckets, native.vega_kt.raw_buckets)
        self.assertEqual(report.vega_kt.full_bucket_covariance, native.vega_kt.full_bucket_covariance)
        saved = json.loads(report.to_json())
        self.assertEqual(saved['vega_kt'], json.loads(native.to_json())['risks']['vega_kt'])
        self.assertEqual(saved, json.loads(rp.ValuationReport.from_json(report.to_json()).to_json()))

    def test_standard_risk_native_equivalence_and_immutable_report(self):
        value = json.loads(fixture('standard_risk'))
        source = rp.PricingSpecification.from_json(json.dumps(value['source']['specification']))
        task = rp.PricingTask.from_specification(source)
        native = source.compile().evaluate()
        self.assertEqual(task.to_json(), rp.PricingTask.from_json(fixture('standard_risk')).to_json())
        before = task.capability('evaluate')
        self.assertEqual((before.status, before.reason), ('conditional', 'compilation_required'))
        plan = task.compile()
        self.assertEqual(plan.capability().status, 'supported')
        result = plan.evaluate()
        report = result.report
        self.assertIsInstance(result.native_result, rp.SpecificationResult)
        self.assertEqual(report.value, native.value)
        self.assertEqual(report.standard_error, native.standard_error)
        self.assertEqual(report.task_fingerprint, task.fingerprint)
        self.assertEqual(report.plan_fingerprint, plan.plan_fingerprint)
        self.assertEqual(report.uncertainty_scope, 'pricing_sampling')
        risks = {r.name: r for r in report.risks}
        self.assertEqual(set(risks), {'delta.raw', 'delta.market_scaled', 'vega.raw', 'vega.market_scaled'})
        self.assertEqual(risks['delta.raw'].unit, 'currency_per_spot')
        self.assertEqual(risks['vega.market_scaled'].unit, 'currency_per_volatility_point')
        self.assertEqual(risks['delta.raw'].values, [native.native_result.delta.raw.value])
        self.assertEqual(risks['delta.raw'].axes, [])
        detached = risks['delta.raw'].values
        detached[0] = 999.
        self.assertNotEqual(detached, risks['delta.raw'].values)
        for obj, name in [(task, 'fingerprint'), (report, 'value'), (risks['delta.raw'], 'unit'), (before, 'status')]:
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(obj, name, 0)

    def test_complete_models_replay_prices_risks_and_metadata(self):
        for name, native_type in [
            ('multi_asset_bs', rp.MultiAssetPrice), ('multi_asset_joint', rp.MultiAssetPrice),
            ('stochastic_dividend_bs', rp.StochasticDividendAadRisk),
            ('stochastic_dividend_hw', rp.StochasticDividendAadRisk),
            ('stochastic_dividend_lsv', rp.StochasticDividendLocalVarianceRisk),
        ]:
            with self.subTest(name=name):
                task = rp.PricingTask.from_json(fixture(name))
                restored = rp.PricingTask.from_json(task.to_pretty_json())
                self.assertEqual(task.fingerprint, restored.fingerprint)
                a, b = task.compile(), restored.compile()
                self.assertEqual(a.plan_fingerprint, b.plan_fingerprint)
                self.assertEqual(a.capability().status, 'supported')
                result = a.evaluate()
                self.assertIsInstance(result.native_result, native_type)
                report = result.report
                self.assertEqual(report.to_json(), b.evaluate().report.to_json())
                saved = rp.ValuationReport.from_json(report.to_pretty_json())
                self.assertEqual(report.to_json(), saved.to_json())
                self.assertTrue(report.risks)
                self.assertTrue(all(r.method and r.convention for r in report.risks))
                price_task = task.with_operation('evaluate')
                self.assertEqual(price_task.compile().evaluate().report.value, report.value)
                self.assertNotEqual(task.fingerprint, price_task.fingerprint)
                if name == 'multi_asset_bs':
                    self.assertEqual(len([r for r in report.risks if r.name == 'gamma']), 4)
                    self.assertEqual(a.capability().gamma_relative_bump, .01)
                if name == 'multi_asset_joint':
                    self.assertEqual(report.calibration_seeds, [74, 513])
                    self.assertEqual(report.uncertainty_scope, 'conditional_on_calibration')
                    self.assertEqual(report.estimator, 'randomized_quasi_monte_carlo')
                    self.assertTrue(any(r.name == 'joint_asset_hw.market_iv.raw' for r in report.risks))
                if name == 'stochastic_dividend_lsv':
                    risk = report.risks[0]
                    self.assertEqual(risk.calibration, 'recalibrated')
                    self.assertEqual(risk.unit, 'currency_per_variance')
                    self.assertIsNone(risk.standard_errors)
                    self.assertNotIn('standard_errors', json.loads(report.to_json())['risks'][0])

    def test_capability_reasons_and_native_preflight(self):
        standard = rp.PricingTask.from_json(fixture('standard_risk'))
        self.assertEqual(standard.capability('aad').reason, 'dedicated_risk_api')
        basket = rp.PricingTask.from_json(fixture('multi_asset_bs'))
        for bump in [0., 1., float('nan'), float('inf')]:
            capability = basket.capability('aad', gamma_relative_bump=bump)
            self.assertEqual((capability.status, capability.reason), ('unsupported', 'invalid_gamma_bump'))
            with self.assertRaises(rp.PricingError):
                basket.with_operation('aad', gamma_relative_bump=bump)
        with self.assertRaises(ValueError):
            basket.capability('evaluate', gamma_relative_bump=.01)
        value = json.loads(fixture('stochastic_dividend_lsv'))
        value['operation'] = {'type': 'evaluate'}
        value['source']['equity']['model']['equity']['particles']['retain_reverse_trace'] = False
        source = rp.PricingTask.from_json(json.dumps(value))
        self.assertEqual(source.capability('local_variance_risk').reason, 'reverse_trace_required')
        self.assertEqual(source.compile().capability().status, 'supported')
        with self.assertRaises(rp.PricingError):
            source.with_operation('local_variance_risk')
        value = json.loads(fixture('multi_asset_joint'))
        value['operation'] = {'type': 'evaluate'}
        value['source']['driver_correlations'][0][1][2] = -.2
        source = rp.PricingTask.from_json(json.dumps(value))
        self.assertEqual(source.capability('evaluate').status, 'conditional')
        with self.assertRaises(rp.PricingError):
            source.compile()

    def test_strict_wire_shapes_versions_and_schema_exports(self):
        original = json.loads(fixture('multi_asset_bs'))
        report = json.loads(rp.PricingTask.from_json(fixture('standard_risk')).compile().evaluate().report.to_json())
        for kind, original, cls, mutations in [
            ('pricing_task', original, rp.PricingTask, [(['schema_version'], 2),
                (['unknown'], 1), (['source', 'numerics', 'maximum_step'], 0),
                (['source', 'lsv', 0], None)]),
            ('valuation_report', report, rp.ValuationReport, [(['schema_version'], 3),
                (['standard_error'], -1), (['evaluated_paths'], '0001'),
                (['risks', 0, 'values'], [1., 2.]), (['risks', 0, 'standard_errors'], []),
                (['risks', 0, 'standard_errors'], None), (['risks', 0, 'unknown'], 1)]),
        ]:
            for path, value in mutations:
                bad = copy.deepcopy(original)
                parent = bad
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                with self.subTest(kind=kind, path=path), self.assertRaises(rp.ValidationError) as caught:
                    cls.from_json(json.dumps(bad))
                self.assertEqual(caught.exception.issues[0].document_kind, kind)
            duplicate = json.dumps(original).replace('{', '{"schema_version":1,', 1)
            with self.assertRaises(rp.ValidationError):
                cls.from_json(duplicate)
        for name, export in [('task-v1/pricing_task', rp.pricing_task_json_schema),
                             ('report-v2/valuation_report', rp.valuation_report_json_schema)]:
            self.assertEqual(json.loads(export()), json.loads((ROOT / f'schemas/{name}.schema.json').read_text()))
        saved = (ROOT / 'fixtures/report-v1/multi_asset_joint.golden.json').read_text()
        restored = rp.ValuationReport.from_json(saved)
        self.assertEqual(restored.task_fingerprint, rp.PricingTask.from_json(fixture('multi_asset_joint')).fingerprint)
        migrated = json.loads(saved)
        migrated['schema_version'] = 2
        self.assertIsNone(restored.calibration_diagnostics)
        self.assertEqual(migrated, json.loads(restored.to_json()))


if __name__ == '__main__':
    unittest.main()
