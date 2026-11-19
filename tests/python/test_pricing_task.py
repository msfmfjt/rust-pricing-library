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

    def test_measures_equal_request_flags_and_type_the_report(self):
        value = json.loads((ROOT / 'fixtures/v3/pricing_request.golden.json').read_text())
        value['engine']['independent_sampling_units'] = 64
        value['risk'] = {'delta': False, 'vega': False,
                         'smile_dynamics': {'type': 'sticky_log_moneyness'}}
        plain = rp.PricingSpecification.from_request(
            rp.PricingRequest.from_json(json.dumps(value)),
            worker_threads=1, reduction_block_size=16)
        value['risk'].update(delta=True, vega=True,
                             gamma={'bump': {'type': 'relative', 'value': 0.01}})
        flagged = rp.PricingSpecification.from_request(
            rp.PricingRequest.from_json(json.dumps(value)),
            worker_threads=1, reduction_block_size=16)
        M = rp.MeasureRequest
        measures = [M.spot_delta(), M.spot_gamma(relative_bump=0.01),
                    M.parallel_market_iv_vega()]

        probe = rp.PricingTask.from_specification(plain)
        self.assertEqual(probe.available_measures(), {
            'price': ['native'], 'spot_delta': ['native', 'bump_and_revalue'],
            'spot_gamma': ['native_spot_bump', 'bump_and_revalue'],
            'parallel_market_iv_vega': ['native', 'bump_and_revalue'],
            'vega_kt': ['native']})
        refused = probe.measure_capabilities([M.spot_gamma(), M.spot_delta()])
        self.assertEqual([(c.measure, c.status, c.reason) for c in refused], [
            ('spot_gamma', 'unsupported', 'no_native_estimator'),
            ('spot_delta', 'conditional', 'compilation_required')])
        self.assertIsNone(probe.measures)

        task = rp.PricingTask.from_specification(
            plain, operation='measures', measures=measures)
        self.assertEqual(task.measures, measures)
        self.assertEqual(json.loads(task.to_json())['schema_version'], 3)
        self.assertEqual(rp.PricingTask.from_json(task.to_json()).fingerprint, task.fingerprint)
        plan = task.compile()
        self.assertEqual(plan.capability().operation, 'measures')
        self.assertEqual(plan.capability().status, 'supported')
        self.assertEqual(
            [c.status for c in plan.measure_capabilities([M.spot_delta(), M.spot_gamma(relative_bump=0.5)])],
            ['supported', 'unsupported'])
        result = plan.evaluate()
        legacy = rp.PricingTask.from_specification(flagged).compile().evaluate()
        self.assertEqual(result.native_result.native_result.to_json(),
                         legacy.native_result.native_result.to_json())

        report = result.report
        typed = [(r.name, r.measure, r.values, r.standard_errors) for r in report.risks]
        named = [(r.name, r.values, r.standard_errors) for r in legacy.report.risks]
        self.assertEqual([(n, v, e) for n, _, v, e in typed], named)
        self.assertTrue(all(r.measure is None for r in legacy.report.risks))
        self.assertEqual(report.risk('spot_delta').name, 'delta.raw')
        self.assertEqual(report.risk('spot_gamma').method_detail,
                         {'type': 'native_spot_bump', 'bump_type': 'relative', 'bump': 0.01})
        self.assertEqual(report.risk('spot_delta').method_detail, {'type': 'native'})
        self.assertIsNone(report.risk('vega_kt'))
        with self.assertRaises(ValueError):
            report.risk('theta')
        saved = json.loads(report.to_json())
        self.assertEqual(saved['schema_version'], 3)
        self.assertEqual(json.loads(legacy.report.to_json())['schema_version'], 2)
        self.assertEqual(rp.ValuationReport.from_json(report.to_json()).to_json(), report.to_json())

        records = report.to_records()
        self.assertEqual(len(records), 1 + len(report.risks))
        self.assertEqual(records[0], {
            'measure': 'price', 'name': 'price', 'underlying_ids': [], 'coordinate': {},
            'value': report.value, 'standard_error': report.standard_error,
            'unit': 'currency', 'method': 'pseudo_monte_carlo',
            'calibration': 'not_applicable', 'uncertainty_scope': 'pricing_sampling',
            'task_fingerprint': task.fingerprint})
        self.assertEqual((records[1]['measure'], records[1]['unit'], records[1]['value']),
                         ('spot_delta', 'currency_per_spot', report.risks[0].values[0]))

        # Risk is selected in exactly one place, and never implicitly.
        with self.assertRaises(rp.PricingError) as caught:
            rp.PricingTask.from_specification(flagged, operation='measures', measures=measures)
        self.assertEqual(caught.exception.reason, 'ambiguous_risk_selection')
        for arguments in [dict(operation='measures'), dict(measures=measures),
                          dict(operation='aad', measures=measures)]:
            with self.assertRaises(ValueError):
                rp.PricingTask.from_specification(plain, **arguments)
        with self.assertRaises((ValueError, rp.PricingError)):
            rp.PricingTask.from_specification(
                plain, operation='measures', measures=[M.spot_delta(), M.spot_delta()])
        with self.assertRaises((ValueError, rp.PricingError)):
            plan.with_operation('evaluate')
        basket = rp.PricingTask.from_json(fixture('multi_asset_bs'))
        self.assertEqual(basket.available_measures(), {
            'price': ['native'],
            'spot_delta': ['native', 'native_calibrated', 'bump_and_revalue'],
            'spot_gamma': ['bump_and_revalue'],
            'cross_gamma': ['native_spot_bump', 'native_spot_bump_calibrated'],
            'correlations': ['bump_and_revalue'],
            'discount_curve': ['native', 'native_calibrated']})
        priced = basket.with_operation('measures', measures=[M.price()]).compile().evaluate()
        self.assertEqual(priced.report.value,
                         basket.with_operation('evaluate').compile().evaluate().report.value)
        self.assertEqual(
            [c.reason for c in basket.measure_capabilities(
                [M.spot_delta(), M.parallel_market_iv_vega()])],
            ['compilation_required', 'dedicated_risk_api'])

    def test_bump_and_revalue_states_its_bump_calibration_and_cost(self):
        value = json.loads((ROOT / 'fixtures/v3/pricing_request.golden.json').read_text())
        value['engine']['independent_sampling_units'] = 64
        value['risk'] = {'delta': False, 'vega': False,
                         'smile_dynamics': {'type': 'sticky_log_moneyness'}}
        source = rp.PricingSpecification.from_request(
            rp.PricingRequest.from_json(json.dumps(value)),
            worker_threads=1, reduction_block_size=16)
        M = rp.MeasureRequest
        delta = M.bump_and_revalue('spot_delta', relative=0.01, calibration='not_applicable')
        self.assertEqual((delta.method, delta.relative_bump, delta.calibration),
                         ('bump_and_revalue', 0.01, 'not_applicable'))
        for arguments in [dict(relative=0.01), dict(calibration='fixed'),
                          dict(relative=0.01, absolute=1.0, calibration='fixed'),
                          dict(relative=0.01, calibration='implicit')]:
            with self.assertRaises((ValueError, TypeError)):
                M.bump_and_revalue('spot_delta', **arguments)
        with self.assertRaises(ValueError):
            M.bump_and_revalue('vega_kt', relative=0.01, calibration='not_applicable')

        probe = rp.PricingTask.from_specification(source)
        gamma = M.bump_and_revalue('spot_gamma', relative=0.01, calibration='not_applicable')
        self.assertEqual(probe.bump_cost([delta, gamma]), {'calibrations': 0, 'evaluations': 3})
        wrong = M.bump_and_revalue('spot_delta', relative=0.01, calibration='recalibrated')
        self.assertEqual(probe.measure_capabilities([wrong])[0].reason, 'bump_calibration_mismatch')

        task = rp.PricingTask.from_specification(
            source, operation='measures', measures=[delta, gamma])
        plan = task.compile()
        result = plan.evaluate()
        self.assertEqual(plan.bump_cost(), {'calibrations': 0, 'evaluations': 3})
        row = result.report.risk('spot_delta')
        self.assertEqual(len(row.standard_errors), 1)
        self.assertGreater(row.standard_errors[0], 0.0)
        self.assertEqual(row.method_detail, {'type': 'bump_and_revalue', 'bump_type': 'relative',
                                             'bump': 0.01, 'calibration': 'not_applicable'})
        self.assertEqual(rp.PricingTask.from_json(task.to_json()).fingerprint, task.fingerprint)
        self.assertEqual(
            rp.ValuationReport.from_json(result.report.to_json()).to_json(),
            result.report.to_json())
        records = [r for r in result.report.to_records() if r['measure'] == 'spot_gamma']
        self.assertEqual(records[0]['standard_error'],
                         result.report.risk('spot_gamma').standard_errors[0])

    def test_rough_family_lsv_native_spot_measures_state_their_calibration(self) -> None:
        value = json.loads(
            (ROOT / 'fixtures/specification-v1/bergomi_lsv.golden.json').read_text())
        value['model']['equity'] = {
            'type': 'rough_local_stochastic_volatility',
            'target': value['model']['equity']['target'],
            'model': {'type': 'rough_heston', 'hurst': 0.2, 'initial_variance': 0.04,
                      'mean_reversion': 0.7, 'long_run_variance': 0.055,
                      'vol_of_vol': 0.15, 'correlation': -0.6},
            'particles': value['model']['equity']['particles']}
        source = rp.PricingSpecification.from_json(json.dumps(value))
        M = rp.MeasureRequest
        delta = M.spot_delta(calibration='recalibrated')
        gamma = M.spot_gamma(relative_bump=0.01, calibration='fixed')
        self.assertEqual((delta.method, delta.calibration),
                         ('native_calibrated', 'recalibrated'))
        self.assertEqual((gamma.method, gamma.relative_bump, gamma.calibration),
                         ('native_spot_bump_calibrated', 0.01, 'fixed'))
        with self.assertRaises(ValueError):
            M.spot_gamma(calibration='fixed')
        with self.assertRaises(ValueError):
            M.spot_delta(calibration='implicit')

        probe = rp.PricingTask.from_specification(source)
        self.assertEqual(probe.source_family, 'rough_family_lsv')
        self.assertEqual(probe.available_measures()['spot_delta'],
                         ['native_calibrated', 'bump_and_revalue'])
        reasons = [c.reason for c in probe.measure_capabilities(
            [M.spot_delta(), M.spot_delta(calibration='not_applicable')])]
        self.assertEqual(reasons, ['no_native_estimator', 'bump_calibration_mismatch'])

        task = rp.PricingTask.from_specification(
            source, operation='measures', measures=[delta, gamma])
        result = task.compile().evaluate()
        native = result.native_result
        self.assertIsInstance(native, rp.RoughFamilyLsvMeasureRisk)
        row = result.report.risk('spot_delta')
        self.assertEqual(row.values, [native.delta.delta])
        self.assertEqual(row.standard_errors, [native.delta.delta_standard_error])
        self.assertEqual(row.calibration, 'recalibrated')
        self.assertEqual(row.method_detail,
                         {'type': 'native_calibrated', 'calibration': 'recalibrated'})
        row = result.report.risk('spot_gamma')
        self.assertEqual(row.values, [native.gamma.gamma])
        self.assertEqual(row.calibration, 'fixed_calibration')
        self.assertEqual(row.method_detail, {
            'type': 'native_spot_bump_calibrated', 'bump_type': 'relative', 'bump': 0.01,
            'calibration': 'fixed'})
        self.assertEqual(rp.PricingTask.from_json(task.to_json()).fingerprint, task.fingerprint)
        self.assertEqual(
            rp.ValuationReport.from_json(result.report.to_json()).to_json(),
            result.report.to_json())

    def test_rough_family_lsv_market_iv_vega_needs_a_reproducing_basis(self) -> None:
        maturities, nodes = [0.2, 0.6, 1.2], [-0.9, -0.45, 0.05, 0.5, 0.9]
        quotes = [0.2 + 0.002 * (i // 5) - 0.003 * nodes[i % 5] + 0.002 * nodes[i % 5] ** 2
                  for i in range(15)]
        surface = rp.MarketIvSurface(maturities, nodes, quotes)
        model = surface.local_volatility_model(
            [0.0, 0.25, 0.5, 0.75, 1.0], [-0.55, -0.13, 0.18, 0.6])
        discount = rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        market = rp.Market.equity(2, 1, 100.0, discount, discount)
        request = rp.PricingRequest(
            '2026-09-04', rp.Product.european_vanilla(1, 2, '2027-09-04', 100.0, 1.0, 'call'),
            market, model, rp.Engine.pseudo_monte_carlo(819, 128), rp.RiskRequest())
        grid = json.loads(request.to_json())['model']['local_variance_grid']
        value = json.loads(
            (ROOT / 'fixtures/specification-v1/bergomi_lsv.golden.json').read_text())
        particles = dict(value['model']['equity']['particles'], retain_reverse_trace=True)
        value['model']['equity'] = {
            'type': 'rough_local_stochastic_volatility',
            'target': {'local_variance_grid': grid, 'reporting_iv_basis': {
                'maturity_nodes': maturities, 'log_forward_moneyness_nodes': nodes,
                'shape': [3, 5], 'implied_volatilities': quotes}},
            'model': {'type': 'rough_heston', 'hurst': 0.2, 'initial_variance': 0.04,
                      'mean_reversion': 0.7, 'long_run_variance': 0.055,
                      'vol_of_vol': 0.15, 'correlation': -0.6},
            'particles': particles}
        source = rp.PricingSpecification.from_json(json.dumps(value))
        M = rp.MeasureRequest
        vega = M.parallel_market_iv_vega(calibration='recalibrated')
        self.assertEqual((vega.method, vega.calibration), ('native_calibrated', 'recalibrated'))
        probe = rp.PricingTask.from_specification(source)
        self.assertEqual(probe.available_measures()['parallel_market_iv_vega'],
                         ['native_calibrated', 'bump_and_revalue'])
        reasons = [c.reason for c in probe.measure_capabilities([
            M.parallel_market_iv_vega(calibration='fixed'),
            M.bump_and_revalue('parallel_market_iv_vega', calibration='recalibrated',
                               relative=0.01)])]
        self.assertEqual(reasons, ['bump_calibration_mismatch', 'model_combination'])

        task = rp.PricingTask.from_specification(source, operation='measures', measures=[vega])
        result = task.compile().evaluate()
        native = result.native_result
        self.assertIsInstance(native, rp.RoughFamilyLsvMeasureRisk)
        self.assertIsNone(native.delta)
        self.assertIsInstance(native.market_iv, rp.RoughFamilyLsvMarketIvRisk)
        row = result.report.risk('parallel_market_iv_vega')
        self.assertEqual(row.values, [native.market_iv.parallel_vega])
        self.assertIsNone(row.standard_errors)
        self.assertEqual(row.calibration, 'recalibrated')
        self.assertEqual(row.method_detail,
                         {'type': 'native_calibrated', 'calibration': 'recalibrated'})

        # Without the reporting basis the IVs are unknown.
        del value['model']['equity']['target']['reporting_iv_basis']
        bare = rp.PricingTask.from_specification(
            rp.PricingSpecification.from_json(json.dumps(value)))
        self.assertEqual(bare.measure_capabilities([vega])[0].reason, 'model_combination')

    def test_rough_family_lsv_model_parameters_are_typed_rows(self) -> None:
        value = json.loads(
            (ROOT / 'fixtures/specification-v1/bergomi_lsv.golden.json').read_text())
        value['model']['equity'] = {
            'type': 'rough_local_stochastic_volatility',
            'target': value['model']['equity']['target'],
            'model': {'type': 'rough_heston', 'hurst': 0.2, 'initial_variance': 0.04,
                      'mean_reversion': 0.7, 'long_run_variance': 0.055,
                      'vol_of_vol': 0.15, 'correlation': -0.6},
            'particles': dict(value['model']['equity']['particles'],
                              retain_reverse_trace=True)}
        source = rp.PricingSpecification.from_json(json.dumps(value))
        M = rp.MeasureRequest
        request = M.model_parameters(['hurst', 'vol_of_vol'], calibration='recalibrated')
        self.assertEqual(request.measure, 'model_parameters')
        self.assertEqual(request.parameters, ['hurst', 'vol_of_vol'])
        for bad in (['volatility'], [], ['hurst', 'hurst']):
            with self.assertRaises(ValueError):
                M.model_parameters(bad, calibration='recalibrated')
        probe = rp.PricingTask.from_specification(source)
        self.assertEqual(probe.available_measures()['model_parameters'], ['native_calibrated'])
        reasons = [c.reason for c in probe.measure_capabilities([
            M.model_parameters(['vol_of_vol'], calibration='fixed'),
            M.model_parameters(['variance_floor'], calibration='recalibrated')])]
        self.assertEqual(reasons, ['bump_calibration_mismatch', 'model_combination'])

        task = rp.PricingTask.from_specification(
            source, operation='measures', measures=[request])
        result = task.compile().evaluate()
        identified = result.native_result.model_parameters
        self.assertIsInstance(identified, rp.RoughFamilyLsvModelParameterRisk)
        risk = identified.risk
        self.assertIsInstance(risk, rp.HestonLsvParameterRisk)
        self.assertEqual(identified.parameters[-1], 'hurst')
        rows = [r for r in result.report.risks if r.measure == 'model_parameters']
        self.assertEqual([r.model_parameter for r in rows], ['hurst', 'vol_of_vol'])
        for row in rows:
            index = identified.parameters.index(row.model_parameter)
            self.assertEqual(row.values, [risk.parameter_adjoints[index]])
            self.assertEqual(row.unit, 'currency_per_unit_parameter')
            self.assertEqual(row.calibration, 'recalibrated')
        self.assertEqual(result.report.model_parameter_risk('vol_of_vol').name,
                         'model_parameters.vol_of_vol.native_calibrated')
        self.assertIsNone(result.report.model_parameter_risk('correlation'))
        self.assertEqual(rp.PricingTask.from_json(task.to_json()).fingerprint, task.fingerprint)
        self.assertEqual(
            rp.ValuationReport.from_json(result.report.to_json()).to_json(),
            result.report.to_json())

    def test_direct_rough_volatility_measures_are_fixed_model_rows(self) -> None:
        value = json.loads(
            (ROOT / 'fixtures/specification-v1/rough_heston.golden.json').read_text())
        source = rp.PricingSpecification.from_json(json.dumps(value))
        M = rp.MeasureRequest
        probe = rp.PricingTask.from_specification(source)
        available = probe.available_measures()
        self.assertEqual(available['spot_delta'], ['native', 'bump_and_revalue'])
        self.assertEqual(available['spot_gamma'], ['native_spot_bump', 'bump_and_revalue'])
        self.assertEqual(available['model_parameters'], ['native'])
        reasons = [c.reason for c in probe.measure_capabilities([
            M.model_parameters(['variance_floor']),
            M.model_parameters(['vol_of_vol'], calibration='recalibrated'),
            M.spot_delta(calibration='fixed')])]
        self.assertEqual(
            reasons, ['model_combination', 'no_native_estimator', 'no_native_estimator'])

        request = M.model_parameters(['hurst', 'correlation'])
        self.assertEqual((request.method, request.calibration), ('native', None))
        task = rp.PricingTask.from_specification(
            source, operation='measures',
            measures=[M.spot_delta(), M.spot_gamma(relative_bump=0.01), request])
        result = task.compile().evaluate()
        native = result.native_result
        self.assertIsInstance(native, rp.RoughVolatilityMeasureRisk)
        self.assertIsInstance(native.delta, rp.RoughVolatilityDelta)
        self.assertIsInstance(native.gamma, rp.RoughVolatilityGamma)
        identified = native.model_parameters
        self.assertIsInstance(identified, rp.RoughVolatilityModelParameterRisk)
        self.assertEqual(identified.parameters[-1], 'hurst')
        self.assertIsInstance(identified.heston, rp.HestonMcParameterRisk)
        self.assertIsInstance(identified.hurst, rp.RoughHestonMcHurstRisk)
        self.assertIsNone(identified.mixed_bergomi)
        report = result.report
        self.assertEqual(report.risk('spot_delta').values, [native.delta.delta])
        self.assertEqual(report.risk('spot_delta').calibration, 'not_applicable')
        self.assertEqual(report.risk('spot_gamma').values, [native.gamma.gamma])
        hurst = report.model_parameter_risk('hurst')
        self.assertEqual(hurst.values, [identified.hurst.hurst_sensitivity])
        self.assertEqual(hurst.standard_errors, [identified.hurst.standard_error])
        self.assertEqual(hurst.name, 'model_parameters.hurst.native')
        correlation = report.model_parameter_risk('correlation')
        self.assertEqual(correlation.values, [identified.heston.parameter_adjoints[4]])
        rows = [r.model_parameter for r in report.risks if r.measure == 'model_parameters']
        self.assertEqual(rows, ['hurst', 'correlation'])
        self.assertEqual(
            rp.ValuationReport.from_json(report.to_json()).to_json(), report.to_json())

    def test_stochastic_dividend_spot_measures_are_native_rows(self) -> None:
        M = rp.MeasureRequest
        base = rp.PricingTask.from_json(fixture('stochastic_dividend_bs'))
        self.assertEqual(base.available_measures()['spot_gamma'],
                         ['native_spot_bump', 'bump_and_revalue'])
        task = base.with_operation(
            'measures', measures=[M.spot_delta(), M.spot_gamma(relative_bump=0.01)])
        result = task.compile().evaluate()
        native = result.native_result
        self.assertIsInstance(native, rp.StochasticDividendMeasureRisk)
        self.assertIsInstance(native.delta, rp.StochasticDividendAadRisk)
        self.assertIsInstance(native.gamma, rp.StochasticDividendGammaRisk)
        delta = result.report.risk('spot_delta')
        self.assertEqual(delta.values, [native.delta.delta])
        self.assertEqual(delta.calibration, 'not_applicable')
        self.assertEqual(result.report.risk('spot_gamma').values, [native.gamma.gamma])

        lsv = rp.PricingTask.from_json(fixture('stochastic_dividend_lsv'))
        self.assertEqual(lsv.available_measures()['spot_delta'],
                         ['native_calibrated', 'bump_and_revalue'])
        reasons = [c.reason for c in lsv.measure_capabilities([
            M.spot_delta(calibration='fixed'), M.spot_delta()])]
        self.assertEqual(reasons, ['no_native_estimator', 'no_native_estimator'])
        result = lsv.with_operation(
            'measures', measures=[M.spot_delta(calibration='recalibrated')]
        ).compile().evaluate()
        self.assertIsInstance(result.native_result.delta, rp.StochasticDividendLsvSpotRisk)
        row = result.report.risk('spot_delta')
        self.assertEqual(row.values, [result.native_result.delta.delta])
        self.assertEqual(row.calibration, 'recalibrated')
        self.assertIsNone(result.native_result.gamma)

    def test_stochastic_dividend_correlations_are_typed_rows(self) -> None:
        M = rp.MeasureRequest
        request = M.correlations(['equity_rate', 'equity_dividend'])
        self.assertEqual((request.measure, request.method), ('correlations', 'native'))
        self.assertEqual(request.correlation_labels, ['equity_rate', 'equity_dividend'])
        for bad in (['equity'], [], ['equity_rate', 'equity_rate'], ['spot_volatility']):
            with self.assertRaises(ValueError):
                M.correlations(bad)
        hw = rp.PricingTask.from_json(fixture('stochastic_dividend_hw'))
        self.assertEqual(hw.available_measures()['correlations'], ['native'])
        bs = rp.PricingTask.from_json(fixture('stochastic_dividend_bs'))
        self.assertEqual(
            bs.measure_capabilities([M.correlations(['equity_rate'])])[0].reason,
            'model_combination')
        lsv = rp.PricingTask.from_json(fixture('stochastic_dividend_lsv'))
        self.assertNotIn('correlations', lsv.available_measures())

        result = hw.with_operation('measures', measures=[request]).compile().evaluate()
        identified = result.native_result.correlations
        self.assertIsInstance(identified, rp.StochasticDividendCorrelationRisk)
        self.assertEqual(identified.correlations,
                         ['equity_dividend', 'equity_rate', 'dividend_rate'])
        risk = identified.risk
        self.assertIsInstance(risk, rp.StochasticDividendAadRisk)
        rows = [r for r in result.report.risks if r.measure == 'correlations']
        self.assertEqual([r.correlation for r in rows], ['equity_rate', 'equity_dividend'])
        for row in rows:
            index = risk.parameter_labels.index(f'{row.correlation}_correlation')
            self.assertEqual(row.values, [risk.derivatives[index]])
            self.assertEqual(row.calibration, 'not_applicable')
        self.assertEqual(result.report.correlation_risk('equity_rate').name,
                         'correlations.equity_rate.native')
        self.assertIsNone(result.report.correlation_risk('dividend_rate'))
        self.assertEqual(
            rp.ValuationReport.from_json(result.report.to_json()).to_json(),
            result.report.to_json())

    def test_stochastic_dividend_cash_is_one_row_on_an_ex_time_axis(self) -> None:
        M = rp.MeasureRequest
        self.assertEqual((M.dividend_cash().measure, M.dividend_cash().method),
                         ('dividend_cash', 'native'))
        bs = rp.PricingTask.from_json(fixture('stochastic_dividend_bs'))
        self.assertEqual(bs.available_measures()['dividend_cash'], ['native'])
        result = bs.with_operation(
            'measures', measures=[M.spot_delta(), M.dividend_cash()]).compile().evaluate()
        native = result.native_result.dividend_cash
        self.assertIsInstance(native, rp.StochasticDividendAadRisk)
        row = result.report.risk('dividend_cash')
        self.assertEqual(row.name, 'dividend_cash.native')
        self.assertEqual(row.axes, [('ex_time', native.cash_times)])
        self.assertEqual(row.values, native.cash_mean_adjoints)
        self.assertEqual(row.calibration, 'not_applicable')

        lsv = rp.PricingTask.from_json(fixture('stochastic_dividend_lsv'))
        self.assertEqual(lsv.available_measures()['dividend_cash'], ['native_calibrated'])
        self.assertEqual(
            lsv.measure_capabilities([M.dividend_cash(calibration='fixed')])[0].reason,
            'no_native_estimator')
        result = lsv.with_operation(
            'measures', measures=[M.dividend_cash(calibration='recalibrated')]
        ).compile().evaluate()
        native = result.native_result.dividend_cash
        self.assertIsInstance(native, rp.StochasticDividendLsvMarketRisk)
        row = result.report.risk('dividend_cash')
        self.assertEqual(row.values, native.cash_mean_adjoints)
        self.assertEqual(row.calibration, 'recalibrated')
        self.assertEqual(
            rp.ValuationReport.from_json(result.report.to_json()).to_json(),
            result.report.to_json())

    def test_stochastic_dividend_model_parameters_are_typed_rows(self) -> None:
        M = rp.MeasureRequest
        request = M.model_parameters(['rate_volatility.0', 'equity_linkage'])
        self.assertEqual(request.parameters, ['rate_volatility.0', 'equity_linkage'])
        for bad in (['rate_volatility'], ['factor_mean_reversion.x']):
            with self.assertRaises(ValueError):
                M.model_parameters(bad)
        hw = rp.PricingTask.from_json(fixture('stochastic_dividend_hw'))
        self.assertEqual(hw.available_measures()['model_parameters'], ['native'])
        bs = rp.PricingTask.from_json(fixture('stochastic_dividend_bs'))
        self.assertEqual(
            bs.measure_capabilities([M.model_parameters(['vol_of_vol'])])[0].reason,
            'model_combination')
        lsv = rp.PricingTask.from_json(fixture('stochastic_dividend_lsv'))
        self.assertNotIn('model_parameters', lsv.available_measures())

        result = hw.with_operation('measures', measures=[request]).compile().evaluate()
        identified = result.native_result.model_parameters
        self.assertIsInstance(identified, rp.StochasticDividendModelParameterRisk)
        self.assertEqual(identified.parameters[:5], [
            'initial_volatility', 'dividend_mean_reversion', 'equity_linkage',
            'dividend_volatility', 'rate_mean_reversion'])
        risk = identified.risk
        rows = [r for r in result.report.risks if r.measure == 'model_parameters']
        self.assertEqual([r.model_parameter for r in rows],
                         ['rate_volatility.0', 'equity_linkage'])
        native = {'rate_volatility.0': 'rate_volatility[0]', 'equity_linkage': 'equity_linkage'}
        for row in rows:
            index = risk.parameter_labels.index(native[row.model_parameter])
            self.assertEqual(row.values, [risk.derivatives[index]])
            self.assertEqual(row.calibration, 'not_applicable')
        self.assertEqual(
            rp.ValuationReport.from_json(result.report.to_json()).to_json(),
            result.report.to_json())

    def test_basket_spot_measures_are_one_row_per_asset_or_pair(self) -> None:
        M = rp.MeasureRequest
        request = M.cross_gamma(relative_bump=0.01)
        self.assertEqual((request.measure, request.method), ('cross_gamma', 'native_spot_bump'))
        with self.assertRaises(ValueError):
            M.cross_gamma(relative_bump=-1.0)
        basket = rp.PricingTask.from_json(fixture('multi_asset_bs'))
        result = basket.with_operation(
            'measures', measures=[M.spot_delta(), request]).compile().evaluate()
        native = result.native_result
        self.assertIsInstance(native, rp.MultiAssetPrice)
        deltas = [r for r in result.report.risks if r.measure == 'spot_delta']
        self.assertEqual([r.underlying_ids for r in deltas],
                         [[i] for i in native.underlying_ids])
        self.assertEqual([r.values for r in deltas],
                         [[risk.delta.value] for risk in native.risks])
        gammas = [r for r in result.report.risks if r.measure == 'cross_gamma']
        self.assertEqual(len(gammas), 4)
        self.assertEqual(gammas[1].values, [native.gamma[0][1].value])
        self.assertEqual(gammas[1].underlying_ids, native.underlying_ids)
        self.assertEqual({r.calibration for r in gammas}, {'not_applicable'})

        joint = rp.PricingTask.from_json(fixture('multi_asset_joint'))
        self.assertEqual(
            [c.reason for c in joint.measure_capabilities(
                [M.spot_delta(), M.spot_delta(calibration='fixed')])],
            ['bump_calibration_mismatch', 'fixed_calibration_unavailable'])
        result = joint.with_operation(
            'measures', measures=[M.spot_delta(calibration='recalibrated')]
        ).compile().evaluate()
        rows = [r for r in result.report.risks if r.measure == 'spot_delta']
        self.assertEqual({r.calibration for r in rows}, {'recalibrated'})
        self.assertEqual(
            rp.ValuationReport.from_json(result.report.to_json()).to_json(),
            result.report.to_json())

    def test_discount_curve_is_log_df_and_dv01_rows(self) -> None:
        M = rp.MeasureRequest
        self.assertEqual((M.discount_curve().measure, M.discount_curve().method),
                         ('discount_curve', 'native'))
        hw = rp.PricingTask.from_json(fixture('stochastic_dividend_hw'))
        self.assertEqual(hw.available_measures()['discount_curve'], ['native'])
        result = hw.with_operation(
            'measures', measures=[M.spot_delta(), M.discount_curve()]).compile().evaluate()
        native = result.native_result.discount_curve
        self.assertIsInstance(native, rp.StochasticDividendAadRisk)
        rows = [r for r in result.report.risks if r.measure == 'discount_curve']
        self.assertEqual([r.name for r in rows],
                         ['discount_curve.log_df.native', 'discount_curve.dv01.native'])
        self.assertEqual(rows[0].axes, [('discount_time', native.discount_times)])
        self.assertEqual(rows[0].values, [
            v for label, v in zip(native.parameter_labels, native.derivatives)
            if label.startswith('discount_log_df[')])
        self.assertEqual(rows[1].values, native.discount_node_dv01)
        self.assertEqual(rows[1].unit, 'currency_per_basis_point')

        lsv = rp.PricingTask.from_json(fixture('stochastic_dividend_lsv'))
        result = lsv.with_operation(
            'measures', measures=[M.discount_curve(calibration='recalibrated')]
        ).compile().evaluate()
        native = result.native_result.discount_curve
        self.assertIsInstance(native, rp.StochasticDividendLsvMarketRisk)
        rows = [r for r in result.report.risks if r.measure == 'discount_curve']
        self.assertEqual(rows[0].values, native.discount_log_df_adjoints)
        self.assertEqual(rows[1].values, native.discount_node_dv01)
        self.assertEqual({r.calibration for r in rows}, {'recalibrated'})

        basket = rp.PricingTask.from_json(fixture('multi_asset_bs'))
        self.assertEqual(
            basket.measure_capabilities([M.discount_curve()])[0].reason,
            'no_native_estimator')
        joint = rp.PricingTask.from_json(fixture('multi_asset_joint'))
        result = joint.with_operation(
            'measures', measures=[M.discount_curve(calibration='recalibrated')]
        ).compile().evaluate()
        curve = result.native_result.hull_white_curve_risk
        rows = [r for r in result.report.risks if r.measure == 'discount_curve']
        self.assertEqual(rows[0].values, [e.value for e in curve.discount_log_df_adjoints])
        self.assertEqual(rows[1].values, [e.value for e in curve.discount_node_dv01])
        self.assertEqual(
            rp.ValuationReport.from_json(result.report.to_json()).to_json(),
            result.report.to_json())

    def test_basket_asset_pair_correlation_is_bump_and_revalue(self) -> None:
        M = rp.MeasureRequest
        request = M.correlations(['asset_pair.1.2'], absolute_bump=0.01,
                                 calibration='not_applicable')
        self.assertEqual((request.measure, request.method, request.correlation_labels),
                         ('correlations', 'bump_and_revalue', ['asset_pair.1.2']))
        for bad in (dict(absolute_bump=0.01), dict(calibration='not_applicable')):
            with self.assertRaises(ValueError):
                M.correlations(['asset_pair.1.2'], **bad)
        with self.assertRaises(ValueError):
            M.correlations(['asset_pair.2.1'])
        basket = rp.PricingTask.from_json(fixture('multi_asset_bs'))
        self.assertEqual(
            basket.measure_capabilities([M.correlations(['asset_pair.1.2'])])[0].reason,
            'no_native_estimator')
        result = basket.with_operation('measures', measures=[request]).compile().evaluate()
        row = result.report.correlation_risk('asset_pair.1.2')
        self.assertEqual(row.name, 'correlations.asset_pair.1.2.bump_and_revalue')
        self.assertEqual((row.underlying_ids, row.correlation), ([1, 2], 'asset_pair.1.2'))
        self.assertEqual(row.unit, 'currency_per_unit_parameter')
        joint = rp.PricingTask.from_json(fixture('multi_asset_joint'))
        self.assertEqual(
            joint.measure_capabilities([M.correlations(
                ['asset_pair.1.2'], absolute_bump=0.01, calibration='recalibrated')])[0].reason,
            'model_combination')
        self.assertEqual(
            rp.ValuationReport.from_json(result.report.to_json()).to_json(),
            result.report.to_json())

    def test_source_family_and_operation_discovery(self):
        expected = {
            'standard_risk': ('standard_single_asset', ['conditional', 'unsupported', 'unsupported']),
            'multi_asset_bs': ('multi_asset', ['conditional', 'conditional', 'unsupported']),
            'stochastic_dividend_bs': ('stochastic_dividend', ['conditional', 'conditional', 'unsupported']),
            'stochastic_dividend_lsv': ('stochastic_dividend_lsv', ['conditional', 'unsupported', 'conditional']),
        }
        for name, (family, statuses) in expected.items():
            with self.subTest(name=name):
                task = rp.PricingTask.from_json(fixture(name))
                self.assertEqual(task.source_family, family)
                listed = task.available_operations()
                self.assertEqual([c.operation for c in listed], ['evaluate', 'aad', 'local_variance_risk'])
                self.assertEqual([c.status for c in listed], statuses)
                for capability in listed:
                    direct = task.capability(capability.operation)
                    self.assertEqual((capability.status, capability.reason, capability.detail),
                                     (direct.status, direct.reason, direct.detail))

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
            ('pricing_task', original, rp.PricingTask, [(['schema_version'], 99),
                (['unknown'], 1), (['source', 'numerics', 'maximum_step'], 0),
                (['source', 'lsv', 0], None)]),
            ('valuation_report', report, rp.ValuationReport, [(['schema_version'], 4),
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
        for name, export in [('task-v3/pricing_task', rp.pricing_task_json_schema),
                             ('report-v3/valuation_report', rp.valuation_report_json_schema)]:
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
