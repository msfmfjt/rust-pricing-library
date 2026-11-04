"""Scale-invariant division AAD through the native graph/task boundary."""
import json
from pathlib import Path
import unittest

import rust_pricing as rp
import test_multi_asset

ROOT = Path(__file__).resolve().parents[2]


class GraphDivisionTests(unittest.TestCase):
    def reciprocal(self, scale, observation, engine, workers):
        b = rp.SourceGraphBuilder()
        spot = b.spot(1, observation)
        c = b.literal(scale)
        amount = b.divide(c, b.multiply(spot, c))
        graph = b.finish([amount], [observation], currency_id=2, underlying_ids=[1])
        r = json.loads((ROOT / 'fixtures/specification-v1/black_scholes.golden.json').read_text())
        r['schema_version'] = 2
        r['product'] = {'type': 'graph', 'graph': json.loads(graph.to_json())['product']}
        r['risk']['delta'] = True
        r['risk']['vega'] = True
        r['engine'] = engine
        r['numerics']['worker_threads'] = workers
        return rp.PricingTask.from_specification(rp.PricingSpecification.from_json(json.dumps(r)))

    def test_reciprocal_delta_and_exact_sampling_replay_across_scales(self):
        engines = [
            dict(type='pseudo_monte_carlo', master_seed=7, independent_sampling_units=64),
            dict(type='randomized_quasi_monte_carlo', master_scramble_seed=7, points_per_scramble=32,
                 scramble_count=4),
        ]
        for engine in engines:
            engine['variance_reduction'] = dict(antithetic=True, brownian_bridge=False)
            for observation in ('2026-09-04', '2027-09-04'):
                expected = None
                for scale in (1., 2.**600, 2.**-600, 1e200, 1e-200):
                    for workers in (1, 3):
                        with self.subTest(engine=engine['type'], observation=observation,
                                          scale=scale, workers=workers):
                            task = self.reciprocal(scale, observation, engine, workers)
                            compiled = task.compile()
                            report = json.loads(compiled.evaluate().report.to_json())
                            delta = next(r for r in report['risks'] if r['name'] == 'delta.raw')
                            self.assertLess(delta['values'][0], 0.)
                            if observation == '2026-09-04':
                                self.assertAlmostEqual(report['value'], .01, delta=2e-18)
                                self.assertAlmostEqual(delta['values'][0], -.0001, delta=3e-20)
                                self.assertEqual(delta['standard_errors'], [0.])
                            # Binary rescaling is exact: preserve price, all risk
                            # blocks and sampling errors, including future paths.
                            numerical = {key: report[key] for key in ('value', 'standard_error', 'risks')}
                            if scale == 1. and workers == 1:
                                expected = numerical
                            if scale in (1., 2.**600, 2.**-600):
                                self.assertEqual(numerical, expected)
                            replay = rp.PricingTask.from_json(task.to_json()).compile()
                            self.assertEqual(replay.evaluate().report.to_json(), compiled.evaluate().report.to_json())

    def test_basket_ratio_reselection_preserves_both_deltas_and_cross_gamma(self):
        helper = test_multi_asset.MultiAssetTests()
        reference = None
        for scale in (1., 2.**600, 2.**-600):
            b = rp.SourceGraphBuilder()
            x, y = b.spot(1, '2027-01-01'), b.spot(2, '2027-01-01')
            c = b.literal(scale)
            ratio = b.divide(b.multiply(x, c), b.multiply(y, c))
            graph = b.finish([ratio], ['2027-01-01'], currency_id=1, underlying_ids=[1, 2])
            source = rp.MultiAssetSpecification('2026-01-01', graph, helper.markets(),
                rp.MultiAssetModelSpec([rp.Model.black_scholes(.2), rp.Model.black_scholes(.3)],
                                      helper.correlation()),
                rp.Engine.randomized_quasi_monte_carlo(32, 702, scramble_count=4, antithetic=True),
                maximum_step=.25, worker_threads=2, reduction_block_size=32)
            price = rp.PricingTask.from_source(source).compile()
            risk = price.with_operation('aad', gamma_relative_bump=.001)
            result = risk.evaluate()
            self.assertEqual(risk.plan_fingerprint, price.plan_fingerprint)
            self.assertEqual(result.report.to_json(), risk.task.compile().evaluate().report.to_json())
            self.assertGreater(result.native_result.risks[0].delta.value, 0.)
            self.assertLess(result.native_result.risks[1].delta.value, 0.)
            # Each simulated ratio is homogeneous: dP/dS1=P/S1,
            # dP/dS2=-P/S2, independently of the normal draws.
            self.assertAlmostEqual(result.native_result.risks[0].delta.value,
                                   result.report.value / 100., delta=4e-17)
            self.assertAlmostEqual(result.native_result.risks[1].delta.value,
                                   -result.report.value / 90., delta=4e-17)
            numerical = json.loads(result.report.to_json())
            for key in ('task_fingerprint', 'plan_fingerprint'):
                numerical.pop(key)
            if reference is None:
                reference = numerical
            self.assertEqual(numerical, reference)


if __name__ == '__main__':
    unittest.main()
