"""Compact smoothing remains exact outside its transition and replays via tasks."""
import json
from pathlib import Path
import unittest

import rust_pricing as rp
import test_multi_asset

ROOT = Path(__file__).resolve().parents[2]


def task(graph, engine, workers):
    source = json.loads((ROOT / 'fixtures/specification-v1/black_scholes.golden.json').read_text())
    source['schema_version'] = 2
    source['market']['spot'] = 101.
    source['product'] = {'type': 'graph', 'graph': json.loads(graph.to_json())['product']}
    source['risk']['delta'] = True
    source['risk']['vega'] = True
    source['engine'] = engine
    source['numerics']['worker_threads'] = workers
    return rp.PricingTask.from_specification(rp.PricingSpecification.from_json(json.dumps(source)))


def graph(observation, operation=None, reverse=False):
    builder = rp.SourceGraphBuilder()
    spot = builder.spot(1, observation)
    if operation is None:
        output = spot
    else:
        bound = builder.literal(1e16 if operation == 'minimum' else -1e16)
        left, right = (bound, spot) if reverse else (spot, bound)
        output = getattr(builder, operation)(left, right, smoothing_half_width=2.)
    return builder.finish([output], [observation], currency_id=2, underlying_ids=[1])


class GraphSmoothingRangeTests(unittest.TestCase):
    def test_saturated_payoffs_match_the_uncapped_spot_for_both_orders_and_engines(self):
        engines = [
            dict(type='pseudo_monte_carlo', master_seed=7, independent_sampling_units=64),
            dict(type='randomized_quasi_monte_carlo', master_scramble_seed=7,
                 points_per_scramble=32, scramble_count=4),
        ]
        for engine in engines:
            engine['variance_reduction'] = dict(antithetic=True, brownian_bridge=False)
            for observation in ('2026-09-04', '2027-09-04'):
                expected = json.loads(task(graph(observation), engine, 1).compile().evaluate().report.to_json())
                for operation in ('minimum', 'maximum'):
                    for reverse in (False, True):
                        for workers in (1, 3):
                            with self.subTest(engine=engine['type'], observation=observation,
                                              operation=operation, reverse=reverse, workers=workers):
                                original = task(graph(observation, operation, reverse), engine, workers)
                                result = original.compile().evaluate().report.to_json()
                                actual = json.loads(result)
                                for field in ('value', 'standard_error', 'risks'):
                                    self.assertEqual(actual[field], expected[field])
                                replay = rp.PricingTask.from_json(original.to_json()).compile()
                                self.assertEqual(replay.evaluate().report.to_json(), result)

    def test_wide_graph_smoothing_has_finite_price_and_aad(self):
        engine = dict(type='pseudo_monte_carlo', master_seed=7, independent_sampling_units=64,
                      variance_reduction=dict(antithetic=True, brownian_bridge=False))
        for width in (2.**1023, float.fromhex('0x1.fffffffffffffp+1023')):
            for operation in ('indicator', 'positive_part'):
                builder = rp.SourceGraphBuilder()
                zero_at_valuation = builder.subtract(builder.spot(1, '2026-09-04'), builder.literal(101.))
                if operation == 'indicator':
                    output = builder.indicator(zero_at_valuation, smoothing_half_width=width)
                    expected_price, expected_delta = .5, .9375 / width
                else:
                    positive = builder.maximum(zero_at_valuation, builder.literal(0.), smoothing_half_width=width)
                    output = builder.divide(positive, builder.literal(width))
                    expected_price, expected_delta = .15625, .5 / width
                source = builder.finish([output], ['2026-09-04'], currency_id=2, underlying_ids=[1])
                result = json.loads(task(source, engine, 1).compile().evaluate().report.to_json())
                with self.subTest(width=width, operation=operation):
                    self.assertEqual(result['value'], expected_price)
                    self.assertEqual(result['standard_error'], 0.)
                    delta = next(r for r in result['risks'] if r['name'] == 'delta.raw')
                    # Mean reduction may round a subnormal derivative; retain
                    # a one-subnormal-ULP bound, never an absolute normal epsilon.
                    self.assertAlmostEqual(delta['values'][0], expected_delta,
                                           delta=float.fromhex('0x0.0000000000001p-1022'))

    def test_basket_aad_and_gamma_reselection_match_the_linear_source(self):
        helper = test_multi_asset.MultiAssetTests()
        expected = None
        for bound in (None, 'minimum', 'maximum'):
            builder = rp.SourceGraphBuilder()
            amount = builder.add(builder.spot(1, '2027-01-01'), builder.spot(2, '2027-01-01'))
            if bound == 'minimum':
                amount = builder.minimum(builder.literal(1e16), amount, smoothing_half_width=2.)
            elif bound == 'maximum':
                amount = builder.maximum(amount, builder.literal(-1e16), smoothing_half_width=2.)
            source = rp.MultiAssetSpecification('2026-01-01',
                builder.finish([amount], ['2027-01-01'], currency_id=1, underlying_ids=[1, 2]),
                helper.markets(),
                rp.MultiAssetModelSpec([rp.Model.black_scholes(.2), rp.Model.black_scholes(.3)],
                                      helper.correlation()),
                rp.Engine.randomized_quasi_monte_carlo(32, 702, scramble_count=4, antithetic=True),
                maximum_step=.25, worker_threads=2, reduction_block_size=32)
            compiled = rp.PricingTask.from_source(source).compile()
            risk = compiled.with_operation('aad', gamma_relative_bump=.001)
            result = risk.evaluate().report.to_json()
            self.assertEqual(risk.plan_fingerprint, compiled.plan_fingerprint)
            self.assertEqual(result, risk.task.compile().evaluate().report.to_json())
            numerical = json.loads(result)
            for key in ('task_fingerprint', 'plan_fingerprint'):
                numerical.pop(key)
            if expected is None:
                expected = numerical
            self.assertEqual(numerical, expected)


if __name__ == '__main__':
    unittest.main()
