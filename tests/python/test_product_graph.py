"""Public arbitrary-product construction, ownership, replay and pricing parity."""
import copy
from datetime import date
import json
import math
from pathlib import Path
import unittest
import rust_pricing as rp
import test_multi_asset

ROOT = Path(__file__).resolve().parents[2]

class ProductGraphTests(unittest.TestCase):
    def vanilla(self):
        b = rp.SourceGraphBuilder()
        spot = b.spot(1, '2027-09-04')
        strike, zero = b.literal(100.), b.literal(0.)
        payoff = b.maximum(b.subtract(spot, strike), zero)
        amount = b.multiply(payoff, b.literal(1.))
        return b, b.finish([amount], ['2027-09-04'], currency_id=2, underlying_ids=[1])

    def source(self, graph):
        source = json.loads((ROOT / 'fixtures/specification-v1/black_scholes.golden.json').read_text())
        source['schema_version'] = 2
        source['product'] = {'type': 'graph', 'graph': json.loads(graph.to_json())['product']}
        return rp.PricingSpecification.from_json(json.dumps(source))

    def test_graph_builder_integrates_with_request_and_specification(self):
        b, graph = self.vanilla()
        native = rp.PricingRequest.from_json((ROOT / 'fixtures/v3/pricing_request.golden.json').read_text())
        curve = rp.DiscountCurve(10, [0., 1.], [1., .95])
        repo = rp.DiscountCurve(11, [0., 1.], [1., .98])
        request = rp.PricingRequest('2026-09-04', rp.Product.from_graph(graph),
            rp.Market.equity(2, 1, 100., curve, repo), rp.Model.black_scholes(.2),
            rp.Engine.pseudo_monte_carlo(7, 1024, antithetic=True), rp.RiskRequest())
        a = rp.PricingPlan.compile(native, worker_threads=2, reduction_block_size=32).evaluate()
        source = rp.PricingSpecification.from_request(request, worker_threads=2, reduction_block_size=32)
        c = source.compile().evaluate()
        self.assertEqual(c.value, a.value)
        self.assertEqual(c.standard_error, a.standard_error)
        self.assertEqual(json.loads(source.to_json())['schema_version'], 2)
        replay = rp.PricingSpecification.from_json(source.to_json())
        self.assertEqual(source.fingerprint, replay.fingerprint)
        self.assertEqual(replay.compile().evaluate().value, c.value)
        with self.assertRaises(rp.PricingError):
            request.to_json()
        frozen = graph.to_json()
        b.literal(123.)
        self.assertEqual(frozen, graph.to_json())
        with self.assertRaises(AttributeError):
            graph.currency_id = 10
        ids = graph.underlying_ids
        ids[0] = 999
        self.assertEqual(graph.underlying_ids, [1])

    def test_all_graph_primitives_round_trip(self):
        b = rp.SourceGraphBuilder()
        spot = b.spot(1, '2027-09-04')
        pre = b.spot(1, '2027-09-04', pre_dividend=True)
        one, two = b.literal(1.), b.literal(2.)
        values = [b.add(spot, pre), b.subtract(spot, one), b.multiply(spot, two),
                  b.divide(spot, two), b.negate(spot), b.minimum(spot, two),
                  b.maximum(spot, two), b.indicator(spot),
                  b.minimum(spot, two, smoothing_half_width=.1),
                  b.maximum(spot, two, smoothing_half_width=.1),
                  b.indicator(spot, smoothing_half_width=.1), b.barrier_hit_state(one, one)]
        graph = b.finish(values, ['2027-09-04'] * len(values), currency_id=2, underlying_ids=[1])
        restored = rp.GraphProduct.from_json(graph.to_pretty_json())
        self.assertEqual(graph.to_json(), restored.to_json())
        self.assertEqual(graph.fingerprint, restored.fingerprint)
        self.assertEqual(graph.currency_id, 2)
        self.assertEqual(len(graph.payment_dates), len(values))

    def test_multi_asset_existing_builders_export_to_generic_source_and_replay(self):
        helper = test_multi_asset.MultiAssetTests()
        basket = rp.MultiAssetProduct.basket([1, 2], [.5, .5], [1., 1.], 'call', 95.,
            '2027-01-01', '2027-01-01', smoothing_half_width=2.)
        graph = basket.to_graph()
        generic = rp.MultiAssetProduct.from_graph(rp.GraphProduct.from_json(graph.to_json()))
        original, replay = helper.plan(basket), helper.plan(generic)
        self.assertEqual(original.fingerprint, replay.fingerprint)
        a, b = original.evaluate_aad(), replay.evaluate_aad()
        self.assertEqual(a.value, b.value)
        self.assertEqual(a.standard_error, b.standard_error)
        self.assertEqual([r.delta.value for r in a.risks], [r.delta.value for r in b.risks])
        self.assertEqual(generic.underlying_ids, [1, 2])

    def test_multi_asset_stochastic_rate_cashflows_replay(self):
        helper = test_multi_asset.MultiAssetTests()
        observations = [rp.AutocallObservation(t, t, coupon_amount=5.,
                        coupon_level=.9, call_level=1.05)
                        for t in ['2026-07-01', '2027-01-01']]
        native = rp.MultiAssetProduct.autocallable([1, 2], [100., 90.], observations,
            '2027-01-01', '2027-01-01', notional=100., final_barrier=.7,
            memory=True, on_autocall='pay', on_maturity='forfeit', smoothing_half_width=.05)
        graph = rp.GraphProduct.from_json(native.to_graph().to_json())
        def plan(product):
            return rp.MultiAssetPlan.compile('2026-01-01', product, helper.markets(),
                [rp.Model.black_scholes(.2), rp.Model.black_scholes(.3)], helper.correlation(),
                rp.Engine.pseudo_monte_carlo(12, 128, antithetic=True), maximum_step=.25,
                rate_model=rp.HullWhiteModel(.1, [0.], [.01]), rate_correlations=[.1, .2])
        original, replay = plan(native), plan(rp.MultiAssetProduct.from_graph(graph))
        self.assertEqual(original.fingerprint, replay.fingerprint)
        self.assertEqual(original.evaluate().value, replay.evaluate().value)
        self.assertEqual(original.evaluate().standard_error, replay.evaluate().standard_error)

    def test_pre_dividend_observation_preserves_cash_jump(self):
        b = rp.SourceGraphBuilder()
        pre = b.spot(1, '2027-09-04', pre_dividend=True)
        post = b.spot(1, '2027-09-04')
        amount = b.subtract(pre, post)
        graph = b.finish([amount], ['2027-09-04'], currency_id=2, underlying_ids=[1])
        source = json.loads(self.source(graph).to_json())
        source['market']['discrete_dividends'] = [
            dict(event_id=1, ex_time=1., quote=dict(type='fixed_cash', amount=10.))]
        source['risk']['delta'] = True
        source['risk']['vega'] = True
        result = rp.PricingSpecification.from_json(json.dumps(source)).compile().evaluate()
        self.assertAlmostEqual(result.value, 9.5, places=12)
        self.assertAlmostEqual(result.standard_error, 0., places=12)

    def test_multi_cashflow_roundtrip_and_validation(self):
        _, graph = self.vanilla()
        value = json.loads(graph.to_json())
        duplicate = copy.deepcopy(value)
        duplicate['product']['cashflows'].append(copy.deepcopy(duplicate['product']['cashflows'][0]))
        doubled = rp.GraphProduct.from_json(json.dumps(duplicate))
        self.assertAlmostEqual(self.source(doubled).compile().evaluate().value,
                               2. * self.source(graph).compile().evaluate().value, places=12)
        for kind in ('unknown', 'cycle', 'future', 'missing'):
            bad = copy.deepcopy(value)
            if kind == 'unknown':
                bad['product']['nodes'][0]['ignored'] = 1
            elif kind == 'cycle':
                bad['product']['nodes'][0]['operation'] = {'type': 'negate', 'input': 0}
            elif kind == 'future':
                bad['product']['cashflows'][0]['payment_date'] = '2026-09-04'
            else:
                bad['product']['cashflows'][0]['output_node'] = 999
            with self.subTest(kind=kind), self.assertRaises(rp.ValidationError) as caught:
                rp.GraphProduct.from_json(json.dumps(bad))
            self.assertEqual(caught.exception.issues[0].document_kind, 'product_graph')
        b = rp.SourceGraphBuilder()
        with self.assertRaises(rp.ValidationError):
            b.literal(float('nan'))
        one = b.literal(1.)
        with self.assertRaises(rp.ValidationError):
            b.finish([one], [], currency_id=1, underlying_ids=[1])
        with self.assertRaises(rp.ValidationError):
            b.maximum(one, one, smoothing_half_width=0.)
        for invalid_date in ['2027-02-30', 42]:
            with self.assertRaises(rp.ValidationError) as caught:
                b.spot(1, invalid_date)
            issue = caught.exception.issues[0]
            self.assertEqual((issue.document_kind, issue.schema_version), ('product_graph', 1))
            self.assertEqual(issue.pointer, '/product/observation_date')
        self.assertEqual(json.loads(rp.product_graph_json_schema()),
                         json.loads((ROOT / 'schemas/product-graph-v1/product_graph.schema.json').read_text()))


    def test_multiple_payment_curve_aad_includes_discount_ratio_pullback(self):
        from test_stochastic_dividends import compile_plan as dividends
        from test_stochastic_dividend_hull_white import compile_plan as dividend_hw
        for family in ('sv', 'hw', 'lsv', 'dividends', 'dividend_hw'):
            for qmc in (False, True):
                # LSV calibration requires a live observation and a grid ending
                # at that observation; the other engines also cover constants.
                for observed in ((True,) if family == 'lsv' else (False, True)):
                    with self.subTest(family=family, qmc=qmc, observed=observed):
                        b = rp.SourceGraphBuilder()
                        first = b.literal(-3.)
                        amount = (b.multiply(b.spot(1, '2027-03-05'), b.literal(.1))
                                  if observed else b.literal(10.))
                        # Repeated output, payment at valuation, interpolation and
                        # extrapolation beyond the last discount-curve pillar.
                        graph = b.finish([first, amount, amount],
                            ['2026-09-04', '2027-03-05', '2028-03-04'],
                            currency_id=2, underlying_ids=[1])
                        def plan(pillar=None, shift=0., workers=1):
                            dfs = [1., .985, .95]
                            if pillar is not None:
                                dfs[pillar] *= math.exp(shift)
                            target = (rp.HullWhiteLsvTarget.flat(.2,
                                [0., .2, .25, (date(2027, 3, 5) - date(2026, 9, 4)).days / 365.],
                                [-1., 0., 1.]) if family == 'lsv' else None)
                            request = rp.PricingRequest('2026-09-04', rp.Product.from_graph(graph),
                                rp.Market.equity(2, 1, 100.,
                                    rp.DiscountCurve(10, [0., .25, 1.], dfs),
                                    rp.DiscountCurve(11, [0., 1.], [1., .98]),
                                    discrete_dividends=[rp.DividendEvent.fixed_cash(1, .2, 4.),
                                                       rp.DividendEvent.fixed_cash(2, 1.4, 3.)]),
                                target.model if family == 'lsv' else rp.Model.black_scholes(.2),
                                (rp.Engine.randomized_quasi_monte_carlo(16, 612, scramble_count=4,
                                     antithetic=True, brownian_bridge=True) if qmc else
                                 rp.Engine.pseudo_monte_carlo(612, 32, antithetic=True)), rp.RiskRequest())
                            common = dict(worker_threads=workers, reduction_block_size=16)
                            rates = rp.HullWhiteModel(.2, [0.], [0.])
                            if family == 'sv':
                                return rp.StochasticVolatilityPlan.compile_bergomi(request,
                                    mean_reversion=.7, vol_of_vol=.3, correlation=-.4,
                                    maximum_step=.25, **common)
                            if family == 'hw':
                                return rp.HullWhiteEquityPlan.compile_bs(request, rates,
                                    equity_rate_correlation=0., maximum_step=.25, **common)
                            if family == 'lsv':
                                return rp.HullWhiteEquityPlan.compile_lsv(request, target, rates,
                                    vol_mean_reversion=.7, vol_of_vol=.2, equity_vol_correlation=-.4,
                                    equity_rate_correlation=0., vol_rate_correlation=0., particle_count=64,
                                    calibration_seed=91, log_bandwidth=.5, minimum_effective_samples=2.,
                                    retain_reverse_trace=True, **common)
                            if family == 'dividends':
                                return dividends(request, maximum_step=.25, **common)
                            return dividend_hw(request, rate_volatilities=[0.], maximum_step=.25, **common)
                        compiled = plan()
                        risk = compiled.evaluate_aad()
                        self.assertEqual(risk.price.value, compiled.evaluate().value)
                        for pillar in (1, 2):
                            index = risk.parameter_labels.index(f'discount_log_df[{pillar}]')
                            for h in (1e-5, 1e-6):
                                fd = (plan(pillar, h).evaluate().value - plan(pillar, -h).evaluate().value)/(2*h)
                                self.assertAlmostEqual(risk.derivatives[index], fd, delta=2e-6)
                        replay = plan(workers=3).evaluate_aad()
                        self.assertEqual(risk.derivatives, replay.derivatives)
                        self.assertEqual(risk.standard_errors, replay.standard_errors)
                        if not observed and risk.standard_errors is not None:
                            self.assertTrue(all(abs(e) < 1e-10 for e in risk.standard_errors))

if __name__ == '__main__':
    unittest.main()
