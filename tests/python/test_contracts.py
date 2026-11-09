"""Rust-owned contract source, lifecycle and native MC interoperability."""
import copy
from datetime import date
import json
from pathlib import Path
import unittest
import rust_pricing as rp
ROOT = Path(__file__).resolve().parents[2]
EXPIRY = '2027-09-04'


def builder(name='option'):
    return rp.ContractBuilder(name, 2, 1, 'exchange.close')


def template(risk=False):
    v = json.loads((ROOT / 'fixtures/v3/pricing_request.golden.json').read_text())
    v['engine']['independent_sampling_units'] = 256
    v['risk']['delta'] = risk
    v['risk']['vega'] = risk
    return rp.PricingSpecification.from_request(rp.PricingRequest.from_json(json.dumps(v)),
        worker_threads=2, reduction_block_size=32)


class ContractTests(unittest.TestCase):
    def test_builder_json_schema_schedule_and_immutable_source(self):
        c = builder().european('call', 100., 1., date(2027, 9, 4), EXPIRY)
        restored = rp.Contract.from_json(c.to_json())
        self.assertEqual(restored.fingerprint, c.fingerprint)
        self.assertEqual(json.loads(c.observations_json())[0]['fixing_source'], 'exchange.close')
        a = json.loads(c.analysis_json())
        self.assertEqual(a['required_observations'][0]['date'], EXPIRY)
        self.assertEqual(a['currency_ids'], [2])
        self.assertEqual(a['horizon'], EXPIRY)
        self.assertEqual(json.loads(rp.Contract.json_schema()), json.loads((ROOT / 'schemas/contract-v1/contract.schema.json').read_text()))
        with self.assertRaises(AttributeError): c.fingerprint = 'changed'
        bad = json.loads(c.to_json())
        bad['root']['cashflow']['amount'] = {'type': 'constant', 'value': True}
        with self.assertRaises(ValueError): rp.Contract.from_json(json.dumps(bad))
        with self.assertRaises(ValueError): rp.Contract.from_json(c.to_json().replace('"schema_version":1', '"schema_version":99'))
        with self.assertRaises(ValueError): builder().forward(1., float('nan'), EXPIRY, EXPIRY)

    def test_native_mc_and_greeks_exact_parity_with_task_replay(self):
        source = template(True)
        c = builder().european('call', 100., 1., EXPIRY, EXPIRY)
        expected = rp.PricingTask.from_specification(source).compile().evaluate().report
        task = c.pricing_task(source)
        actual = task.compile().evaluate().report
        self.assertEqual((actual.value, actual.standard_error), (expected.value, expected.standard_error))
        self.assertEqual([(r.name, r.values, r.standard_errors) for r in actual.risks],
                         [(r.name, r.values, r.standard_errors) for r in expected.risks])
        self.assertEqual(actual.to_json(), rp.PricingTask.from_json(task.to_json()).compile().evaluate().report.to_json())
        graph = c.to_graphs('2026-09-04', [1])[0]
        self.assertEqual(graph.currency_id, 2)
        self.assertEqual(graph.payment_dates, [EXPIRY])

    def test_lifecycle_fixing_idempotency_checkpoint_and_explicit_settlement(self):
        c = builder().forward(100., 2., EXPIRY, '2027-09-06')
        o = json.loads(c.observations_json())[0]
        event = json.loads(c.fixing_event_json('fix', o['id'], 120.))
        r = c.advance(json.dumps([event, event]), EXPIRY)
        self.assertEqual(r.applied_event_ids, ['fix'])
        self.assertEqual(json.loads(r.determined_cashflows_json())[0]['amount'], 40.)
        self.assertEqual(json.loads(r.remaining_observations_json()), [])
        restored = rp.ContractLifecycle.from_json(r.to_json())
        self.assertEqual(restored.residual_contract.fingerprint, r.residual_contract.fingerprint)
        outstanding = restored.advance('[]', '2027-09-07')
        self.assertEqual(len(json.loads(outstanding.outstanding_payments_json())), 1)
        settlement = {'id': 'paid', 'effective_date': '2027-09-06',
                      'event': {'type': 'settlement', 'cashflow_id': 'option/forward'}}
        settled = restored.advance(json.dumps([settlement]), '2027-09-07')
        self.assertEqual(json.loads(settled.outstanding_payments_json()), [])
        with self.assertRaises(ValueError) as error: c.advance('[]', '2027-09-05')
        self.assertEqual(error.exception.code, 'missing_fixing')
        conflict = copy.deepcopy(event)
        conflict['event']['value'] = 121.
        with self.assertRaises(ValueError): c.advance(json.dumps([event, conflict]), EXPIRY)
        future = c.advance(json.dumps([event]), '2026-09-04')
        self.assertEqual(future.applied_event_ids, [])

    def test_exotic_builders_and_historical_rate_fixings(self):
        dates = ['2027-03-04', EXPIRY]
        b = builder('exotic')
        for aggregate in ('arithmetic_average', 'geometric_average', 'running_minimum', 'running_maximum'):
            c = b.asian(aggregate, 'call', 80., 1., dates, EXPIRY)
            events = [json.loads(c.fixing_event_json(str(i), o['id'], v))
                      for i, (o, v) in enumerate(zip(json.loads(c.observations_json()), [80., 125.]))]
            result = c.advance(json.dumps(events), EXPIRY)
            expected = {'arithmetic_average': 22.5, 'geometric_average': 20., 'running_minimum': 0., 'running_maximum': 45.}[aggregate]
            self.assertAlmostEqual(json.loads(result.determined_cashflows_json())[0]['amount'], expected)
        barrier = b.discrete_barrier('call', 90., 1., EXPIRY, EXPIRY, dates, 'up', 'knock_out', 120., 3.)
        o = json.loads(barrier.observations_json())[0]
        result = barrier.advance('[' + barrier.fixing_event_json('hit', o['id'], 120.) + ']', dates[0])
        self.assertEqual(json.loads(result.barrier_states_json()), {'exotic/barrier': True})
        self.assertEqual(json.loads(result.remaining_observations_json()), [])
        swap = b.swap(1000., .03, [EXPIRY], ['2028-03-04'], [.5])
        with self.assertRaises(ValueError): swap.pricing_task(template())
        o = json.loads(swap.observations_json())[0]
        result = swap.advance('[' + swap.fixing_event_json('rate', o['id'], .05) + ']', EXPIRY)
        self.assertEqual(sum(c['amount'] for c in json.loads(result.determined_cashflows_json())), 10.)

    def test_autocall_cancellation_and_bundle_currency_guard(self):
        dates = ['2027-03-04', EXPIRY]
        c = builder('autocall').autocall(100., 100., 70., dates, dates, [100.] * 2, [80.] * 2, [.05] * 2)
        o = json.loads(c.observations_json())[0]
        r = c.advance('[' + c.fixing_event_json('called', o['id'], 100.) + ']', dates[0])
        self.assertEqual(sum(x['amount'] for x in json.loads(r.outstanding_payments_json())), 105.)
        self.assertEqual(json.loads(r.remaining_observations_json()), [])
        a = builder('a').forward(100., 1., EXPIRY, EXPIRY)
        b = rp.ContractBuilder('b', 3, 1, 'exchange.close').forward(100., 1., EXPIRY, EXPIRY)
        combined = rp.Contract.bundle('two_currencies', [a, b])
        self.assertEqual([g.currency_id for g in combined.to_graphs('2026-09-04', [1])], [2, 3])
        with self.assertRaises(ValueError): combined.pricing_task(template())
        reverse = a.reversed('reverse').scaled('scaled', 2.)
        self.assertAlmostEqual(reverse.pricing_task(template()).compile().evaluate().report.value,
                               -2. * a.pricing_task(template()).compile().evaluate().report.value)
        with self.assertRaises(ValueError): builder().autocall(100., 100., 70., dates, dates, [100.], [80.], [.05])

    def test_discontinuous_risk_refused_and_new_wire_round_trips(self):
        c = builder().digital('call', 100., 7., EXPIRY, EXPIRY)
        task = c.pricing_task(template())
        self.assertEqual(json.loads(task.to_json())['schema_version'], 2)
        self.assertGreater(task.compile().evaluate().report.value, 0.)
        self.assertEqual(task.fingerprint, rp.PricingTask.from_json(task.to_json()).fingerprint)
        with self.assertRaises((ValueError, rp.PricingError)):
            c.pricing_task(template(True)).compile()
        doc = json.loads(c.to_json())
        doc['root'] = {'type': 'exercise', 'id': 'exercise', 'date': EXPIRY, 'contract': doc['root']}
        with self.assertRaises(ValueError): rp.Contract.from_json(json.dumps(doc))

    def test_existing_stochastic_dividend_source_keeps_native_aad(self):
        native = rp.PricingTask.from_json((ROOT / 'fixtures/task-v1/stochastic_dividend_bs.golden.json').read_text())
        c = builder().european('call', 100., 1., EXPIRY, EXPIRY)
        from test_typed_pricing_sources import dividend
        source = json.loads(native.to_json())['source']
        task = c.pricing_task(dividend(source), operation='aad')
        a, b = task.compile().evaluate().report, native.compile().evaluate().report
        self.assertEqual((a.value, a.standard_error), (b.value, b.standard_error))
        self.assertEqual([(r.name, r.values) for r in a.risks], [(r.name, r.values) for r in b.risks])

    def test_existing_volatility_and_rate_model_sources_preserve_mc_outputs(self):
        for name in ('black_scholes', 'rough_heston', 'bergomi_lsv', 'bergomi_hull_white'):
            with self.subTest(model=name):
                value = json.loads((ROOT / f'fixtures/specification-v1/{name}.golden.json').read_text())
                p = value['product']
                source = rp.PricingSpecification.from_json(json.dumps(value))
                c = rp.ContractBuilder('model-check', p['currency_id'], p['underlying_id'], 'close').european(
                    p['side']['type'], p['strike'], p['notional'], p['expiry'], p.get('payment_date', p['expiry']))
                expected = rp.PricingTask.from_specification(source).compile().evaluate().report
                actual = c.pricing_task(source).compile().evaluate().report
                self.assertAlmostEqual(actual.value, expected.value, places=11)
                self.assertAlmostEqual(actual.standard_error, expected.standard_error, places=11)
                self.assertEqual([(r.name, r.values) for r in actual.risks], [(r.name, r.values) for r in expected.risks])
