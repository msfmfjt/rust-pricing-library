"""Terminal Gamma boundary through the real native PricingPlan."""
import json
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backtesting'))
from rust_pricing_backtest.numerics import black_scholes_call, flat_request

POLICY = json.loads((ROOT / 'fixtures/backtesting/numerical-policy-v0.1.json').read_text())


def make_request(**kwargs):
    return flat_request(POLICY, **(dict(seed=7, samples=64, steps=8, nodes=17) | kwargs))


def compile_request(request, workers=1):
    return rp.PricingPlan.compile(request, worker_threads=workers, reduction_block_size=16)


def with_reporting_days(payload, days):
    """Specify market dates independently of the left-Euler simulation grid."""
    p = json.loads(json.dumps(payload))
    grid = p['model']['local_variance_grid']
    basis = p['model']['reporting_iv_basis']
    width = len(grid['log_forward_moneyness_nodes'])
    quote_width = len(basis['log_forward_moneyness_nodes'])
    times = grid['time_nodes']
    maturities = [d / 365 for d in days]
    ivs = []
    for t in maturities:
        total = math.fsum(grid['values'][i * width] * max(0., min(t, b) - a)
                          for i, (a, b) in enumerate(zip(times, times[1:])))
        ivs.extend([math.sqrt(total / t)] * quote_width)
    basis.update(maturity_nodes=maturities, shape=[len(days), quote_width],
                 implied_volatilities=ivs)
    start = date.fromisoformat(p['valuation_date'])
    p['risk']['vega_kt']['maturity_nodes'] = [(start + timedelta(days=d)).isoformat() for d in days]
    return p


class TerminalMeasureTests(unittest.TestCase):
    def test_sparse_and_offstep_reporting_dates_keep_same_samples_and_covariance(self):
        import jsonschema
        for term_structure in [False, True]:
            for side in ['call', 'put']:
                with self.subTest(term_structure=term_structure, side=side):
                    p = json.loads(make_request(strike=105., samples=1024).to_json())
                    p['product']['side']['type'] = side
                    p['risk']['vega_kt']['full_bucket_covariance'] = True
                    if term_structure:
                        p['model']['local_variance_grid']['values'] = [
                            .01 + .01*i for i in range(9) for _ in range(17)]
                        p = with_reporting_days(p, list(range(16, 129, 16)))
                    dense = compile_request(rp.PricingRequest.from_json(json.dumps(p))).evaluate()
                    for days in [[64, 128], [11, 43, 128], [1, 127, 128]]:
                        q = with_reporting_days(p, days)
                        plan = compile_request(rp.PricingRequest.from_json(json.dumps(q)), workers=4)
                        result = plan.evaluate()
                        a, b = dense.vega_kt, result.vega_kt
                        self.assertEqual(b.policy_label, 'flat_variance_terminal_measure_pathwise_v1')
                        self.assertEqual(result.value, dense.value)
                        self.assertEqual(result.delta_raw, dense.delta_raw)
                        self.assertEqual(result.vega_raw, dense.vega_raw)
                        self.assertEqual(list(b.raw_buckets[:-17]), [0.]*(len(b.raw_buckets)-17))
                        self.assertEqual(list(a.raw_buckets[-17:]), list(b.raw_buckets[-17:]))
                        self.assertEqual(b.projection.signed_residual, 0.)
                        for j in range(17):
                            x, y = a.estimates[-17+j], b.estimates[-17+j]
                            self.assertEqual(x.sample_variance, y.sample_variance)
                            self.assertEqual(x.price_covariance, y.price_covariance)
                            for k in range(17):
                                ai = (len(a.raw_buckets)-17+j)*len(a.raw_buckets)+len(a.raw_buckets)-17+k
                                bi = (len(b.raw_buckets)-17+j)*len(b.raw_buckets)+len(b.raw_buckets)-17+k
                                self.assertEqual(a.full_bucket_covariance[ai], b.full_bucket_covariance[bi])
                        self.assertEqual(rp.PricingResult.from_json(result.to_json()).to_json(), result.to_json())
                        jsonschema.validate(json.loads(result.to_json()), json.loads(rp.result_json_schema()))
                        diagnostic = plan.evaluate_vega_kt_terminal_diagnostic()
                        t = 128/365
                        sigma = q['model']['reporting_iv_basis']['implied_volatilities'][-1]
                        oracle = black_scholes_call(100., 105., t, sigma)['vega']
                        self.assertAlmostEqual(diagnostic.scalar_vega, oracle, places=11)

    def test_fixed_reporting_dates_across_time_refinement_match_independent_price_bumps(self):
        days = [11, 43, 128]
        for steps in [2, 8, 16]:
            p = with_reporting_days(json.loads(make_request(strike=105., steps=steps).to_json()), days)
            plan = compile_request(rp.PricingRequest.from_json(json.dumps(p)))
            result = plan.evaluate()
            self.assertEqual(result.vega_kt.policy_label, 'flat_variance_terminal_measure_pathwise_v1')
            d = plan.evaluate_vega_kt_terminal_diagnostic()
            nodes = d.log_moneyness_nodes
            x = math.log(105/100)
            i = max(j for j, n in enumerate(nodes) if n <= x)
            weight = (x-nodes[i])/(nodes[i+1]-nodes[i])
            # Reprice independent quoted IV bumps at each reporting coordinate.
            h = 1e-5
            expected = [0.] * len(d.raw_buckets)
            for j, w in [(i, 1-weight), (i+1, weight)]:
                expected[-len(nodes)+j] = (
                    black_scholes_call(100., 105., 128/365, .2+h*w)['price']
                    - black_scholes_call(100., 105., 128/365, .2-h*w)['price'])/(2*h)
            for actual, ref in zip(d.raw_buckets, expected):
                self.assertAlmostEqual(actual, ref, delta=2e-8)

    def test_sparse_reporting_keeps_rqmc_scramble_statistics(self):
        p = json.loads(make_request().to_json())
        p['engine'] = {'type': 'randomized_quasi_monte_carlo', 'points_per_scramble': 128,
                       'scramble_count': 4, 'master_scramble_seed': 7,
                       'variance_reduction': {'antithetic': True, 'brownian_bridge': True}}
        p['risk']['vega_kt']['full_bucket_covariance'] = True
        dense = compile_request(rp.PricingRequest.from_json(json.dumps(p))).evaluate()
        q = with_reporting_days(p, [11, 43, 128])
        result = compile_request(rp.PricingRequest.from_json(json.dumps(q))).evaluate()
        self.assertEqual(result.vega_kt.policy_label, 'flat_variance_terminal_measure_pathwise_v1')
        self.assertEqual(result.value, dense.value)
        self.assertEqual(list(result.vega_kt.raw_buckets[-17:]), list(dense.vega_kt.raw_buckets[-17:]))
        for j in range(17):
            self.assertEqual(result.vega_kt.estimates[-17+j].sample_variance,
                             dense.vega_kt.estimates[-17+j].sample_variance)
        variance = math.fsum(result.vega_kt.full_bucket_covariance)
        self.assertGreater(variance, 0.)
        self.assertAlmostEqual(variance, result.vega.raw.standard_error**2 * 4, places=11)

    def test_legacy_initial_loss_probe_is_not_a_pass_for_terminal_measure(self):
        from rust_pricing_backtest.allocation_probe import run_probe
        policy = json.loads((ROOT / 'fixtures/backtesting/allocation-probe-policy-v0.1.json').read_text())
        policy.update(seeds=[7], time_steps=[2], sampling_units=64)
        report = run_probe(policy, POLICY)
        self.assertEqual(report['status'], 'not_applicable')
        self.assertEqual(report['paper_vega_kt_acceptance'], 'not_established')
        self.assertEqual(report['checks'], [])
        self.assertEqual(report['observations'], [])
        self.assertEqual(len(report['not_applicable']), 1)
        self.assertEqual(report['not_applicable'][0]['policy_label'],
                         'flat_variance_terminal_measure_pathwise_v1')

    def test_ordinary_report_uses_same_sample_atom_risk_and_wire_roundtrips(self):
        p = json.loads(make_request(strike=105., samples=1024).to_json())
        p['risk']['vega_kt']['full_bucket_covariance'] = True
        result = compile_request(rp.PricingRequest.from_json(json.dumps(p))).evaluate()
        kt = result.vega_kt
        self.assertEqual(kt.policy_label, 'flat_variance_terminal_measure_pathwise_v1')
        self.assertEqual(kt.projection.signed_residual, 0.)
        self.assertAlmostEqual(sum(kt.raw_buckets), result.vega_raw, places=11)
        self.assertEqual(sum(v != 0 for v in kt.raw_buckets), 2)
        self.assertGreater(max(e.sample_variance for e in kt.estimates), 0.)
        self.assertGreater(max(abs(e.price_covariance) for e in kt.estimates), 0.)
        variance = math.fsum(kt.full_bucket_covariance)
        self.assertAlmostEqual(variance, result.vega.raw.standard_error**2*1024, delta=2e-10)
        expected = result.to_json()
        self.assertEqual(rp.PricingResult.from_json(expected).to_json(), expected)
        # The immutable v3 wire schema already carries these sample statistics.
        import jsonschema
        jsonschema.validate(json.loads(expected), json.loads(rp.result_json_schema()))

    def test_exact_terminal_node_with_auditable_time_zero_and_unchanged_ordinary_result(self):
        plan = compile_request(make_request())
        before = plan.evaluate().to_json()
        d = plan.evaluate_vega_kt_terminal_diagnostic()
        obj = d.to_dict()
        self.assertEqual(d.gamma_atoms, [(100., 1.)])
        self.assertEqual(d.policy_label, 'flat_variance_terminal_measure_diagnostic_v1')
        self.assertEqual(obj['calculation_kind'], 'deterministic')
        self.assertTrue(obj['diagnostic_only'])
        self.assertNotIn('accepted', obj)
        self.assertNotIn('standard_errors', obj)
        self.assertEqual(obj['request_fingerprint'], plan.request_fingerprint)
        self.assertEqual(obj['plan_fingerprint'], plan.plan_fingerprint)
        self.assertEqual(sum(v != 0 for v in d.raw_buckets), 1)
        self.assertEqual(d.hedge_quantities[-9], -1.)
        oracle = black_scholes_call(100., 100., 128/365, .2)['vega']
        self.assertAlmostEqual(d.scalar_vega, oracle, places=12)
        self.assertAlmostEqual(d.expected_local_parallel_vega, oracle, places=12)
        first = obj['local_rows'][0]
        self.assertEqual(first['source_maturity'], 0.)
        self.assertEqual(sum(v != 0 for v in first['expected_source_hat_vegas']), 1)
        self.assertAlmostEqual(sum(first['expected_source_hat_vegas']), oracle/8, places=12)
        json.dumps(obj, allow_nan=False)
        self.assertEqual(before, plan.evaluate().to_json())
        with self.assertRaises(AttributeError):
            d.scalar_vega = 0.

    def test_offnode_calls_puts_and_price_differences(self):
        for strike in [95., 100., 105.]:
            for side in ['call', 'put']:
                p = json.loads(make_request(strike=strike, samples=16384).to_json())
                p['product']['side']['type'] = side
                plan = compile_request(rp.PricingRequest.from_json(json.dumps(p)))
                d = plan.evaluate_vega_kt_terminal_diagnostic()
                obj = d.to_dict()
                self.assertEqual(d.gamma_atoms, [(strike, 1.)])
                self.assertEqual(obj['affine_slope'], -1. if side == 'put' else 0.)
                oracle = black_scholes_call(100., strike, 128/365, .2)['vega']
                self.assertAlmostEqual(d.scalar_vega, oracle, places=12)
                # Bump independent quoted IV prices, not the native projection.
                x = math.log(strike/100.)
                nodes = d.log_moneyness_nodes
                i = max(j for j, n in enumerate(nodes) if n <= x)
                h = (x-nodes[i])/(nodes[i+1]-nodes[i])
                expected = [0.] * len(d.raw_buckets)
                expected[-len(nodes)+i] = oracle*(1-h)
                expected[-len(nodes)+i+1] = oracle*h
                for actual, ref in zip(d.raw_buckets, expected):
                    self.assertAlmostEqual(actual, ref, places=11)
                bump = 1e-5
                fd = (black_scholes_call(100., strike, 128/365, .2+bump)['price']
                      - black_scholes_call(100., strike, 128/365, .2-bump)['price'])/(2*bump)
                self.assertAlmostEqual(fd, d.scalar_vega, delta=2e-8)
                native = plan.evaluate()
                self.assertLess(abs(native.vega_raw-d.expected_local_parallel_vega),
                                6*native.vega.raw.standard_error)

    def test_deterministic_result_does_not_depend_on_sampling_settings(self):
        def numeric(d):
            p = d.to_dict()
            for key in ['request_fingerprint', 'plan_fingerprint']:
                p.pop(key)
            return p
        base = numeric(compile_request(make_request()).evaluate_vega_kt_terminal_diagnostic())
        for seed, samples, workers in [(97, 1024, 4), (41, 2, 1)]:
            p = json.loads(make_request(seed=seed, samples=samples).to_json())
            p['engine']['variance_reduction']['brownian_bridge'] = True
            p['risk']['vega_kt']['full_bucket_covariance'] = True
            d = compile_request(rp.PricingRequest.from_json(json.dumps(p)), workers).evaluate_vega_kt_terminal_diagnostic()
            self.assertEqual(base, numeric(d))


if __name__ == '__main__':
    unittest.main()
