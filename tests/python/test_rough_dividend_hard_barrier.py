"""Public hard Barrier Spot-risk binding, retained references and API boundaries."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest

import rust_pricing as rp

from rough_dividend_barrier_styles import inputs
from test_stochastic_dividends import compile_lsv

CFG, BASES, MARKET = inputs("rough-barrier-rebates-reference.json")
TARGET = json.loads((Path(__file__).resolve().parents[2] /
    "fixtures/stochastic-dividends/rough-conditional-barrier-reference.json").read_text())["target_variances"]


def request(case, *, spot=100., points=64, scrambles=4, engine=None, risk=None,
            product=None, monitoring_dates=None):
    c, base = case['contract'], BASES[case['base_case']]
    return rp.PricingRequest(
        '2026-09-04', product or rp.Product.barrier(
            1, 2, '2027-09-03', c['strike'], c['barrier'], c['notional'],
            c['side'], c['direction'], c['style'], 'discrete',
            monitoring_dates or base['monitoring_dates'], '2027-12-04', rebate=c['rebate']),
        rp.Market.equity(2, 1, spot,
            rp.DiscountCurve(10, [0., 1.], [1., MARKET['annual_discount']]),
            rp.DiscountCurve(11, [0., 1.], [1., MARKET['annual_carry']]),
            discrete_dividends=[rp.DividendEvent.fixed_cash(i+1, t, q)
                for i, (t, q) in enumerate(zip(MARKET['cash_times'], MARKET['cash_means']))]),
        rp.Model.local_volatility_from_grid(
            [0., MARKET['fixing_time'], MARKET['expiry_time']], [-.5, 0., .5],
            TARGET, 1e-8, 4.),
        engine or rp.Engine.randomized_quasi_monte_carlo(
            points, 193, scramble_count=scrambles, antithetic=True, brownian_bridge=True),
        risk or rp.RiskRequest())


def compile_plan(case, req=None, **changes):
    return compile_lsv(req or request(case), rough=True,
        **(dict(hurst=BASES[case['base_case']]['hurst'], particle_count=64,
            retain_reverse_trace=False, maximum_step=MARKET['expiry_time']/8*(1+8*math.ulp(1.)),
            worker_threads=1, reduction_block_size=64) | changes))


def snapshot(risk, *, with_fingerprint=True):
    p = risk.price
    values = (p.value, p.standard_error, risk.delta, risk.delta_standard_error,
            p.plan_fingerprint, p.independent_sampling_units, p.evaluated_paths,
            risk.method, risk.coordinate, risk.uncertainty_scope)
    return values if with_fingerprint else values[:4] + values[5:]


class HardBarrierSpotRisk(unittest.TestCase):
    def test_retained_references_metadata_and_immutable_results(self):
        # Use the existing production counts and gates for two binding-level
        # controls; the Rust integration suite covers all sixteen reference cases.
        for case in (CFG['cases'][0], CFG['cases'][-1]):
            with self.subTest(case=case['id']):
                typed = request(case, points=16384, scrambles=16)
                restored = rp.PricingRequest.from_json(typed.to_json())
                self.assertEqual(typed.fingerprint, restored.fingerprint)
                plan = compile_plan(case, restored)
                base = BASES[case['base_case']]
                self.assertEqual(plan.time_nodes, base['times'])
                for actual, expected in zip(plan.lsv_squared_leverage, base['squared_leverage']):
                    self.assertAlmostEqual(actual, expected, delta=2e-13)
                original = plan.evaluate()
                risk = plan.evaluate_lsv_hard_barrier_spot_risk()
                self.assertIsInstance(risk, rp.StochasticDividendLsvSpotRisk)
                self.assertIsInstance(risk.price, rp.StochasticDividendPrice)
                for q, value, se in [('price', risk.price.value, risk.price.standard_error),
                                     ('delta', risk.delta, risk.delta_standard_error)]:
                    bound = abs(value-case[q])+4*math.hypot(se, case[q+'_se'])
                    self.assertLess(bound, CFG['acceptance'][q+'_bound'])
                    self.assertLess(se, CFG['acceptance']['valuation_'+q+'_se'])
                self.assertEqual(risk.method, 'buehler-rough-residual-lsv-hard-barrier-survival-spot-v2')
                self.assertEqual(risk.coordinate, 'physical_spot_with_residual_lsv_reanchoring')
                self.assertEqual(risk.uncertainty_scope, 'pricing_sampling_only_scale_invariant_calibration')
                self.assertEqual(risk.price.uncertainty_scope, 'pricing_conditional_on_calibration')
                self.assertEqual(risk.price.independent_sampling_units, 16)
                self.assertEqual(risk.price.evaluated_paths, 524288)
                self.assertNotEqual(risk.price.plan_fingerprint, plan.plan_fingerprint)
                self.assertEqual(plan.evaluate().value, original.value)
                self.assertEqual(plan.evaluate().standard_error, original.standard_error)
                self.assertEqual(original.plan_fingerprint, plan.plan_fingerprint)
                for obj, name in [(risk, 'delta'), (risk, 'method'), (risk.price, 'value')]:
                    with self.assertRaises(AttributeError):
                        setattr(obj, name, 0.)

    def test_recompiled_spot_bumps_and_worker_replay(self):
        for case in CFG['cases']:
            with self.subTest(case=case['id']):
                plan = compile_plan(case)
                risk = plan.evaluate_lsv_hard_barrier_spot_risk()
                for bump in (1e-3, 5e-4):
                    up = compile_plan(case, request(case, spot=100+bump))
                    down = compile_plan(case, request(case, spot=100-bump))
                    fd = (up.evaluate_lsv_hard_barrier_spot_risk().price.value
                        - down.evaluate_lsv_hard_barrier_spot_risk().price.value)/(2*bump)
                    self.assertAlmostEqual(risk.delta, fd, delta=2e-6)
                parallel = compile_plan(case, worker_threads=3)
                # Execution policy participates in the fingerprint, while
                # numerical results and sampling metadata replay exactly.
                self.assertNotEqual(plan.plan_fingerprint, parallel.plan_fingerprint)
                self.assertEqual(snapshot(risk, with_fingerprint=False),
                    snapshot(parallel.evaluate_lsv_hard_barrier_spot_risk(), with_fingerprint=False))
        # Concurrent calls share an immutable compiled plan and release the GIL.
        with ThreadPoolExecutor(max_workers=2) as executor:
            values = list(executor.map(lambda _: snapshot(plan.evaluate_lsv_hard_barrier_spot_risk()), range(2)))
        self.assertEqual(values, [snapshot(risk)]*2)

    def test_pseudo_mc_sampling_counts_with_and_without_antithetics(self):
        for antithetic in (False, True):
            case = CFG['cases'][0]
            engine = rp.Engine.pseudo_monte_carlo(193, 128,
                antithetic=antithetic, brownian_bridge=True)
            plan = compile_plan(case, request(case, engine=engine))
            risk = plan.evaluate_lsv_hard_barrier_spot_risk()
            self.assertEqual(risk.price.independent_sampling_units, 128)
            self.assertEqual(risk.price.evaluated_paths, 256 if antithetic else 128)
            for se in (risk.price.standard_error, risk.delta_standard_error):
                self.assertTrue(math.isfinite(se) and se > 0)
            self.assertEqual(snapshot(risk), snapshot(plan.evaluate_lsv_hard_barrier_spot_risk()))

    def test_initial_monitoring_preserves_unhit_paths_and_fixed_rebates(self):
        discount = MARKET['annual_discount']**MARKET['payment_time']
        for case in CFG['cases']:
            with self.subTest(case=case['id']):
                dates = ['2026-09-04'] + BASES[case['base_case']]['monitoring_dates']
                # Each retained contract is unhit at Spot=100.
                future = compile_plan(case).evaluate_lsv_hard_barrier_spot_risk()
                initial = compile_plan(case, request(case, monitoring_dates=dates))
                risk = initial.evaluate_lsv_hard_barrier_spot_risk()
                self.assertEqual(snapshot(risk, with_fingerprint=False),
                                 snapshot(future, with_fingerprint=False))
                self.assertNotEqual(risk.price.plan_fingerprint, future.price.plan_fingerprint)
                c = case['contract']
                # No remaining observation and unhit: knock-in pays the rebate.
                # Already hit: knock-out pays the rebate despite future observations.
                fixed = case if c['style'] == 'knock_in' else dict(case, contract=dict(c,
                    barrier=95. if c['direction'] == 'up' else 105.))
                fixed_dates = ['2026-09-04'] if c['style'] == 'knock_in' else dates
                fixed_plan = compile_plan(fixed, request(fixed, monitoring_dates=fixed_dates))
                fixed_risk = fixed_plan.evaluate_lsv_hard_barrier_spot_risk()
                self.assertAlmostEqual(fixed_risk.price.value, discount*c['rebate'], delta=2e-14)
                self.assertEqual(fixed_plan.evaluate().value, fixed_risk.price.value)
                self.assertEqual((fixed_risk.price.standard_error, fixed_risk.delta,
                                  fixed_risk.delta_standard_error), (0., 0., 0.))
        with self.assertRaises(rp.ValidationError):
            request(CFG['cases'][0], monitoring_dates=['2026-09-03', '2027-09-03'])

    def test_initial_hit_vanilla_matches_independent_parity_reference(self):
        for i in (0, 14):  # H=.1 Up Call and H=.3 Down Put, both knock-in.
            ki, ko = CFG['cases'][i:i+2]
            c = ki['contract']
            initial_hit = dict(ki, contract=dict(c,
                barrier=95. if c['direction'] == 'up' else 105.))
            dates = ['2026-09-04'] + BASES[ki['base_case']]['monitoring_dates']
            plan = compile_plan(initial_hit, request(initial_hit, points=16384,
                scrambles=16, monitoring_dates=dates))
            risk = plan.evaluate_lsv_hard_barrier_spot_risk()
            # The retained KI/KO references use common inputs. Combine each
            # batch before estimating uncertainty, including its covariance.
            discount = MARKET['annual_discount']**MARKET['payment_time']
            for j, q, value, se in [(0, 'price', risk.price.value, risk.price.standard_error),
                                     (1, 'delta', risk.delta, risk.delta_standard_error)]:
                batches = [a[j]+b[j]-(discount*c['rebate'] if j == 0 else 0.)
                    for a, b in zip(ki['batch_means'], ko['batch_means'])]
                expected = sum(batches)/len(batches)
                reference_se = math.sqrt(sum((x-expected)**2 for x in batches)
                    /(len(batches)*(len(batches)-1)))
                self.assertLess(abs(value-expected)+4*math.hypot(se, reference_se),
                                CFG['acceptance'][q+'_bound'])
                self.assertLess(se, CFG['acceptance']['valuation_'+q+'_se'])
                self.assertLess(reference_se, CFG['acceptance']['reference_'+q+'_se'])

    def test_scope_errors_preserve_existing_methods(self):
        case = CFG['cases'][0]
        hard = compile_plan(case)
        with self.assertRaises(rp.PricingError):
            hard.evaluate_lsv_spot_risk()
        smooth = compile_plan(case, request(case,
            risk=rp.RiskRequest(payoff_smoothing_half_width=.5)))
        ordinary = smooth.evaluate_lsv_spot_risk()
        self.assertTrue(math.isfinite(ordinary.delta))
        vanilla = rp.Product.european_vanilla(1, 2, '2027-09-03', 80., 2., 'call')
        unsupported = [smooth, compile_plan(case, request(case, product=vanilla)),
            compile_plan(case, request(case, spot=case['contract']['barrier'],
                monitoring_dates=['2026-09-04', '2027-09-03'])),
            compile_lsv(request(case), rough=False, particle_count=64, retain_reverse_trace=False),
            compile_plan(case, correlation=1., equity_dividend_correlation=0.,
                dividend_volatility_correlation=0.)]
        for plan in unsupported:
            with self.assertRaises(rp.PricingError):
                plan.evaluate_lsv_hard_barrier_spot_risk()
        self.assertEqual(snapshot(ordinary), snapshot(smooth.evaluate_lsv_spot_risk()))
        with self.assertRaises(rp.ValidationError):
            request(case, risk=rp.RiskRequest(delta=True))
        with self.assertRaises(rp.PricingError):
            compile_plan(case, request(case,
                risk=rp.RiskRequest(delta=True, payoff_smoothing_half_width=.5)))
        # An omitted rebate is also supported through the typed constructor.
        no_rebate = dict(case, contract=dict(case['contract'], rebate=None))
        self.assertTrue(math.isfinite(compile_plan(no_rebate).evaluate_lsv_hard_barrier_spot_risk().delta))


if __name__ == '__main__':
    unittest.main()
