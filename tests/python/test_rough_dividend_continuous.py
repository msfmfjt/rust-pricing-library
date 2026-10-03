"""Opt-in continuous rough-LSV bridge: public API and independent references."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np
import rust_pricing as rp

from test_rough_dividend_hard_barrier import CFG, request
from rough_dividend_continuous_reference import DIRECTORY, inputs, path_values

FIXTURE = json.loads((DIRECTORY/'rough-continuous-barrier-reference.json').read_text())
BASES, MARKET = inputs()


def make_request(case, *, points=64, scrambles=4, risk=None, monitoring='continuous',
                 history=None, dates=None):
    contract, market = case['contract'], MARKET | case['market']
    product = rp.Product.barrier(1, 2, '2027-09-03', market['strike'], market['barrier'],
        contract['notional'], contract['side'], contract['direction'], contract['style'],
        monitoring, dates or BASES[case['base_case']]['monitoring_dates'], '2027-12-04',
        rebate=contract['rebate'] or None, historical_hit=history)
    return request(CFG['cases'][0], product=product, points=points, scrambles=scrambles, risk=risk)


def compile_plan(case, req=None, **changes):
    return rp.StochasticDividendContinuousBarrierPlan.compile_rough_bergomi_lsv(
        req or make_request(case), **(dict(hurst=BASES[case['base_case']]['hurst'],vol_of_vol=.6,
        correlation=-.4,dividend_mean_reversion=.7,equity_linkage=.6,dividend_volatility=.35,
        equity_dividend_correlation=-.25,dividend_volatility_correlation=.15,
        particle_count=64,calibration_seed=42,log_bandwidth=.35,minimum_effective_samples=5.,
        maximum_step=MARKET['expiry_time']/8*(1+8*math.ulp(1.)),worker_threads=1,
        reduction_block_size=64) | changes))


class ContinuousBarrierApproximation(unittest.TestCase):
    def test_independent_numpy_references_and_calibration_inputs(self):
        for case in FIXTURE['cases']:
            with self.subTest(case=case['id']):
                typed = make_request(case, points=8192, scrambles=16)
                restored = rp.PricingRequest.from_json(typed.to_json())
                self.assertEqual(typed.fingerprint, restored.fingerprint)
                plan = compile_plan(case, restored)
                base = BASES[case['base_case']]
                self.assertEqual(plan.time_nodes, base['times'])
                self.assertEqual(plan.lsv_time_nodes, plan.time_nodes)
                self.assertEqual(plan.lsv_log_moneyness_nodes, base['log_nodes'])
                np.testing.assert_allclose(plan.lsv_squared_leverage, base['squared_leverage'], rtol=0, atol=2e-13)
                value = plan.evaluate()
                means = np.asarray(case['batch_means'])[:, 0]
                reference_se = means.std(ddof=1)/math.sqrt(len(means))
                bound = abs(value.value-means.mean())+4*math.hypot(value.standard_error, reference_se)
                self.assertLess(bound, FIXTURE['acceptance']['price_difference_plus_4se'])
                self.assertLess(value.standard_error, FIXTURE['acceptance']['production_price_se'])
                self.assertEqual(value.plan_fingerprint, plan.plan_fingerprint)
                self.assertEqual(value.scheme, plan.scheme)
                self.assertIn('continuous-physical-log-bridge-approx-v1', value.scheme)
                self.assertEqual(value.independent_sampling_units, 16)
                self.assertEqual(value.evaluated_paths, 262144)
                self.assertEqual(plan.random_factor_count, 4)
                self.assertEqual(plan.risky_spot, plan.lsv_initial_residual_equity)

    def test_price_only_contract_immutable_plan_and_worker_replay(self):
        case = FIXTURE['cases'][0]
        req = make_request(case)
        plan = compile_plan(case, req)
        value = plan.evaluate()
        replay = compile_plan(case, req, worker_threads=3).evaluate()
        self.assertEqual((value.value,value.standard_error),(replay.value,replay.standard_error))
        self.assertNotEqual(value.plan_fingerprint,replay.plan_fingerprint)
        for name in ('evaluate_lsv_spot_risk','evaluate_aad','evaluate_lsv_gamma','evaluate_local_variance_risk'):
            self.assertFalse(hasattr(plan,name))
        with self.assertRaises(AttributeError):
            plan.scheme = 'exact'
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(lambda _: plan.evaluate().value,range(2))),[value.value]*2)
        for risk in (rp.RiskRequest(delta=True),rp.RiskRequest(vega=True),
                     rp.RiskRequest(gamma_relative_bump=.01),rp.RiskRequest(payoff_smoothing_half_width=2.)):
            with self.assertRaisesRegex(rp.PricingError,'price-only'):
                compile_plan(case,make_request(case,risk=risk))
        with self.assertRaises(rp.PricingError):
            compile_plan(case,make_request(case,monitoring='discrete'))

    def test_history_and_monitoring_end_do_not_infer_past_hits(self):
        case = FIXTURE['cases'][0]
        past=['2026-09-03']
        knocked = compile_plan(case,make_request(case,history=True,dates=past))
        result=knocked.evaluate()
        self.assertEqual((result.value,result.standard_error),(0.,0.))
        # Explicit false after monitoring ended selects vanilla even if later
        # simulated paths cross the barrier.
        unhit=compile_plan(case,make_request(case,history=False,dates=past))
        self.assertGreater(unhit.evaluate().value,compile_plan(case).evaluate().value)
        initial=compile_plan(case,make_request(case,dates=['2026-09-04']))
        self.assertEqual(initial.evaluate().value,unhit.evaluate().value)
        self.assertNotEqual(initial.plan_fingerprint,unhit.plan_fingerprint)
        future=make_request(case,history=False,dates=['2026-09-03']+BASES[case['base_case']]['monitoring_dates'])
        self.assertEqual(compile_plan(case,future).evaluate().value,compile_plan(case).evaluate().value)

    def test_independent_path_parity_and_refinement_evidence(self):
        for case in FIXTURE['cases']:
            base,market=BASES[case['base_case']],MARKET|case['market']
            z=np.random.default_rng(37).standard_normal((128,len(base['times'])-1,4))
            ko=path_values(base,market,z,**case['contract'])
            ki=path_values(base,market,z,**(case['contract']|dict(style='knock_in')))
            vanilla=path_values(base,market,z,**case['contract'],monitoring_end=-1)
            np.testing.assert_allclose(ki+ko,vanilla,rtol=0,atol=2e-14)
            self.assertTrue(np.all(ko[:,0] <= ko[:,1]+1e-14))
        for row in FIXTURE['refinement']:
            means=np.asarray(row['batch_means'])
            delta=means[:,2]-means[:,0]
            bound=abs(delta.mean())+4*delta.std(ddof=1)/math.sqrt(len(delta))
            if row['coarse_steps']==64:
                self.assertLess(bound,FIXTURE['acceptance']['last_paired_price_change_bound'])
            # Endpoint-only monitoring overprices a zero-rebate knock-out on
            # each coupled path; it is retained as a distinct diagnostic.
            self.assertTrue(np.all(means[:,0] <= means[:,1]+1e-13))
            self.assertTrue(np.all(means[:,2] <= means[:,3]+1e-13))


if __name__=='__main__':
    unittest.main()
