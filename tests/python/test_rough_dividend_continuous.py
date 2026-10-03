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

    def test_spot_bump_risk_against_independent_numpy(self):
        gates=FIXTURE['spot_bump_acceptance']
        for case in FIXTURE['cases']:
            with self.subTest(case=case['id']):
                plan=compile_plan(case,make_request(case,points=8192,scrambles=16))
                risk=plan.evaluate_spot_bump_risk(spot_absolute_bump=1.)
                means=np.asarray(case['spot_bump_batch_means'])
                reference=means.mean(axis=0)
                errors=means.std(axis=0,ddof=1)/math.sqrt(len(means))
                actual=np.array(risk.delta_estimates+risk.bump_differences)
                se=np.array(risk.delta_standard_errors+risk.bump_difference_standard_errors)
                bounds=np.abs(actual-reference)+4*np.hypot(errors,se)
                self.assertLess(max(bounds[:3]),gates['delta_difference_plus_4se'])
                self.assertLess(max(bounds[3:]),gates['gap_difference_plus_4se'])
                self.assertLess(max(se[:3]),gates['production_delta_se'])
                self.assertLess(max(errors[:3]),gates['reference_delta_se'])
                self.assertEqual(risk.spot_bumps,[.5,1.,2.])
                self.assertEqual(risk.delta,risk.delta_estimates[1])
                self.assertEqual(risk.standard_error,risk.delta_standard_errors[1])
                self.assertEqual(risk.payoff_evaluations,7*risk.price.evaluated_paths)
                self.assertEqual(risk.price.plan_fingerprint,plan.plan_fingerprint)
                self.assertEqual(risk.price.scheme,plan.scheme)
                self.assertEqual(risk.price.value,plan.evaluate().value)
                self.assertIn('continuous-bridge-crn-spot-bump-v1',risk.method)
                self.assertEqual(risk.uncertainty_scope,'sampling_only_fixed_calibration_grid_bridge_and_bump')

    def test_spot_bump_api_replay_immutability_and_validation(self):
        case=FIXTURE['cases'][0]
        plan=compile_plan(case)
        risk=plan.evaluate_spot_bump_risk(spot_absolute_bump=1.)
        relative=plan.evaluate_spot_bump_risk(spot_relative_bump=.01)
        self.assertEqual(risk.delta_estimates,relative.delta_estimates)
        self.assertNotEqual(risk.risk_fingerprint,relative.risk_fingerprint)
        self.assertNotEqual(risk.risk_fingerprint,plan.evaluate_spot_bump_risk(spot_absolute_bump=2.).risk_fingerprint)
        with self.assertRaises(AttributeError):
            risk.delta=0.
        detached=risk.delta_estimates
        detached[0]=999.
        self.assertNotEqual(risk.delta_estimates[0],999.)
        parallel=compile_plan(case,worker_threads=3).evaluate_spot_bump_risk(spot_absolute_bump=1.)
        self.assertEqual(risk.delta_estimates,parallel.delta_estimates)
        self.assertEqual(risk.delta_standard_errors,parallel.delta_standard_errors)
        self.assertEqual(risk.bump_difference_standard_errors,parallel.bump_difference_standard_errors)
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(lambda _: plan.evaluate_spot_bump_risk(spot_absolute_bump=1.).delta,range(2))),[risk.delta]*2)
        for kwargs in ({},{'spot_absolute_bump':1.,'spot_relative_bump':.01},
                       *({'spot_absolute_bump':h} for h in [0.,-1.,float('nan'),float('inf'),1e-300,45.,50.])):
            with self.subTest(kwargs=kwargs),self.assertRaises((rp.ValidationError,rp.PricingError)):
                plan.evaluate_spot_bump_risk(**kwargs)
        # Frozen past hit pays fixed cash and has exactly zero Spot-bump risk.
        rebated=case|dict(contract=case['contract']|dict(notional=1e18,rebate=7.))
        fixed=compile_plan(rebated,make_request(rebated,history=True,dates=['2026-09-03'])).evaluate_spot_bump_risk(spot_absolute_bump=1.)
        self.assertEqual(fixed.delta_estimates,[0.,0.,0.])
        self.assertEqual(fixed.delta_standard_errors,[0.,0.,0.])
        self.assertEqual(fixed.bump_difference_standard_errors,[0.,0.])

    def test_gamma_bump_risk_against_independent_numpy(self):
        gates=FIXTURE['gamma_bump_acceptance']
        sampling=FIXTURE['gamma_production_sampling']
        for case in FIXTURE['cases']:
            with self.subTest(case=case['id']):
                plan=compile_plan(case,make_request(case,points=sampling['points_per_scramble'],scrambles=sampling['scramble_count']))
                risk=plan.evaluate_gamma_bump_risk(spot_absolute_bump=2.)
                means=np.asarray(case['gamma_bump_batch_means'])
                errors=means.std(axis=0,ddof=1)/math.sqrt(len(means))
                actual=np.array(risk.gamma_estimates+risk.bump_differences)
                se=np.array(risk.gamma_standard_errors+risk.bump_difference_standard_errors)
                bounds=np.abs(actual-means.mean(axis=0))+4*np.hypot(errors,se)
                self.assertLess(max(bounds[:3]),gates['gamma_difference_plus_4se'])
                self.assertLess(max(bounds[3:]),gates['gap_difference_plus_4se'])
                self.assertLess(max(se[:3]),gates['production_gamma_se'])
                self.assertLess(max(errors[:3]),gates['reference_gamma_se'])
                self.assertEqual(risk.spot_bumps,[1.,2.,4.])
                self.assertEqual(risk.gamma,risk.gamma_estimates[1])
                self.assertEqual(risk.standard_error,risk.gamma_standard_errors[1])
                self.assertEqual(risk.payoff_evaluations,7*risk.price.evaluated_paths)
                self.assertEqual(risk.price.plan_fingerprint,plan.plan_fingerprint)
                self.assertEqual(risk.price.scheme,plan.scheme)
                self.assertIn('continuous-bridge-crn-price-gamma-v1',risk.method)
                self.assertEqual(risk.uncertainty_scope,'sampling_only_fixed_calibration_grid_bridge_and_bump')

    def test_gamma_bump_api_preserves_spot_risk_and_replays(self):
        case=FIXTURE['cases'][0]
        plan=compile_plan(case)
        delta=plan.evaluate_spot_bump_risk(spot_absolute_bump=2.)
        risk=plan.evaluate_gamma_bump_risk(spot_absolute_bump=2.)
        self.assertEqual((risk.price.value,risk.price.standard_error),(delta.price.value,delta.price.standard_error))
        self.assertEqual((risk.delta,risk.delta_standard_error),(delta.delta,delta.standard_error))
        self.assertNotEqual(risk.risk_fingerprint,delta.risk_fingerprint)
        relative=plan.evaluate_gamma_bump_risk(spot_relative_bump=.02)
        self.assertEqual(risk.gamma_estimates,relative.gamma_estimates)
        self.assertNotEqual(risk.risk_fingerprint,relative.risk_fingerprint)
        self.assertNotEqual(risk.risk_fingerprint,plan.evaluate_gamma_bump_risk(spot_absolute_bump=1.).risk_fingerprint)
        with self.assertRaises(AttributeError):
            risk.gamma=0.
        detached=risk.gamma_estimates
        detached[0]=999.
        self.assertNotEqual(risk.gamma_estimates[0],999.)
        parallel=compile_plan(case,worker_threads=3).evaluate_gamma_bump_risk(spot_absolute_bump=2.)
        self.assertEqual(risk.gamma_estimates,parallel.gamma_estimates)
        self.assertEqual(risk.gamma_standard_errors,parallel.gamma_standard_errors)
        self.assertEqual(risk.bump_difference_standard_errors,parallel.bump_difference_standard_errors)
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(lambda _: plan.evaluate_gamma_bump_risk(spot_absolute_bump=2.).gamma,range(2))),[risk.gamma]*2)
        for kwargs in ({},{'spot_absolute_bump':1.,'spot_relative_bump':.01},
                       *({'spot_absolute_bump':h} for h in [0.,-1.,float('nan'),float('inf'),1e-300,45.,50.])):
            with self.subTest(kwargs=kwargs),self.assertRaises((rp.ValidationError,rp.PricingError)):
                plan.evaluate_gamma_bump_risk(**kwargs)
        rebated=case|dict(contract=case['contract']|dict(notional=1e18,rebate=7.))
        fixed=compile_plan(rebated,make_request(rebated,history=True,dates=['2026-09-03'])).evaluate_gamma_bump_risk(spot_absolute_bump=2.)
        self.assertEqual(fixed.gamma_estimates,[0.,0.,0.])
        self.assertEqual(fixed.gamma_standard_errors,[0.,0.,0.])
        self.assertEqual(fixed.bump_difference_standard_errors,[0.,0.])

    def test_parallel_local_volatility_against_independent_valuation(self):
        fixture=json.loads((DIRECTORY/'rough-continuous-local-vol-reference.json').read_text())
        gates,sampling=fixture['acceptance'],fixture['production_sampling']
        for case in fixture['cases']:
            with self.subTest(case=case['id']):
                req=make_request(case,points=sampling['points_per_scramble'],scrambles=sampling['scramble_count'])
                plan=compile_plan(case,req)
                result=plan.evaluate_parallel_local_volatility_risk(local_volatility_bump=fixture['sampling']['bump'])
                # Retained scenario inputs come from separate request compilation;
                # the NumPy oracle independently values those inputs only.
                data=json.loads(req.to_json())
                self.assertEqual(data['model']['local_variance_grid']['values'],case['original_target_variances'])
                for scenario in case['scenarios']:
                    data['model']['local_variance_grid']['values']=scenario['target_variances']
                    external=compile_plan(case,rp.PricingRequest.from_json(json.dumps(data)))
                    np.testing.assert_allclose(external.lsv_squared_leverage,scenario['squared_leverage'],rtol=0,atol=2e-13)
                means=np.asarray(case['batch_means'])
                errors=means.std(axis=0,ddof=1)/math.sqrt(len(means))
                actual=np.array(result.vega_estimates+result.bump_differences)
                se=np.array(result.vega_standard_errors+result.bump_difference_standard_errors)
                bounds=np.abs(actual-means.mean(axis=0))+4*np.hypot(errors,se)
                self.assertLess(max(bounds[:3]),gates['vega_difference_plus_4se'])
                self.assertLess(max(bounds[3:]),gates['gap_difference_plus_4se'])
                self.assertLess(max(se[:3]),gates['production_vega_se'])
                self.assertLess(max(errors[:3]),gates['reference_vega_se'])
                self.assertEqual(result.local_volatility_bumps,[.005,.01,.02])
                self.assertEqual(result.vega,result.vega_estimates[1])
                self.assertEqual(result.standard_error,result.vega_standard_errors[1])
                self.assertEqual(result.vega_per_vol_point,.01*result.vega)
                self.assertEqual(result.standard_error_per_vol_point,.01*result.standard_error)
                self.assertEqual(result.scenario_evaluated_paths,7*result.price.evaluated_paths)
                self.assertEqual(result.payoff_evaluations,result.scenario_evaluated_paths)
                self.assertEqual(result.recalibration_count,6)
                self.assertEqual(result.price.plan_fingerprint,plan.plan_fingerprint)
                self.assertIn('parallel-local-vol-recalibrated-crn-v1',result.method)
                self.assertEqual(result.uncertainty_scope,'pricing_only_fixed_calibration_seed_grid_bridge_and_bump')

    def test_parallel_local_volatility_api_replay_history_and_bounds(self):
        case=FIXTURE['cases'][0]
        plan=compile_plan(case)
        original=plan.evaluate()
        risk=plan.evaluate_parallel_local_volatility_risk(local_volatility_bump=.01)
        self.assertEqual((risk.price.value,risk.price.standard_error),(original.value,original.standard_error))
        self.assertEqual(plan.evaluate().value,original.value)
        self.assertNotEqual(risk.risk_fingerprint,plan.evaluate_parallel_local_volatility_risk(local_volatility_bump=.005).risk_fingerprint)
        with self.assertRaises(AttributeError):
            risk.vega=0.
        detached=risk.vega_estimates;detached[0]=999.
        self.assertNotEqual(risk.vega_estimates[0],999.)
        parallel=compile_plan(case,worker_threads=3).evaluate_parallel_local_volatility_risk(local_volatility_bump=.01)
        self.assertEqual(risk.vega_estimates,parallel.vega_estimates)
        self.assertEqual(risk.vega_standard_errors,parallel.vega_standard_errors)
        self.assertEqual(risk.bump_difference_standard_errors,parallel.bump_difference_standard_errors)
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(lambda _: plan.evaluate_parallel_local_volatility_risk(local_volatility_bump=.01).vega,range(2))),[risk.vega]*2)
        with self.assertRaises(TypeError):
            plan.evaluate_parallel_local_volatility_risk()
        for h in [0.,-1.,float('nan'),float('inf'),1e-300,.2]:
            with self.subTest(bump=h),self.assertRaises(rp.PricingError):
                plan.evaluate_parallel_local_volatility_risk(local_volatility_bump=h)
        for field,value in [('floor',.03),('cap',.06)]:
            data=json.loads(make_request(case).to_json());data['model']['local_variance_grid'][field]=value
            edge=compile_plan(case,rp.PricingRequest.from_json(json.dumps(data)))
            with self.assertRaises(rp.PricingError):
                edge.evaluate_parallel_local_volatility_risk(local_volatility_bump=.01)
        rebated=case|dict(contract=case['contract']|dict(notional=1e18,rebate=7.))
        fixed=compile_plan(rebated,make_request(rebated,history=True,dates=['2026-09-03'])).evaluate_parallel_local_volatility_risk(local_volatility_bump=.01)
        self.assertEqual(fixed.vega_estimates,[0.,0.,0.])
        self.assertEqual(fixed.vega_standard_errors,[0.,0.,0.])
        self.assertEqual(fixed.bump_difference_standard_errors,[0.,0.])

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
