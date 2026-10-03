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


def reporting_request(case, **kwargs):
    data=json.loads(make_request(case,**kwargs).to_json())
    data['model']['reporting_iv_basis']=dict(maturity_nodes=[91/365,364/365],
        log_forward_moneyness_nodes=[-.25,.25],shape=[2,2],implied_volatilities=[.2]*4)
    return rp.PricingRequest.from_json(json.dumps(data))


QUOTE_TIMES=[.25,1.25]
QUOTE_X=[-.75,0.,.75]
QUOTE_IV=[.22,.20,.21,.24,.22,.23]

def iv_request(case, surface):
    c,m=case['contract'],MARKET|case['market']
    return rp.PricingRequest('2026-09-04',
        rp.Product.barrier(1,2,'2027-09-03',m['strike'],m['barrier'],c['notional'],
            c['side'],c['direction'],c['style'],'continuous',BASES[case['base_case']]['monitoring_dates'],
            '2027-12-04',rebate=c['rebate'] or None),
        rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.],[1.,m['annual_discount']]),
            rp.DiscountCurve(11,[0.,1.],[1.,m['annual_carry']]),
            discrete_dividends=[rp.DividendEvent.fixed_cash(i+1,t,q) for i,(t,q) in enumerate(zip(m['cash_times'],m['cash_means']))]),
        surface.local_volatility_model([0.,m['fixing_time'],m['expiry_time']],[-.5,0.,.5]),
        rp.Engine.randomized_quasi_monte_carlo(32,193,scramble_count=4,antithetic=True,brownian_bridge=True),
        rp.RiskRequest())

def independent_quote_target(quotes):
    # Solve the natural cubic system directly; do not use the Rust coefficient
    # map, surface queries or finite-difference Dupire derivatives.
    times=np.asarray(QUOTE_TIMES);xs=np.asarray(QUOTE_X);q=np.asarray(quotes).reshape(2,3)
    h=np.diff(xs);matrix=np.eye(3)
    matrix[1]=[h[0],2*(h[0]+h[1]),h[1]]
    w=times[:,None]*q*q;rhs=np.zeros_like(w)
    rhs[:,1]=6*(np.diff(w,axis=1)[:,1]/h[1]-np.diff(w,axis=1)[:,0]/h[0])
    second=np.linalg.solve(matrix,rhs.T).T
    values=[]
    for t in [MARKET['fixing_time'],MARKET['fixing_time'],MARKET['expiry_time']]:
        for x in [-.5,0.,.5]:
            i=min(np.searchsorted(xs,x,side='right')-1,1)
            b=(x-xs[i])/h[i];a=1-b
            v=a*w[:,i]+b*w[:,i+1]+((a**3-a)*second[:,i]+(b**3-b)*second[:,i+1])*h[i]**2/6
            dx=(w[:,i+1]-w[:,i])/h[i]+((1-3*a*a)*second[:,i]+(3*b*b-1)*second[:,i+1])*h[i]/6
            dxx=a*second[:,i]+b*second[:,i+1]
            wt=(v[1]-v[0])/(times[1]-times[0]);b=(t-times[0])/(times[1]-times[0])
            v,dx,dxx=[(1-b)*y[0]+b*y[1] for y in (v,dx,dxx)]
            density=(1-x*dx/(2*v))**2-dx*dx/4*(1/v+.25)+dxx/2
            values.append(wt/density)
    return values


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

    def test_bucketed_local_volatility_against_independent_numpy(self):
        fixture=json.loads((DIRECTORY/'rough-continuous-bucketed-local-vol-reference.json').read_text())
        gates=fixture['acceptance'];sampling=fixture['production_sampling']
        for case in fixture['cases']:
            with self.subTest(case=case['id']):
                req=make_request(case,points=sampling['points_per_scramble'],scrambles=sampling['scramble_count'])
                plan=compile_plan(case,req)
                result=plan.evaluate_bucketed_local_volatility_risk(local_volatility_bump=fixture['sampling']['bump'],node_indices=case['node_indices'])
                data=json.loads(req.to_json());grid=data['model']['local_variance_grid']
                self.assertEqual(grid['values'],case['original_target_variances'])
                self.assertEqual(result.time_nodes,grid['time_nodes'])
                self.assertEqual(result.log_moneyness_nodes,grid['log_forward_moneyness_nodes'])
                self.assertEqual(result.node_indices,case['node_indices'])
                panels=[]
                for i,node in enumerate(case['nodes']):
                    for scenario in node['scenarios']:
                        grid['values']=scenario['target_variances']
                        external=compile_plan(case,rp.PricingRequest.from_json(json.dumps(data)))
                        np.testing.assert_allclose(external.lsv_squared_leverage,scenario['squared_leverage'],rtol=0,atol=2e-13)
                    panels.append((node['batch_means'],result.vega_estimates[i]+result.bump_differences[i],result.vega_standard_errors[i]+result.bump_difference_standard_errors[i]))
                panels.append((case['sum_batch_means'],result.sum_vega_estimates+result.sum_bump_differences,result.sum_vega_standard_errors+result.sum_bump_difference_standard_errors))
                for batches,actual,se in panels:
                    means=np.asarray(batches);errors=means.std(axis=0,ddof=1)/math.sqrt(len(means))
                    bounds=np.abs(np.asarray(actual)-means.mean(axis=0))+4*np.hypot(errors,se)
                    self.assertLess(max(bounds[:3]),gates['vega_difference_plus_4se'])
                    self.assertLess(max(bounds[3:]),gates['gap_difference_plus_4se'])
                    self.assertLess(max(se[:3]),gates['production_vega_se'])
                    self.assertLess(max(errors[:3]),gates['reference_vega_se'])
                self.assertEqual(result.recalibration_count,12)
                self.assertEqual(result.scenario_evaluated_paths,13*result.price.evaluated_paths)
                self.assertEqual(result.payoff_evaluations,result.scenario_evaluated_paths)
                self.assertEqual(result.price.plan_fingerprint,plan.plan_fingerprint)
                self.assertIn('bucketed-local-vol-recalibrated-crn-v1',result.method)
                self.assertEqual(result.uncertainty_scope,'pricing_only_fixed_calibration_seed_grid_bridge_and_bump')

    def test_bucketed_local_volatility_api_order_replay_and_validation(self):
        case=FIXTURE['cases'][0];plan=compile_plan(case)
        kwargs=dict(local_volatility_bump=.01,node_indices=[4,0])
        risk=plan.evaluate_bucketed_local_volatility_risk(**kwargs)
        base=plan.evaluate()
        self.assertEqual((risk.price.value,risk.price.standard_error),(base.value,base.standard_error))
        self.assertEqual(risk.local_volatility_bumps,[.005,.01,.02])
        self.assertLess(len(risk.time_nodes),len(plan.time_nodes))
        reverse=plan.evaluate_bucketed_local_volatility_risk(**(kwargs|dict(node_indices=[0,4])))
        self.assertEqual(risk.vega_estimates,reverse.vega_estimates[::-1])
        self.assertNotEqual(risk.risk_fingerprint,reverse.risk_fingerprint)
        one=plan.evaluate_bucketed_local_volatility_risk(**(kwargs|dict(node_indices=[4])))
        self.assertEqual(one.vega_estimates[0],risk.vega_estimates[0])
        self.assertEqual(one.sum_vega_standard_errors,one.vega_standard_errors[0])
        with self.assertRaises(AttributeError):
            risk.node_indices=[0]
        for name in ('node_indices','time_nodes','log_moneyness_nodes','sum_vega_estimates'):
            detached=getattr(risk,name);detached[0]=999
            self.assertNotEqual(getattr(risk,name)[0],999)
        detached=risk.vega_estimates;detached[0][0]=999
        self.assertNotEqual(risk.vega_estimates[0][0],999)
        parallel=compile_plan(case,worker_threads=3).evaluate_bucketed_local_volatility_risk(**kwargs)
        self.assertEqual(risk.vega_estimates,parallel.vega_estimates)
        self.assertEqual(risk.vega_standard_errors,parallel.vega_standard_errors)
        self.assertEqual(risk.sum_vega_standard_errors,parallel.sum_vega_standard_errors)
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(lambda _: plan.evaluate_bucketed_local_volatility_risk(**kwargs).sum_vega_estimates,range(2))),[risk.sum_vega_estimates]*2)
        for invalid in [[],[0,0],[9],[-1]]:
            with self.subTest(nodes=invalid),self.assertRaises((rp.PricingError,OverflowError)):
                plan.evaluate_bucketed_local_volatility_risk(**(kwargs|dict(node_indices=invalid)))
        for h in [0.,-1.,float('nan'),float('inf'),1e-300,.2]:
            with self.subTest(bump=h),self.assertRaises(rp.PricingError):
                plan.evaluate_bucketed_local_volatility_risk(**(kwargs|dict(local_volatility_bump=h)))
        with self.assertRaises(TypeError):
            plan.evaluate_bucketed_local_volatility_risk(local_volatility_bump=.01)
        rebated=case|dict(contract=case['contract']|dict(notional=1e18,rebate=7.))
        fixed=compile_plan(rebated,make_request(rebated,history=True,dates=['2026-09-03'])).evaluate_bucketed_local_volatility_risk(**kwargs)
        self.assertEqual(fixed.vega_estimates,[[0.]*3]*2)
        self.assertEqual(fixed.sum_vega_estimates,[0.]*3)
        self.assertEqual(fixed.sum_vega_standard_errors,[0.]*3)
        self.assertEqual(fixed.sum_bump_difference_standard_errors,[0.]*2)

    def test_market_iv_rebuild_against_independent_numpy_dupire(self):
        source=rp.MarketIvSurface(QUOTE_TIMES,QUOTE_X,QUOTE_IV)
        for case in FIXTURE['cases']:
            with self.subTest(case=case['id']):
                req=iv_request(case,source);data=json.loads(req.to_json())
                np.testing.assert_allclose(data['model']['local_variance_grid']['values'],independent_quote_target(QUOTE_IV),rtol=0,atol=3e-15)
                plain=compile_plan(case,req)
                plan=compile_plan(case,req,market_iv_surface=source)
                r=plan.evaluate_parallel_market_iv_risk(implied_volatility_bump=.01)
                self.assertFalse(plain.supports_market_iv_risk);self.assertTrue(plan.supports_market_iv_risk)
                self.assertNotEqual(plan.plan_fingerprint,plain.plan_fingerprint)
                self.assertEqual((r.price.value,r.price.standard_error),(plain.evaluate().value,plain.evaluate().standard_error))
                self.assertEqual(r.price.plan_fingerprint,plan.plan_fingerprint)
                for j,h in enumerate(r.implied_volatility_bumps):
                    prices=[]
                    for shift in [-h,h]:
                        data['model']['local_variance_grid']['values']=independent_quote_target(np.asarray(QUOTE_IV)+shift)
                        prices.append(compile_plan(case,rp.PricingRequest.from_json(json.dumps(data))).evaluate().value)
                    self.assertAlmostEqual(r.vega_estimates[j],(prices[1]-prices[0])/(2*h),delta=2e-9)
                local=plan.evaluate_parallel_local_volatility_risk(local_volatility_bump=.01)
                self.assertGreater(abs(local.vega-r.vega),1e-5)
                self.assertEqual(r.quote_maturity_nodes,QUOTE_TIMES);self.assertEqual(r.quote_log_moneyness_nodes,QUOTE_X)
                self.assertEqual(r.implied_volatilities,QUOTE_IV)
                self.assertEqual(r.implied_volatility_bumps,[.005,.01,.02])
                self.assertEqual(r.recalibration_count,6);self.assertEqual(r.payoff_evaluations,7*r.price.evaluated_paths)
                self.assertEqual(r.scenario_evaluated_paths,r.payoff_evaluations)
                self.assertEqual(r.vega_per_vol_point,.01*r.vega)
                self.assertEqual(r.standard_error_per_vol_point,.01*r.standard_error)
                self.assertEqual(r.interpolation,'natural-cubic-w-linear-time-v1')
                self.assertEqual(r.uncertainty_scope,'pricing_only_fixed_calibration_seed_grid_bridge_and_quote_interpolation')

    def test_market_iv_api_replay_history_and_validation(self):
        source=rp.MarketIvSurface(QUOTE_TIMES,QUOTE_X,QUOTE_IV)
        self.assertEqual(source.maturity_nodes,QUOTE_TIMES);self.assertEqual(source.log_moneyness_nodes,QUOTE_X)
        self.assertEqual(source.implied_volatilities,QUOTE_IV);self.assertEqual(source.interpolation,'natural-cubic-w-linear-time-v1')
        with self.assertRaises(AttributeError):source.implied_volatilities=[]
        detached=source.implied_volatilities;detached[0]=999.;self.assertEqual(source.implied_volatilities,QUOTE_IV)
        case=FIXTURE['cases'][0];req=iv_request(case,source)
        plan=compile_plan(case,req,market_iv_surface=source)
        kwargs=dict(implied_volatility_bump=.01)
        r=plan.evaluate_parallel_market_iv_risk(**kwargs)
        worker=compile_plan(case,req,market_iv_surface=source,worker_threads=3).evaluate_parallel_market_iv_risk(**kwargs)
        self.assertEqual(r.vega_estimates,worker.vega_estimates);self.assertEqual(r.vega_standard_errors,worker.vega_standard_errors)
        self.assertEqual(r.bump_difference_standard_errors,worker.bump_difference_standard_errors)
        with self.assertRaises(AttributeError):r.vega=0.
        detached=r.implied_volatilities;detached[0]=999.;self.assertEqual(r.implied_volatilities,QUOTE_IV)
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(lambda _:plan.evaluate_parallel_market_iv_risk(**kwargs).vega_estimates,range(2))),[r.vega_estimates]*2)
        with self.assertRaises(rp.PricingError):compile_plan(case,req).evaluate_parallel_market_iv_risk(**kwargs)
        wrong=rp.MarketIvSurface(QUOTE_TIMES,QUOTE_X,(np.asarray(QUOTE_IV)+.001).tolist())
        with self.assertRaises(rp.PricingError):compile_plan(case,req,market_iv_surface=wrong)
        for h in [0.,-.01,.2,1e-300,float('nan'),float('inf')]:
            with self.assertRaises(rp.PricingError):plan.evaluate_parallel_market_iv_risk(implied_volatility_bump=h)
        for changes in [dict(floor=.2),dict(cap=.01)]:
            with self.assertRaises(rp.ValidationError):source.local_volatility_model([0.,.5,1.],[-.5,0.,.5],**changes)
        with self.assertRaises(rp.ValidationError):source.local_volatility_model([0.,.5,1.],[-.8,0.,.5])
        for times,xs,iv in [(QUOTE_TIMES,QUOTE_X,[.2]*5),([0.,1.],QUOTE_X,[.2]*6),
                            (QUOTE_TIMES,QUOTE_X,[float('nan')]*6),([.5,1.],QUOTE_X,[.5]*3+[.1]*3)]:
            with self.assertRaises(rp.ValidationError):rp.MarketIvSurface(times,xs,iv)
        # Only the largest positive ladder shift exceeds this cap: no partial
        # scenario valuation or clipping is allowed to turn it into a result.
        flat=rp.MarketIvSurface(QUOTE_TIMES,QUOTE_X,[.2]*6)
        capped=json.loads(iv_request(case,flat).to_json());capped['model']['local_variance_grid']['cap']=.045
        capped_plan=compile_plan(case,rp.PricingRequest.from_json(json.dumps(capped)),market_iv_surface=flat)
        with self.assertRaises(rp.PricingError):capped_plan.evaluate_parallel_market_iv_risk(**kwargs)
        self.assertNotEqual(r.risk_fingerprint,plan.evaluate_parallel_market_iv_risk(implied_volatility_bump=.005).risk_fingerprint)
        data=json.loads(req.to_json());data['product'].update(style=dict(type='knock_out'),historical_hit=True,
            monitoring_dates=['2026-09-03','2027-09-03'],notional=1e18,rebate=7.)
        fixed=compile_plan(case,rp.PricingRequest.from_json(json.dumps(data)),market_iv_surface=source).evaluate_parallel_market_iv_risk(**kwargs)
        self.assertEqual(fixed.vega_estimates,[0.]*3);self.assertEqual(fixed.vega_standard_errors,[0.]*3)
        self.assertEqual(fixed.bump_difference_standard_errors,[0.]*2)

    def test_reporting_iv_projection_against_independent_numpy(self):
        fixture=json.loads((DIRECTORY/'rough-continuous-reporting-iv-reference.json').read_text())
        case=fixture['case'];gates=fixture['acceptance'];sampling=fixture['production_sampling']
        req=reporting_request(case,points=sampling['points_per_scramble'],scrambles=sampling['scramble_count'])
        plan=compile_plan(case,req)
        # Separate request compiles establish retained recalibration inputs.
        data=json.loads(req.to_json())
        for node in case['nodes']:
            for scenario in node['scenarios']:
                data['model']['local_variance_grid']['values']=scenario['target_variances']
                external=compile_plan(case,rp.PricingRequest.from_json(json.dumps(data)))
                np.testing.assert_allclose(external.lsv_squared_leverage,scenario['squared_leverage'],rtol=0,atol=2e-13)
        for panel in fixture['projections']:
            with self.subTest(threshold=panel['threshold']):
                r=plan.evaluate_reporting_iv_projection(local_volatility_bump=.01,relative_density_threshold=panel['threshold'])
                actual=np.r_[np.c_[r.bucket_estimates,r.bump_differences].ravel(),r.pre_projection_estimates,r.projected_sum_estimates,r.residual_estimates]
                se=np.r_[np.c_[r.bucket_standard_errors,r.bump_difference_standard_errors].ravel(),r.pre_projection_standard_errors,r.projected_sum_standard_errors,r.residual_standard_errors]
                means=np.asarray(panel['batch_means']);errors=means.std(axis=0,ddof=1)/math.sqrt(len(means))
                bounds=np.abs(actual-means.mean(axis=0))+4*np.hypot(errors,se)
                self.assertLess(max(bounds),gates['difference_plus_4se'])
                self.assertLess(max(se),gates['production_se']);self.assertLess(max(errors),gates['reference_se'])
                self.assertEqual(r.active_domain_start_indices,[d[0] for d in panel['active_domains']])
                self.assertEqual(r.active_domain_end_indices,[d[1] for d in panel['active_domains']])
                np.testing.assert_allclose(r.excluded_probability_masses,panel['excluded_probability_masses'],rtol=0,atol=1e-14)
                self.assertEqual(r.reporting_maturity_nodes,[91/365,364/365])
                self.assertEqual(r.reporting_log_moneyness_nodes,[-.25,.25])
                self.assertEqual(r.reporting_implied_volatilities,[.2]*4)
                self.assertEqual(r.positive_target_time_nodes,[182/365,364/365])
                self.assertEqual(r.target_log_moneyness_nodes,[-.5,0.,.5])
                self.assertEqual(r.local_volatility_bumps,[.005,.01,.02])
                self.assertEqual(r.recalibration_count,54)
                self.assertEqual(r.scenario_evaluated_paths,55*r.price.evaluated_paths)
                self.assertEqual(r.payoff_evaluations,r.scenario_evaluated_paths)
                np.testing.assert_allclose(np.asarray(r.projected_sum_estimates)+r.residual_estimates,r.pre_projection_estimates,rtol=0,atol=2e-12)
                self.assertEqual(r.projection_policy,'local_vega_density_reporting_iv_projection_v1')
                self.assertEqual(r.uncertainty_scope,'pricing_only_fixed_calibration_seed_grid_bridge_bump_and_projection')

    def test_reporting_iv_joint_estimator_covariance(self):
        case=FIXTURE['cases'][0]
        kwargs=dict(local_volatility_bump=.01,relative_density_threshold=1e-8)
        labels=['price']
        for b in range(4):
            labels.extend(f'bucket_estimates[{b}][{j}]' for j in range(3))
            labels.extend(f'bump_differences[{b}][{j}]' for j in range(2))
        for name in ('pre_projection_estimates','projected_sum_estimates','residual_estimates'):
            labels.extend(f'{name}[{j}]' for j in range(3))
        for qmc in (False,True):
            for antithetic in (False,True):
                with self.subTest(qmc=qmc,antithetic=antithetic):
                    data=json.loads(reporting_request(case,points=16).to_json())
                    if not qmc:
                        data['engine']=dict(type='pseudo_monte_carlo',independent_sampling_units=64,master_seed=193)
                    data['engine']['variance_reduction']=dict(antithetic=antithetic,brownian_bridge=True)
                    req=rp.PricingRequest.from_json(json.dumps(data))
                    plan=compile_plan(case,req,reduction_block_size=8)
                    plain=plan.evaluate_reporting_iv_projection(**kwargs)
                    r=plan.evaluate_reporting_iv_projection(**kwargs,full_covariance=True)
                    self.assertIsNone(plain.estimator_covariance)
                    self.assertEqual(r.covariance_labels,labels)
                    self.assertEqual(plain.covariance_labels,labels)
                    for name in ('bucket_estimates','bucket_standard_errors','bump_differences',
                                 'bump_difference_standard_errors','pre_projection_estimates',
                                 'pre_projection_standard_errors','projected_sum_estimates',
                                 'projected_sum_standard_errors','residual_estimates','residual_standard_errors',
                                 'payoff_evaluations','recalibration_count'):
                        self.assertEqual(getattr(r,name),getattr(plain,name))
                    self.assertEqual((r.price.value,r.price.standard_error),(plain.price.value,plain.price.standard_error))
                    self.assertNotEqual(r.risk_fingerprint,plain.risk_fingerprint)
                    covariance=np.asarray(r.estimator_covariance)
                    errors=np.r_[r.price.standard_error,np.c_[r.bucket_standard_errors,r.bump_difference_standard_errors].ravel(),
                                 r.pre_projection_standard_errors,r.projected_sum_standard_errors,r.residual_standard_errors]
                    self.assertEqual(covariance.shape,(30,30))
                    np.testing.assert_array_equal(covariance,covariance.T)
                    np.testing.assert_allclose(covariance.diagonal(),errors**2,rtol=2e-14,atol=1e-15)
                    self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(),-1e-12*max(1.,np.trace(covariance)))
                    # Reconstruct sums/gaps/residual errors from their component
                    # covariance, including negative cross-covariances.
                    for j in range(3):
                        w=np.zeros(30);w[1+j:21:5]=1.
                        self.assertAlmostEqual(float(w@covariance@w),r.projected_sum_standard_errors[j]**2,delta=1e-10)
                        w=np.zeros(30);w[21+j]=1.;w[24+j]=-1.
                        self.assertAlmostEqual(float(w@covariance@w),r.residual_standard_errors[j]**2,delta=1e-10)
                    for b in range(4):
                        for j in range(2):
                            w=np.zeros(30);w[1+5*b+j]=1.;w[2+5*b+j]=-1.
                            self.assertAlmostEqual(float(w@covariance@w),r.bump_difference_standard_errors[b][j]**2,delta=1e-10)
                    worker=compile_plan(case,req,reduction_block_size=8,worker_threads=3).evaluate_reporting_iv_projection(**kwargs,full_covariance=True)
                    self.assertEqual(r.estimator_covariance,worker.estimator_covariance)
                    with self.assertRaises(AttributeError):r.estimator_covariance=[]
                    detached=r.estimator_covariance;detached[0][0]=999.
                    self.assertNotEqual(r.estimator_covariance[0][0],999.)
                    detached=r.covariance_labels;detached[0]='changed'
                    self.assertEqual(r.covariance_labels[0],'price')
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        replay=list(pool.map(lambda _:plan.evaluate_reporting_iv_projection(**kwargs,full_covariance=True).estimator_covariance,range(2)))
                    self.assertEqual(replay,[r.estimator_covariance]*2)
        rebated=case|dict(contract=case['contract']|dict(notional=1e18,rebate=7.))
        fixed=compile_plan(rebated,reporting_request(rebated,points=16,history=True,dates=['2026-09-03'])).evaluate_reporting_iv_projection(**kwargs,full_covariance=True)
        self.assertEqual(fixed.estimator_covariance,[[0.]*30]*30)

    def test_reporting_iv_api_history_replay_and_validation(self):
        case=FIXTURE['cases'][0];req=reporting_request(case,points=16)
        plan=compile_plan(case,req);kwargs=dict(local_volatility_bump=.01,relative_density_threshold=.9)
        r=plan.evaluate_reporting_iv_projection(**kwargs)
        p=plan.evaluate()
        self.assertEqual((p.value,p.standard_error),(r.price.value,r.price.standard_error))
        self.assertEqual(p.value,compile_plan(case,make_request(case,points=16)).evaluate().value)
        worker=compile_plan(case,req,worker_threads=3).evaluate_reporting_iv_projection(**kwargs)
        self.assertEqual(r.bucket_estimates,worker.bucket_estimates)
        self.assertEqual(r.bucket_standard_errors,worker.bucket_standard_errors)
        self.assertEqual(r.residual_standard_errors,worker.residual_standard_errors)
        self.assertEqual(r.price.plan_fingerprint,plan.plan_fingerprint)
        with self.assertRaises(AttributeError):r.relative_density_threshold=1.
        a=r.bucket_estimates;a[0][0]=999.;self.assertNotEqual(r.bucket_estimates[0][0],999.)
        a=r.active_domain_start_indices;a[0]=999;self.assertNotEqual(r.active_domain_start_indices[0],999)
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(lambda _:plan.evaluate_reporting_iv_projection(**kwargs).residual_estimates,range(2))),[r.residual_estimates]*2)
        for changes in [dict(local_volatility_bump=.005),dict(relative_density_threshold=1e-8)]:
            self.assertNotEqual(r.risk_fingerprint,plan.evaluate_reporting_iv_projection(**(kwargs|changes)).risk_fingerprint)
        with self.assertRaises(rp.PricingError):compile_plan(case).evaluate_reporting_iv_projection(**kwargs)
        for t in [0.,-1.,1.01,float('nan'),float('inf')]:
            with self.assertRaises(rp.PricingError):plan.evaluate_reporting_iv_projection(**(kwargs|dict(relative_density_threshold=t)))
        for h in [0.,-1.,.2,1e-300,float('nan'),float('inf')]:
            with self.assertRaises(rp.PricingError):plan.evaluate_reporting_iv_projection(**(kwargs|dict(local_volatility_bump=h)))
        with self.assertRaises(TypeError):plan.evaluate_reporting_iv_projection(local_volatility_bump=.01)
        bad=json.loads(req.to_json());bad['model']['reporting_iv_basis']['maturity_nodes']=[.75,1.]
        with self.assertRaises(rp.PricingError):compile_plan(case,rp.PricingRequest.from_json(json.dumps(bad))).evaluate_reporting_iv_projection(**kwargs)
        rebated=case|dict(contract=case['contract']|dict(notional=1e18,rebate=7.))
        fixed=compile_plan(rebated,reporting_request(rebated,points=16,history=True,dates=['2026-09-03'])).evaluate_reporting_iv_projection(**kwargs)
        self.assertEqual(fixed.bucket_estimates,[[0.]*3]*4)
        self.assertEqual(fixed.residual_standard_errors,[0.]*3)

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
