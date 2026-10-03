"""Independent NumPy physical log-Spot bridge approximation and mesh refinement.

No production paths, payoffs or random numbers are used. Leverage values are
retained calibration inputs. The reference validates a finite-grid approximation;
it does not provide an exact nonlinear/rough continuous hitting law.
"""
import json
import math
from pathlib import Path

import numpy as np

from rough_dividend_survival_reference import hybrid_weights
from rough_dividend_hard_refinement import coarsen, refine_case

DIRECTORY = Path(__file__).resolve().parents[2] / 'fixtures/stochastic-dividends'


def inputs():
    source = json.loads((DIRECTORY / 'rough-survival-barrier-reference.json').read_text())
    market = json.loads((DIRECTORY / source['market_contract_fixture']).read_text())
    return {c['id']: c for c in source['cases']}, market


def path_values(base, market, z, *, direction='up', side='call', style='knock_out',
                notional=1., rebate=0., monitoring_end=None, historical_hit=False):
    """Return (bridge price, endpoint-only price); z layout is (D,Vorth,near,Eorth)."""
    times = np.asarray(base['times'])
    dt = np.diff(times)
    n, steps, factors = z.shape
    assert factors == 4 and steps == len(dt)
    surface = np.asarray(base['squared_leverage']).reshape(len(times), -1)
    h, eta, kappa, alpha = (base[k] for k in ('hurst', 'eta', 'kappa', 'equity_linkage'))
    rho, nu = market['equity_dividend_correlation'], market['dividend_volatility']
    sv, dv = base['equity_volatility_correlation'], base['dividend_volatility_correlation']
    orth = (sv-rho*dv)/math.sqrt(1-dv*dv)
    eq = rho*z[:, :, 0] + orth*z[:, :, 1] + math.sqrt(1-rho*rho-orth*orth)*z[:, :, 3]
    weights, residual, variance = hybrid_weights(times, h)
    increments = np.sqrt(dt)*(dv*z[:, :, 0]+math.sqrt(1-dv*dv)*z[:, :, 1])
    drivers = increments @ weights.T
    drivers[:, 1:] += residual*z[:, :, 2]
    growth = market['annual_carry']/market['annual_discount']
    funded = market['spot']-sum(q/growth**t for q, t in zip(market['cash_means'], market['cash_times']))

    def coefficients(t):
        a, b, c, cash = funded*growth**t, 0., 0., 0.
        for q, ex in zip(market['cash_means'], market['cash_times']):
            if ex == t:
                cash += q
            elif ex > t:
                amount = q*growth**(t-ex)
                decay = math.exp(-kappa*(ex-t))
                a += amount*(1-decay)*alpha
                b += amount*decay
                c += amount*(1-decay)*(1-alpha)
        return a, b, c, cash

    end = times[-1] if monitoring_end is None else monitoring_end
    hit = lambda s: s >= market['barrier'] if direction == 'up' else s <= market['barrier']
    f, y = np.ones(n), np.ones(n)
    a, b, c, _ = coefficients(0.)
    post = a*f+b*y+c
    initial = historical_hit or (end >= 0 and hit(market['spot']))
    survival, discrete = np.full(n, float(not initial)), np.full(n, float(not initial))
    for i, step in enumerate(dt):
        sigma = np.sqrt(np.interp(np.log(f), base['log_nodes'], surface[i])) * np.exp(
            eta*drivers[:, i]/2-eta*eta*variance[i]/4)
        # Two correlated stochastic loadings; never use residual variance alone.
        e, d = a*f*sigma/post, b*y*nu/post
        log_variance = e*e+d*d+2*rho*e*d
        next_f = f*np.exp(-sigma*sigma*step/2+sigma*math.sqrt(step)*eq[:, i])
        half = math.exp(-kappa*step/2)
        mid_y = half*y+(1-half)*(alpha*f+1-alpha)
        next_y = half*mid_y*np.exp(nu*math.sqrt(step)*z[:, i, 0]-nu*nu*step/2)+(1-half)*(alpha*next_f+1-alpha)
        a, b, c, cash = coefficients(times[i+1])
        next_post = a*next_f+b*next_y+c
        pre = next_post+cash*next_y
        if times[i+1] <= end and not historical_hit:
            safe = ~(hit(post) | hit(pre) | hit(next_post))
            interval = np.zeros(n)
            positive = safe & (log_variance > 0)
            exponent = -2*np.log(market['barrier']/post[positive])*np.log(market['barrier']/pre[positive])/(log_variance[positive]*step)
            interval[positive] = -np.expm1(exponent)
            interval[safe & (log_variance == 0)] = 1.
            survival *= interval
            discrete *= safe
        f, y, post = next_f, next_y, next_post
    intrinsic = np.maximum(post-market['strike'] if side == 'call' else market['strike']-post, 0.)
    probabilities = np.stack((survival, discrete), axis=1)
    active = probabilities if style == 'knock_out' else 1-probabilities
    return (active*notional*intrinsic[:, None]+(1-active)*rebate)*market['annual_discount']**market['payment_time']


def batch_means(base, market, contract, *, seed=20261003, batches=32, pairs=8192):
    rng = np.random.Generator(np.random.PCG64(seed))
    means = []
    for _ in range(batches):
        z = rng.standard_normal((pairs, len(base['times'])-1, 4))
        mean = (path_values(base, market, z, **contract)+path_values(base, market, -z, **contract))/2
        means.append(mean.mean(axis=0))
    return np.asarray(means)


def spot_bump_means(base, market, contract, *, bump=1., seed=20261005, batches=32, pairs=8192):
    """Finite central price differences on paired PCG64 paths, no production risk."""
    rng = np.random.Generator(np.random.PCG64(seed))
    means = []
    for _ in range(batches):
        z = rng.standard_normal((pairs, len(base['times'])-1, 4))
        columns = []
        for h in [bump/2, bump, 2*bump]:
            scenarios = []
            for spot in [market['spot']-h, market['spot']+h]:
                shifted = market | dict(spot=spot)
                scenarios.append((path_values(base, shifted, z, **contract)[:,0]
                                  +path_values(base, shifted, -z, **contract)[:,0])/2)
            columns.append((scenarios[1]-scenarios[0])/(2*h))
        columns.extend([columns[0]-columns[1], columns[1]-columns[2]])
        means.append(np.stack(columns,axis=1).mean(axis=0))
    return np.asarray(means)


def gamma_bump_means(base, market, contract, *, bump=2., seed=20261006, batches=32, pairs=8192):
    """Independent three-price second differences, paired before batch reduction."""
    rng = np.random.Generator(np.random.PCG64(seed))
    means = []
    for _ in range(batches):
        z = rng.standard_normal((pairs, len(base['times'])-1, 4))
        center = (path_values(base, market, z, **contract)[:,0]
                  +path_values(base, market, -z, **contract)[:,0])/2
        columns = []
        for h in [bump/2, bump, 2*bump]:
            scenarios = []
            for spot in [market['spot']-h, market['spot']+h]:
                shifted = market | dict(spot=spot)
                scenarios.append((path_values(base, shifted, z, **contract)[:,0]
                                  +path_values(base, shifted, -z, **contract)[:,0])/2)
            columns.append((scenarios[1]+scenarios[0]-2*center)/(h*h))
        columns.extend([columns[0]-columns[1], columns[1]-columns[2]])
        means.append(np.stack(columns,axis=1).mean(axis=0))
    return np.asarray(means)


def local_volatility_bump_means(base, market, contract, scenarios, *, bump=.01, seed=20261007, batches=32, pairs=8192):
    """Independent valuation conditional on retained, separately recalibrated surfaces.

    This does not independently implement the particle calibration algorithm.
    Every scenario re-evolves f/Y with the same PCG64 normals and its own leverage.
    """
    rng=np.random.Generator(np.random.PCG64(seed))
    means=[]
    for _ in range(batches):
        z=rng.standard_normal((pairs,len(base['times'])-1,4))
        values=[]
        for scenario in scenarios:
            shifted=base | dict(squared_leverage=scenario['squared_leverage'])
            values.append((path_values(shifted,market,z,**contract)[:,0]
                           +path_values(shifted,market,-z,**contract)[:,0])/2)
        columns=[(values[2*j+1]-values[2*j])/(2*h) for j,h in enumerate([bump/2,bump,2*bump])]
        columns.extend([columns[0]-columns[1],columns[1]-columns[2]])
        means.append(np.stack(columns,axis=1).mean(axis=0))
    return np.asarray(means)


def verify_local_volatility_fixture():
    fixture=json.loads((DIRECTORY/'rough-continuous-local-vol-reference.json').read_text())
    bases,market=inputs()
    for case in fixture['cases']:
        means=local_volatility_bump_means(bases[case['base_case']],market|case['market'],case['contract'],case['scenarios'],**fixture['sampling'])
        np.testing.assert_allclose(means,case['batch_means'],rtol=0,atol=1e-9)
        errors=means.std(axis=0,ddof=1)/math.sqrt(len(means))
        assert max(errors[:3])<fixture['acceptance']['reference_vega_se']
        print(json.dumps(dict(scope=fixture['scope'],case=case['id'],estimates=means.mean(axis=0).tolist(),standard_errors=errors.tolist())),flush=True)


def verify_bucketed_local_volatility_fixture():
    fixture=json.loads((DIRECTORY/'rough-continuous-bucketed-local-vol-reference.json').read_text())
    bases,market=inputs()
    for case in fixture['cases']:
        panels=[]
        for node in case['nodes']:
            means=local_volatility_bump_means(bases[case['base_case']],market|case['market'],case['contract'],node['scenarios'],**fixture['sampling'])
            np.testing.assert_allclose(means,node['batch_means'],rtol=0,atol=1e-9)
            panels.append(means)
            errors=means.std(axis=0,ddof=1)/math.sqrt(len(means))
            assert max(errors[:3])<fixture['acceptance']['reference_vega_se']
            print(json.dumps(dict(scope=fixture['scope'],case=case['id'],node_index=node['node_index'],estimates=means.mean(axis=0).tolist(),standard_errors=errors.tolist())),flush=True)
        summed=np.sum(panels,axis=0)
        np.testing.assert_allclose(summed,case['sum_batch_means'],rtol=0,atol=1e-9)
        errors=summed.std(axis=0,ddof=1)/math.sqrt(len(summed))
        assert max(errors[:3])<fixture['acceptance']['reference_vega_se']
        print(json.dumps(dict(scope=fixture['scope'],case=case['id'],node_indices=case['node_indices'],sum_estimates=summed.mean(axis=0).tolist(),sum_standard_errors=errors.tolist())),flush=True)


def reporting_iv_weights(target_times, target_x, reporting_times, reporting_x, volatility, threshold):
    """Independent flat-IV density, lumped hats and bilinear reporting map.

    This reproduces the reporting convention only; it is not an IV/Dupire Jacobian.
    """
    x=np.asarray(target_x);areas=np.r_[np.diff(x)[0]/2,(x[2:]-x[:-2])/2,np.diff(x)[-1]/2]
    weights=np.zeros((len(target_times)*len(x),len(reporting_times)*len(reporting_x)))
    domains=[];excluded=[]
    for ti,t in enumerate(target_times):
        if t==0: continue
        std=volatility*math.sqrt(t);d2=-x/std-std/2
        density=np.exp(-d2*d2/2)/(math.sqrt(2*math.pi)*np.exp(x)*std)
        qualifies=density/density.max()>=threshold
        forward=int(np.argmin(np.abs(x)));assert qualifies[forward]
        lo=hi=forward
        while lo>0 and qualifies[lo-1]: lo-=1
        while hi+1<len(x) and qualifies[hi+1]: hi+=1
        domains.append([lo,hi])
        excluded.append(float(sum(density[i]*math.exp(x[i])*areas[i] for i in range(len(x)) if not lo<=i<=hi)))
        k=int(np.clip(np.searchsorted(reporting_times,t,side='right')-1,0,len(reporting_times)-2))
        tw=(t-reporting_times[k])/(reporting_times[k+1]-reporting_times[k])
        for i in range(lo,hi+1):
            q=float(np.clip(x[i],reporting_x[0],reporting_x[-1]))
            l=int(np.clip(np.searchsorted(reporting_x,q,side='right')-1,0,len(reporting_x)-2))
            xw=(q-reporting_x[l])/(reporting_x[l+1]-reporting_x[l])
            for rt,w in [(k,1-tw),(k+1,tw)]:
                for rx,u in [(l,1-xw),(l+1,xw)]: weights[ti*len(x)+i,rt*len(reporting_x)+rx]+=w*u/areas[i]
    return weights,domains,excluded


def reporting_iv_projection_batches(node_batches, weights):
    # [node,batch,half/base/double/gaps] -> [batch,reporting bucket,ladder/gaps].
    nodes=np.asarray(node_batches)
    buckets=np.einsum('ijk,ib->jbk',nodes[:,:,:3],weights)
    gaps=np.stack([buckets[:,:,0]-buckets[:,:,1],buckets[:,:,1]-buckets[:,:,2]],axis=2)
    columns=np.concatenate([buckets,gaps],axis=2)
    pre=nodes[:,:,:3].sum(axis=0);projected=buckets.sum(axis=1)
    return np.concatenate([columns.reshape(len(pre),-1),pre,projected,pre-projected],axis=1)


def verify_reporting_iv_fixture():
    fixture=json.loads((DIRECTORY/'rough-continuous-reporting-iv-reference.json').read_text())
    bases,market=inputs();case=fixture['case'];base=bases[case['base_case']]
    nodes=[]
    for node in case['nodes']:
        means=local_volatility_bump_means(base,market|case['market'],case['contract'],node['scenarios'],**fixture['sampling'])
        np.testing.assert_allclose(means,node['batch_means'],rtol=0,atol=1e-9);nodes.append(means)
    for panel in fixture['projections']:
        weights,domains,excluded=reporting_iv_weights(**fixture['reporting'],threshold=panel['threshold'])
        np.testing.assert_allclose(weights,panel['weights'],rtol=0,atol=1e-14)
        assert domains==panel['active_domains']
        np.testing.assert_allclose(excluded,panel['excluded_probability_masses'],rtol=0,atol=1e-14)
        means=reporting_iv_projection_batches(nodes,weights)
        np.testing.assert_allclose(means,panel['batch_means'],rtol=0,atol=1e-9)
        print(json.dumps(dict(scope=fixture['scope'],threshold=panel['threshold'],estimates=means.mean(axis=0).tolist(),standard_errors=(means.std(axis=0,ddof=1)/math.sqrt(len(means))).tolist())),flush=True)


def refinement_means(base, market, contract, coarse_steps, *, seed=20261004, batches=16, pairs=2048):
    """Coupled coarse/fine Brownian increments and exact newest-cell integrals."""
    fine = refine_case(base, 2*coarse_steps)
    coarse = refine_case(base, coarse_steps)
    rng = np.random.Generator(np.random.PCG64(seed))
    means = []
    for _ in range(batches):
        z = rng.standard_normal((pairs, 2*coarse_steps, 4))
        extra = rng.standard_normal((pairs, coarse_steps))
        cz = coarsen(z, extra, base['hurst'], 2, base['dividend_volatility_correlation'])
        values = [(path_values(c, market, normals, **contract)+path_values(c, market, -normals, **contract))/2
                  for c, normals in ((coarse, cz), (fine, z))]
        means.append(np.concatenate((values[0].mean(axis=0), values[1].mean(axis=0))))
    return np.asarray(means)


def verify_fixture():
    fixture = json.loads((DIRECTORY / 'rough-continuous-barrier-reference.json').read_text())
    bases, market = inputs()
    for case in fixture['cases']:
        inputs_market = market | case['market']
        means = batch_means(bases[case['base_case']], inputs_market, case['contract'], **fixture['sampling'])
        np.testing.assert_allclose(means, case['batch_means'], rtol=0, atol=1e-9)
        se = means.std(axis=0, ddof=1)/math.sqrt(len(means))
        assert se[0] < fixture['acceptance']['reference_price_se']
        print(json.dumps(dict(scope=fixture['scope'], case=case['id'], price=float(means[:, 0].mean()),
                              standard_error=float(se[0]))), flush=True)
    for case in fixture['cases']:
        means = spot_bump_means(bases[case['base_case']], market | case['market'],
                               case['contract'], **fixture['spot_bump_sampling'])
        np.testing.assert_allclose(means, case['spot_bump_batch_means'], rtol=0, atol=1e-9)
        errors = means.std(axis=0, ddof=1)/math.sqrt(len(means))
        assert np.max(errors[:3]) < fixture['spot_bump_acceptance']['reference_delta_se']
        print(json.dumps(dict(scope='finite_bump_continuous_bridge_spot_risk', case=case['id'],
            spot_bumps=[.5,1.,2.], estimates=means.mean(axis=0).tolist(), standard_errors=errors.tolist())), flush=True)
    for case in fixture['cases']:
        means = gamma_bump_means(bases[case['base_case']], market | case['market'],
                                case['contract'], **fixture['gamma_bump_sampling'])
        np.testing.assert_allclose(means, case['gamma_bump_batch_means'], rtol=0, atol=1e-9)
        errors = means.std(axis=0, ddof=1)/math.sqrt(len(means))
        assert np.max(errors[:3]) < fixture['gamma_bump_acceptance']['reference_gamma_se']
        print(json.dumps(dict(scope='finite_bump_continuous_bridge_gamma', case=case['id'],
            spot_bumps=[1.,2.,4.], estimates=means.mean(axis=0).tolist(), standard_errors=errors.tolist())), flush=True)
    for case in fixture['refinement']:
        source = next(c for c in fixture['cases'] if c['id'] == case['case'])
        means = refinement_means(bases[source['base_case']], market | source['market'], source['contract'],
                                 case['coarse_steps'], **fixture['refinement_sampling'])
        np.testing.assert_allclose(means, case['batch_means'], rtol=0, atol=1e-9)
        delta = means[:, 2]-means[:, 0]
        bound = abs(delta.mean())+4*delta.std(ddof=1)/math.sqrt(len(delta))
        if case['coarse_steps'] == 64:
            assert bound < fixture['acceptance']['last_paired_price_change_bound']
        print(json.dumps(dict(scope='fixed_surface_continuous_bridge_refinement', case=case['case'],
            steps=[case['coarse_steps'],2*case['coarse_steps']], bridge_change=float(delta.mean()),
            abs_change_plus_4se=float(bound))), flush=True)


if __name__ == '__main__':
    verify_fixture()
    verify_local_volatility_fixture()
    verify_bucketed_local_volatility_fixture()
    verify_reporting_iv_fixture()
