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
