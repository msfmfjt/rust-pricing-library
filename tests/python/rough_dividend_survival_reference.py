"""Independent hard Barrier price/Delta by sequential survival conditioning.

NumPy evolves a frozen rough-LSV discrete law. Each monitored equity normal
is sampled within its survival region; the changing probability and inverse-CDF
transport are differentiated analytically. The final equity normal is
integrated exactly. No production path, payoff, risk or RNG calls are used.
"""
import math
from statistics import NormalDist

import numpy as np

from rough_dividend_barrier_reference import cdf
from rough_dividend_conditional_barrier import density


def inverse_cdf(p):
    p = np.asarray(p)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError('survival quantiles must be strictly inside (0, 1)')
    return np.fromiter(map(NormalDist().inv_cdf, p.flat), float, count=p.size).reshape(p.shape)


def hybrid_weights(times, h):
    """L2 cell averages plus an independent newest-cell residual."""
    dt = np.diff(times)
    weights = np.zeros((len(dt) + 1, len(dt)))
    residual = dt ** h * (0.5 - h) / (h + 0.5)
    for i in range(1, len(times)):
        weights[i, :i] = (math.sqrt(2 * h) / (h + 0.5)
            * ((times[i] - times[:i]) ** (h + 0.5)
               - (times[i] - times[1:i + 1]) ** (h + 0.5)) / dt[:i])
    variance = (weights * weights) @ dt
    variance[1:] += residual * residual
    return weights, residual, variance


def path_values(case, market, normals, uniforms, *, spot=None, direction="up",
                side="call", style="knock_in", notional=1., rebate=0.):
    """Per-path price/analytic Delta; normals have shape (paths, steps, 3)."""
    if direction not in ('up', 'down') or side not in ('call', 'put') or style not in ('knock_in', 'knock_out'):
        raise ValueError('invalid Barrier direction, side or style')
    up, call = direction == 'up', side == 'call'
    spot = market['spot'] if spot is None else spot
    times = np.asarray(case['times'])
    dt = np.diff(times)
    n, steps = uniforms.shape
    assert normals.shape == (n, steps, 3)
    nodes = np.asarray(case['log_nodes'])
    surface = np.asarray(case['squared_leverage']).reshape(len(times), len(nodes))
    h, eta, kappa = case['hurst'], case['eta'], case['kappa']
    alpha, rho_sv, rho_dv = (case[k] for k in
        ('equity_linkage', 'equity_volatility_correlation', 'dividend_volatility_correlation'))
    rho, nu = market['equity_dividend_correlation'], market['dividend_volatility']
    orth = (rho_sv - rho * rho_dv) / math.sqrt(1 - rho_dv * rho_dv)
    conditional_root = math.sqrt(1 - rho * rho - orth * orth)
    growth = market['annual_carry'] / market['annual_discount']
    funded = spot - sum(q / growth ** t for q, t in zip(market['cash_means'], market['cash_times']))
    if funded <= 0:
        raise ValueError('positive funded residual equity required')
    weights, residual, variance = hybrid_weights(times, h)
    dv = np.sqrt(dt) * (rho_dv * normals[:, :, 0] + math.sqrt(1 - rho_dv * rho_dv) * normals[:, :, 1])
    # Only previous-node history drives a step. Newest residuals from older
    # nodes are not reused by the hybrid scheme's older-cell averages.
    drivers = dv @ weights.T
    drivers[:, 1:] += residual * normals[:, :, 2]
    monitors = set(case['monitoring_indices'])
    if 0 in monitors and spot == market['barrier']:
        raise ValueError('Spot Delta is undefined at the initial monitoring boundary')
    initial_hit = 0 in monitors and (spot > market['barrier'] if up else spot < market['barrier'])

    def coefficients(t):
        a, b, c, cash = funded * growth ** t, 0., 0., 0.
        for q, ex in zip(market['cash_means'], market['cash_times']):
            if ex == t:
                cash += q
            elif ex > t:
                amount = q * growth ** (t - ex)
                decay = math.exp(-kappa * (ex - t))
                a += amount * (1 - decay) * alpha
                b += amount * decay
                c += amount * (1 - decay) * (1 - alpha)
        return a, b, c, cash

    def leg(knockout):
        if knockout and initial_hit:
            return (np.zeros((n, 2)), np.zeros((n, 2)),
                    np.stack((np.ones(n), np.zeros(n)), axis=1))
        f, y = np.ones(n), np.ones(n)
        df, dy = np.zeros(n), np.zeros(n)
        survival, dsurvival = np.ones(n), np.zeros(n)
        hit = np.zeros(n)
        for i in range(steps):
            log_f = np.log(f)
            cell = np.clip(np.searchsorted(nodes, log_f, side='right') - 1, 0, len(nodes) - 2)
            l2 = np.interp(log_f, nodes, surface[i])
            slope = (surface[i, cell + 1] - surface[i, cell]) / (nodes[cell + 1] - nodes[cell])
            slope = np.where((log_f > nodes[0]) & (log_f < nodes[-1]), slope, 0.)
            sigma = np.sqrt(l2) * np.exp(eta * drivers[:, i] / 2 - eta * eta * variance[i] / 4)
            dsigma = sigma * slope * df / (2 * l2 * f)
            m = rho * normals[:, i, 0] + orth * normals[:, i, 1]
            mu = log_f - sigma * sigma * dt[i] / 2 + sigma * math.sqrt(dt[i]) * m
            dmu = df / f + dsigma * (math.sqrt(dt[i]) * m - sigma * dt[i])
            s, ds = sigma * math.sqrt(dt[i]) * conditional_root, dsigma * math.sqrt(dt[i]) * conditional_root
            half = math.exp(-kappa * dt[i] / 2)
            noise = np.exp(nu * math.sqrt(dt[i]) * normals[:, i, 0] - nu * nu * dt[i] / 2)
            link = (1 - half) * alpha
            u = half * (half * y + (1 - half) * (alpha * f + 1 - alpha)) * noise + (1 - half) * (1 - alpha)
            du = half * (half * dy + (1 - half) * alpha * df) * noise
            a, b, c, cash = coefficients(times[i + 1])
            g = growth ** times[i + 1]
            monitored_b = b + cash if up else b
            monitored_q = a + monitored_b * link
            boundary = (market['barrier'] - monitored_b * u - c) / monitored_q
            dboundary = -monitored_b * du / monitored_q - boundary * g / monitored_q

            # The digital probability has no exercise boundary. Evaluate both
            # tails directly so rare-hit rebates survive CDF rounding near one.
            probability, dprobability, hit_probability = np.ones(n), np.zeros(n), np.zeros(n)
            if knockout and i + 1 in monitors:
                positive = boundary > 0
                cutoff, dcutoff = np.full(n, -np.inf), np.zeros(n)
                cutoff[positive] = (np.log(boundary[positive]) - mu[positive]) / s[positive]
                dcutoff[positive] = (dboundary[positive] / boundary[positive] - dmu[positive] - cutoff[positive] * ds[positive]) / s[positive]
                sign = 1. if up else -1.
                probability, dprobability = cdf(sign * cutoff), sign * density(cutoff) * dcutoff
                hit_probability = cdf(-sign * cutoff)

            if i == steps - 1:
                q = a + b * link
                constant, dconstant = b * u + c - market['strike'], b * du
                exercise = -constant / q
                dexercise = -dconstant / q - exercise * g / q
                lo, dlo = np.zeros(n), np.zeros(n)
                hi, dhi = np.full(n, np.inf), np.zeros(n)
                if call:
                    lo = np.maximum(exercise, 0.)
                    dlo = np.where(exercise > 0, dexercise, 0.)
                else:
                    hi, dhi = exercise, dexercise
                if knockout and i + 1 in monitors:
                    if up:
                        dhi = np.where(boundary < hi, dboundary, dhi)
                        hi = np.minimum(hi, boundary)
                    else:
                        dlo = np.where(boundary > lo, dboundary, dlo)
                        lo = np.maximum(lo, boundary)
                mean = np.exp(mu + s * s / 2)
                dmean = mean * (dmu + s * ds)

                def tail(cut, dcut):
                    finite = (cut > 0) & np.isfinite(cut)
                    z, dz = np.where(cut <= 0, np.inf, -np.inf), np.zeros(n)
                    z[finite] = (mu[finite] - np.log(cut[finite])) / s[finite]
                    dz[finite] = (dmu[finite] - dcut[finite] / cut[finite] - z[finite] * ds[finite]) / s[finite]
                    sign = 1. if call else -1.
                    probability = cdf(sign * z)
                    dprobability = sign * density(z) * dz
                    moment = mean * cdf(sign * (z + s))
                    dmoment = dmean * cdf(sign * (z + s)) + sign * mean * density(z + s) * (dz + ds)
                    return (sign * (q * moment + constant * probability),
                            sign * (g * moment + q * dmoment + dconstant * probability + constant * dprobability))

                first, last = (tail(lo, dlo), tail(hi, dhi)) if call else (tail(hi, dhi), tail(lo, dlo))
                value = np.where(hi > lo, first[0] - last[0], 0.)
                tangent = np.where(hi > lo, first[1] - last[1], 0.)
                payoff = np.stack((survival * value, dsurvival * value + survival * tangent), axis=1)
                tangent = dsurvival * probability + survival * dprobability
                surviving = np.stack((survival * probability, tangent), axis=1)
                hitting = np.stack((hit + survival * hit_probability, -tangent), axis=1)
                return payoff, surviving, hitting

            z = inverse_cdf(uniforms[:, i])
            dz = np.zeros(n)
            if knockout and i + 1 in monitors:
                sign = 1. if up else -1.
                active = probability > 0
                uniform = uniforms[:, i] if up else 1 - uniforms[:, i]
                p = uniform * probability
                # Zero numerical survival is an absorbing zero-weight path.
                # Positive probabilities are never floored or clipped.
                z[active] = sign * inverse_cdf(p[active])
                dz[active] = sign * uniform[active] * dprobability[active] / density(z[active])
                hit += survival * hit_probability
                dsurvival = dsurvival * probability + survival * dprobability
                survival *= probability
            f = np.exp(mu + s * z)
            df = f * (dmu + ds * z + s * dz)
            y, dy = u + link * f, du + link * df
        raise AssertionError('missing terminal step')

    ko, survival, hit = leg(True)
    payoff, inactive = (leg(False)[0] - ko, survival) if style == 'knock_in' else (ko, hit)
    value = notional * payoff + rebate * inactive
    return value * market['annual_discount'] ** market['payment_time']


def batch_means(case, market, *, seed=20261003, batches=32, pairs=8192):
    """Independent PCG64 batches; each sampling unit is an antithetic pair."""
    rng = np.random.Generator(np.random.PCG64(seed))
    steps = len(case['times']) - 1
    means = []
    for _ in range(batches):
        z = rng.standard_normal((pairs, steps, 3))
        u = rng.random((pairs, steps))
        a = path_values(case, market, z, u)
        b = path_values(case, market, -z, 1 - u)
        means.append(((a + b) / 2).mean(axis=0))
    return np.asarray(means)


def verify_fixture():
    """Regenerate every retained batch on CI; never overwrite reference data."""
    import json
    from pathlib import Path
    directory = Path(__file__).resolve().parents[2] / 'fixtures/stochastic-dividends'
    fixture = json.loads((directory / 'rough-survival-barrier-reference.json').read_text())
    market = json.loads((directory / fixture['market_contract_fixture']).read_text())
    sampling = fixture['sampling']
    assert sampling['rng'] == 'numpy.random.PCG64'
    for case in fixture['cases']:
        means = batch_means(case, market, seed=sampling['seed'], batches=sampling['batches'],
                            pairs=sampling['antithetic_pairs_per_batch'])
        np.testing.assert_allclose(means, case['batch_means'], rtol=0, atol=1e-9)
        for j, quantity in enumerate(('price', 'delta')):
            value = float(means[:, j].mean())
            se = float(means[:, j].std(ddof=1) / math.sqrt(len(means)))
            assert abs(value - case[quantity]) < 1e-9
            assert abs(se - case[quantity + '_se']) < 1e-9
            assert se < fixture['acceptance']['reference_' + quantity + '_se']
            print(json.dumps(dict(scope=fixture['scope'], case=case['id'], quantity=quantity,
                                  value=value, batch_se=se, **sampling)), flush=True)


if __name__ == '__main__':
    verify_fixture()
